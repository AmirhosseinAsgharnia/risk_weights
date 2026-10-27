"""
The feasibility surrogate p_hat(theta) -- fit on the accumulated ROLLOUT-
level binary outcomes (see learning.feasibility_dataset), one model per
(scenario_family, blocker_side) -- never a single model across both sides,
and never trained on the per-scenario fractional (k_i/n_i) rate directly
(see module docstring below for why).

Why rollout-level, not one row per theta: standard scikit-learn classifiers
want binary labels, not a fractional target -- but every target here is
genuinely a binomial observation (k_i successes out of n_i rollouts at the
SAME theta), not an exact probability. Expanding each theta into its n_i
individual 0/1 rollout rows and grouping by scenario_id for
cross-validation/bootstrapping keeps that binomial structure honest (a
theta with n_i=100 correctly outweighs one with n_i=10 through sheer row
count, rather than needing an ad hoc sample_weight) while staying squarely
inside what these classifiers actually expect.

Every prediction distinguishes two different kinds of uncertainty:
  - rollout/binomial uncertainty at a KNOWN theta -- that's the dataset's own
    Wilson CI on (k_i, n_i), already computed by learning.eval_batch, no
    surrogate needed.
  - surrogate EPISTEMIC uncertainty from limited theta coverage -- estimated
    here via a seeded bootstrap ensemble over whole scenario_id groups (never
    individual rollouts -- resampling rollouts alone would understate this,
    since it never removes/duplicates an entire theta's worth of evidence).
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class ConstantClassifier:
    """Degenerate fallback for a dataset (or bootstrap resample) where every
    observed rollout is the same class -- every real classifier here
    (including sklearn's) refuses to fit that, correctly, since there is
    nothing to discriminate. Rather than crash (a single-class dataset is a
    perfectly legitimate, if uninformative, thing to encounter early in a
    campaign, or for a scenario family that's uniformly easy/hard), this
    just predicts the observed rate everywhere -- an honest "we have no
    evidence of variation" answer, not a failure. fit/predict_proba only;
    never registered with sklearn's own estimator machinery."""

    def __init__(self):
        self.rate = 0.5

    def fit(self, X, y):
        self.rate = float(np.clip(np.mean(y), 0.0, 1.0))
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.rate), np.full(n, self.rate)])


def _candidate_models() -> dict[str, object]:
    """Sensible, lightweight, non-neural-network candidates (per this
    pipeline's own requirement not to reach for a neural net automatically
    on a small tabular dataset) -- a regularized logistic baseline with
    bounded-degree polynomial terms, an extra-trees ensemble, and histogram
    gradient boosting. Fresh instances each call (these are stateful once
    fit)."""
    return {
        "logistic_poly2": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3),
        "hist_gb": HistGradientBoostingClassifier(max_depth=4, min_samples_leaf=5),
    }


@dataclass(frozen=True)
class TrainingRow:
    """One rollout, expanded to feature/label/group form. theta values are
    ALREADY normalized to [0, 1] per-field (see normalize_theta) -- the
    persisted artifact keeps theta_bounds so a caller predicting on a new,
    raw theta gets normalized identically at inference time."""
    scenario_id: str
    theta_normalized: np.ndarray
    y: int


def normalize_theta(theta: dict[str, float], bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    """theta dict -> a fixed-order (sorted field name) vector in
    approximately [0, 1] per dimension -- NOT clipped, so a query outside
    the training bounds produces a value outside [0, 1] rather than being
    silently folded back in; see extrapolation_warning for how that's
    surfaced instead of hidden."""
    names = sorted(bounds)
    return np.array([(theta[name] - bounds[name][0]) / (bounds[name][1] - bounds[name][0])
                      for name in names], dtype=float)


def _build_training_rows(rollout_records: list[dict], bounds: dict[str, tuple[float, float]]) -> list[TrainingRow]:
    return [TrainingRow(scenario_id=r["scenario_id"], theta_normalized=normalize_theta(r["theta"], bounds),
                         y=int(r["y"]))
            for r in rollout_records]


def _grouped_oof_predictions(model_factory, rows: list[TrainingRow], n_splits: int) -> np.ndarray:
    """Out-of-fold predicted probabilities for every row, via GroupKFold by
    scenario_id -- a row is only ever scored by a model that never saw its
    theta's OTHER rollouts during fitting, so this is a fair estimate of
    generalization to a new theta, not just a new rollout of an already-seen one."""
    X = np.stack([r.theta_normalized for r in rows])
    y = np.array([r.y for r in rows])
    groups = np.array([r.scenario_id for r in rows])

    oof = np.full(len(rows), np.nan)
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(X, y, groups):
        # Grouped by construction -- assert it, since a silent leak here would invalidate every
        # metric downstream (this is exactly the mistake the module docstring exists to prevent).
        assert not set(groups[train_idx]) & set(groups[test_idx]), \
            "GroupKFold leaked a scenario_id across train/validation -- this must never happen"
        model = model_factory()
        if len(set(y[train_idx])) < 2:
            # A degenerate fold (all-success or all-failure training split, possible at small N) --
            # predict the training fold's empirical rate rather than fitting a classifier on one class.
            oof[test_idx] = y[train_idx].mean()
            continue
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return oof


@dataclass
class ModelReport:
    name: str
    brier: float
    log_loss: float
    roc_auc: float | None   # None when the dataset doesn't have both classes represented
    calibration: dict       # {"prob_true": [...], "prob_pred": [...]} from sklearn.calibration_curve
    n_rows: int
    n_scenarios: int


def evaluate_candidates(rollout_records: list[dict], bounds: dict[str, tuple[float, float]],
                         n_splits: int = 5) -> list[ModelReport]:
    """Grouped-CV comparison of every candidate in _candidate_models(),
    sorted best (lowest Brier score) first. Requires at least 2 distinct
    scenario_id groups (GroupKFold's own minimum); n_splits is clamped down
    to the number of available groups if fewer than requested."""
    rows = _build_training_rows(rollout_records, bounds)
    n_groups = len(set(r.scenario_id for r in rows))
    if n_groups < 2:
        raise ValueError(f"need rollouts from at least 2 distinct scenarios to cross-validate, got {n_groups}")
    n_splits = min(n_splits, n_groups)

    y = np.array([r.y for r in rows])
    both_classes_present = len(set(y)) == 2

    if not both_classes_present:
        # Every rollout in the dataset is the same class -- see ConstantClassifier's own docstring
        # for why this is a legitimate, non-crashing outcome rather than an error. Brier/log-loss
        # against the constant rate are still meaningful (they're just 0 and near-0 respectively,
        # since the "prediction" exactly matches every observation by construction); ROC-AUC is
        # undefined with only one class present, so left None like the normal per-candidate path does.
        rate = float(y.mean())
        oof = np.full(len(rows), rate)
        oof_clipped = np.clip(oof, 1e-6, 1 - 1e-6)
        return [ModelReport(
            name="constant", brier=float(brier_score_loss(y, oof)),
            log_loss=float(log_loss(y, oof_clipped, labels=[0, 1])), roc_auc=None,
            calibration={"prob_true": [rate], "prob_pred": [rate]},
            n_rows=len(rows), n_scenarios=n_groups,
        )]

    reports = []
    for name, _ in _candidate_models().items():
        oof = _grouped_oof_predictions(lambda n=name: _candidate_models()[n], rows, n_splits)
        oof_clipped = np.clip(oof, 1e-6, 1 - 1e-6)   # log_loss is undefined at exactly 0/1
        prob_true, prob_pred = calibration_curve(y, oof, n_bins=min(10, n_groups), strategy="quantile")
        reports.append(ModelReport(
            name=name,
            brier=float(brier_score_loss(y, oof)),
            log_loss=float(log_loss(y, oof_clipped, labels=[0, 1])),
            roc_auc=float(roc_auc_score(y, oof)) if both_classes_present else None,
            calibration={"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()},
            n_rows=len(rows), n_scenarios=n_groups,
        ))
    return sorted(reports, key=lambda r: r.brier)


def _bootstrap_ensemble(model_name: str, rows: list[TrainingRow], n_bootstrap: int, seed: int) -> list:
    """Refit `model_name` n_bootstrap times, each on a resample of WHOLE
    scenario_id groups (with replacement) -- never individual rollouts (see
    module docstring: that would understate epistemic uncertainty, since it
    can never fully drop or duplicate a theta's entire body of evidence).
    Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    scenario_ids = sorted(set(r.scenario_id for r in rows))
    by_scenario: dict[str, list[TrainingRow]] = {sid: [] for sid in scenario_ids}
    for r in rows:
        by_scenario[r.scenario_id].append(r)

    models = []
    for _ in range(n_bootstrap):
        chosen = rng.choice(scenario_ids, size=len(scenario_ids), replace=True)
        resampled_rows = [row for sid in chosen for row in by_scenario[sid]]
        X = np.stack([r.theta_normalized for r in resampled_rows])
        y = np.array([r.y for r in resampled_rows])
        if model_name != "constant" and len(set(y)) < 2:
            continue   # degenerate resample (all one class) -- skip rather than fit a broken model.
                       # ConstantClassifier itself has no such restriction (see its own docstring) --
                       # a degenerate resample is exactly the case it exists for.
        model = ConstantClassifier() if model_name == "constant" else _candidate_models()[model_name]
        model.fit(X, y)
        models.append(model)
    return models


@dataclass
class SurrogateArtifact:
    """Everything persisted for one (family, blocker_side) surrogate --
    saved/loaded as one joblib file (see save/load below)."""
    family: str
    blocker_side: int
    model_name: str
    model: object
    bootstrap_models: list
    theta_bounds: dict[str, tuple[float, float]]
    n_scenarios: int
    n_rollouts: int
    report: ModelReport
    preliminary: bool   # True whenever n_scenarios is small enough that this should not be
                         # presented as a finished result -- see PRELIMINARY_SCENARIO_THRESHOLD

    def predict(self, theta: dict[str, float]) -> dict:
        """p_hat, a bootstrap-derived epistemic interval, and an
        extrapolation/sparse-support warning for one query theta."""
        x = normalize_theta(theta, self.theta_bounds).reshape(1, -1)
        p_hat = float(self.model.predict_proba(x)[0, 1])

        boot_preds = np.array([float(m.predict_proba(x)[0, 1]) for m in self.bootstrap_models])
        lo, hi = (float(np.percentile(boot_preds, 2.5)), float(np.percentile(boot_preds, 97.5))
                  ) if len(boot_preds) > 0 else (None, None)

        warnings_out = []
        if self.preliminary:
            warnings_out.append(
                f"PRELIMINARY: this surrogate was fit on only {self.n_scenarios} scenarios -- "
                f"treat p_hat as a rough pipeline check, not a validated estimate.")
        if np.any(x < 0) or np.any(x > 1):
            warnings_out.append("EXTRAPOLATION: theta falls outside the training data's bounds "
                                 "for at least one field -- p_hat is unreliable here.")
        return {"p_hat": max(0.0, min(1.0, p_hat)), "epistemic_interval": (lo, hi), "warnings": warnings_out}

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str) -> "SurrogateArtifact":
        return joblib.load(path)


PRELIMINARY_SCENARIO_THRESHOLD = 50   # below this many distinct thetas, every artifact is marked
                                       # preliminary -- see the module/class docstrings; not a
                                       # claim that 50 is "enough," just a floor below which a
                                       # smooth-looking plot must not be mistaken for proof.


def fit_surrogate(
        family: str, blocker_side: int, rollout_records: list[dict], bounds: dict[str, tuple[float, float]],
        *, n_splits: int = 5, n_bootstrap: int = 200, bootstrap_seed: int = 0,
) -> SurrogateArtifact:
    """Compare candidates (evaluate_candidates), refit the best one on the
    FULL dataset, build its bootstrap ensemble, and return one persistable
    SurrogateArtifact. Still produces an artifact -- explicitly marked
    preliminary -- even when the dataset is too small to be a credible
    model (per this pipeline's own "don't withhold pipeline-testing output,
    but never present a smooth plot as proof" requirement)."""
    reports = evaluate_candidates(rollout_records, bounds, n_splits=n_splits)
    best = reports[0]
    rows = _build_training_rows(rollout_records, bounds)

    X = np.stack([r.theta_normalized for r in rows])
    y = np.array([r.y for r in rows])
    final_model = ConstantClassifier() if best.name == "constant" else _candidate_models()[best.name]
    final_model.fit(X, y)

    n_scenarios = len(set(r.scenario_id for r in rows))
    preliminary = n_scenarios < PRELIMINARY_SCENARIO_THRESHOLD
    if best.name == "constant":
        warnings.warn(f"fit_surrogate({family}, side={blocker_side}): every rollout in the dataset "
                       f"is the same outcome -- falling back to a constant predictor (see "
                       f"ConstantClassifier). This is not a fit failure, but there is no evidence "
                       f"of variation across theta yet.")
    if preliminary:
        warnings.warn(f"fit_surrogate({family}, side={blocker_side}): only {n_scenarios} scenarios -- "
                       f"artifact marked preliminary (see SurrogateArtifact.preliminary).")

    bootstrap_models = _bootstrap_ensemble(best.name, rows, n_bootstrap, bootstrap_seed)

    return SurrogateArtifact(
        family=family, blocker_side=blocker_side, model_name=best.name, model=final_model,
        bootstrap_models=bootstrap_models, theta_bounds=dict(bounds), n_scenarios=n_scenarios,
        n_rollouts=len(rows), report=best, preliminary=preliminary,
    )
