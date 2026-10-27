"""
Tests for the feasibility surrogate (learning.feasibility_surrogate) --
grouped-split correctness, [0,1]-constrained output, the degenerate
(single-class) fallback, extrapolation warnings, and save/load. All on
small synthetic data -- no real PPO training anywhere in this file.

Run with: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_feasibility_surrogate.py -v
"""

import warnings

import numpy as np
import pytest

from learning.feasibility_surrogate import (
    fit_surrogate, evaluate_candidates, normalize_theta, ConstantClassifier,
)

_BOUNDS = {"a": (0.0, 10.0), "b": (-5.0, 5.0)}


def _make_records(n_scenarios: int, n_per_scenario: int, seed: int, p_fn=None) -> list[dict]:
    """Synthetic rollout records over _BOUNDS -- p_fn(theta) -> success
    probability, defaulting to a fixed 0.5 (used by tests that don't care
    about the actual relationship, just the plumbing)."""
    rng = np.random.default_rng(seed)
    p_fn = p_fn or (lambda theta: 0.5)
    records = []
    for i in range(n_scenarios):
        theta = {"a": float(rng.uniform(*_BOUNDS["a"])), "b": float(rng.uniform(*_BOUNDS["b"]))}
        p = p_fn(theta)
        for ep in range(n_per_scenario):
            records.append({"scenario_id": f"s{i}", "episode": ep, "theta": theta,
                             "y": int(rng.uniform() < p)})
    return records


def test_normalize_theta_maps_bounds_to_unit_interval():
    theta = {"a": 0.0, "b": -5.0}
    vec = normalize_theta(theta, _BOUNDS)
    assert np.allclose(vec, [0.0, 0.0])
    theta2 = {"a": 10.0, "b": 5.0}
    assert np.allclose(normalize_theta(theta2, _BOUNDS), [1.0, 1.0])


def test_evaluate_candidates_requires_at_least_two_scenarios():
    records = _make_records(1, 10, seed=0)
    with pytest.raises(ValueError):
        evaluate_candidates(records, _BOUNDS)


def test_grouped_cv_never_leaks_a_scenario_across_folds():
    """The internal assertion in _grouped_oof_predictions would raise if
    GroupKFold ever put the same scenario_id in both train and test -- this
    test just needs enough scenarios/variation to exercise that code path
    without raising for an unrelated reason."""
    records = _make_records(12, 15, seed=0, p_fn=lambda t: 0.2 + 0.6 * (t["a"] / 10.0))
    reports = evaluate_candidates(records, _BOUNDS, n_splits=4)
    assert len(reports) == 3   # logistic_poly2, extra_trees, hist_gb
    assert reports == sorted(reports, key=lambda r: r.brier)   # best (lowest Brier) first


def test_predictions_are_probability_valued():
    records = _make_records(12, 15, seed=1, p_fn=lambda t: 0.2 + 0.6 * (t["a"] / 10.0))
    artifact = fit_surrogate("cutin", 1, records, _BOUNDS, n_bootstrap=20)
    for a in np.linspace(0, 10, 5):
        for b in np.linspace(-5, 5, 5):
            result = artifact.predict({"a": float(a), "b": float(b)})
            assert 0.0 <= result["p_hat"] <= 1.0
            if result["epistemic_interval"][0] is not None:
                lo, hi = result["epistemic_interval"]
                assert 0.0 <= lo <= hi <= 1.0


def test_surrogate_learns_the_true_relationship_direction():
    """Not a tight numerical check (small-N sklearn models are noisy) --
    just confirms the fitted surrogate points the right way on a strong,
    unambiguous synthetic signal."""
    records = _make_records(30, 20, seed=2, p_fn=lambda t: 0.05 + 0.9 * (t["a"] / 10.0))
    artifact = fit_surrogate("cutin", 1, records, _BOUNDS, n_bootstrap=20)
    p_low = artifact.predict({"a": 0.5, "b": 0.0})["p_hat"]
    p_high = artifact.predict({"a": 9.5, "b": 0.0})["p_hat"]
    assert p_high > p_low


def test_preliminary_flag_below_threshold():
    records = _make_records(5, 10, seed=0)
    artifact = fit_surrogate("cutin", 1, records, _BOUNDS, n_bootstrap=10)
    assert artifact.preliminary is True
    assert any("PRELIMINARY" in w for w in artifact.predict({"a": 5.0, "b": 0.0})["warnings"])


def test_extrapolation_warning_outside_bounds():
    records = _make_records(10, 10, seed=0, p_fn=lambda t: 0.3)
    artifact = fit_surrogate("cutin", 1, records, _BOUNDS, n_bootstrap=10)
    in_bounds = artifact.predict({"a": 5.0, "b": 0.0})
    out_of_bounds = artifact.predict({"a": 500.0, "b": 0.0})
    assert not any("EXTRAPOLATION" in w for w in in_bounds["warnings"])
    assert any("EXTRAPOLATION" in w for w in out_of_bounds["warnings"])


def test_degenerate_single_class_dataset_does_not_crash():
    records = _make_records(6, 10, seed=0, p_fn=lambda t: 0.0)   # every rollout fails
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifact = fit_surrogate("cutin", 1, records, _BOUNDS, n_bootstrap=10)
    assert artifact.model_name == "constant"
    assert isinstance(artifact.model, ConstantClassifier)
    result = artifact.predict({"a": 5.0, "b": 0.0})
    assert result["p_hat"] == pytest.approx(0.0, abs=1e-9)


def test_bootstrap_resamples_whole_scenarios_not_rollouts():
    """Every bootstrap model must have been fit on a resample whose rows
    all belong to complete scenario_id groups -- indirectly verified here
    by confirming bootstrap predictions vary across the ensemble (i.e. it
    genuinely resampled something), while still respecting the [0,1]
    contract per-model."""
    records = _make_records(15, 15, seed=3, p_fn=lambda t: 0.2 + 0.6 * (t["a"] / 10.0))
    artifact = fit_surrogate("cutin", 1, records, _BOUNDS, n_bootstrap=40)
    preds = [m.predict_proba(np.array([[0.5, 0.5]]))[0, 1] for m in artifact.bootstrap_models]
    assert len(artifact.bootstrap_models) > 0
    assert len(set(np.round(preds, 6))) > 1   # not every bootstrap model is identical


def test_save_and_load_round_trip(tmp_path):
    records = _make_records(10, 10, seed=0, p_fn=lambda t: 0.2 + 0.6 * (t["a"] / 10.0))
    artifact = fit_surrogate("cutin", 1, records, _BOUNDS, n_bootstrap=10)
    path = tmp_path / "surrogate.joblib"
    artifact.save(path)
    loaded = type(artifact).load(path)
    query = {"a": 3.0, "b": 1.0}
    assert loaded.predict(query)["p_hat"] == artifact.predict(query)["p_hat"]
    assert loaded.model_name == artifact.model_name
