import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from learning.closed_loop import ScenarioSpec, Stratum
from learning.feasibility_cutin import THETA_BOUNDS_CUTIN, CutinConfig
from learning.feasibility_sampling import sample_cutin
from learning.feasibility_surrogate import fit_surrogate
from learning.minimax_feasibility import (
    ADMITTED, REJECTED, UNCERTAIN, FeasibilityScreen, ScreeningConfig, demonstrated,
    demonstrated_scenarios_from_dataset,
)

CUTIN1 = Stratum("cutin", 1)


def _spec(**theta):
    return ScenarioSpec.make(CUTIN1, {**CutinConfig().theta(), **theta})


class _StubArtifact:
    def __init__(self, p, lo, hi, warns=()):
        self.p, self.lo, self.hi, self.warns = p, lo, hi, list(warns)

    def predict(self, theta):
        return {"p_hat": self.p, "epistemic_interval": (self.lo, self.hi), "warnings": list(self.warns)}


def _screen(policy, art, tau=0.8):
    s = FeasibilityScreen(ScreeningConfig(policy=policy, tau=tau))
    s.artifacts[CUTIN1.label] = art
    s.identities[CUTIN1.label] = {"sha256": "stub"}
    return s


@pytest.mark.parametrize("policy,p,lo,hi,expected", [
    ("point", 0.9, 0.85, 0.95, ADMITTED),
    ("conservative", 0.9, 0.85, 0.95, ADMITTED),
    ("point", 0.85, 0.6, 0.95, ADMITTED),        # point admits on p_hat
    ("conservative", 0.85, 0.6, 0.95, UNCERTAIN),  # ...conservative does not: interval straddles tau
    ("conservative", 0.7, 0.6, 0.85, UNCERTAIN),
    ("point", 0.3, 0.1, 0.5, REJECTED),
    ("conservative", 0.3, 0.1, 0.5, REJECTED),
])
def test_screening_policies(policy, p, lo, hi, expected):
    d = _screen(policy, _StubArtifact(p, lo, hi)).classify(_spec())
    assert d.status == expected and d.evidence == "surrogate"


def test_extrapolation_and_missing_interval_are_never_admitted():
    d = _screen("point", _StubArtifact(0.99, 0.95, 1.0, ["EXTRAPOLATION: outside"])).classify(_spec())
    assert d.status == UNCERTAIN
    d = _screen("conservative", _StubArtifact(0.99, None, None)).classify(_spec())
    assert d.status == UNCERTAIN and any("no bootstrap interval" in w for w in d.warnings)


def test_preliminary_warning_is_carried_into_the_decision():
    d = _screen("conservative", _StubArtifact(0.99, 0.9, 1.0, ["PRELIMINARY: only 24"])).classify(_spec())
    assert d.status == ADMITTED and any(w.startswith("PRELIMINARY") for w in d.warnings)


def test_oracle_evidence_overrides_surrogate_and_failure_is_unverified_not_rejected():
    s = _screen("conservative", _StubArtifact(0.1, 0.0, 0.2))
    spec = _spec()
    assert s.classify(spec).status == REJECTED
    assert s.record_oracle_evidence(spec, 100, 100) == ADMITTED
    d = s.classify(spec)
    assert d.status == ADMITTED and d.evidence == "oracle"
    spec2 = _spec(road_mu=0.5)
    assert s.record_oracle_evidence(spec2, 10, 100) == UNCERTAIN
    assert s.classify(spec2).status == UNCERTAIN   # unverified -- never "rejected"/"infeasible"


def test_demonstrated_rule_is_wilson_lower_bound():
    assert demonstrated(100, 100, 0.8) and not demonstrated(83, 100, 0.8) and not demonstrated(0, 0, 0.5)


def test_unscreened_policy_admits_everything():
    s = FeasibilityScreen(ScreeningConfig(policy="none"))
    assert s.classify(_spec()).status == ADMITTED and s.classify(_spec()).evidence == "unscreened"


# -- real SurrogateArtifact files --
def _fit(tmp_path: Path, family="cutin", side=1, single_class=False, name="s.joblib", bounds=THETA_BOUNDS_CUTIN):
    cfgs, _ = sample_cutin(12, 1, seed=0)
    rows = []
    for c in cfgs:
        for ep in range(4):
            y = 1 if single_class else int(c.road_mu > 0.7)
            rows.append({"scenario_id": c.scenario_id(), "theta": c.theta(), "y": y})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        art = fit_surrogate(family, side, rows, dict(bounds), n_splits=2, n_bootstrap=5)
    path = tmp_path / name
    art.save(path)
    return path


def test_real_artifact_load_identity_and_classification(tmp_path):
    path = _fit(tmp_path)
    s = FeasibilityScreen(ScreeningConfig(policy="point", tau=0.5), {CUTIN1.label: str(path)})
    ident = s.identities[CUTIN1.label]
    assert len(ident["sha256"]) == 64 and ident["preliminary"] is True
    d = s.classify(_spec(road_mu=1.0))
    assert d.surrogate_sha256 == ident["sha256"] and d.p_hat is not None
    assert any(w.startswith("PRELIMINARY") for w in d.warnings)


def test_missing_mismatched_and_preliminary_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        FeasibilityScreen(ScreeningConfig(), {CUTIN1.label: str(tmp_path / "nope.joblib")})
    path = _fit(tmp_path)
    with pytest.raises(ValueError, match="never mixed"):
        FeasibilityScreen(ScreeningConfig(), {"cutin_sideneg1": str(path)})
    with pytest.raises(ValueError, match="PRELIMINARY"):
        FeasibilityScreen(ScreeningConfig(allow_preliminary=False), {CUTIN1.label: str(path)})
    other = dict(THETA_BOUNDS_CUTIN, road_mu=(0.3, 1.0))
    p2 = _fit(tmp_path, name="b.joblib", bounds=other)
    with pytest.raises(ValueError, match="bounds"):
        FeasibilityScreen(ScreeningConfig(), {CUTIN1.label: str(p2)})
    with pytest.raises(KeyError):
        FeasibilityScreen(ScreeningConfig()).classify(_spec())


def test_single_class_surrogate_fallback(tmp_path):
    path = _fit(tmp_path, single_class=True)
    s = FeasibilityScreen(ScreeningConfig(policy="conservative", tau=0.8), {CUTIN1.label: str(path)})
    d = s.classify(_spec())
    assert d.p_hat == pytest.approx(1.0) and d.status == ADMITTED


def test_demonstrated_scenarios_from_existing_dataset():
    path = Path("artifacts/feasibility/datasets/cutin/side_pos1/scenarios.jsonl")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    specs = demonstrated_scenarios_from_dataset(rows, 0.8)
    expected = sum(demonstrated(r["k"], r["n"], 0.8) for r in rows)
    assert len(specs) == expected > 0
    assert all(s.stratum == CUTIN1 for s in specs)
    assert {s.scenario_id for s in specs} <= {r["scenario_id"] for r in rows}   # ids match the PPO dataset
