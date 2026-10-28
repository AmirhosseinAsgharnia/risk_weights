import hashlib
import json

import numpy as np
import pytest

from learning.feasibility_audit import (
    BOUNDS_HISTORY, audit_rows, curated_dataset, relabel_dir, validate_admission,
)
from learning.feasibility_cutin import CutinConfig, THETA_BOUNDS_CUTIN
from learning.feasibility_dataset import rollouts_path, scenarios_path
from learning.feasibility_surrogate import fit_surrogate

THETA = CutinConfig().theta()


def _row(sid, theta=None, k=90, n=100, n_rollovers=0, **extra):
    return {"scenario_id": sid, "family": "cutin", "blocker_side": 1, "mode": "robust", "theta": theta or THETA,
            "k": k, "n": n, "n_collisions": 0, "n_rollovers": n_rollovers, "n_offroad": 0, "n_timeouts": n - k - n_rollovers,
            **extra}


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _dataset(root, rows):
    _write(scenarios_path("cutin", 1, root), rows)
    _write(rollouts_path("cutin", 1, root), [{"scenario_id": r["scenario_id"], "episode": e, "y": int(e < r["k"]),
                                             "theta": r["theta"]} for r in rows for e in range(r["n"])])


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_audit_categories_curation_and_originals_untouched(tmp_path):
    oob = {**THETA, "front_gap_m": 10.0}                       # valid under cutin-bounds-v0 only
    assert BOUNDS_HISTORY["cutin"]["cutin-bounds-v0"]["front_gap_m"][0] <= 10.0 < THETA_BOUNDS_CUTIN["front_gap_m"][0]
    rows = [_row("a"), _row("b", theta=oob), _row("c", k=40, n_rollovers=30),
            _row("d", k=95, rollover_event_model="physical_v2", theta_bounds_version="cutin-bounds-v1"),
            _row("e", k=60, n_rollovers=10)]
    _dataset(tmp_path, rows)
    before = (_sha(scenarios_path("cutin", 1, tmp_path)), _sha(rollouts_path("cutin", 1, tmp_path)))
    # a relabel for "e" that reproduced the legacy evaluation, none for "c"
    _write(relabel_dir("cutin", 1, tmp_path) / "scenarios.jsonl",
           [{**_row("e", k=88), "rollover_event_model": "physical_v2", "legacy_reproduced": True,
             "n_success_with_friction_circle_exceeded": 3}])
    _write(relabel_dir("cutin", 1, tmp_path) / "rollouts.jsonl",
           [{"scenario_id": "e", "episode": i, "y": int(i < 88), "theta": THETA} for i in range(100)])

    a = {r["scenario_id"]: r for r in audit_rows("cutin", 1, tmp_path)}
    assert a["a"]["label_category"] == "legacy_no_rollover" and a["a"]["included"]
    assert a["a"]["rollover_event_model"] == "legacy_v1" and "inferred" in a["a"]["rollover_event_model_source"]
    assert a["b"]["label_category"] == "out_of_bounds" and not a["b"]["included"] and "front_gap_m" in a["b"]["exclusion_reason"]
    assert a["b"]["theta"] == oob                                                    # never clamped
    assert a["c"]["label_category"] == "legacy_rollover_label" and not a["c"]["included"]
    assert a["d"]["label_category"] == "stamped" and a["d"]["theta_bounds_version_source"] == "stamped"
    assert a["e"]["label_category"] == "relabeled" and a["e"]["relabel"]["k"] == 88

    cur = curated_dataset("cutin", 1, tmp_path)
    assert {s["scenario_id"] for s in cur["scenarios"]} == {"a", "d", "e"}
    assert next(s for s in cur["scenarios"] if s["scenario_id"] == "e")["k"] == 88   # NEW labels, not the legacy 60
    assert sum(r["y"] for r in cur["rollouts"] if r["scenario_id"] == "e") == 88
    assert not {r["scenario_id"] for r in cur["rollouts"]} & {"b", "c"}
    assert cur["provenance"]["exclusion_reasons"] == {"out_of_bounds": 1, "legacy_rollover_label": 1}
    assert (_sha(scenarios_path("cutin", 1, tmp_path)), _sha(rollouts_path("cutin", 1, tmp_path))) == before


def test_unreproducible_relabel_is_excluded(tmp_path):
    _dataset(tmp_path, [_row("e", k=60, n_rollovers=10)])
    _write(relabel_dir("cutin", 1, tmp_path) / "scenarios.jsonl",
           [{**_row("e", k=88), "rollover_event_model": "physical_v2", "legacy_reproduced": False}])
    a = audit_rows("cutin", 1, tmp_path)[0]
    assert a["label_category"] == "relabel_not_reproducible" and not a["included"]


def _synthetic(tmp_path, n_scen=30):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_scen):
        th = {k: float(lo + rng.random() * (hi - lo)) for k, (lo, hi) in THETA_BOUNDS_CUTIN.items()}
        x = (th["front_gap_m"] - 25.0) / 75.0                        # solvable iff the front gap is large
        k = int(round(100 * (0.98 if x > 0.5 else 0.2)))
        rows.append(_row(f"s{i:02d}", theta=th, k=k))
    _dataset(tmp_path, rows)
    return rows


def test_surrogate_provenance_and_admission_validation(tmp_path):
    rows = _synthetic(tmp_path)
    cur = curated_dataset("cutin", 1, tmp_path)
    art = fit_surrogate("cutin", 1, cur["rollouts"], THETA_BOUNDS_CUTIN, n_bootstrap=5, provenance=cur["provenance"])
    assert art.provenance["n_included"] == len(rows) and art.provenance["theta_bounds_version"] == "cutin-bounds-v1"
    v = validate_admission("cutin", 1, tmp_path, n_splits=5, n_bootstrap=10)
    sc = v["status_counts"]
    assert v["n_scenarios"] == len(rows) and sum(sc.values()) == len(rows)
    assert v["false_admissions"] <= sc["admitted"] and v["false_rejections"] <= sc["rejected"]
    assert v["n_demonstrated"] == sum(r["k"] >= 98 for r in rows)
    assert all(r["lower"] <= r["p_hat"] + 1e-9 or r["status"] != "admitted" for r in v["per_scenario"])
    assert 0.0 <= v["interval_coverage_of_empirical_rate"] <= 1.0
