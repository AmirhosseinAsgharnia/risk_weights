import numpy as np
import pytest

from learning.closed_loop import ScenarioSpec, Stratum, build_controller
from learning.constrained_adversary import ConstrainedScenarioAdversary
from learning.constrained_pipeline import ConstrainedExperiment, normalize_config
from learning.minimax_adversary import AdversaryConfig
from learning.minimax_feasibility import FeasibilityScreen, ScreeningConfig
from learning.minimax_store import ExperimentStore
from learning.paralo_mpc import pw_from_unit
from learning.physical_eval import DEFAULT_CONSTRAINTS, constraints_from_config, deterministic_part, run_physical_episode

CUTIN1 = Stratum("cutin", 1)


def _cfg(tmp_path, **over):
    raw = {"experiment_id": "t", "artifacts_root": str(tmp_path), "strata": [{"family": "cutin", "blocker_side": 1}],
           "screening": {"policy": "none"}, "realization_seeds": [1000],
           "master": {"n_init": 1, "n_starts": 1, "maxfev_per_start": 2},
           "exchange": {"max_iterations": 2},
           "adversary": {"n_global": 2, "max_global_evals": 1, "local_top": 0, "n_local": 0},
           "heldout": {"sampling_seed": 77, "n_per_stratum": 1, "realization_seeds": [5000]},
           "n_workers": 4}
    raw.update(over)
    return normalize_config(raw)


def test_config_has_no_evaluation_weights_and_explicit_constraints(tmp_path):
    c = _cfg(tmp_path)
    assert "cost" not in c and c["controller"]["name"] == "paralo_mpc"
    assert [x["name"] for x in c["constraints"]] == [x["name"] for x in DEFAULT_CONSTRAINTS]
    assert all(x["status"] == "provisional" for x in c["constraints"])
    c2 = _cfg(tmp_path, constraints=[{"name": "collision", "metric": "collision_rate", "threshold": 0.0,
                                      "units": "1/episode"}])
    assert len(c2["constraints"]) == 1   # a constraint list replaces the defaults; never merged silently
    with pytest.raises(AssertionError):
        _cfg(tmp_path, controller={"name": "mpc"})


def test_inadmissible_scenarios_are_never_simulated():
    class Stub:
        def predict(self, theta):
            return {"p_hat": 0.7, "epistemic_interval": (0.55, 0.85), "warnings": []}   # straddles tau -> UNCERTAIN
    screen = FeasibilityScreen(ScreeningConfig(policy="conservative", tau=0.8))
    screen.artifacts[CUTIN1.label], screen.identities[CUTIN1.label] = Stub(), {"sha256": "stub"}
    calls = []
    adv = ConstrainedScenarioAdversary([CUTIN1], screen, lambda u, s: calls.append(s), constraints_from_config(
        DEFAULT_CONSTRAINTS), AdversaryConfig(n_global=4, max_global_evals=4), 0)
    res = adv.propose(np.zeros(7), 0)
    assert res.performance is None and res.constraints == {} and calls == []


def test_end_to_end_optimize_interrupt_resume_compare(tmp_path):
    full_cfg = _cfg(tmp_path, experiment_id="full")
    full = ConstrainedExperiment(full_cfg, ExperimentStore(tmp_path, "full"))
    full.store.init_config(full_cfg)
    full.write_manifest()
    res_full = full.run()
    assert res_full["status"] in ("converged_feasible", "no_feasible_found", "budget_exhausted")
    assert res_full["status"] != "converged_feasible" or res_full["feasible_on_final_set"]

    cut_cfg = _cfg(tmp_path, experiment_id="cut", exchange={"max_iterations": 1})
    store = ExperimentStore(tmp_path, "cut")
    store.init_config(cut_cfg)
    exp = ConstrainedExperiment(cut_cfg, store)
    exp.write_manifest()
    exp.run()
    store.append("adversarial_candidates", {"iteration": 1, "stray": True})
    resumed_cfg = _cfg(tmp_path, experiment_id="cut", exchange={"max_iterations": 2})
    store.check_config(resumed_cfg)
    exp2 = ConstrainedExperiment(resumed_cfg, store)
    exp2.check_manifest()
    res = exp2.run(resume=True)
    its, its_full = store.read_log("iterations"), full.store.read_log("iterations")
    assert [r["iteration"] for r in its] == list(range(len(its)))
    assert [r["w"] for r in its] == [r["w"] for r in its_full] and res["status"] == res_full["status"]
    assert not any(r.get("stray") for r in store.read_log("adversarial_candidates"))
    assert exp2.store.cache_hits > 0

    # a cached episode equals a direct re-simulation
    s0 = ScenarioSpec.from_dict(store.read_json("checkpoints/state.json")["scenarios"][0])
    w = pw_from_unit(np.array(its[0]["w"]))
    cached = exp2.episode_records(exp2.controller_spec, w, [s0], [1000])[(s0.key, 1000)]
    direct = run_physical_episode(build_controller(exp2.controller_spec, w), s0, 1000)
    assert deterministic_part(cached) == deterministic_part(direct)

    out = exp2.compare("heldout", resumed_cfg["heldout"])
    labels = [r["label"] for r in out["rows"]]
    assert labels[:3] == ["original_mpc_legacy_events", "original_mpc", "paralo_mpc_w0"]
    assert any(l.startswith("paralo_mpc_best") for l in labels)
    ev = {r["label"]: (r["event_model"], r["primary_comparison"]) for r in out["rows"]}
    assert ev["original_mpc_legacy_events"] == ("legacy_v1", False) and ev["original_mpc"] == ("physical_v2", True)
    assert out["versions"]["rollover_event_model"] == "physical_v2"
    assert not set(out["pool_keys"]) & {r["key"] for r in store.read_log("adversarial_candidates")}
    with pytest.raises(AssertionError, match="overlap"):
        exp2.compare("bad", {**resumed_cfg["heldout"], "realization_seeds": [1000]})
