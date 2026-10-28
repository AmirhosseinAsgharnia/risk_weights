import json

import numpy as np
import pytest

from learning.closed_loop import EVALUATOR_VERSION, EvalCostConfig, ScenarioSpec, Stratum
from learning.feasibility_cutin import CutinConfig, THETA_BOUNDS_CUTIN
from learning.feasibility_sandwich import THETA_BOUNDS_SANDWICH
from learning.minimax_adversary import AdversaryConfig, ScenarioAdversary
from learning.minimax_feasibility import ADMITTED, FeasibilityScreen, ScreeningConfig, UNCERTAIN
from learning.minimax_oracle import OracleRefinementConfig, OracleRefiner
from learning.minimax_pipeline import Experiment, normalize_config
from learning.minimax_store import ExperimentStore, canonical_hash, episode_cache_key, scientific_config
from learning.mpc import MPCWeights

CUTIN1, SAND_NEG = Stratum("cutin", 1), Stratum("sandwich", -1)


# -- store / cache keys --
def test_cache_key_covers_every_cost_affecting_input():
    base = dict(controller_spec={"name": "mpc", "config": {"horizon_s": 3.0}}, w=[1.0, 2.0], scenario_key="a/b",
                realization_seed=1, cost_config={"alpha": 1}, evaluator_version="v1", schema_version=1)
    k0 = episode_cache_key(**base)
    for field, new in [("controller_spec", {"name": "mpc", "config": {"horizon_s": 2.0}}), ("w", [1.0, 2.0 + 1e-15]),
                       ("scenario_key", "a/c"), ("realization_seed", 2), ("cost_config", {"alpha": 2}),
                       ("evaluator_version", "v2"), ("schema_version", 2)]:
        assert episode_cache_key(**{**base, field: new}) != k0, field
    assert episode_cache_key(**base) == k0


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


def test_scientific_hash_ignores_budgets_and_workers_but_not_science(tmp_path):
    c = _cfg(tmp_path)
    h = canonical_hash(scientific_config(c))
    c2 = json.loads(json.dumps(c))
    c2["n_workers"] = 32
    c2["exchange"]["max_iterations"] = 99
    assert canonical_hash(scientific_config(c2)) == h
    c3 = json.loads(json.dumps(c))
    c3["cost"]["penalty_collision"] = 200.0
    assert canonical_hash(scientific_config(c3)) != h
    store = ExperimentStore(tmp_path, "t")
    store.init_config(c)
    store.check_config(c2)
    with pytest.raises(ValueError, match="scientific configuration differs"):
        store.check_config(c3)


def test_truncate_logs_drops_uncheckpointed_iterations(tmp_path):
    s = ExperimentStore(tmp_path, "t")
    for it in (-1, 0, 1, 1, 2):
        s.append("adversarial_candidates", {"iteration": it})
    s.truncate_logs_to(2)
    assert [r["iteration"] for r in s.read_log("adversarial_candidates")] == [-1, 0, 1, 1]
    s.truncate_logs_to(0)
    assert s.read_log("adversarial_candidates") == []


# -- adversary (no simulation: synthetic cost) --
def _synthetic_cost(w, specs):
    # harder when the cutter is close and fast; independent of w here
    return np.array([s.theta_dict.get("cutter_relative_speed_mps", 0.0) - 0.1 * s.theta_dict.get("cutter_gap_m", 0.0)
                     if s.family == "cutin" else s.theta_dict["rear_relative_speed_mps"] for s in specs])


def test_adversary_keeps_strata_schemas_separate_and_returns_costliest_admitted():
    logs = []
    adv = ScenarioAdversary([CUTIN1, SAND_NEG], FeasibilityScreen(ScreeningConfig(policy="none")), _synthetic_cost,
                            AdversaryConfig(n_global=6, max_global_evals=6, local_top=1, n_local=3), seed=0,
                            log=logs.append)
    res = adv.propose(np.zeros(7), 0)
    for r in logs:
        expected = THETA_BOUNDS_CUTIN if r["family"] == "cutin" else THETA_BOUNDS_SANDWICH
        assert set(r["theta"]) == set(expected)
        assert (r["family"], r["blocker_side"]) in {("cutin", 1), ("sandwich", -1)}
    keys = [r["key"] for r in logs]
    assert len(keys) == len(set(keys))
    evaluated = [r for r in logs if r["cost"] is not None]
    assert res.cost == max(r["cost"] for r in evaluated)
    assert {r["phase"] for r in logs} >= {"global", "local"}


class _Stub:
    def __init__(self, status_of):
        self.status_of = status_of

    def predict(self, theta):
        p = self.status_of(theta)
        return {"p_hat": p, "epistemic_interval": (p - 0.15, p + 0.15), "warnings": []}


def _stub_screen(p_of_theta):
    s = FeasibilityScreen(ScreeningConfig(policy="conservative", tau=0.8))
    s.artifacts[CUTIN1.label] = _Stub(p_of_theta)
    s.identities[CUTIN1.label] = {"sha256": "stub"}
    return s


def test_rejected_never_evaluated_and_uncertain_only_with_refinement():
    # every candidate is UNCERTAIN (0.65 < 0.8 <= 0.95)
    screen = _stub_screen(lambda th: 0.8)
    calls = []
    cost = lambda w, specs: (calls.append(len(specs)), _synthetic_cost(w, specs))[1]
    adv = ScenarioAdversary([CUTIN1], screen, cost, AdversaryConfig(n_global=4, max_global_evals=4), seed=0)
    res = adv.propose(np.zeros(7), 0)
    assert res.scenario is None and calls == []   # nothing admitted, nothing simulated
    screen2 = _stub_screen(lambda th: 0.1)        # all rejected
    adv2 = ScenarioAdversary([CUTIN1], screen2, cost, AdversaryConfig(n_global=4), seed=0)
    assert adv2.propose(np.zeros(7), 0).scenario is None and calls == []


def _fake_train(results_by_seed, calls):
    def train(family, cfg, params, verbose=0):
        calls.append(params.ppo_seed)
        k = results_by_seed[params.ppo_seed]
        return {"evaluation": {"summary": {"success_rate": k / 50, "n_success": k, "n_episodes": 50}},
                "stop_reason": "fake", "timesteps_actual": 1, "evaluated_checkpoint": "best"}
    return train


def test_oracle_refinement_admits_on_independent_validation_and_never_retrains(tmp_path):
    screen = _stub_screen(lambda th: 0.8)       # all uncertain
    train_calls, val_calls = [], []
    validate = lambda model, spec, seed, n, rank: (val_calls.append((model, rank)), [{"success": True, "outcome": "finished"}] * n)[1]
    ref = OracleRefiner(OracleRefinementConfig(enabled=True, max_refinements=1, ppo_seeds=(0, 1),
                                               validation_episodes=100), tmp_path / "oracle",
                        train_fn=_fake_train({0: 10, 1: 40}, train_calls), validate_fn=validate)
    adv = ScenarioAdversary([CUTIN1], screen, _synthetic_cost, AdversaryConfig(n_global=3, max_global_evals=3), seed=0,
                            refiner=ref)
    res = adv.propose(np.zeros(7), 0)
    assert train_calls == [0, 1] and len(val_calls) == 1
    assert val_calls[0][0].endswith("ppo_seed1/model") and val_calls[0][1] == 2_000_000   # best seed, independent stream
    rec = res.records[0]
    assert rec["status"] == ADMITTED and rec["selected_ppo_seed"] == 1 and rec["validation"] == {
        **rec["validation"], "k": 100, "n": 100}
    assert res.scenario is not None and res.scenario.key == rec["scenario"]["family"] + "_sidepos1/" + rec["scenario"]["scenario_id"]
    # same scenario again: reloaded from disk, no retraining, budget untouched
    ref2 = OracleRefiner(ref.config, tmp_path / "oracle", train_fn=_fake_train({0: 0, 1: 0}, train_calls),
                         validate_fn=validate)
    spec = ScenarioSpec.from_dict(rec["scenario"])
    again = ref2.refine(spec, _stub_screen(lambda th: 0.8), iteration=5, mpc_cost=1.0)
    assert train_calls == [0, 1] and ref2.n_used == 0 and again["status"] == ADMITTED


def test_failed_refinement_leaves_scenario_unverified(tmp_path):
    screen = _stub_screen(lambda th: 0.8)
    ref = OracleRefiner(OracleRefinementConfig(enabled=True, ppo_seeds=(0,), validation_episodes=20), tmp_path,
                        train_fn=_fake_train({0: 0}, []),
                        validate_fn=lambda m, s, seed, n, r: [{"success": False, "outcome": "timeout"}] * n)
    spec = ScenarioSpec.make(CUTIN1, CutinConfig().theta())
    rec = ref.refine(spec, screen, iteration=0, mpc_cost=3.0)
    assert rec["status"] == UNCERTAIN and screen.classify(spec).status == UNCERTAIN


def test_seed_pool_is_admitted_on_dataset_oracle_evidence(tmp_path):
    th = CutinConfig().theta()
    rows = [{"scenario_id": ScenarioSpec.make(CUTIN1, {**th, **t}).scenario_id, "family": "cutin", "blocker_side": 1,
             "mode": "robust", "k": k, "n": 100, "theta": {**th, **t}} for _, k, t in [("a", 98, {}), ("b", 95, {"front_gap_m": 80.0}), ("c", 40, {"front_gap_m": 50.0}),
                                                     ("old", 99, {"front_gap_m": 10.0})]]   # "old": cutin-bounds-v0
    path = tmp_path / "scenarios.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    exp = Experiment(_cfg(tmp_path, seed_pool_dataset=str(path)), ExperimentStore(tmp_path, "t"))
    assert len(exp.adversary.pool) == 2
    for s in exp.adversary.pool:
        d = exp.screen.classify(s)
        assert d.status == ADMITTED and d.evidence == "oracle" and d.lower >= 0.8
        s.validate()                                   # out-of-bounds (older bound version) rows never enter
    assert len(exp.pool_skipped_invalid) == 1


# -- end-to-end on the real simulator (tiny budgets) --
def test_end_to_end_optimize_interrupt_resume_and_evaluate(tmp_path):
    full_cfg = _cfg(tmp_path, experiment_id="full")
    full = Experiment(full_cfg, ExperimentStore(tmp_path, "full"))
    full.store.init_config(full_cfg)
    full.write_manifest()
    res_full = full.run()

    # interrupted run: 1 iteration, plus a stray half-written iteration-1 row as if the process died
    cut_cfg = _cfg(tmp_path, experiment_id="cut", exchange={"max_iterations": 1})
    store = ExperimentStore(tmp_path, "cut")
    store.init_config(cut_cfg)
    exp = Experiment(cut_cfg, store)
    exp.write_manifest()
    exp.run()
    store.append("adversarial_candidates", {"iteration": 1, "stray": True})

    resumed_cfg = _cfg(tmp_path, experiment_id="cut", exchange={"max_iterations": 2})
    store.check_config(resumed_cfg)   # budgets may change on resume
    exp2 = Experiment(resumed_cfg, store)
    exp2.check_manifest()
    res = exp2.run(resume=True)

    its_full = full.store.read_log("iterations")
    its = store.read_log("iterations")
    assert [r["iteration"] for r in its] == list(range(len(its)))       # no duplicated iteration rows
    assert [r["w"] for r in its] == [r["w"] for r in its_full]          # identical to the uninterrupted run
    assert not any(r.get("stray") for r in store.read_log("adversarial_candidates"))
    assert res["best_w"] == res_full["best_w"]
    assert exp2.store.cache_hits > 0

    out = exp2.evaluate(np.array(res["best_w_unit"]), "heldout", resumed_cfg["heldout"])
    assert out["n_scenarios"] >= 1 and 0.0 <= out["success_rate"] <= 1.0
    assert not set(out["pool_keys"]) & {r["key"] for r in store.read_log("adversarial_candidates")}
    with pytest.raises(AssertionError, match="overlap"):
        exp2.evaluate(np.array(res["best_w_unit"]), "bad", {**resumed_cfg["heldout"], "realization_seeds": [1000]})
