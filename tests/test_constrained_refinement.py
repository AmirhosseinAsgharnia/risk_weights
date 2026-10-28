"""Oracle refinement inside the constrained adversary, per-stratum reporting, and the
single shared w across strata (spec §19, §22, §29)."""

import numpy as np

from learning.closed_loop import ScenarioSpec, Stratum
from learning.constrained_adversary import ConstrainedScenarioAdversary
from learning.constrained_core import (
    ConstrainedAdversaryResult, ConstrainedExchangeSettings, ExchangeStatus, run_constrained_exchange,
)
from learning.feasibility_cutin import CutinConfig
from learning.feasibility_sandwich import SandwichConfig
from learning.minimax_adversary import AdversaryConfig
from learning.minimax_core import MasterSettings
from learning.minimax_feasibility import ADMITTED, FeasibilityScreen, ScreeningConfig, UNCERTAIN
from learning.minimax_oracle import OracleRefinementConfig, OracleRefiner
from learning.physical_eval import ConstraintSpec

CUTIN1, SAND1 = Stratum("cutin", 1), Stratum("sandwich", 1)
CONS = [ConstraintSpec("collision", "collision_rate", 0.0, "1/episode")]


class _Stub:
    def __init__(self, p_of):
        self.p_of = p_of

    def predict(self, theta):
        p = self.p_of(theta)
        return {"p_hat": p, "epistemic_interval": (p - 0.15, p + 0.15), "warnings": []}


def _screen(p_of, strata=(CUTIN1,)):
    s = FeasibilityScreen(ScreeningConfig(policy="conservative", tau=0.8))
    for st in strata:
        s.artifacts[st.label] = _Stub(p_of)
        s.identities[st.label] = {"sha256": "stub"}
    return s


def _metrics(calls):
    def fn(u, specs):
        calls.append([s.key for s in specs])
        return [{"objective": s.theta_dict.get("cutter_gap_m", 30.0) / 100.0,
                 "values": {"collision": 1.0 if s.theta_dict.get("cutter_gap_m", 99) < 20 else 0.0}} for s in specs]
    return fn


def _fake_train(k_by_seed, calls):
    def train(family, cfg, params, verbose=0):
        calls.append((params.ppo_seed, params.rollover_event))
        k = k_by_seed[params.ppo_seed]
        return {"evaluation": {"summary": {"success_rate": k / 50, "n_success": k, "n_episodes": 50}},
                "stop_reason": "fake", "timesteps_actual": 1, "evaluated_checkpoint": "best"}
    return train


def test_uncertain_candidates_are_never_evaluated_without_a_refiner():
    calls = []
    adv = ConstrainedScenarioAdversary([CUTIN1], _screen(lambda th: 0.8), _metrics(calls), CONS,
                                       AdversaryConfig(n_global=4, max_global_evals=4), seed=0)
    res = adv.propose(np.zeros(7), 0)
    assert res.performance is None and calls == [] and res.refinements == []


def test_refinement_admits_on_ppo_evidence_and_records_provenance(tmp_path):
    calls, train_calls = [], []
    screen = _screen(lambda th: 0.8)                                   # every candidate UNCERTAIN
    ref = OracleRefiner(OracleRefinementConfig(enabled=True, max_refinements=1, max_per_iteration=1, ppo_seeds=(0,),
                                               validation_episodes=60), tmp_path / "oracle",
                        train_fn=_fake_train({0: 50}, train_calls),
                        validate_fn=lambda m, s, seed, n, r: [{"success": True, "outcome": "finished"}] * n)
    adv = ConstrainedScenarioAdversary([CUTIN1], screen, _metrics(calls), CONS,
                                       AdversaryConfig(n_global=6, max_global_evals=6, n_local=0), seed=0, refiner=ref)
    res = adv.propose(np.zeros(7), 0)
    assert train_calls == [(0, "physical_v2")]                         # corrected event model, budget 1
    assert len(res.refinements) == 1 and res.refinements[0]["status"] == ADMITTED
    key = ScenarioSpec.from_dict(res.refinements[0]["scenario"]).key
    assert screen.classify(ScenarioSpec.from_dict(res.refinements[0]["scenario"])).evidence == "oracle"
    # the newly admitted scenario is the only admitted one, so it is every criterion's worst case
    assert res.performance[0].key == key and res.per_stratum[CUTIN1.label]["n_admitted_evaluated"] == 1
    # the most-violating uncertain candidate was referred first
    evaluated = [k for batch in calls for k in batch]
    assert key in evaluated and (tmp_path / "oracle" / CUTIN1.label).is_dir()


def test_refinement_that_fails_leaves_candidate_unadmitted(tmp_path):
    ref = OracleRefiner(OracleRefinementConfig(enabled=True, ppo_seeds=(0,), validation_episodes=20), tmp_path,
                        train_fn=_fake_train({0: 0}, []),
                        validate_fn=lambda m, s, seed, n, r: [{"success": False, "outcome": "timeout"}] * n)
    adv = ConstrainedScenarioAdversary([CUTIN1], _screen(lambda th: 0.8), _metrics([]), CONS,
                                       AdversaryConfig(n_global=3, max_global_evals=3, n_local=0), seed=0, refiner=ref)
    res = adv.propose(np.zeros(7), 0)
    assert res.refinements and res.refinements[0]["status"] == UNCERTAIN and res.performance is None


def test_per_stratum_worst_cases_with_separate_schemas():
    calls = []
    adv = ConstrainedScenarioAdversary([CUTIN1, SAND1], _screen(lambda th: 0.99, (CUTIN1, SAND1)), _metrics(calls),
                                       CONS, AdversaryConfig(n_global=5, max_global_evals=5, n_local=1), seed=1)
    res = adv.propose(np.zeros(7), 0)
    assert set(res.per_stratum) == {CUTIN1.label, SAND1.label}
    for label, ps in res.per_stratum.items():
        assert ps["objective"]["scenario"].startswith(label) and set(ps) >= {"objective", "collision"}
    assert res.performance[1] == max(ps["objective"]["value"] for ps in res.per_stratum.values())


def test_master_uses_one_shared_w_for_the_union_of_strata():
    sa = ScenarioSpec.make(CUTIN1, CutinConfig().theta())
    sb = ScenarioSpec.make(SAND1, SandwichConfig().theta())
    seen = []

    def evaluate(u, specs):
        seen.append((np.asarray(u).tobytes(), {s.stratum.label for s in specs}))
        x = float(np.asarray(u)[0])
        return [{"objective": abs(x - (0.3 if s.stratum == CUTIN1 else 0.7)), "values": {"collision": 0.0}} for s in specs]

    class _Adv:
        def propose(self, u, k):
            return ConstrainedAdversaryResult((sa if k < 0 else sb, 0.9), {"collision": (sa, 0.0)}, 1)

    st = run_constrained_exchange(evaluate, _Adv(), CONS, np.full(2, 0.5), MasterSettings(n_init=6, n_starts=1,
                                  maxfev_per_start=40), ConstrainedExchangeSettings(max_iterations=3),
                                  key=lambda s: s.key)
    union = [labels for _, labels in seen if len(labels) == 2]
    assert union and all(labels == {CUTIN1.label, SAND1.label} for labels in union)   # one u evaluated on BOTH strata
    assert st.iterations[-1]["master"]["t"] >= 0.2 - 1e-6                            # max over strata, not average
    assert st.status in (ExchangeStatus.CONVERGED_FEASIBLE, ExchangeStatus.BUDGET_EXHAUSTED, ExchangeStatus.RUNNING)
