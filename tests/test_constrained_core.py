import numpy as np
import pytest

from learning.closed_loop import Stratum
from learning.constrained_adversary import ConstrainedScenarioAdversary
from learning.constrained_core import (
    ConstrainedAdversaryResult, OBJ_MAX, ConstrainedExchangeSettings, ExchangeStatus, assess, run_constrained_exchange,
    select_final,
)
from learning.feasibility_cutin import THETA_BOUNDS_CUTIN
from learning.feasibility_sandwich import THETA_BOUNDS_SANDWICH
from learning.minimax_adversary import AdversaryConfig
from learning.minimax_core import MasterSettings
from learning.minimax_feasibility import FeasibilityScreen, ScreeningConfig
from learning.physical_eval import ConstraintSpec

GRID = np.linspace(0.0, 1.0, 101)
R_MAX = ConstraintSpec("r", "collision_rate", 0.3, "1/episode")   # metric only labels it here


def _toy(r_fn):
    ev = lambda u, S: [{"objective": abs(u[0] - th), "values": {"r": r_fn(u[0], th)}} for th in S]

    class Adv:
        def propose(self, u, k):
            J = [abs(u[0] - t) for t in GRID]
            R = [r_fn(u[0], t) for t in GRID]
            return ConstrainedAdversaryResult((float(GRID[int(np.argmax(J))]), max(J)),
                                              {"r": (float(GRID[int(np.argmax(R))]), max(R))}, len(GRID))
    return ev, Adv()


def _run(r_fn, **settings):
    ev, adv = _toy(r_fn)
    st = run_constrained_exchange(ev, adv, [R_MAX], np.array([0.2]), MasterSettings(n_init=8, xatol=1e-7),
                                  ConstrainedExchangeSettings(**{"objective_tol": 1e-4, **settings}), key=repr)
    return st, select_final(st, ev, [R_MAX])


def test_toy_constrained_minimax_reaches_the_known_solution():
    # min_w max_theta |w - theta|  s.t.  max_theta theta (1 - w) <= 0.3   ->  w* = 0.7, J* = 0.7
    st, fin = _run(lambda w, th: th * (1 - w))
    assert st.status == ExchangeStatus.CONVERGED_FEASIBLE
    assert fin["u"][0] == pytest.approx(0.7, abs=1e-4) and fin["assessment"].t == pytest.approx(0.7, abs=1e-4)
    assert fin["feasible_on_final_set"] and sorted(st.scenarios) == [0.0, 1.0]


def test_infeasible_problem_is_never_reported_feasible():
    st, fin = _run(lambda w, th: th)   # theta = 1 violates R <= 0.3 for every w
    assert st.status == ExchangeStatus.NO_FEASIBLE_FOUND
    assert not fin["feasible_on_final_set"] and not fin["assessment"].feasible
    assert st.iterations[-1]["scenarios_never_satisfied"]


def test_budget_exhaustion_is_reported_separately_from_convergence():
    st, _ = _run(lambda w, th: th * (1 - w), max_iterations=1)
    assert st.status == ExchangeStatus.BUDGET_EXHAUSTED and "iteration budget" in st.stop_reason


def test_worst_case_aggregation_and_lexicographic_merit():
    c = [ConstraintSpec("a", "collision_rate", 0.0, "1/episode"), ConstraintSpec("b", "peak_abs_jerk", None, "m/s^3")]
    outs = [{"objective": 0.1, "values": {"a": 0.0, "b": 50.0}}, {"objective": 0.4, "values": {"a": 0.0, "b": 5.0}}]
    a = assess(outs, c)
    assert a.feasible and a.t == 0.4 and a.worst_objective_index == 1 and a.worst["b"]["index"] == 0
    bad = assess([{"objective": 0.0, "values": {"a": 0.5, "b": 0.0}}], c)
    assert not bad.feasible and bad.n_violated == 1 and bad.merit > 1.0 >= a.merit   # any feasible beats any infeasible
    worst_feasible = assess([{"objective": OBJ_MAX, "values": {"a": 0.0, "b": 0.0}}], c)      # completion_time range
    two_bad = assess([{"objective": 0.0, "values": {"a": 0.5, "b": 0.0}}] * 2, c)
    one_bad_slow = assess([{"objective": OBJ_MAX, "values": {"a": 0.5, "b": 0.0}}], c)
    assert worst_feasible.merit < bad.merit < one_bad_slow.merit < two_bad.merit              # exact lexicographic order
    with pytest.raises(AssertionError):
        assess([{"objective": OBJ_MAX + 0.5, "values": {"a": 0.0, "b": 0.0}}], c)


def test_constrained_adversary_returns_independent_counterexamples_per_criterion():
    cons = [ConstraintSpec("fast", "collision_rate", 0.0, "1/episode")]

    def metrics(u, specs):   # objective and constraint deliberately disagree on which scenario is worst
        return [{"objective": s.theta_dict["cutter_gap_m"] / 100.0 if s.family == "cutin" else 0.0,
                 "values": {"fast": s.theta_dict.get("cutter_relative_speed_mps", s.theta_dict.get("rear_relative_speed_mps"))}}
                for s in specs]
    logs = []
    adv = ConstrainedScenarioAdversary([Stratum("cutin", 1), Stratum("sandwich", -1)],
                                       FeasibilityScreen(ScreeningConfig(policy="none")), metrics, cons,
                                       AdversaryConfig(n_global=6, max_global_evals=6, local_top=1, n_local=2), 0,
                                       log=logs.append)
    res = adv.propose(np.zeros(7), 0)
    done = [r for r in logs if r["objective"] is not None]
    assert res.performance[1] == max(r["objective"] for r in done)
    assert res.constraints["fast"][1] == max(r["values"]["fast"] for r in done)
    for r in logs:
        assert set(r["theta"]) == set(THETA_BOUNDS_CUTIN if r["family"] == "cutin" else THETA_BOUNDS_SANDWICH)
    assert {r["phase"] for r in logs} >= {"global", "local:objective"}
