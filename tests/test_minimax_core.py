import numpy as np
import pytest

from learning.minimax_core import (
    AdversaryResult, BoxAdversary, CountingOracle, ExchangeSettings, ExchangeState, FiniteAdversary,
    MasterSettings, best_iterate, random_search, run_exchange, solve_master, solve_master_finite, solve_maximin,
)


def _toy_f(w, th):
    return float((np.asarray(w)[0] - np.asarray(th)[0]) ** 2)


def _toy_oracle(w, scenarios):
    return np.array([_toy_f(w, s) for s in scenarios])


def _master(settings):
    return lambda obj, inc, k: solve_master(obj, [-1.0], [1.0], settings, inc, k)


def test_continuous_toy_minimax_reaches_w0_j1():
    oracle = CountingOracle(_toy_oracle)
    adv = BoxAdversary(_toy_f, [-1.0], [1.0], n_samples=16, seed=0)
    state = run_exchange(oracle, adv, _master(MasterSettings(n_init=8, maxfev_per_start=80, xatol=1e-6)),
                         np.array([0.5]), ExchangeSettings(max_iterations=10, improvement_tol=1e-4), key=repr)
    best = best_iterate(state)
    assert abs(best["w"][0]) < 1e-2
    assert best["estimate"] == pytest.approx(1.0, abs=2e-2)
    # both endpoints were discovered and kept
    ends = sorted(round(s[0]) for s in state.scenarios)
    assert ends[0] == -1 and ends[-1] == 1
    assert state.stop_reason.startswith("no admissible scenario beats")


# Payoff matrix J[w_i, theta_j]: minimax picks row 1 (value 2), maximin = 1 -- they differ.
M = np.array([[3.0, 0.0], [1.0, 2.0]])


def _matrix_oracle(w, scenarios):
    return np.array([M[int(round(w[0])), j] for j in scenarios])


def test_discrete_minimax_differs_from_maximin_and_order_is_correct():
    oracle = CountingOracle(_matrix_oracle)
    master = lambda obj, inc, k: solve_master_finite(obj, [np.array([0.0]), np.array([1.0])])
    state = run_exchange(oracle, FiniteAdversary(_matrix_oracle, [0, 1]), master, np.array([0.0]),
                         ExchangeSettings(max_iterations=5), key=repr)
    best = best_iterate(state)
    assert best["w"] == [1.0] and best["estimate"] == 2.0
    mm = solve_maximin(_matrix_oracle, [0, 1], master)
    assert mm["value"] == 1.0 and mm["argmax_index"] == 0
    assert mm["value"] < best["estimate"]


def test_same_w_used_for_every_scenario_in_master():
    objective_calls = []

    def master(obj, inc, k):
        res = solve_master_finite(lambda w: (objective_calls.append(w.copy()), obj(w))[1],
                                  [np.array([0.0]), np.array([1.0])])
        return res

    state = run_exchange(CountingOracle(_matrix_oracle), FiniteAdversary(_matrix_oracle, [0, 1]), master,
                         np.array([0.0]), ExchangeSettings(max_iterations=5), key=repr)
    # the master value of a candidate w is the max over the WHOLE current set at that single w
    for rec in state.iterations:
        w = np.array(rec["w"])
        S = [int(k) for k in rec["set_keys"]]
        assert rec["set_costs"] == _matrix_oracle(w, S).tolist()
        assert rec["F_S"] == max(_matrix_oracle(w, S))
    assert state.iterations[-1]["set_size"] == 2


class _ScriptedAdversary:
    """Returns pre-scripted scenarios; scenario = shift c in J = (w - c)^2."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def propose(self, w, k):
        c = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return AdversaryResult((c,), _toy_f(w, (c,)), 1)


def test_previous_scenarios_kept_duplicates_not_readded_and_new_scenario_moves_w():
    oracle = CountingOracle(_toy_oracle)
    adv = _ScriptedAdversary([0.8, -0.8, 0.8])
    state = run_exchange(oracle, adv, _master(MasterSettings(n_init=8, xatol=1e-6, maxfev_per_start=80)),
                         np.array([0.0]), ExchangeSettings(max_iterations=6), key=repr)
    keys = [repr(s) for s in state.scenarios]
    assert len(keys) == len(set(keys)) == 2          # no duplicates
    ws = [r["w"][0] for r in state.iterations]
    assert ws[0] == pytest.approx(0.8, abs=1e-2)     # S = {0.8}
    assert ws[1] == pytest.approx(0.0, abs=1e-2)     # -0.8 added -> w moves to the middle
    assert [r["set_size"] for r in state.iterations] == sorted(r["set_size"] for r in state.iterations)


def test_worst_case_cost_over_set_is_max():
    oracle = CountingOracle(_toy_oracle)
    adv = _ScriptedAdversary([0.5, -1.0])
    state = run_exchange(oracle, adv, lambda obj, inc, k: solve_master_finite(obj, [np.array([0.0])]),
                         np.array([0.0]), ExchangeSettings(max_iterations=3), key=repr)
    r = state.iterations[-1]
    assert r["F_S"] == pytest.approx(max(r["set_costs"]))
    assert r["F_S"] == pytest.approx(1.0)


def test_iteration_and_eval_budgets_stop():
    adv = BoxAdversary(_toy_f, [-1.0], [1.0], n_samples=8)
    s1 = run_exchange(CountingOracle(_toy_oracle), adv, _master(MasterSettings(n_init=4, maxfev_per_start=10)),
                      np.array([0.5]), ExchangeSettings(max_iterations=1), key=repr)
    assert len(s1.iterations) == 1 and s1.stop_reason in (
        "iteration budget exhausted",
        "no admissible scenario beats the master worst case by more than improvement_tol")
    o = CountingOracle(_toy_oracle)
    s2 = run_exchange(o, adv, _master(MasterSettings(n_init=4, maxfev_per_start=10)), np.array([0.5]),
                      ExchangeSettings(max_iterations=50, max_oracle_evals=5), key=repr)
    assert s2.stop_reason == "oracle-evaluation budget exhausted"
    t = iter(range(1000))
    s3 = run_exchange(CountingOracle(_toy_oracle), adv, _master(MasterSettings(n_init=4, maxfev_per_start=10)),
                      np.array([0.5]), ExchangeSettings(max_iterations=50, max_wall_s=0.5), key=repr,
                      clock=lambda: float(next(t)))
    assert s3.stop_reason == "wall-clock budget exhausted"


def test_resume_from_state_matches_uninterrupted_run():
    def run(stop_after=None):
        adv = BoxAdversary(_toy_f, [-1.0], [1.0], n_samples=8, seed=3)
        snaps = []
        st = run_exchange(CountingOracle(_toy_oracle), adv, _master(MasterSettings(n_init=4, seed=1)),
                          np.array([0.7]), ExchangeSettings(max_iterations=stop_after or 10), key=repr,
                          on_iteration=lambda s: snaps.append(ExchangeState(list(s.scenarios), s.k, s.w_prev,
                                                                            list(s.iterations), None)))
        return st, snaps

    full, _ = run()
    _, snaps = run(stop_after=1)
    adv = BoxAdversary(_toy_f, [-1.0], [1.0], n_samples=8, seed=3)
    resumed = run_exchange(CountingOracle(_toy_oracle), adv, _master(MasterSettings(n_init=4, seed=1)),
                           np.array([0.7]), ExchangeSettings(max_iterations=10), key=repr, state=snaps[0])
    assert [r["w"] for r in resumed.iterations] == [r["w"] for r in full.iterations]


def test_counting_oracle_caches():
    calls = []
    o = CountingOracle(lambda w, s: (calls.append(len(s)), _toy_oracle(w, s))[1])
    o(np.array([0.1]), [(0.0,), (1.0,)])
    o(np.array([0.1]), [(1.0,), (0.0,)])
    assert o.n_evals == 2 and calls == [2]


def test_random_search_and_master_respect_bounds():
    r = random_search(lambda x: float(np.sum(x ** 2)), [-1, -1], [1, 1], n=20, seed=0)
    assert np.all(np.abs(r.x) <= 1)
    m = solve_master(lambda x: float(-x[0]), [0.0], [2.0], MasterSettings(n_init=4))
    assert m.x[0] == pytest.approx(2.0, abs=1e-6)
