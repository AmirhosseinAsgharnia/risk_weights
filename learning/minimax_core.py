"""
Generic exchange (cutting-set) algorithm for

    min_{w in W} max_{theta in Theta_f} J(w, theta)

with no knowledge of vehicles, MPC or surrogates -- those enter only
through the two plug-ins:

  oracle(w, scenarios) -> np.ndarray   J(w, theta) for each theta (same w)
  adversary.propose(w, k) -> AdversaryResult   approx. argmax_theta J(w, theta)

Each iteration k:
  A. master:      w_k = argmin_w max_{theta in S_k} J(w, theta)   (bounded derivative-free)
  B. adversary:   theta_new ~ argmax_{theta in Theta_f} J(w_k, theta)
  C. update:      S_{k+1} = S_k + {theta_new} if it beats max_S J(w_k, .) by > tol
                   (previous scenarios are never dropped; duplicates are never re-added)
  D. stop on:     no improving scenario, iteration / oracle-eval / wall-clock budget, stagnation.

Both A and B are approximate, so the result is an EMPIRICAL minimax point:
`estimate_k = max(F_S(w_k), J(w_k, theta_new))` is the worst cost actually
observed at w_k, and the returned w is the iterate with the lowest such
estimate. No global-convergence claim is made.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


# ── Plug-in protocols ───────────────────────────────────────────────────────
@dataclass
class AdversaryResult:
    scenario: Any | None        # None => no admissible scenario found
    cost: float                 # J(w, scenario); -inf when scenario is None
    n_evals: int = 0            # oracle evaluations spent
    records: list = field(default_factory=list)   # free-form per-candidate log rows


class Adversary(Protocol):
    def propose(self, w: np.ndarray, iteration: int) -> AdversaryResult:
        ...


Oracle = Callable[[np.ndarray, list], np.ndarray]


class CountingOracle:
    """Wraps an oracle with an in-memory cache keyed on (w bytes, scenario
    key) and counts (w, theta) evaluations actually computed."""

    def __init__(self, oracle: Oracle, key: Callable[[Any], str] = repr):
        self.oracle, self.key = oracle, key
        self.cache: dict[tuple[bytes, str], float] = {}
        self.n_evals = 0

    def __call__(self, w: np.ndarray, scenarios: list) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        wk = w.tobytes()
        missing = [s for s in scenarios if (wk, self.key(s)) not in self.cache]
        if missing:
            for s, c in zip(missing, np.asarray(self.oracle(w, missing), dtype=float)):
                self.cache[(wk, self.key(s))] = float(c)
            self.n_evals += len(missing)
        return np.array([self.cache[(wk, self.key(s))] for s in scenarios])


# ── Master problem ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MasterSettings:
    n_init: int = 16              # seeded LHS points evaluated before local search
    n_starts: int = 3             # Nelder-Mead restarts from the best initial points
    maxfev_per_start: int = 60
    xatol: float = 1e-3
    fatol: float = 1e-6
    initial_step: float = 0.1     # simplex edge, as a fraction of each box side
    seed: int = 0


@dataclass
class MasterResult:
    x: np.ndarray
    f: float
    n_objective_calls: int
    history: list                 # (x list, f) per objective call


def solve_master(objective: Callable[[np.ndarray], float], lo, hi, settings: MasterSettings,
                 incumbents: list | None = None, iteration: int = 0) -> MasterResult:
    """min_x objective(x) over the box [lo, hi] -- seeded LHS + bounded
    Nelder-Mead restarts. Deterministic given (settings.seed, iteration)."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    history: list = []

    def f(x):
        x = np.clip(np.asarray(x, dtype=float), lo, hi)
        val = float(objective(x))
        history.append((x.tolist(), val))
        return val

    starts = [np.clip(np.asarray(x, dtype=float), lo, hi) for x in (incumbents or [])]
    if settings.n_init > 0:
        cube = qmc.LatinHypercube(d=len(lo), seed=settings.seed + 7919 * iteration).random(settings.n_init)
        starts += [lo + c * (hi - lo) for c in cube]
    scored = sorted(((f(x), i, x) for i, x in enumerate(starts)), key=lambda t: (t[0], t[1]))

    for f0, _, x0 in scored[:settings.n_starts]:
        step = settings.initial_step * (hi - lo)
        simplex = [x0] + [np.clip(x0 + np.eye(len(lo))[j] * (step[j] if x0[j] + step[j] <= hi[j] else -step[j]), lo, hi)
                          for j in range(len(lo))]
        minimize(f, x0, method="Nelder-Mead", bounds=list(zip(lo, hi)),
                 options={"maxfev": settings.maxfev_per_start, "xatol": settings.xatol,
                          "fatol": settings.fatol, "initial_simplex": np.array(simplex)})
    best_x, best_f = min(history, key=lambda h: h[1])
    return MasterResult(np.array(best_x), best_f, len(history), history)


def solve_master_finite(objective: Callable[[np.ndarray], float], candidates: list) -> MasterResult:
    """Exact master over a finite w set (ties -> first candidate)."""
    history = [(np.asarray(c, dtype=float).tolist(), float(objective(np.asarray(c, dtype=float)))) for c in candidates]
    best_x, best_f = min(history, key=lambda h: h[1])
    return MasterResult(np.array(best_x), best_f, len(history), history)


# ── Exchange loop ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExchangeSettings:
    max_iterations: int = 20
    improvement_tol: float = 1e-6        # add theta_new only if J(w_k, theta_new) > F_S(w_k) + tol
    max_oracle_evals: int | None = None  # (w, theta) evaluations, counted by CountingOracle
    max_wall_s: float | None = None
    stagnation_iterations: int | None = None   # stop if best estimate hasn't improved by > tol this long


@dataclass
class ExchangeState:
    """Everything needed to resume at the start of iteration `k`."""
    scenarios: list = field(default_factory=list)
    k: int = 0
    w_prev: list | None = None
    iterations: list = field(default_factory=list)   # per-iteration dict records
    stop_reason: str | None = None


def run_exchange(oracle: CountingOracle, adversary: Adversary, master: Callable, w0: np.ndarray,
                 settings: ExchangeSettings, key: Callable[[Any], str], *,
                 state: ExchangeState | None = None, on_iteration: Callable[[ExchangeState], None] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> ExchangeState:
    """master(objective, incumbents, iteration) -> MasterResult over the
    caller's w box. w0 seeds the first adversary call when S is empty and
    is always offered to the master as an incumbent."""
    state = state or ExchangeState()
    t_start = clock()
    w0 = np.asarray(w0, dtype=float)

    def budget_exceeded() -> str | None:
        if settings.max_oracle_evals is not None and oracle.n_evals >= settings.max_oracle_evals:
            return "oracle-evaluation budget exhausted"
        if settings.max_wall_s is not None and clock() - t_start >= settings.max_wall_s:
            return "wall-clock budget exhausted"
        return None

    if not state.scenarios:
        res = adversary.propose(w0, -1)
        if res.scenario is None:
            state.stop_reason = "adversary found no admissible scenario at w0"
            return state
        state.scenarios.append(res.scenario)

    while state.stop_reason is None:
        if state.k >= settings.max_iterations:
            state.stop_reason = "iteration budget exhausted"
            break
        S = list(state.scenarios)
        objective = lambda w: float(np.max(oracle(w, S)))
        incumbents = [w0] + ([np.asarray(state.w_prev)] if state.w_prev is not None else [])
        m = master(objective, incumbents, state.k)
        w_k = np.asarray(m.x, dtype=float)
        costs_S = oracle(w_k, S)
        F_S = float(np.max(costs_S))

        adv = adversary.propose(w_k, state.k)
        estimate = max(F_S, adv.cost) if adv.scenario is not None else F_S
        rec = {"k": state.k, "w": w_k.tolist(), "F_S": F_S, "set_size": len(S),
               "set_keys": [key(s) for s in S], "set_costs": costs_S.tolist(),
               "argmax_set_key": key(S[int(np.argmax(costs_S))]),
               "adversary_key": key(adv.scenario) if adv.scenario is not None else None,
               "adversary_cost": adv.cost if adv.scenario is not None else None,
               "estimate": estimate, "master_calls": m.n_objective_calls,
               "oracle_evals_total": oracle.n_evals, "added": False}

        gap = (adv.cost - F_S) if adv.scenario is not None else -np.inf
        if adv.scenario is None or gap <= settings.improvement_tol:
            state.stop_reason = "no admissible scenario beats the master worst case by more than improvement_tol"
        elif key(adv.scenario) in set(key(s) for s in S):
            state.stop_reason = "adversary returned an already-included scenario"
        else:
            state.scenarios.append(adv.scenario)
            rec["added"] = True

        state.iterations.append(rec)
        state.w_prev = w_k.tolist()
        state.k += 1

        if state.stop_reason is None:
            state.stop_reason = budget_exceeded()
        if state.stop_reason is None and settings.stagnation_iterations:
            n = settings.stagnation_iterations
            ests = [r["estimate"] for r in state.iterations]
            if len(ests) > n and min(ests[-n:]) >= min(ests[:-n]) - settings.improvement_tol:
                state.stop_reason = f"stagnation: best estimate not improved in {n} iterations"
        if on_iteration is not None:
            on_iteration(state)
    return state


def best_iterate(state: ExchangeState) -> dict:
    """The iterate with the lowest empirically observed worst-case cost."""
    assert state.iterations, "no completed iterations"
    return min(state.iterations, key=lambda r: (r["estimate"], r["k"]))


# ── Baselines / secondary benchmark ────────────────────────────────────────
def random_search(objective: Callable[[np.ndarray], float], lo, hi, n: int, seed: int,
                  include: list | None = None) -> MasterResult:
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    pts = [np.asarray(x, dtype=float) for x in (include or [])]
    pts += [lo + u * (hi - lo) for u in np.random.default_rng(seed).random((n, len(lo)))]
    return solve_master_finite(objective, pts)


def solve_maximin(oracle: Oracle, scenarios: list, master: Callable) -> dict:
    """max_theta min_w J(w, theta) over a FINITE scenario list -- secondary
    benchmark only (generally <= the minimax value; equal only at a saddle)."""
    rows = []
    for i, s in enumerate(scenarios):
        m = master(lambda w, s=s: float(oracle(w, [s])[0]), [], i)
        rows.append({"index": i, "w": np.asarray(m.x).tolist(), "min_cost": m.f})
    best = max(rows, key=lambda r: (r["min_cost"], -r["index"]))
    return {"value": best["min_cost"], "argmax_index": best["index"], "w_at_argmax": best["w"], "per_scenario": rows}


# ── Toy plug-ins (tests, `pilot` mode) ─────────────────────────────────────
class BoxAdversary:
    """argmax over a 1-D/ n-D box of a cheap function f(w, theta): seeded
    LHS + bounded Nelder-Mead on -f. Scenarios are tuples (hashable)."""

    def __init__(self, f: Callable, lo, hi, n_samples: int = 32, maxfev: int = 80, seed: int = 0):
        self.f, self.lo, self.hi = f, np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        self.n_samples, self.maxfev, self.seed = n_samples, maxfev, seed

    def propose(self, w, iteration):
        settings = MasterSettings(n_init=self.n_samples, n_starts=2, maxfev_per_start=self.maxfev,
                                  seed=self.seed + 104729)
        m = solve_master(lambda th: -self.f(w, th), self.lo, self.hi, settings, iteration=iteration + 1)
        theta = tuple(np.round(m.x, 12).tolist())
        return AdversaryResult(theta, float(self.f(w, np.array(theta))), m.n_objective_calls)


class FiniteAdversary:
    """Exact argmax over a finite scenario list (ties -> first)."""

    def __init__(self, oracle: Oracle, scenarios: list):
        self.oracle, self.scenarios = oracle, list(scenarios)

    def propose(self, w, iteration):
        costs = np.asarray(self.oracle(np.asarray(w, dtype=float), self.scenarios), dtype=float)
        i = int(np.argmax(costs))
        return AdversaryResult(self.scenarios[i], float(costs[i]), len(self.scenarios))
