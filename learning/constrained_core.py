"""
Generic constrained exchange (cutting-set) algorithm for

    min_{w in W}  max_{theta in Theta}  J(w, theta)
    s.t.          max_{theta in Theta}  R_i(w, theta) <= eps_i      for every enforced i

with no knowledge of vehicles, MPC or surrogates. Plug-ins:

  evaluate(u, scenarios) -> list[{"objective": float in [0, 1], "values": {constraint name: float}}]
  adversary.propose(u, k) -> ConstrainedAdversaryResult
  constraints: objects with .name, .enforced, .threshold, .violated(value), .violation(value)

Master (iteration k, working set S_k):
    min_{w,t} t  s.t.  J(w, theta_j) <= t,  R_i(w, theta_j) <= eps_i  for all theta_j in S_k
solved by seeded LHS + bounded Nelder-Mead (learning.minimax_core.solve_master) on a
LEXICOGRAPHIC order -- Nelder-Mead only ever COMPARES function values, so the order
is encoded exactly as a scalar ("merit"):
    feasible on S_k   -> merit = t                          in [0, 1]
    infeasible on S_k -> merit = 2 + n_violated + t/2      (n_violated = number of violated
                                                             (constraint, scenario) pairs)
Every feasible candidate ranks above every infeasible one, feasible candidates are
ranked by worst-case objective, infeasible ones by how many requirements they
break, then by objective. No violation magnitude of one unit is ever traded
against another unit or against the objective, and feasibility is tracked as its
own flag, never inferred from the merit value.

Why not scipy's constrained solvers (COBYLA/COBYQA/SLSQP/trust-constr): each builds
linear or quadratic models of the constraints, but here every R_i is an observed
event frequency over a few seeds -- piecewise constant in w -- so those models are
flat almost everywhere and their steps carry no information; an order-based
search with explicit feasibility bookkeeping is the honest choice.

Exchange status (ExchangeStatus): this solver never produces MASTER_INFEASIBLE_CERTIFIED
(a sampling search cannot certify infeasibility); NO_FEASIBLE_FOUND means only that the
search found no feasible w on the working set.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from learning.minimax_core import MasterSettings, solve_master


class ExchangeStatus:
    CONVERGED_FEASIBLE = "converged_feasible"          # feasible w found; adversary found neither a worse
                                                       # objective nor any violation beyond tolerance
    NO_FEASIBLE_FOUND = "no_feasible_found"            # the master search found no w feasible on S_k
    MASTER_INFEASIBLE_CERTIFIED = "master_infeasible_certified"   # never produced by this solver (no certificate)
    BUDGET_EXHAUSTED = "budget_exhausted"              # iteration / evaluation / wall-clock budget hit
    NO_ADMISSIBLE_SCENARIOS = "no_admissible_scenarios"
    RUNNING = "running"


OBJ_MAX = 2.0   # progress_shortfall is in [0, 1]; completion_time in [0, 2] (learning.physical_eval)


# ── Assessment of one w on a finite scenario set ───────────────────────────
@dataclass
class Assessment:
    t: float                                  # worst-case objective over S
    worst_objective_index: int
    values: dict                              # name -> per-scenario values (list)
    worst: dict                               # name -> {"value", "index", "violation", "violated", "enforced"}
    violated_pairs: list                      # (name, scenario index) for enforced, violated pairs
    feasible: bool
    n_scenarios: int

    @property
    def n_violated(self) -> int:
        return len(self.violated_pairs)

    @property
    def merit(self) -> float:
        # lexicographic (feasible first, then fewer violated pairs, then t), exact for any t in [0, OBJ_MAX]
        return self.t if self.feasible else OBJ_MAX + 1.0 + self.n_violated + self.t / (OBJ_MAX + 1.0)

    def to_dict(self, keys: list[str] | None = None) -> dict:
        name = (lambda i: keys[i]) if keys else (lambda i: i)
        return {"t": self.t, "worst_objective_scenario": name(self.worst_objective_index), "feasible": self.feasible,
                "n_violated": self.n_violated,
                "worst": {n: {**w, "scenario": name(w["index"])} for n, w in self.worst.items()},
                "violated_pairs": [[n, name(i)] for n, i in self.violated_pairs]}


def assess(outcomes: list[dict], constraints) -> Assessment:
    assert outcomes, "empty scenario set"
    obj = np.array([o["objective"] for o in outcomes], dtype=float)
    assert np.all((obj >= 0.0) & (obj <= OBJ_MAX)), f"objective must lie in [0, {OBJ_MAX}]: {obj}"
    values, worst, violated = {}, {}, []
    for c in constraints:
        v = np.array([o["values"][c.name] for o in outcomes], dtype=float)
        i = int(np.argmax(v))
        values[c.name] = v.tolist()
        worst[c.name] = {"value": float(v[i]), "index": i, "violation": c.violation(float(v[i])),
                         "violated": c.violated(float(v[i])), "enforced": c.enforced}
        violated += [(c.name, j) for j in range(len(v)) if c.violated(float(v[j]))]
    j = int(np.argmax(obj))
    return Assessment(float(obj[j]), j, values, worst, violated, not violated, len(outcomes))


# ── Master ──────────────────────────────────────────────────────────────────
@dataclass
class MasterOutcome:
    u: np.ndarray
    assessment: Assessment
    n_candidates: int
    n_feasible_candidates: int
    never_satisfied: list                     # scenario indices no evaluated w satisfied (enforced constraints)


def solve_constrained_master(assess_fn: Callable[[np.ndarray], Assessment], dim: int, settings: MasterSettings,
                             incumbents: list | None = None, iteration: int = 0) -> MasterOutcome:
    seen: dict[bytes, Assessment] = {}

    def merit(u):
        u = np.asarray(u, dtype=float)
        a = seen.get(u.tobytes())
        if a is None:
            a = seen[u.tobytes()] = assess_fn(u)
        return a.merit

    res = solve_master(merit, np.zeros(dim), np.ones(dim), settings, incumbents, iteration)
    best_u = np.asarray(res.x, dtype=float)
    best = seen[best_u.tobytes()]
    n_s = best.n_scenarios
    ever_ok = np.zeros(n_s, dtype=bool)
    for a in seen.values():
        bad = {j for _, j in a.violated_pairs}
        ever_ok |= np.array([j not in bad for j in range(n_s)])
    return MasterOutcome(best_u, best, len(seen), sum(a.feasible for a in seen.values()),
                         [int(j) for j in np.flatnonzero(~ever_ok)])


# ── Adversary protocol ──────────────────────────────────────────────────────
@dataclass
class ConstrainedAdversaryResult:
    performance: tuple | None                 # (scenario, objective) -- worst objective found, or None
    constraints: dict                         # name -> (scenario, value) worst value found per constraint
    n_evals: int = 0
    records: list = field(default_factory=list)
    per_stratum: dict = field(default_factory=dict)   # stratum label -> {criterion: {"scenario", "value"}, ...}
    refinements: list = field(default_factory=list)   # PPO oracle-refinement provenance records


class ConstrainedAdversary(Protocol):
    def propose(self, u: np.ndarray, iteration: int) -> ConstrainedAdversaryResult:
        ...


# ── Exchange loop ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ConstrainedExchangeSettings:
    max_iterations: int = 15
    objective_tol: float = 0.01               # add theta_perf only if J(w_k, theta) > t_S + tol (shortfall units)
    max_scenario_evals: int | None = None     # (w, scenario) evaluations, counted by the caller's oracle
    max_wall_s: float | None = None


@dataclass
class ConstrainedExchangeState:
    scenarios: list
    k: int = 0
    u_prev: list | None = None
    iterations: list = field(default_factory=list)
    status: str = ExchangeStatus.RUNNING
    stop_reason: str | None = None


def run_constrained_exchange(evaluate, adversary: ConstrainedAdversary, constraints, u0: np.ndarray,
                             master_settings: MasterSettings, settings: ConstrainedExchangeSettings, *,
                             key: Callable[[Any], str], n_evals: Callable[[], int] = lambda: 0,
                             state: ConstrainedExchangeState | None = None, on_iteration=None,
                             master=None) -> ConstrainedExchangeState:
    """`master(assess_fn, incumbents, k) -> MasterOutcome` may be supplied (e.g. to
    set logging context); default: solve_constrained_master with master_settings."""
    t0 = time.monotonic()
    u0 = np.asarray(u0, dtype=float)
    master = master or (lambda fn, inc, k: solve_constrained_master(fn, len(u0), master_settings, inc, k))
    st = state or ConstrainedExchangeState([])
    if st.status != ExchangeStatus.RUNNING:
        return st

    def add_counterexamples(res: ConstrainedAdversaryResult, t_ref: float | None) -> list[dict]:
        keys = {key(s) for s in st.scenarios}
        added = []
        cands = []
        if res.performance is not None and (t_ref is None or res.performance[1] > t_ref + settings.objective_tol):
            cands.append(("objective", res.performance[0], res.performance[1]))
        for c in constraints:
            hit = res.constraints.get(c.name)
            if hit is not None and c.violated(hit[1]):
                cands.append((c.name, hit[0], hit[1]))
        for reason, sc, val in cands:
            if key(sc) not in keys:
                st.scenarios.append(sc)
                keys.add(key(sc))
                added.append({"reason": reason, "scenario": key(sc), "value": float(val)})
        return added

    if not st.scenarios:   # seed S_0 with the adversary's counterexamples at w0
        res = adversary.propose(u0, -1)
        add_counterexamples(res, None)
        if not st.scenarios:
            st.status, st.stop_reason = ExchangeStatus.NO_ADMISSIBLE_SCENARIOS, "adversary found no admissible scenario"
            return st

    while True:
        if st.k >= settings.max_iterations:
            st.status, st.stop_reason = ExchangeStatus.BUDGET_EXHAUSTED, f"iteration budget ({settings.max_iterations})"
            break
        if settings.max_scenario_evals is not None and n_evals() >= settings.max_scenario_evals:
            st.status, st.stop_reason = ExchangeStatus.BUDGET_EXHAUSTED, f"evaluation budget ({settings.max_scenario_evals})"
            break
        if settings.max_wall_s is not None and time.monotonic() - t0 >= settings.max_wall_s:
            st.status, st.stop_reason = ExchangeStatus.BUDGET_EXHAUSTED, f"wall-clock budget ({settings.max_wall_s}s)"
            break

        k, S = st.k, list(st.scenarios)
        keys = [key(s) for s in S]
        assess_fn = lambda u: assess(evaluate(u, S), constraints)
        incumbents = [u0] + ([np.asarray(st.u_prev)] if st.u_prev is not None else [])
        m = master(assess_fn, incumbents, k)
        rec = {"iteration": k, "w": m.u.tolist(), "master": m.assessment.to_dict(keys),
               "master_candidates": m.n_candidates, "master_feasible_candidates": m.n_feasible_candidates,
               "scenarios_never_satisfied": [keys[j] for j in m.never_satisfied], "working_set": keys}
        st.u_prev = m.u.tolist()
        if not m.assessment.feasible:
            rec.update({"added": [], "counterexample": False, "status": ExchangeStatus.NO_FEASIBLE_FOUND})
            st.iterations.append(rec)
            st.k += 1
            st.status = ExchangeStatus.NO_FEASIBLE_FOUND
            st.stop_reason = (f"no w feasible on the working set was found ({m.n_candidates} candidates); "
                              f"{m.assessment.n_violated} violated (constraint, scenario) pairs at the best candidate")
            if on_iteration:
                on_iteration(st)
            break

        res = adversary.propose(m.u, k)
        added = add_counterexamples(res, m.assessment.t)
        adv = {"performance": None if res.performance is None else
               {"scenario": key(res.performance[0]), "objective": float(res.performance[1])},
               "constraints": {n: {"scenario": key(sc), "value": float(v),
                                   "violation": next(c for c in constraints if c.name == n).violation(float(v))}
                               for n, (sc, v) in res.constraints.items()},
               "n_evals": res.n_evals, "per_stratum": res.per_stratum,
               "refinements": [{"scenario": r["scenario"]["scenario_id"], "status": r["status"],
                                "validation": {k: r["validation"][k] for k in ("k", "n")}} for r in res.refinements]}
        counterexample = any(a["reason"] != "objective" for a in added)
        rec.update({"adversary": adv, "added": added, "counterexample": counterexample,
                    "estimate_t": max(m.assessment.t, adv["performance"]["objective"] if adv["performance"] else 0.0)})
        st.iterations.append(rec)
        st.k += 1
        if not added:
            rec["status"] = st.status = ExchangeStatus.CONVERGED_FEASIBLE
            st.stop_reason = "adversary found no worse objective and no constraint violation"
            if on_iteration:
                on_iteration(st)
            break
        rec["status"] = ExchangeStatus.RUNNING
        if on_iteration:
            on_iteration(st)
    return st


def select_final(state: ConstrainedExchangeState, evaluate, constraints) -> dict:
    """The reported solution. CONVERGED: the last iterate (it survived the
    adversary). Otherwise every iterate is re-assessed on the FINAL working set
    and the lexicographically best one is reported, with its feasibility flag
    -- an infeasible candidate is never labelled feasible."""
    if not state.iterations:
        return {"u": None, "assessment": None, "feasible_on_final_set": False}
    S = state.scenarios
    if state.status == ExchangeStatus.CONVERGED_FEASIBLE:
        u = np.array(state.iterations[-1]["w"])
        a = assess(evaluate(u, S), constraints)
        return {"u": u, "assessment": a, "feasible_on_final_set": a.feasible, "iteration": state.iterations[-1]["iteration"]}
    best = None
    for rec in state.iterations:
        u = np.array(rec["w"])
        a = assess(evaluate(u, S), constraints)
        if best is None or a.merit < best[1].merit:
            best = (u, a, rec["iteration"])
    return {"u": best[0], "assessment": best[1], "feasible_on_final_set": best[1].feasible, "iteration": best[2]}


__all__ = ["ExchangeStatus", "Assessment", "assess", "MasterOutcome", "solve_constrained_master",
           "ConstrainedAdversaryResult", "ConstrainedExchangeSettings", "ConstrainedExchangeState",
           "run_constrained_exchange", "select_final"]
