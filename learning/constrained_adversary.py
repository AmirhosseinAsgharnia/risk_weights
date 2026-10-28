"""
Scenario adversary for the constrained exchange (learning.constrained_core).
At a fixed w_k it searches, within each stratum's own theta schema and only
over PPO-ADMITTED scenarios, for

  performance:  argmax_theta J_progress(w_k, theta)
  constraint i: argmax_theta R_i(w_k, theta)        (each enforced constraint separately)

One closed-loop evaluation of a scenario yields every metric at once, so the
searches share their global sample; local refinement then runs per criterion
(the objective and each enforced constraint), around that criterion's own top
candidates. Criteria are never combined: each returns its own worst scenario
and value in its own units.

Steps per call (as learning.minimax_adversary, whose sampling, screening and
local-perturbation code is reused unchanged):
  1. global: seeded space-filling batch per stratum + seed pool; screened; up to
             max_global_evals ADMITTED candidates (+ pool) evaluated
  2. local:  for each criterion, Gaussian perturbations around its top admitted
             candidates (constraints only when their worst value is > 0 --
             perturbing around zero-event scenarios carries no signal)
  3. return: per criterion, the worst admitted candidate (ties -> scenario key)

  4. refine:  (only with an OracleRefiner and budget left) UNCERTAIN surrogate-screened
             candidates are evaluated too -- for REFERRAL only, never as counterexamples --
             and those worse than every admitted candidate on some criterion (objective
             above the best admitted one, or an enforced constraint violated) are sent to
             the PPO oracle (learning.minimax_oracle.OracleRefiner: trains + validates PPO,
             admission by the unchanged Wilson rule), at most max_per_iteration per call,
             most-violating first. A newly ADMITTED one then competes like any other.
             Oracle results are versioned files in the experiment's oracle/ dir; the
             feasibility datasets are never modified.

Rejected candidates are logged and never evaluated. Every result also carries the
per-stratum worst case of every criterion (each stratum searched with its own sampler
and surrogate; the shared w is the same everywhere).
"""

import numpy as np

from learning.closed_loop import ScenarioSpec, Stratum
from learning.constrained_core import ConstrainedAdversaryResult
from learning.feasibility_sampling import sample_cutin, sample_sandwich
from learning.minimax_adversary import AdversaryConfig, ScenarioAdversary
from learning.minimax_feasibility import ADMITTED, UNCERTAIN
from learning.scenario import derive_seed

_SAMPLERS = {"cutin": sample_cutin, "sandwich": sample_sandwich}


class ConstrainedScenarioAdversary(ScenarioAdversary):
    def __init__(self, strata, screen, metrics_fn, constraints, config: AdversaryConfig, seed: int, *,
                 pool=None, log=None, refiner=None):
        """metrics_fn(u, [ScenarioSpec]) -> list[{"objective", "values": {name: value}}]."""
        super().__init__(strata, screen, None, config, seed, pool=pool, refiner=refiner, log=log)
        self.metrics_fn, self.constraints = metrics_fn, list(constraints)

    def _evaluate_metrics(self, u, entries):
        if entries:
            for e, m in zip(entries, self.metrics_fn(u, [e["spec"] for e in entries])):
                e["metrics"] = m

    @staticmethod
    def _crit(e, name):
        return e["metrics"]["objective"] if name == "objective" else e["metrics"]["values"][name]

    def propose(self, u, iteration: int) -> ConstrainedAdversaryResult:
        cfg = self.config
        entries, seen = [], set()

        def add(spec, phase):
            if spec.key in seen:
                return None
            seen.add(spec.key)
            e = {"spec": spec, "decision": self._classify(spec), "metrics": None, "phase": phase}
            entries.append(e)
            return e

        to_eval = []
        for si, stratum in enumerate(self.strata):
            configs, _ = _SAMPLERS[stratum.family](cfg.n_global, stratum.blocker_side,
                                                   derive_seed(self.seed, si, iteration + 1), method=cfg.doe_method)
            fresh = [e for e in (add(ScenarioSpec.make(stratum, c.theta()), "global") for c in configs) if e]
            pooled = [e for e in (add(s, "pool") for s in self.pool if s.stratum == stratum) if e][:cfg.max_pool_evals]
            to_eval += [e for e in fresh if e["decision"].status == ADMITTED][:cfg.max_global_evals]
            to_eval += [e for e in pooled if e["decision"].status == ADMITTED]
            if self.refiner is not None and self.refiner.budget_left() > 0:
                to_eval += [e for e in fresh if e["decision"].status == UNCERTAIN
                            and e["decision"].evidence == "surrogate"][:cfg.max_global_evals]
        self._evaluate_metrics(u, to_eval)

        criteria = ["objective"] + [c.name for c in self.constraints if c.enforced]
        rng = np.random.default_rng(derive_seed(self.seed, 99_991, iteration + 1))
        local = []
        for stratum in self.strata:
            done = [e for e in entries if e["metrics"] is not None and e["spec"].stratum == stratum
                    and e["decision"].status == ADMITTED]
            for name in criteria:
                top = sorted(done, key=lambda e: (-self._crit(e, name), e["spec"].key))[:cfg.local_top]
                for e in top:
                    if name != "objective" and self._crit(e, name) <= 0.0:
                        continue
                    for cand in self._local(stratum, e["spec"], rng):
                        ne = add(cand, f"local:{name}")
                        if ne is not None and ne["decision"].status == ADMITTED:
                            local.append(ne)
        self._evaluate_metrics(u, local)
        refinements = self._refine(entries, iteration)

        for e in entries:
            d, m = e["decision"], e["metrics"]
            self.log({"iteration": iteration, "phase": e["phase"], **e["spec"].to_dict(), "key": e["spec"].key,
                      "status": d.status, "evidence": d.evidence, "p_hat": d.p_hat, "lower": d.lower, "upper": d.upper,
                      "warnings": d.warnings, "surrogate_sha256": d.surrogate_sha256,
                      "objective": None if m is None else m["objective"], "values": None if m is None else m["values"],
                      "refined": e.get("refined", False)})

        done = [e for e in entries if e["metrics"] is not None and e["decision"].status == ADMITTED]
        n_evals = sum(e["metrics"] is not None for e in entries)
        if not done:
            return ConstrainedAdversaryResult(None, {}, n_evals, refinements=refinements)
        names = ["objective"] + [c.name for c in self.constraints]
        worst = lambda pool, name: max(pool, key=lambda e: (self._crit(e, name), e["spec"].key))
        perf = worst(done, "objective")
        cons = {n: (worst(done, n)["spec"], self._crit(worst(done, n), n)) for n in names[1:]}
        per_stratum = {}
        for stratum in self.strata:
            sd = [e for e in done if e["spec"].stratum == stratum]
            if sd:
                per_stratum[stratum.label] = {"n_admitted_evaluated": len(sd), **{
                    n: {"scenario": worst(sd, n)["spec"].key, "value": float(self._crit(worst(sd, n), n))} for n in names}}
        return ConstrainedAdversaryResult((perf["spec"], perf["metrics"]["objective"]), cons, n_evals,
                                          per_stratum=per_stratum, refinements=refinements)

    def _refine(self, entries, iteration: int) -> list[dict]:
        if self.refiner is None:
            return []
        admitted = [e for e in entries if e["metrics"] is not None and e["decision"].status == ADMITTED]
        best_obj = max((e["metrics"]["objective"] for e in admitted), default=-np.inf)
        enforced = [c for c in self.constraints if c.enforced]

        def n_violated(e):
            return sum(c.violated(e["metrics"]["values"][c.name]) for c in enforced)

        unc = [e for e in entries if e["metrics"] is not None and e["decision"].status == UNCERTAIN
               and e["decision"].evidence == "surrogate" and (e["metrics"]["objective"] > best_obj or n_violated(e) > 0)]
        unc.sort(key=lambda e: (-n_violated(e), -e["metrics"]["objective"], e["spec"].key))
        out = []
        for e in unc[:self.refiner.config.max_per_iteration]:
            if self.refiner.budget_left() <= 0 and not self.refiner.result_path(e["spec"]).exists():
                break
            out.append(self.refiner.refine(e["spec"], self.screen, iteration=iteration,
                                           mpc_cost=e["metrics"]["objective"]))
            e["decision"] = self.screen.classify(e["spec"])
            e["refined"] = True
        return out


__all__ = ["ConstrainedScenarioAdversary"]
