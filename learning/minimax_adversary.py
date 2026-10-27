"""
Scenario adversary for the exchange algorithm (learning.minimax_core):
approximately solves

    theta_{k+1} = argmax_{theta in Theta_f} J_eval(w_k, theta)

over one or more strata (family x blocker side), each in its OWN theta
schema -- candidates of different strata are only ever compared through
their scalar J_eval, never merged into a joint feature vector.

Per call (iteration k, fixed w_k):
  1. global:  per stratum, a seeded space-filling batch from
              learning.feasibility_sampling (family validators applied there),
              plus any seed pool (e.g. PPO-demonstrated dataset scenarios);
  2. screen:  every candidate classified by FeasibilityScreen (cheap);
  3. evaluate: J_eval for up to max_global_evals ADMITTED candidates per stratum
              (expensive -- one closed-loop batch over the shared realization seeds),
              and, if oracle refinement is on, for UNCERTAIN ones too;
  4. local:   Gaussian perturbations (normalized theta units) around the top
              admitted candidates; re-validated, re-screened, evaluated;
  5. refine:  UNCERTAIN candidates costlier than the best admitted one may be
              sent to the PPO oracle (OracleRefiner) within budget; if it
              demonstrates success they become admitted;
  6. return:  the costliest ADMITTED candidate.
"""

from dataclasses import dataclass, asdict

import numpy as np

from learning.closed_loop import ScenarioSpec, Stratum
from learning.feasibility_common import ScenarioFamilyConfigError
from learning.feasibility_sampling import sample_cutin, sample_sandwich
from learning.minimax_core import AdversaryResult
from learning.minimax_feasibility import FeasibilityScreen, ADMITTED, UNCERTAIN
from learning.scenario import derive_seed

_SAMPLERS = {"cutin": sample_cutin, "sandwich": sample_sandwich}


@dataclass(frozen=True)
class AdversaryConfig:
    n_global: int = 32            # space-filling candidates per stratum per iteration (screened, cheap)
    max_global_evals: int = 8     # of which at most this many are closed-loop evaluated per stratum
    local_top: int = 2            # admitted candidates refined locally
    n_local: int = 4              # perturbations per refined candidate
    local_sigma: float = 0.08     # perturbation std, fraction of each theta field's range
    doe_method: str = "lhs"
    max_pool_evals: int = 8       # seed-pool scenarios evaluated per iteration (per stratum)


class ScenarioAdversary:
    def __init__(self, strata: list[Stratum], screen: FeasibilityScreen, cost_fn, config: AdversaryConfig,
                 seed: int, *, pool: list[ScenarioSpec] | None = None, refiner=None, log=None):
        """cost_fn(w, [ScenarioSpec]) -> np.ndarray of J_eval (the experiment's
        caching/counting oracle). log(record) receives one dict per candidate."""
        self.strata, self.screen, self.cost_fn, self.config, self.seed = strata, screen, cost_fn, config, seed
        self.pool = list(pool or [])
        self.refiner = refiner
        self.log = log or (lambda rec: None)
        self.n_surrogate_calls = 0

    def _classify(self, spec):
        self.n_surrogate_calls += 1
        return self.screen.classify(spec)

    def _evaluate(self, w, entries):
        """One batched closed-loop evaluation (parallelized by cost_fn)."""
        if not entries:
            return
        costs = self.cost_fn(w, [e["spec"] for e in entries])
        for e, c in zip(entries, costs):
            e["cost"] = float(c)

    def _local(self, stratum: Stratum, spec: ScenarioSpec, rng: np.random.Generator) -> list[ScenarioSpec]:
        bounds = stratum.bounds
        out = []
        for _ in range(self.config.n_local):
            theta = {}
            for name, (lo, hi) in bounds.items():
                v = spec.theta_dict[name] + self.config.local_sigma * (hi - lo) * rng.standard_normal()
                theta[name] = float(np.clip(v, lo, hi))
            cand = ScenarioSpec.make(stratum, theta)
            try:
                cand.validate()
            except ScenarioFamilyConfigError:
                continue
            out.append(cand)
        return out

    def propose(self, w, iteration: int) -> AdversaryResult:
        cfg = self.config
        entries: list[dict] = []
        seen: set[str] = set()

        def add(spec, phase):
            if spec.key in seen:
                return None
            seen.add(spec.key)
            d = self._classify(spec)
            e = {"spec": spec, "decision": d, "cost": None, "phase": phase}
            entries.append(e)
            return e

        to_eval = []
        for si, stratum in enumerate(self.strata):
            k_seed = derive_seed(self.seed, si, iteration + 1)
            configs, _rejected = _SAMPLERS[stratum.family](cfg.n_global, stratum.blocker_side, k_seed,
                                                            method=cfg.doe_method)
            fresh = [add(ScenarioSpec.make(stratum, c.theta()), "global") for c in configs]
            pooled = [add(s, "pool") for s in self.pool if s.stratum == stratum][:cfg.max_pool_evals]
            fresh = [e for e in fresh if e is not None]
            pooled = [e for e in pooled if e is not None]
            want = UNCERTAIN if (self.refiner is not None and self.refiner.budget_left() > 0) else None
            to_eval += [e for e in fresh if e["decision"].status == ADMITTED][:cfg.max_global_evals]
            to_eval += [e for e in pooled if e["decision"].status == ADMITTED]
            if want:
                to_eval += [e for e in fresh if e["decision"].status == UNCERTAIN][:cfg.max_global_evals]
        self._evaluate(w, to_eval)

        # local refinement around the costliest admitted candidates, per stratum
        rng = np.random.default_rng(derive_seed(self.seed, 99_991, iteration + 1))
        local_eval = []
        for stratum in self.strata:
            top = sorted((e for e in entries if e["cost"] is not None and e["decision"].status == ADMITTED
                          and e["spec"].stratum == stratum), key=lambda e: -e["cost"])[:cfg.local_top]
            for e in top:
                for cand in self._local(stratum, e["spec"], rng):
                    ne = add(cand, "local")
                    if ne is not None and ne["decision"].status == ADMITTED:
                        local_eval.append(ne)
        self._evaluate(w, local_eval)

        # oracle refinement of uncertain, high-cost candidates
        admitted = [e for e in entries if e["cost"] is not None and e["decision"].status == ADMITTED]
        best_admitted = max((e["cost"] for e in admitted), default=-np.inf)
        refinements = []
        if self.refiner is not None:
            unc = sorted((e for e in entries if e["cost"] is not None and e["decision"].status == UNCERTAIN
                          and e["decision"].evidence == "surrogate" and e["cost"] > best_admitted),
                         key=lambda e: -e["cost"])
            for e in unc[:self.refiner.config.max_per_iteration]:
                if self.refiner.budget_left() <= 0 and not self.refiner.result_path(e["spec"]).exists():
                    break
                rec = self.refiner.refine(e["spec"], self.screen, iteration=iteration, mpc_cost=e["cost"])
                refinements.append(rec)
                e["decision"] = self.screen.classify(e["spec"])
                e["refined"] = True

        for e in entries:
            d = e["decision"]
            self.log({"iteration": iteration, "phase": e["phase"], **e["spec"].to_dict(), "key": e["spec"].key,
                      "status": d.status, "evidence": d.evidence, "p_hat": d.p_hat, "lower": d.lower,
                      "upper": d.upper, "warnings": d.warnings, "surrogate_sha256": d.surrogate_sha256,
                      "cost": e["cost"], "refined": e.get("refined", False)})

        admitted = [e for e in entries if e["cost"] is not None and e["decision"].status == ADMITTED]
        n_evals = sum(e["cost"] is not None for e in entries)
        if not admitted:
            return AdversaryResult(None, -np.inf, n_evals, refinements)
        best = max(admitted, key=lambda e: (e["cost"], e["spec"].key))
        return AdversaryResult(best["spec"], best["cost"], n_evals, refinements)

    def admitted_pool(self, iteration: int, n_per_stratum: int) -> list[ScenarioSpec]:
        """A fixed, screened scenario sample (no evaluation) -- used by the
        random-search / maximin baselines and held-out evaluation."""
        out = []
        for si, stratum in enumerate(self.strata):
            configs, _ = _SAMPLERS[stratum.family](n_per_stratum * 4, stratum.blocker_side,
                                                    derive_seed(self.seed, si, iteration + 1), method=self.config.doe_method)
            specs = [ScenarioSpec.make(stratum, c.theta()) for c in configs]
            out += [s for s in specs if self._classify(s).status == ADMITTED][:n_per_stratum]
        return out


def adversary_config_dict(c: AdversaryConfig) -> dict:
    return asdict(c)
