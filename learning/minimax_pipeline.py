"""
Surrogate-assisted minimax tuning of ONE shared MPC parameter vector w:

    w* = argmin_{w in W} max_{(f, b, theta) admitted} J_eval(w, theta)

See docs/minimax_design.md for the definitions of w, J_eval and the
admissible set, and learning.minimax_core for the exchange algorithm.

Modes:
    pilot     toy minimax/maximin problems (no simulation), plus an optional
              closed-loop throughput benchmark (--benchmark-workers 1,2,4,8)
    optimize  start a new experiment from a JSON config (method: minimax |
              baseline | random_search | maximin)
    resume    continue an interrupted experiment (scientific config must match)
    evaluate  evaluate a fixed w (an experiment's best, or --w-json) on held-out
              scenarios and realization seeds
    report    summaries/plots from saved results; --compare <evaluation.json ...>

Examples:
    python -m learning.minimax_pipeline --mode pilot
    python -m learning.minimax_pipeline --mode optimize --config configs/minimax/smoke_unscreened.json
    python -m learning.minimax_pipeline --mode resume --experiment smoke_unscreened
    python -m learning.minimax_pipeline --mode evaluate --experiment smoke_unscreened --label heldout
    python -m learning.minimax_pipeline --mode report --experiment smoke_unscreened

Overrides: --set key.sub=<json value> (repeatable), e.g. --set n_workers=8 --set exchange.max_iterations=20
"""

import os

# Same reasoning as learning.feasibility_pipeline: pin BLAS/OMP to one thread per process before numpy
# is imported, so process-level parallelism (run_episode_jobs) never oversubscribes cores.
for _env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from learning.closed_loop import (
    EVALUATOR_VERSION, EvalCostConfig, ScenarioSpec, Stratum, run_episode_jobs, summarize_episodes,
)
from learning.feasibility_common import SCHEMA_VERSION
from learning.feasibility_dataset import read_jsonl
from learning.minimax_adversary import AdversaryConfig, ScenarioAdversary
from learning.minimax_core import (
    BoxAdversary, CountingOracle, ExchangeSettings, ExchangeState, FiniteAdversary, MasterSettings,
    best_iterate, random_search, run_exchange, solve_master, solve_master_finite, solve_maximin,
)
from learning.minimax_feasibility import (
    ADMITTED, FeasibilityScreen, ScreeningConfig, demonstrated_scenarios_from_dataset,
)
from learning.minimax_oracle import OracleRefinementConfig, OracleRefiner
from learning.minimax_store import ExperimentStore, episode_cache_key, git_commit
from learning.mpc import MPCConfig, MPCWeights, W_NAMES, w_from_unit, w_to_unit

DEFAULT_CONFIG = {
    "experiment_id": None,
    "artifacts_root": "artifacts/minimax",
    "method": "minimax",                    # minimax | baseline | random_search | maximin
    "seed": 0,
    "strata": [{"family": "cutin", "blocker_side": 1,
                "surrogate": "artifacts/feasibility/surrogates/cutin_side1.joblib"}],
    "controller": {"name": "mpc", "config": {}},
    "w0": None,                             # natural-unit dict; None -> MPCWeights() defaults
    "cost": {},
    "screening": {},
    "realization_seeds": [1000, 1001, 1002, 1003],
    "master": {},
    "exchange": {},
    "adversary": {},
    "oracle_refinement": {},
    "seed_pool_dataset": None,              # optional PPO scenarios.jsonl: Wilson-demonstrated rows seed the adversary
    "random_search": {"n_candidates": 32, "n_scenarios_per_stratum": 8},
    "heldout": {"sampling_seed": 10_000, "n_per_stratum": 16, "realization_seeds": list(range(5000, 5010))},
    "n_workers": 1,
}

BUDGET_STOP_PREFIXES = ("iteration budget", "oracle-evaluation budget", "wall-clock budget", "stagnation")


# ── Config ──────────────────────────────────────────────────────────────────
def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _apply_set(config: dict, assignment: str) -> None:
    path, _, raw = assignment.partition("=")
    keys = path.split(".")
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    try:
        d[keys[-1]] = json.loads(raw)
    except json.JSONDecodeError:
        d[keys[-1]] = raw


def normalize_config(config: dict) -> dict:
    """Fill every sub-config with its explicit defaults, so the stored config
    (and its hash) pins the values actually used even if code defaults change."""
    c = _deep_merge(DEFAULT_CONFIG, config)
    assert c["experiment_id"], "config needs an experiment_id"
    assert c["method"] in ("minimax", "baseline", "random_search", "maximin"), c["method"]
    assert c["controller"]["name"] == "mpc", "the pipeline tunes the MPC controller (see learning.mpc)"
    c["controller"]["config"] = MPCConfig.from_dict(c["controller"]["config"]).to_dict()
    c["w0"] = asdict(MPCWeights(**(c["w0"] or {})))
    c["cost"] = asdict(EvalCostConfig(**c["cost"]))
    c["screening"] = asdict(ScreeningConfig(**c["screening"]))
    c["master"] = asdict(MasterSettings(**c["master"]))
    c["exchange"] = asdict(ExchangeSettings(**c["exchange"]))
    c["adversary"] = asdict(AdversaryConfig(**c["adversary"]))
    orc = asdict(OracleRefinementConfig.from_dict(c["oracle_refinement"]))
    orc["ppo_seeds"] = list(orc["ppo_seeds"])
    c["oracle_refinement"] = orc
    c["realization_seeds"] = [int(s) for s in c["realization_seeds"]]
    assert len(set(c["realization_seeds"])) == len(c["realization_seeds"]) > 0
    labels = [Stratum(s["family"], s["blocker_side"]).label for s in c["strata"]]
    assert len(set(labels)) == len(labels), f"duplicate strata: {labels}"
    if c["screening"]["policy"] != "none":
        missing = [l for l, s in zip(labels, c["strata"]) if not s.get("surrogate")]
        assert not missing, f"screening policy {c['screening']['policy']!r} needs a surrogate for {missing}"
    return c


# ── Experiment ──────────────────────────────────────────────────────────────
class Experiment:
    def __init__(self, config: dict, store: ExperimentStore, *, train_fn=None, validate_fn=None):
        self.config, self.store = config, store
        self.strata = [Stratum(s["family"], s["blocker_side"]) for s in config["strata"]]
        self.cost = EvalCostConfig(**config["cost"])
        self.cost_dict = asdict(self.cost)
        self.controller_spec = config["controller"]
        self.seeds = config["realization_seeds"]
        self.n_workers = int(config["n_workers"])
        sc = ScreeningConfig(**config["screening"])
        paths = {Stratum(s["family"], s["blocker_side"]).label: s["surrogate"]
                 for s in config["strata"] if sc.policy != "none" and s.get("surrogate")}
        self.screen = FeasibilityScreen(sc, paths)
        self.w0_unit = w_to_unit(MPCWeights(**config["w0"]).to_vector())
        self.oracle = CountingOracle(self._J, key=lambda s: s.key)
        self.n_episodes_simulated = 0
        self.context = {"iteration": -1, "phase": "init"}
        orc = OracleRefinementConfig.from_dict(config["oracle_refinement"])
        self.refiner = (OracleRefiner(orc, store.path("oracle"), train_fn=train_fn, validate_fn=validate_fn)
                        if orc.enabled else None)
        pool = []
        if config["seed_pool_dataset"]:
            # PPO-demonstrated dataset scenarios are admitted on that oracle evidence (Wilson rule), not re-screened
            rows = read_jsonl(Path(config["seed_pool_dataset"]))
            kn = {r["scenario_id"]: (r["k"], r["n"]) for r in rows}
            for s in demonstrated_scenarios_from_dataset(rows, sc.tau_demonstrated):
                if s.stratum in self.strata:
                    self.screen.record_oracle_evidence(s, *kn[s.scenario_id], source=config["seed_pool_dataset"])
                    pool.append(s)
        self.adversary = ScenarioAdversary(self.strata, self.screen, self._adversary_cost,
                                           AdversaryConfig(**config["adversary"]), config["seed"], pool=pool,
                                           refiner=self.refiner, log=self._log_candidate)

    # -- oracle: J_eval(w, scenarios), unit-cube w, episode-level disk cache --
    def _w(self, u) -> np.ndarray:
        return w_from_unit(u)

    def episode_records(self, w: np.ndarray, specs: list[ScenarioSpec], seeds: list[int]) -> dict:
        jobs, slots, out = [], [], {}
        for sp in specs:
            for seed in seeds:
                key = episode_cache_key(self.controller_spec, w, sp.key, seed, self.cost_dict,
                                        EVALUATOR_VERSION, SCHEMA_VERSION)
                rec = self.store.cache_get(key)
                if rec is None:
                    jobs.append((self.controller_spec, w.tolist(), sp.to_dict(), seed, self.cost_dict))
                    slots.append((sp.key, seed, key))
                else:
                    out[(sp.key, seed)] = rec
        for (skey, seed, key), rec in zip(slots, run_episode_jobs(jobs, self.n_workers)):
            self.store.cache_put(key, rec)
            out[(skey, seed)] = rec
        self.n_episodes_simulated += len(jobs)
        return out

    def _J(self, u, specs: list[ScenarioSpec]) -> np.ndarray:
        w = self._w(u)
        recs = self.episode_records(w, specs, self.seeds)
        costs = np.array([np.mean([recs[(s.key, seed)]["cost"] for seed in self.seeds]) for s in specs])
        self.store.append("controller_candidates", {
            **self.context, "w": w.tolist(), "w_unit": np.asarray(u).tolist(), "scenario_keys": [s.key for s in specs],
            "costs": costs.tolist(), "max_cost": float(costs.max()),
            "n_success": [sum(recs[(s.key, seed)]["success"] for seed in self.seeds) for s in specs]})
        return costs

    def _adversary_cost(self, u, specs):
        return self.oracle(u, specs)

    def _log_candidate(self, rec):
        self.store.append("adversarial_candidates", rec)

    # -- manifest / checkpoint --
    def write_manifest(self) -> None:
        self.store.write_json("manifest.json", {
            "git_commit": git_commit(), "created_unix": time.time(), "evaluator_version": EVALUATOR_VERSION,
            "scenario_schema_version": SCHEMA_VERSION, "w_names": list(W_NAMES),
            "surrogates": self.screen.identities, "screening": self.config["screening"],
            "realization_seeds": self.seeds,
            "note": "Empirical minimax over PPO-screened scenarios; no robustness certificate is implied."})

    def check_manifest(self) -> None:
        m = self.store.read_json("manifest.json")
        assert m is not None, "missing manifest.json"
        for label, ident in self.screen.identities.items():
            old = m["surrogates"].get(label)
            if old is None or old["sha256"] != ident["sha256"]:
                raise ValueError(f"surrogate for {label} changed since the experiment started "
                                 f"({(old or {}).get('sha256', 'none')[:12]} -> {ident['sha256'][:12]}); "
                                 f"start a new experiment")
        if m["evaluator_version"] != EVALUATOR_VERSION:
            raise ValueError(f"evaluator version changed ({m['evaluator_version']} -> {EVALUATOR_VERSION})")

    def save_state(self, state: ExchangeState) -> None:
        self.store.write_json("checkpoints/state.json", {
            "scenarios": [s.to_dict() for s in state.scenarios], "k": state.k, "w_prev": state.w_prev,
            "iterations": state.iterations, "stop_reason": state.stop_reason,
            "oracle_evidence": self.screen.oracle_evidence, "oracle_n_evals": self.oracle.n_evals,
            "refinements_used": self.refiner.n_used if self.refiner else 0})

    def load_state(self) -> ExchangeState | None:
        d = self.store.read_json("checkpoints/state.json")
        if d is None:
            return None
        self.screen.oracle_evidence.update(d["oracle_evidence"])
        self.oracle.n_evals = d["oracle_n_evals"]
        if self.refiner:
            self.refiner.n_used = d["refinements_used"]
        return ExchangeState([ScenarioSpec.from_dict(s) for s in d["scenarios"]], d["k"], d["w_prev"],
                             d["iterations"], d["stop_reason"])

    # -- methods --
    def _master(self, objective, incumbents, k):
        self.context = {"iteration": k, "phase": "master"}
        return solve_master(objective, np.zeros(len(W_NAMES)), np.ones(len(W_NAMES)),
                            MasterSettings(**self.config["master"]), incumbents, k)

    def run(self, resume: bool = False) -> dict:
        t0 = time.monotonic()
        method = self.config["method"]
        if method == "minimax":
            result = self._run_minimax(resume)
        else:
            result = self._run_fixed_pool(method)
        result["runtime_s"] = time.monotonic() - t0
        result["counters"] = {"episodes_simulated": self.n_episodes_simulated, "cache_hits": self.store.cache_hits,
                              "oracle_pair_evals": self.oracle.n_evals,
                              "surrogate_calls": self.adversary.n_surrogate_calls,
                              "oracle_refinements": self.refiner.n_used if self.refiner else 0,
                              "solver_failures": self._solver_failures()}
        self.store.write_json("summary.json", result)
        self.store.write_json("best_weights.json", {"w": result["best_w"], "w_unit": result["best_w_unit"],
                                                    "names": list(W_NAMES), "method": method})
        return result

    def _solver_failures(self) -> int:
        n = 0
        for p in (self.store.dir / "cache").rglob("*.json") if (self.store.dir / "cache").exists() else []:
            n += json.loads(p.read_text()).get("solver_failures", 0)
        return n

    def _run_minimax(self, resume: bool) -> dict:
        state = None
        if resume:
            state = self.load_state()
            self.store.truncate_logs_to(state.k if state else 0)
            if state and state.stop_reason and state.stop_reason.startswith(BUDGET_STOP_PREFIXES):
                state.stop_reason = None
        settings = ExchangeSettings(**self.config["exchange"])

        def adversary_proxy_propose(w, k):
            self.context = {"iteration": k, "phase": "adversary"}
            res = self.adversary.propose(w, k)
            for rec in res.records:
                self.store.append("oracle_refinements", rec)
            return res

        proxy = type("AdvProxy", (), {"propose": staticmethod(adversary_proxy_propose)})()

        def on_iteration(st):
            self.store.append("iterations", {"iteration": st.k - 1, **st.iterations[-1]})
            self.save_state(st)

        state = run_exchange(self.oracle, proxy, self._master, self.w0_unit, settings, key=lambda s: s.key,
                             state=state, on_iteration=on_iteration)
        self.save_state(state)
        if not state.iterations:
            return {"method": "minimax", "stop_reason": state.stop_reason, "best_w": None, "best_w_unit": None}
        best = best_iterate(state)
        u = np.array(best["w"])
        self.context = {"iteration": state.k, "phase": "summary"}   # dropped (and recomputed) on a later resume
        final_costs = self.oracle(u, state.scenarios)
        worst_i = int(np.argmax(final_costs))
        return {"method": "minimax", "stop_reason": state.stop_reason, "iterations": len(state.iterations),
                "best_iteration": best["k"], "best_w": dict(zip(W_NAMES, self._w(u).tolist())),
                "best_w_unit": u.tolist(), "worst_observed_cost_at_best_w": best["estimate"],
                "worst_cost_over_final_set_at_best_w": float(final_costs.max()),
                "worst_scenario": state.scenarios[worst_i].to_dict(), "final_set_size": len(state.scenarios),
                "final_set": [s.key for s in state.scenarios],
                "claim": "approximate empirical minimax point; inner and outer problems solved approximately"}

    def _run_fixed_pool(self, method: str) -> dict:
        self.context = {"iteration": 0, "phase": "pool"}
        rs = self.config["random_search"]
        pool = self.adversary.admitted_pool(0, rs["n_scenarios_per_stratum"])
        assert pool, "no admitted scenarios for the fixed pool"
        objective = lambda u: float(np.max(self.oracle(u, pool)))
        d = len(W_NAMES)
        if method == "baseline":
            m = solve_master_finite(objective, [self.w0_unit])
        elif method == "random_search":
            m = random_search(objective, np.zeros(d), np.ones(d), rs["n_candidates"], self.config["seed"],
                              include=[self.w0_unit])
        else:   # maximin -- secondary benchmark, NOT the primary objective
            mm = solve_maximin(self.oracle, pool, self._master)
            return {"method": "maximin", "maximin_value": mm["value"], "argmax_scenario": pool[mm["argmax_index"]].key,
                    "best_w": dict(zip(W_NAMES, self._w(mm["w_at_argmax"]).tolist())), "best_w_unit": mm["w_at_argmax"],
                    "per_scenario": mm["per_scenario"], "pool": [s.key for s in pool],
                    "note": "max_theta min_w J (secondary benchmark); the w shown is only optimal for its own theta"}
        costs = self.oracle(m.x, pool)
        return {"method": method, "best_w": dict(zip(W_NAMES, self._w(m.x).tolist())), "best_w_unit": m.x.tolist(),
                "worst_cost_over_pool_at_best_w": float(costs.max()), "worst_scenario": pool[int(np.argmax(costs))].to_dict(),
                "pool": [s.key for s in pool], "n_candidates": m.n_objective_calls}

    # -- held-out evaluation of a fixed w --
    def heldout_pool(self, heldout: dict) -> list[ScenarioSpec]:
        adv = ScenarioAdversary(self.strata, self.screen, None, AdversaryConfig(**self.config["adversary"]),
                                heldout["sampling_seed"])
        return adv.admitted_pool(0, heldout["n_per_stratum"])

    def evaluate(self, u, label: str, heldout: dict) -> dict:
        t0 = time.monotonic()
        seeds = [int(s) for s in heldout["realization_seeds"]]
        overlap = set(seeds) & set(self.seeds)
        assert not overlap, f"held-out realization seeds overlap the optimization seeds: {sorted(overlap)}"
        pool = self.heldout_pool(heldout)
        seen = {r["key"] for r in self.store.read_log("adversarial_candidates")}
        pool = [s for s in pool if s.key not in seen]   # truly held-out scenarios only
        w = self._w(u)
        recs = self.episode_records(w, pool, seeds)
        per = []
        for s in pool:
            summ = summarize_episodes([recs[(s.key, seed)] for seed in seeds])
            per.append({"key": s.key, "theta": s.theta_dict, **{k: v for k, v in summ.items() if k != "per_episode_costs"}})
        means = np.array([p["mean_cost"] for p in per])
        allep = [recs[(s.key, seed)] for s in pool for seed in seeds]
        agg = summarize_episodes(allep)
        out = {"label": label, "w": dict(zip(W_NAMES, w.tolist())), "w_unit": np.asarray(u).tolist(),
               "heldout": heldout, "pool_keys": [s.key for s in pool], "n_scenarios": len(pool),
               "worst_scenario_cost": float(means.max()), "mean_scenario_cost": float(means.mean()),
               "q90_scenario_cost": float(np.quantile(means, 0.9)), "q95_scenario_cost": float(np.quantile(means, 0.95)),
               "success_rate": agg["success_rate"], "collision_rate": agg["n_collisions"] / agg["n_episodes"],
               "rollover_rate": agg["n_rollovers"] / agg["n_episodes"], "offroad_rate": agg["n_offroad"] / agg["n_episodes"],
               "timeout_rate": agg["n_timeouts"] / agg["n_episodes"], "n_episodes": agg["n_episodes"],
               "runtime_s": time.monotonic() - t0, "per_scenario": per}
        self.store.write_json(f"evaluations/{label}.json", out)
        return out


# ── Report ──────────────────────────────────────────────────────────────────
def report(store: ExperimentStore) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    its = store.read_log("iterations")
    summ = store.read_json("summary.json") or {}
    cand = store.read_log("adversarial_candidates")
    status_counts = {}
    for c in cand:
        status_counts[c["status"]] = status_counts.get(c["status"], 0) + 1
    out = {"experiment": store.dir.name, "summary": summ, "n_iterations": len(its),
           "candidate_status_counts": status_counts, "n_oracle_refinements": len(store.read_log("oracle_refinements"))}
    if its:
        (store.dir / "plots").mkdir(exist_ok=True)
        k = [r["iteration"] for r in its]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(k, [r["F_S"] for r in its], "o-", label="F_S(w_k) = max over current set")
        ax[0].plot(k, [r["adversary_cost"] if r["adversary_cost"] is not None else np.nan for r in its], "s--",
                   label="adversary's best J(w_k, theta_new)")
        ax[0].plot(k, [r["estimate"] for r in its], "k:", label="worst observed at w_k")
        ax[0].set_xlabel("iteration"); ax[0].set_ylabel("J_eval"); ax[0].legend(fontsize=8)
        W = np.array([r["w"] for r in its])
        for j, n in enumerate(W_NAMES):
            ax[1].plot(k, W[:, j], "o-", label=n)
        ax[1].set_xlabel("iteration"); ax[1].set_ylabel("w (unit-cube coordinates)"); ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(store.dir / "plots" / "convergence.png", dpi=120)
        plt.close(fig)
    return out


def compare(eval_paths: list[str]) -> list[dict]:
    rows = []
    pools = set()
    for p in eval_paths:
        e = json.loads(Path(p).read_text())
        pools.add(tuple(sorted(e["pool_keys"])) + tuple(e["heldout"]["realization_seeds"]))
        rows.append({"file": p, **{k: e[k] for k in ("label", "n_scenarios", "worst_scenario_cost", "mean_scenario_cost",
                                                      "q90_scenario_cost", "q95_scenario_cost", "success_rate",
                                                      "collision_rate", "rollover_rate", "offroad_rate", "timeout_rate")}})
    if len(pools) > 1:
        print("WARNING: these evaluations used DIFFERENT held-out scenario pools or seeds -- not directly comparable.")
    return rows


# ── Pilot ───────────────────────────────────────────────────────────────────
def pilot(benchmark_workers: list[int], benchmark_episodes: int) -> dict:
    f = lambda w, th: float((w[0] - th[0]) ** 2)
    st = run_exchange(CountingOracle(lambda w, S: np.array([f(w, s) for s in S])),
                      BoxAdversary(f, [-1.0], [1.0], n_samples=16),
                      lambda obj, inc, k: solve_master(obj, [-1.0], [1.0], MasterSettings(n_init=8, xatol=1e-6), inc, k),
                      np.array([0.5]), ExchangeSettings(max_iterations=10, improvement_tol=1e-4), key=repr)
    b = best_iterate(st)
    print(f"[toy] J=(w-theta)^2 on [-1,1]^2: w*={b['w'][0]:+.4f} (exact 0), J*={b['estimate']:.4f} (exact 1), "
          f"set={sorted(round(s[0], 3) for s in st.scenarios)}, stop: {st.stop_reason}")
    M = np.array([[3.0, 0.0], [1.0, 2.0]])
    mo = lambda w, S: np.array([M[int(round(w[0])), j] for j in S])
    fm = lambda obj, inc, k: solve_master_finite(obj, [np.array([0.0]), np.array([1.0])])
    st2 = run_exchange(CountingOracle(mo), FiniteAdversary(mo, [0, 1]), fm, np.array([0.0]), ExchangeSettings(), key=repr)
    mm = solve_maximin(mo, [0, 1], fm)
    print(f"[toy] matrix {M.tolist()}: minimax={best_iterate(st2)['estimate']} at w={best_iterate(st2)['w']}, "
          f"maximin={mm['value']} -- order matters")
    out = {"toy_w": b["w"][0], "toy_J": b["estimate"]}
    if benchmark_workers:
        from learning.feasibility_cutin import CutinConfig
        spec = ScenarioSpec.make(Stratum("cutin", 1), CutinConfig().theta())
        jobs = [({"name": "mpc", "config": {}}, MPCWeights().to_vector().tolist(), spec.to_dict(), 100 + i,
                 asdict(EvalCostConfig())) for i in range(benchmark_episodes)]
        out["benchmark"] = {}
        for n in benchmark_workers:
            t = time.monotonic()
            run_episode_jobs(jobs, n)
            dt = time.monotonic() - t
            out["benchmark"][n] = benchmark_episodes / dt
            print(f"[bench] n_workers={n:>3}: {benchmark_episodes / dt:6.2f} closed-loop episodes/s")
        best_n = max(out["benchmark"], key=out["benchmark"].get)
        print(f"[bench] highest throughput at n_workers={best_n} -- set \"n_workers\" accordingly")
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["pilot", "optimize", "resume", "evaluate", "report"])
    ap.add_argument("--config", help="experiment JSON (optimize)")
    ap.add_argument("--experiment", help="experiment id (resume/evaluate/report)")
    ap.add_argument("--artifacts-root", default=DEFAULT_CONFIG["artifacts_root"])
    ap.add_argument("--set", action="append", default=[], help="override: key.sub=<json>")
    ap.add_argument("--label", default="heldout", help="evaluate: output name")
    ap.add_argument("--w-json", help="evaluate: JSON file with natural-unit w ({name: value}) instead of best_weights")
    ap.add_argument("--heldout-config", help="evaluate: JSON with a 'heldout' section (use the same one to compare methods)")
    ap.add_argument("--compare", nargs="+", help="report: evaluation JSON files to tabulate")
    ap.add_argument("--benchmark-workers", default="", help="pilot: e.g. 1,2,4,8")
    ap.add_argument("--benchmark-episodes", type=int, default=16)
    args = ap.parse_args()

    if args.mode == "pilot":
        pilot([int(x) for x in args.benchmark_workers.split(",") if x], args.benchmark_episodes)
        return

    if args.mode == "report" and args.compare:
        rows = compare(args.compare)
        cols = list(rows[0].keys())
        print("\t".join(cols))
        for r in rows:
            print("\t".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols))
        return

    if args.mode == "optimize":
        assert args.config, "--config is required for optimize"
        raw = json.loads(Path(args.config).read_text())
        for s in args.set:
            _apply_set(raw, s)
        config = normalize_config(raw)
        store = ExperimentStore(config["artifacts_root"], config["experiment_id"])
        if store.exists():
            raise SystemExit(f"experiment {config['experiment_id']} already exists at {store.dir} -- use --mode resume")
        store.init_config(config)
        exp = Experiment(config, store)
        exp.write_manifest()
        print(json.dumps(exp.run(), indent=2, default=str))
        return

    assert args.experiment, "--experiment is required"
    store = ExperimentStore(args.artifacts_root, args.experiment)
    stored = store.read_json("config.json")
    assert stored is not None, f"no experiment at {store.dir}"
    config = copy.deepcopy(stored["config"])
    for s in args.set:
        _apply_set(config, s)
    config = normalize_config(config)

    if args.mode == "report":
        print(json.dumps(report(store), indent=2, default=str))
        return

    store.check_config(config)
    exp = Experiment(config, store)
    exp.check_manifest()
    if args.mode == "resume":
        print(json.dumps(exp.run(resume=True), indent=2, default=str))
    else:   # evaluate
        if args.w_json:
            u = w_to_unit(MPCWeights(**json.loads(Path(args.w_json).read_text())).to_vector())
        else:
            bw = store.read_json("best_weights.json")
            assert bw is not None and bw["w_unit"] is not None, "no best_weights.json -- run optimize first or pass --w-json"
            u = np.array(bw["w_unit"])
        heldout = json.loads(Path(args.heldout_config).read_text())["heldout"] if args.heldout_config else config["heldout"]
        res = exp.evaluate(u, args.label, heldout)
        print(json.dumps({k: v for k, v in res.items() if k != "per_scenario"}, indent=2, default=str))


if __name__ == "__main__":
    main()
