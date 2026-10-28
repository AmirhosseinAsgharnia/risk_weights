"""
Constrained minimax tuning of ONE shared PARALO-MPC weight vector w
(docs/constrained_design.md):

    w* = argmin_w  max_{theta in Theta_f} J_progress(w, theta)
         s.t.      max_{theta in Theta_f} R_i(w, theta) <= eps_i   for every enforced constraint i

J_progress is the mean progress shortfall over fixed realization seeds and R_i
are physical event rates / peaks (learning.physical_eval) -- there is no
evaluation weight vector. Theta_f is the PPO-admitted set (learning.minimax_feasibility,
unchanged). The original MPC + J_eval pipeline (learning.minimax_pipeline) is
kept untouched as the baseline.

Modes:
    optimize  start a new experiment from a JSON config
    resume    continue an interrupted / budget-stopped experiment (scientific config must match)
    compare   evaluate the original MPC, the PARALO MPC at w0, and at the experiment's best w
              on the SAME held-out admitted scenarios and seeds, with the same physical metrics
    report    summary, per-iteration table and plots

Examples:
    python -m learning.constrained_pipeline --mode optimize --config configs/constrained/smoke_unscreened.json
    python -m learning.constrained_pipeline --mode resume --experiment c_smoke --set exchange.max_iterations=4
    python -m learning.constrained_pipeline --mode compare --experiment c_smoke --label heldout
    python -m learning.constrained_pipeline --mode report --experiment c_smoke

Overrides: --set key.sub=<json value> (repeatable), e.g. --set n_workers=8 --set exchange.max_iterations=20
"""

import os

for _env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from learning.closed_loop import ScenarioSpec, Stratum
from learning.constrained_adversary import ConstrainedScenarioAdversary
from learning.constrained_core import (
    ConstrainedExchangeSettings, ConstrainedExchangeState, ExchangeStatus, run_constrained_exchange,
    select_final, solve_constrained_master,
)
from learning.feasibility_common import SCHEMA_VERSION, ScenarioFamilyConfigError
from learning.feasibility_dataset import read_jsonl
from learning.minimax_adversary import AdversaryConfig
from learning.minimax_core import MasterSettings
from learning.minimax_feasibility import FeasibilityScreen, ScreeningConfig, demonstrated_scenarios_from_dataset
from learning.minimax_pipeline import _apply_set, _deep_merge
from learning.minimax_oracle import OracleRefinementConfig, OracleRefiner
from learning.minimax_store import ExperimentStore, episode_cache_key, git_commit
from learning.mpc import MPCWeights, W_NAMES
from learning.paralo_mpc import PARALO_MPC_VERSION, ParaloMPCConfig, ParaloWeights, PW_NAMES, pw_from_unit, pw_to_unit
from learning.physical_eval import (
    DEFAULT_CONSTRAINTS, DIAG_PEAKS, EVENTS, OBJECTIVES, PEAKS, PHYSICAL_EVALUATOR_VERSION, constraints_from_config,
    objective_value, run_physical_jobs, summarize_scenario,
)
from learning.vehicle_physics import ROLLOVER_EVENT_MODELS, VEHICLE_PHYSICS_VERSION

DEFAULT_CONFIG = {
    "experiment_id": None,
    "artifacts_root": "artifacts/constrained",
    "seed": 0,
    "strata": [{"family": "cutin", "blocker_side": 1,
                "surrogate": "artifacts/feasibility/surrogates/cutin_side1.joblib"}],
    "controller": {"name": "paralo_mpc", "config": {}},
    "event_model": "physical_v2",          # simulator ROLLOVER detector for every evaluation (learning.vehicle_physics)
    "objective": "progress_shortfall",     # or "completion_time" (lexicographic, see learning.physical_eval)
    "w0": None,                             # natural-unit dict; None -> ParaloWeights() defaults
    "constraints": [dict(c) for c in DEFAULT_CONSTRAINTS],
    "screening": {},
    "seed_pool_dataset": None,             # raw scenarios.jsonl path (rows filtered by the CURRENT validator only)
    "seed_pool_curated": False,            # True: per-stratum curated dataset (learning.feasibility_audit) instead
    "oracle_refinement": {},
    "realization_seeds": [1000, 1001, 1002, 1003],
    "master": {},
    "exchange": {},
    "adversary": {},
    "heldout": {"sampling_seed": 10_000, "n_per_stratum": 16, "realization_seeds": list(range(5000, 5010))},
    "baselines": [{"label": "original_mpc_legacy_events", "controller": {"name": "mpc", "config": {}}, "w": None,
                   "event_model": "legacy_v1"},
                  {"label": "original_mpc", "controller": {"name": "mpc", "config": {}}, "w": None}],
    "n_workers": 1,
}


def _evaluator_id(event_model: str) -> dict:
    return {"evaluator": "physical", "version": PHYSICAL_EVALUATOR_VERSION, "event_model": event_model}


def normalize_config(config: dict) -> dict:
    c = _deep_merge(DEFAULT_CONFIG, config)
    if "constraints" in config:   # a list replaces the defaults wholesale (no element-wise merge)
        c["constraints"] = copy.deepcopy(config["constraints"])
    assert c["experiment_id"], "config needs an experiment_id"
    assert c["controller"]["name"] == "paralo_mpc", "this pipeline tunes the PARALO MPC (learning.paralo_mpc)"
    c["controller"]["config"] = ParaloMPCConfig.from_dict(c["controller"]["config"]).to_dict()
    c["w0"] = asdict(ParaloWeights(**(c["w0"] or {})))
    c["constraints"] = [asdict(s) for s in constraints_from_config(c["constraints"])]
    c["screening"] = asdict(ScreeningConfig(**c["screening"]))
    c["master"] = asdict(MasterSettings(**c["master"]))
    c["exchange"] = asdict(ConstrainedExchangeSettings(**c["exchange"]))
    c["adversary"] = asdict(AdversaryConfig(**c["adversary"]))
    assert c["event_model"] in ROLLOVER_EVENT_MODELS and c["objective"] in OBJECTIVES
    orc = asdict(OracleRefinementConfig.from_dict({"rollover_event": c["event_model"], **c["oracle_refinement"]}))
    assert orc["rollover_event"] == c["event_model"], "oracle refinement must use the evaluator's rollover event model"
    orc["ppo_seeds"] = list(orc["ppo_seeds"])
    c["oracle_refinement"] = orc
    for b in c["baselines"]:
        b.setdefault("event_model", c["event_model"])
        assert b["event_model"] in ROLLOVER_EVENT_MODELS
    c["realization_seeds"] = [int(s) for s in c["realization_seeds"]]
    assert len(set(c["realization_seeds"])) == len(c["realization_seeds"]) > 0
    labels = [Stratum(s["family"], s["blocker_side"]).label for s in c["strata"]]
    assert len(set(labels)) == len(labels), f"duplicate strata: {labels}"
    if c["screening"]["policy"] != "none":
        missing = [l for l, s in zip(labels, c["strata"]) if not s.get("surrogate")]
        assert not missing, f"screening policy {c['screening']['policy']!r} needs a surrogate for {missing}"
    return c


class ConstrainedExperiment:
    def __init__(self, config: dict, store: ExperimentStore, *, train_fn=None, validate_fn=None):
        self.config, self.store = config, store
        self.strata = [Stratum(s["family"], s["blocker_side"]) for s in config["strata"]]
        self.constraints = constraints_from_config(config["constraints"])
        self.controller_spec = config["controller"]
        self.seeds = config["realization_seeds"]
        self.event_model, self.objective = config["event_model"], config["objective"]
        self.n_workers = int(config["n_workers"])
        sc = ScreeningConfig(**config["screening"])
        paths = {Stratum(s["family"], s["blocker_side"]).label: s["surrogate"]
                 for s in config["strata"] if sc.policy != "none" and s.get("surrogate")}
        self.screen = FeasibilityScreen(sc, paths)
        self.u0 = pw_to_unit(ParaloWeights(**config["w0"]).to_vector())
        self.cache: dict[tuple[bytes, str], dict] = {}
        self.n_scenario_evals = 0
        self.n_episodes_simulated = 0
        self.context = {"iteration": -1, "phase": "init"}
        pool, self.pool_skipped_invalid, self.pool_provenance = [], [], {}
        sources = []
        if config["seed_pool_dataset"]:
            sources.append((config["seed_pool_dataset"], read_jsonl(Path(config["seed_pool_dataset"]))))
        if config["seed_pool_curated"]:
            from learning.feasibility_audit import curated_dataset
            for st in self.strata:
                cur = curated_dataset(st.family, st.blocker_side)
                self.pool_provenance[st.label] = cur["provenance"]
                sources.append((f"curated:{st.label}", cur["scenarios"]))
        for src, rows in sources:
            kn = {r["scenario_id"]: (r["k"], r["n"]) for r in rows}
            for s in demonstrated_scenarios_from_dataset(rows, sc.tau_demonstrated):
                try:
                    s.validate()   # rows sampled under older, wider theta bounds fail the current validator
                except ScenarioFamilyConfigError:
                    self.pool_skipped_invalid.append(s.key)
                    continue
                if s.stratum in self.strata and s not in pool:
                    self.screen.record_oracle_evidence(s, *kn[s.scenario_id], source=src)
                    pool.append(s)
        orc = OracleRefinementConfig.from_dict(config["oracle_refinement"])
        self.refiner = (OracleRefiner(orc, store.path("oracle"), train_fn=train_fn, validate_fn=validate_fn)
                        if orc.enabled else None)
        self.adversary = ConstrainedScenarioAdversary(
            self.strata, self.screen, self._adversary_metrics, self.constraints, AdversaryConfig(**config["adversary"]),
            config["seed"], pool=pool, log=lambda r: self.store.append("adversarial_candidates", r), refiner=self.refiner)

    # -- physical oracle: cached per episode on disk, per (w, scenario) in memory --
    def episode_records(self, controller_spec: dict, w, specs: list[ScenarioSpec], seeds: list[int],
                        event_model: str | None = None) -> dict:
        event_model = event_model or self.event_model
        jobs, slots, out = [], [], {}
        for sp in specs:
            for seed in seeds:
                key = episode_cache_key(controller_spec, w, sp.key, seed, _evaluator_id(event_model),
                                        PHYSICAL_EVALUATOR_VERSION, SCHEMA_VERSION)
                rec = self.store.cache_get(key)
                if rec is None:
                    jobs.append((controller_spec, list(map(float, w)), sp.to_dict(), seed, event_model))
                    slots.append((sp.key, seed, key))
                else:
                    out[(sp.key, seed)] = rec
        for (skey, seed, key), rec in zip(slots, run_physical_jobs(jobs, self.n_workers)):
            self.store.cache_put(key, rec)
            out[(skey, seed)] = rec
        self.n_episodes_simulated += len(jobs)
        return out

    def metrics(self, u, specs: list[ScenarioSpec]) -> list[dict]:
        u = np.asarray(u, dtype=float)
        missing = [s for s in specs if (u.tobytes(), s.key) not in self.cache]
        if missing:
            w = pw_from_unit(u)
            recs = self.episode_records(self.controller_spec, w, missing, self.seeds)
            for s in missing:
                summ = summarize_scenario([recs[(s.key, seed)] for seed in self.seeds])
                self.cache[(u.tobytes(), s.key)] = {"objective": objective_value(summ, self.objective),
                                                    "values": {c.name: c.value(summ) for c in self.constraints},
                                                    "summary": summ}
            self.n_scenario_evals += len(missing)
        out = [self.cache[(u.tobytes(), s.key)] for s in specs]
        if self.context["phase"] in ("master", "final"):
            self.store.append("controller_candidates", {
                **self.context, "w": pw_from_unit(u).tolist(), "w_unit": u.tolist(), "scenario_keys": [s.key for s in specs],
                "objectives": [o["objective"] for o in out],
                "values": {c.name: [o["values"][c.name] for o in out] for c in self.constraints}})
        return out

    def _adversary_metrics(self, u, specs):
        return self.metrics(u, specs)

    # -- provenance / checkpoint --
    def versions(self) -> dict:
        from learning.feasibility_pipeline import CURRENT_BOUNDS_VERSION
        return {"physical_evaluator": PHYSICAL_EVALUATOR_VERSION, "rollover_event_model": self.event_model,
                "vehicle_physics": VEHICLE_PHYSICS_VERSION, "mpc": PARALO_MPC_VERSION,
                "theta_bounds": {st.label: CURRENT_BOUNDS_VERSION[st.family] for st in self.strata},
                "surrogates": {l: {"sha256": i["sha256"], "provenance": self.screen.artifacts[l].provenance}
                               for l, i in self.screen.identities.items()},
                "oracle_refinement": self.config["oracle_refinement"], "objective": self.objective}

    def write_manifest(self) -> None:
        self.store.write_json("manifest.json", {
            "versions": self.versions(), "seed_pool_provenance": self.pool_provenance,
            "git_commit": git_commit(), "created_unix": time.time(), "evaluator_version": PHYSICAL_EVALUATOR_VERSION,
            "scenario_schema_version": SCHEMA_VERSION, "w_names": list(PW_NAMES), "surrogates": self.screen.identities,
            "screening": self.config["screening"], "realization_seeds": self.seeds,
            "constraints": self.config["constraints"], "seed_pool_skipped_invalid": self.pool_skipped_invalid,
            "note": "Constrained empirical minimax over PPO-screened scenarios; no robustness or safety certificate."})

    def check_manifest(self) -> None:
        m = self.store.read_json("manifest.json")
        assert m is not None, "missing manifest.json"
        for label, ident in self.screen.identities.items():
            old = m["surrogates"].get(label)
            if old is None or old["sha256"] != ident["sha256"]:
                raise ValueError(f"surrogate for {label} changed since the experiment started; start a new experiment")
        if m["evaluator_version"] != PHYSICAL_EVALUATOR_VERSION:
            raise ValueError(f"evaluator version changed ({m['evaluator_version']} -> {PHYSICAL_EVALUATOR_VERSION})")
        old = m.get("versions", {})
        for k in ("rollover_event_model", "vehicle_physics", "mpc"):
            if old.get(k) != self.versions()[k]:
                raise ValueError(f"{k} changed since the experiment started ({old.get(k)} -> {self.versions()[k]})")

    def save_state(self, st: ConstrainedExchangeState) -> None:
        self.store.write_json("checkpoints/state.json", {
            "scenarios": [s.to_dict() for s in st.scenarios], "k": st.k, "u_prev": st.u_prev,
            "iterations": st.iterations, "status": st.status, "stop_reason": st.stop_reason,
            "oracle_evidence": self.screen.oracle_evidence, "n_scenario_evals": self.n_scenario_evals,
            "refinements_used": self.refiner.n_used if self.refiner else 0})

    def load_state(self) -> ConstrainedExchangeState | None:
        d = self.store.read_json("checkpoints/state.json")
        if d is None:
            return None
        self.screen.oracle_evidence.update(d["oracle_evidence"])
        self.n_scenario_evals = d["n_scenario_evals"]
        if self.refiner:
            self.refiner.n_used = d.get("refinements_used", 0)
        return ConstrainedExchangeState([ScenarioSpec.from_dict(s) for s in d["scenarios"]], d["k"], d["u_prev"],
                                        d["iterations"], d["status"], d["stop_reason"])

    # -- run --
    def run(self, resume: bool = False) -> dict:
        t0 = time.monotonic()
        st = None
        if resume:
            st = self.load_state()
            self.store.truncate_logs_to(st.k if st else 0)
            if st and st.status == ExchangeStatus.BUDGET_EXHAUSTED:
                st.status, st.stop_reason = ExchangeStatus.RUNNING, None   # continue with the (possibly raised) budget

        def master(fn, incumbents, k):
            self.context = {"iteration": k, "phase": "master"}
            out = solve_constrained_master(fn, len(PW_NAMES), MasterSettings(**self.config["master"]), incumbents, k)
            self.context = {"iteration": k, "phase": "adversary"}
            return out

        def on_iteration(s):
            self.store.append("iterations", s.iterations[-1])
            self.save_state(s)

        self.context = {"iteration": -1, "phase": "adversary"}
        st = run_constrained_exchange(self.metrics, self.adversary, self.constraints, self.u0,
                                      MasterSettings(**self.config["master"]),
                                      ConstrainedExchangeSettings(**self.config["exchange"]),
                                      key=lambda s: s.key, n_evals=lambda: self.n_scenario_evals, state=st,
                                      on_iteration=on_iteration, master=master)
        self.save_state(st)
        self.context = {"iteration": st.k, "phase": "final"}
        fin = select_final(st, self.metrics, self.constraints)
        result = self._summary(st, fin)
        result["runtime_s"] = time.monotonic() - t0
        result["counters"] = {"scenario_evals": self.n_scenario_evals, "episodes_simulated": self.n_episodes_simulated,
                              "cache_hits": self.store.cache_hits, "surrogate_calls": self.adversary.n_surrogate_calls,
                              "oracle_refinements": self.refiner.n_used if self.refiner else 0}
        result["versions"] = self.versions()
        self.store.write_json("summary.json", result)
        self.store.write_json("best_weights.json", {"w": result["best_w"], "w_unit": result["best_w_unit"],
                                                    "names": list(PW_NAMES), "status": result["status"],
                                                    "feasible_on_final_set": result["feasible_on_final_set"]})
        return result

    def _summary(self, st: ConstrainedExchangeState, fin: dict) -> dict:
        out = {"status": st.status, "stop_reason": st.stop_reason, "iterations": len(st.iterations),
               "final_set_size": len(st.scenarios), "final_set": [s.key for s in st.scenarios],
               "best_w": None, "best_w_unit": None, "feasible_on_final_set": False,
               "claim": ("best solution found under the documented budget; inner and outer problems solved "
                         "approximately by sampling and local search -- no global optimality, robustness or "
                         "safety certification is implied")}
        if fin["u"] is None:
            return out
        a = fin["assessment"]
        sc = st.scenarios
        out.update({
            "best_w": dict(zip(PW_NAMES, pw_from_unit(fin["u"]).tolist())), "best_w_unit": fin["u"].tolist(),
            "best_iteration": fin["iteration"], "feasible_on_final_set": fin["feasible_on_final_set"],
            "worst_progress_shortfall": a.t, "worst_progress_scenario": sc[a.worst_objective_index].to_dict(),
            "constraints": {n: {"worst_value": w["value"], "violation": w["violation"], "violated": w["violated"],
                                "enforced": w["enforced"], "worst_scenario": sc[w["index"]].to_dict()}
                            for n, w in a.worst.items()},
            "violated_pairs": [[n, sc[i].key] for n, i in a.violated_pairs],
            "scenarios_never_satisfied_last_master": st.iterations[-1].get("scenarios_never_satisfied", []),
            "per_stratum_final_set": self._per_stratum(fin["u"], sc),
        })
        return out

    def _per_stratum(self, u, specs: list[ScenarioSpec]) -> dict:
        """Worst case of every criterion within each stratum of the final set, at the SAME (shared) w."""
        ms = self.metrics(u, specs)
        out = {}
        for st in self.strata:
            idx = [i for i, s in enumerate(specs) if s.stratum == st]
            if not idx:
                out[st.label] = None
                continue
            j = max(idx, key=lambda i: ms[i]["objective"])
            out[st.label] = {"n_scenarios": len(idx), "objective": {"value": ms[j]["objective"], "scenario": specs[j].key},
                             **{c.name: {"value": max(ms[i]["values"][c.name] for i in idx),
                                         "violated": any(c.violated(ms[i]["values"][c.name]) for i in idx)}
                                for c in self.constraints}}
        return out

    # -- held-out comparison --
    def heldout_pool(self, heldout: dict) -> list[ScenarioSpec]:
        adv = ConstrainedScenarioAdversary(self.strata, self.screen, None, self.constraints,
                                           AdversaryConfig(**self.config["adversary"]), heldout["sampling_seed"])
        seen = {r["key"] for r in self.store.read_log("adversarial_candidates")}
        return [s for s in adv.admitted_pool(0, heldout["n_per_stratum"]) if s.key not in seen]

    def compare(self, label: str, heldout: dict, extra_w: dict | None = None) -> dict:
        seeds = [int(s) for s in heldout["realization_seeds"]]
        overlap = set(seeds) & set(self.seeds)
        assert not overlap, f"held-out realization seeds overlap the optimization seeds: {sorted(overlap)}"
        pool = self.heldout_pool(heldout)
        assert pool, "no admitted held-out scenarios"
        entries = [(b["label"], b["controller"], b["w"], b["event_model"]) for b in self.config["baselines"]]
        entries.append(("paralo_mpc_w0", self.controller_spec, self.config["w0"], self.event_model))
        bw = self.store.read_json("best_weights.json")
        if bw and bw["w"]:
            entries.append((f"paralo_mpc_best[{bw['status']}]", self.controller_spec, bw["w"], self.event_model))
        if extra_w:
            entries.append(("paralo_mpc_custom", self.controller_spec, extra_w, self.event_model))
        rows = []
        for name, spec, w, event_model in entries:
            wv = (MPCWeights(**(w or {})).to_vector() if spec["name"] == "mpc" else ParaloWeights(**(w or {})).to_vector())
            recs = self.episode_records(spec, wv, pool, seeds, event_model)
            per = [summarize_scenario([recs[(s.key, seed)] for seed in seeds]) for s in pool]
            eps = [recs[(s.key, seed)] for s in pool for seed in seeds]
            n = len(eps)
            ms = [e["timing"]["ms_per_plan"] for e in eps if e["timing"]["ms_per_plan"] is not None]
            names = W_NAMES if spec["name"] == "mpc" else PW_NAMES
            row = {"label": name, "controller": spec["name"], "event_model": event_model,
                   "primary_comparison": event_model == self.event_model, "w": dict(zip(names, wv.tolist())),
                   "n_scenarios": len(pool), "n_episodes": n,
                   "completion_rate": sum(e["success"] for e in eps) / n,
                   **{f"{ev}_rate": sum(e["events"][ev] for e in eps) / n for ev in EVENTS},
                   "worst_progress_shortfall": max(p["progress_shortfall"] for p in per),
                   "mean_progress_shortfall": float(np.mean([p["progress_shortfall"] for p in per])),
                   "completion_time": float(np.mean([p["completion_time"] for p in per])),
                   **{p: max(e[p] for e in eps) for p in PEAKS + DIAG_PEAKS},
                   "mean_peak_abs_accel": float(np.mean([e["peak_abs_accel"] for e in eps])),
                   "mean_peak_abs_jerk": float(np.mean([e["peak_abs_jerk"] for e in eps])),
                   "ms_per_replan_mean": float(np.mean(ms)) if ms else None,
                   "constraints": {c.name: {"worst_value": max(c.value(p) for p in per),
                                            "violated": any(c.violated(c.value(p)) for p in per),
                                            "enforced": c.enforced} for c in self.constraints}}
            rows.append(row)
        out = {"label": label, "heldout": heldout, "pool_keys": [s.key for s in pool], "rows": rows,
               "versions": self.versions(),
               "note": "all controllers measured with the same physical metrics on the same scenarios and seeds; rows "
                       "with primary_comparison=false use the LEGACY event model (historical definitions, known "
                       "rollover false positives) and are not comparable as safety results; ms_per_replan is "
                       "wall-clock and machine-dependent"}
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
    counts = {}
    for c in cand:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    table = [{"iteration": r["iteration"], "status": r.get("status"), "t_S": r["master"]["t"],
              "feasible_on_S": r["master"]["feasible"], "n_violated": r["master"]["n_violated"],
              "adv_objective": ((r.get("adversary") or {}).get("performance") or {}).get("objective"),
              "added": [a["reason"] for a in r.get("added", [])], "set_size": len(r["working_set"])} for r in its]
    if its:
        (store.dir / "plots").mkdir(exist_ok=True)
        k = [r["iteration"] for r in its]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(k, [r["master"]["t"] for r in its], "o-", label="worst shortfall on S_k at w_k")
        ax[0].plot(k, [r["adversary"]["performance"]["objective"] if r.get("adversary") and r["adversary"]["performance"]
                       else np.nan for r in its], "s--", label="adversary's worst shortfall at w_k")
        ax[0].set_xlabel("iteration"); ax[0].set_ylabel("progress shortfall"); ax[0].legend(fontsize=8)
        names = list(its[0]["master"]["worst"])
        for n in names:
            ax[1].plot(k, [r["master"]["worst"][n]["value"] for r in its], "o-", label=n)
        ax[1].set_xlabel("iteration"); ax[1].set_ylabel("worst value on S_k (own units)"); ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(store.dir / "plots" / "convergence.png", dpi=120)
        plt.close(fig)
    return {"experiment": store.dir.name, "summary": summ, "iterations": table, "candidate_status_counts": counts}


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["optimize", "resume", "compare", "report"])
    ap.add_argument("--config", help="experiment JSON (optimize)")
    ap.add_argument("--experiment", help="experiment id (resume/compare/report)")
    ap.add_argument("--artifacts-root", default=DEFAULT_CONFIG["artifacts_root"])
    ap.add_argument("--set", action="append", default=[], help="override: key.sub=<json>")
    ap.add_argument("--label", default="heldout", help="compare: output name")
    ap.add_argument("--heldout-config", help="compare: JSON with a 'heldout' section")
    ap.add_argument("--w-json", help="compare: extra natural-unit PARALO w ({name: value}) to include")
    args = ap.parse_args()

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
        exp = ConstrainedExperiment(config, store)
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
    exp = ConstrainedExperiment(config, store)
    exp.check_manifest()
    if args.mode == "resume":
        print(json.dumps(exp.run(resume=True), indent=2, default=str))
        return
    heldout = json.loads(Path(args.heldout_config).read_text())["heldout"] if args.heldout_config else config["heldout"]
    extra = json.loads(Path(args.w_json).read_text()) if args.w_json else None
    res = exp.compare(args.label, heldout, extra)
    cols = ["label", "event_model", "completion_rate", *[f"{e}_rate" for e in EVENTS], "worst_progress_shortfall",
            "mean_progress_shortfall", *PEAKS, "ms_per_replan_mean"]
    print("\t".join(cols))
    for r in res["rows"]:
        print("\t".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols))


if __name__ == "__main__":
    main()
