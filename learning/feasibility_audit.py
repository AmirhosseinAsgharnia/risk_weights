"""
PPO feasibility-dataset audit, relabelling, curation and surrogate validation
(docs/physics_audit.md §A3, Part B). The original scenarios.jsonl /
rollouts.jsonl are NEVER modified: the audit and the relabels are separate,
versioned files next to them, and curation only decides what a surrogate
fit may use.

Per row the audit records family, blocker side, theta, the theta-bound
version (stamped, or recovered from git history for rows that predate the
stamp), current-bound validity, the rollover event model its labels were
produced under, PPO counts, and an inclusion decision with its reason:

  out_of_bounds             theta outside the CURRENT bounds -> excluded (never clamped)
  legacy_rollover_label     labels from the legacy_v1 ROLLOVER event with >= 1 rollover-terminated
                            episode -> excluded until re-evaluated under physical_v2
  legacy_no_rollover        legacy_v1 labels, zero rollover-terminated episodes -> included: every
                            trajectory is identical under physical_v2 up to the first step where
                            |a_y| >= SSF g (not checkable without the policy; flagged). The policy was
                            TRAINED under the legacy event, so this is lower-bound evidence.
  relabeled                 policy re-evaluated under physical_v2 here (after reproducing its original
                            legacy_v1 k exactly) -> included with the NEW labels
  stamped                   produced under the stamped event model == the current one -> included

Modes:
    audit     write audit_<date-free version>.jsonl and print the category table
    relabel   re-evaluate every locally available policy of an in-bounds row under physical_v2
    fit       fit a surrogate on the curated dataset (same as feasibility_pipeline --mode fit-only)
    validate  grouped-CV validation of the surrogate's ADMISSION decisions on the curated dataset

    python -m learning.feasibility_audit --mode audit --scenario-family cutin --blocker-side 1
    python -m learning.feasibility_audit --mode relabel --scenario-family cutin --blocker-side 1 --workers 10
    python -m learning.feasibility_audit --mode fit --scenario-family cutin --blocker-side 1 --surrogate-out artifacts/feasibility/surrogates/cutin_side1.joblib
    python -m learning.feasibility_audit --mode validate --scenario-family cutin --blocker-side 1
"""

import os

for _env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import argparse
import json
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from learning.eval_batch import wilson_ci
from learning.feasibility_cutin import THETA_BOUNDS_CUTIN, THETA_BOUNDS_VERSION_CUTIN
from learning.feasibility_sandwich import THETA_BOUNDS_SANDWICH, THETA_BOUNDS_VERSION_SANDWICH
from learning.feasibility_dataset import (
    ARTIFACTS_ROOT, _atomic_write_bytes, _append_line, dataset_dir, read_jsonl, rollouts_path, scenarios_path,
)
from learning.minimax_feasibility import demonstrated

AUDIT_VERSION = "dataset-audit-v1"
RELABEL_EVENT_MODEL = "physical_v2"

# Every bound version ever used, oldest first (git history; docs/physics_audit.md §A3).
BOUNDS_HISTORY = {
    "cutin": {
        "cutin-bounds-v0": {**THETA_BOUNDS_CUTIN, "cutter_relative_speed_mps": (-12.0, 5.0), "front_gap_m": (8.0, 60.0)},
        THETA_BOUNDS_VERSION_CUTIN: dict(THETA_BOUNDS_CUTIN),
    },
    "sandwich": {THETA_BOUNDS_VERSION_SANDWICH: dict(THETA_BOUNDS_SANDWICH)},
}
CURRENT = {"cutin": THETA_BOUNDS_VERSION_CUTIN, "sandwich": THETA_BOUNDS_VERSION_SANDWICH}
# Rows written before the theta_bounds_version stamp: the commit that first contained them decides
# (rows of 75b33de were generated before eb2edbb changed the cut-in bounds).
LEGACY_BATCHES = {("cutin", 1): [("75b33de", "cutin-bounds-v0"), ("be97cd3", "cutin-bounds-v1")]}


def _side_dir(blocker_side: int) -> str:
    return f"side_{'pos' if blocker_side > 0 else 'neg'}1"


def relabel_dir(family: str, blocker_side: int, root: Path = ARTIFACTS_ROOT) -> Path:
    return dataset_dir(family, blocker_side, root) / f"relabel_{RELABEL_EVENT_MODEL}"


def policy_dir(family: str, blocker_side: int, scenario_id: str, root: Path = ARTIFACTS_ROOT) -> Path:
    return Path(root) / "policies" / family / _side_dir(blocker_side) / scenario_id


def _git_batch_ids(family: str, blocker_side: int, root: Path) -> dict[str, tuple[str, str]]:
    """scenario_id -> (bounds version, 'git:<commit>') for pre-stamp rows."""
    out = {}
    rel = scenarios_path(family, blocker_side, root)
    for commit, version in LEGACY_BATCHES.get((family, blocker_side), []):
        try:
            text = subprocess.run(["git", "show", f"{commit}:{rel}"], capture_output=True, text=True, check=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        for line in text.splitlines():
            if line.strip():
                out.setdefault(json.loads(line)["scenario_id"], (version, f"git:{commit}"))
    return out


def _out_of_bounds(theta: dict, bounds: dict) -> list[str]:
    return sorted(k for k, (lo, hi) in bounds.items() if not lo <= theta[k] <= hi)


# ── audit ───────────────────────────────────────────────────────────────────
def audit_rows(family: str, blocker_side: int, root: Path = ARTIFACTS_ROOT) -> list[dict]:
    rows = read_jsonl(scenarios_path(family, blocker_side, root))
    relabels = {r["scenario_id"]: r for r in read_jsonl(relabel_dir(family, blocker_side, root) / "scenarios.jsonl")}
    git_ids = _git_batch_ids(family, blocker_side, root)
    bounds = BOUNDS_HISTORY[family][CURRENT[family]]
    out = []
    for i, r in enumerate(rows):
        sid = r["scenario_id"]
        if r.get("theta_bounds_version"):
            bver, bsrc = r["theta_bounds_version"], "stamped"
        else:
            bver, bsrc = git_ids.get(sid, ("unknown", "unknown"))
        oob = _out_of_bounds(r["theta"], bounds)
        ev = r.get("rollover_event_model") or "legacy_v1"
        ev_src = "stamped" if r.get("rollover_event_model") else "inferred (pre-stamp rows used the env default)"
        pol = policy_dir(family, blocker_side, sid, root) / "model.zip"
        rl = relabels.get(sid)
        rec = {"audit_version": AUDIT_VERSION, "row_index": i, "scenario_id": sid, "family": family,
               "blocker_side": blocker_side, "theta": r["theta"], "theta_bounds_version": bver,
               "theta_bounds_version_source": bsrc, "current_bounds_version": CURRENT[family],
               "current_bounds_valid": not oob, "out_of_bounds_fields": oob,
               "rollover_event_model": ev, "rollover_event_model_source": ev_src,
               "k": r["k"], "n": r["n"], "wilson_95ci": list(wilson_ci(r["k"], r["n"])),
               "demonstrated": demonstrated(r["k"], r["n"], 0.8),
               **{c: r.get(c) for c in ("n_collisions", "n_rollovers", "n_offroad", "n_timeouts")},
               "policy_available_locally": pol.is_file(), "relabel": None}
        if oob:
            cat, inc, why = "out_of_bounds", False, f"outside {CURRENT[family]}: {', '.join(oob)}"
        elif rl is not None and rl["legacy_reproduced"]:
            cat, inc, why = "relabeled", True, None
            rec["relabel"] = {k: rl[k] for k in ("k", "n", "rollover_event_model", "n_rollovers", "n_collisions",
                                                  "n_offroad", "n_timeouts", "n_success_with_friction_circle_exceeded")}
        elif rl is not None:
            cat, inc, why = "relabel_not_reproducible", False, "legacy_v1 re-evaluation did not reproduce the stored k"
        elif ev == RELABEL_EVENT_MODEL:
            cat, inc, why = "stamped", True, None
        elif (r.get("n_rollovers") or 0) > 0:
            cat, inc = "legacy_rollover_label", False
            why = ("labels include legacy_v1 ROLLOVER terminations (false-positive detector); re-evaluation "
                   + ("pending (`--mode relabel`)" if pol.is_file() else "pending -- policy weights not available locally"))
        else:
            cat, inc, why = "legacy_no_rollover", True, None
        rec.update({"label_category": cat, "included": inc, "exclusion_reason": why})
        out.append(rec)
    return out


def write_audit(family: str, blocker_side: int, root: Path = ARTIFACTS_ROOT) -> tuple[Path, list[dict]]:
    rows = audit_rows(family, blocker_side, root)
    path = dataset_dir(family, blocker_side, root) / f"audit_{AUDIT_VERSION}.jsonl"
    _atomic_write_bytes(path, "".join(json.dumps(r) + "\n" for r in rows).encode())
    return path, rows


def category_table(rows: list[dict]) -> dict:
    tab = {}
    for r in rows:
        key = f"{r['theta_bounds_version']}|{r['label_category']}"
        t = tab.setdefault(key, {"rows": 0, "demonstrated": 0, "included": 0, "policy_local": 0})
        t["rows"] += 1
        t["demonstrated"] += int(r["demonstrated"])
        t["included"] += int(r["included"])
        t["policy_local"] += int(r["policy_available_locally"])
    return dict(sorted(tab.items()))


# ── relabel (re-evaluate stored policies under the corrected event model) ───
def _evaluate_policy(args) -> dict:
    family, blocker_side, sid, root, event_model = args
    import torch
    from stable_baselines3 import PPO
    from learning.env import EgoTrafficEnv
    from learning.feasibility_train_one import CONFIG_CLASSES, RUNTIME_CLASSES
    from learning.vehicle_physics import friction_demand
    torch.set_num_threads(1)
    pdir = policy_dir(family, blocker_side, sid, Path(root))
    meta = json.loads((pdir / "model.meta.json").read_text())
    cfg = CONFIG_CLASSES[family](**meta["theta"], blocker_side=meta["blocker_side"], seed=meta["scenario_seed"],
                                 mode=meta["mode"], goal_distance_m=meta["goal_distance_m"],
                                 max_episode_seconds=meta["max_episode_seconds"], min_progress_m=meta["min_progress_m"])
    assert cfg.scenario_id() == sid, f"{sid}: reconstructed config has id {cfg.scenario_id()}"
    model = PPO.load(str(pdir / "model.zip"), device="cpu")
    ev = meta["evaluation"]
    assert ev["deterministic"] and ev["eval_seed"] is None
    env = EgoTrafficEnv(arena=RUNTIME_CLASSES[family](cfg), mode="distribution", worker_rank=0, rollover_event=event_model)
    eps = []
    for e in range(ev["episodes"]):
        obs, info = env.reset()
        done, peak_fd, peak_ay = False, 0.0, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            ts = env.ego_tire_state()
            peak_fd, peak_ay = max(peak_fd, friction_demand(ts, env.ego_car.vehicle_params)), max(peak_ay, abs(ts.a_y))
        eps.append({"episode": e, "success": bool(info["success"]), "outcome": info["outcome"],
                    "failure_reason": info["failure_reason"], "steps": env.step_count,
                    "peak_friction_demand": peak_fd, "peak_abs_lat_accel": peak_ay})
    return {"scenario_id": sid, "event_model": event_model, "episodes": eps, "meta": meta}


def _counts(eps: list[dict]) -> dict:
    fr = [e["failure_reason"] for e in eps]
    return {"k": sum(e["success"] for e in eps), "n": len(eps), "n_collisions": fr.count("collision"),
            "n_rollovers": fr.count("rollover"), "n_offroad": fr.count("off_road"),
            "n_timeouts": sum(e["outcome"] == "timeout" for e in eps)}


def relabel(family: str, blocker_side: int, root: Path = ARTIFACTS_ROOT, n_workers: int = 1) -> list[dict]:
    """Re-evaluate every locally available policy of an in-bounds row: first under legacy_v1 (must
    reproduce the stored k exactly, or the evaluation is not trusted), then under physical_v2."""
    rows = {r["scenario_id"]: r for r in read_jsonl(scenarios_path(family, blocker_side, root))}
    out_dir = relabel_dir(family, blocker_side, root)
    done = {r["scenario_id"] for r in read_jsonl(out_dir / "scenarios.jsonl")}
    todo = [a["scenario_id"] for a in audit_rows(family, blocker_side, root)
            if a["current_bounds_valid"] and a["policy_available_locally"] and a["scenario_id"] not in done
            and a["rollover_event_model"] != RELABEL_EVENT_MODEL]
    jobs = [(family, blocker_side, sid, str(root), m) for sid in todo for m in ("legacy_v1", RELABEL_EVENT_MODEL)]
    if n_workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(n_workers, mp_context=multiprocessing.get_context("fork")) as ex:
            res = list(ex.map(_evaluate_policy, jobs))
    else:
        res = [_evaluate_policy(j) for j in jobs]
    by = {(r["scenario_id"], r["event_model"]): r for r in res}
    written = []
    for sid in todo:
        old, leg, new = rows[sid], by[(sid, "legacy_v1")], by[(sid, RELABEL_EVENT_MODEL)]
        cl, cn = _counts(leg["episodes"]), _counts(new["episodes"])
        rec = {"scenario_id": sid, "family": family, "blocker_side": blocker_side, "theta": old["theta"],
               "rollover_event_model": RELABEL_EVENT_MODEL, **cn, "wilson_95ci": list(wilson_ci(cn["k"], cn["n"])),
               "legacy_reproduced": cl["k"] == old["k"] and cl["n"] == old["n"] and cl["n_rollovers"] == old["n_rollovers"],
               "legacy_reevaluation": cl, "original": {k: old[k] for k in ("k", "n", "n_rollovers", "n_collisions",
                                                                         "n_offroad", "n_timeouts")},
               "n_success_with_friction_circle_exceeded": sum(e["success"] and e["peak_friction_demand"] >= 1.0
                                                              for e in new["episodes"]),
               "max_peak_abs_lat_accel": max(e["peak_abs_lat_accel"] for e in new["episodes"]),
               "policy": str(policy_dir(family, blocker_side, sid, root) / "model.zip"),
               "policy_trained_under": new["meta"].get("rollover_event_model") or "legacy_v1",
               "audit_version": AUDIT_VERSION}
        out_dir.mkdir(parents=True, exist_ok=True)
        for e in new["episodes"]:
            _append_line(out_dir / "rollouts.jsonl", {"scenario_id": sid, "episode": e["episode"], "y": int(e["success"]),
                                                       "outcome": e["outcome"], "failure_reason": e["failure_reason"],
                                                       "theta": old["theta"], "rollover_event_model": RELABEL_EVENT_MODEL,
                                                       "peak_friction_demand": e["peak_friction_demand"]})
        _append_line(out_dir / "scenarios.jsonl", rec)
        written.append(rec)
    return written


# ── curation ────────────────────────────────────────────────────────────────
def curated_dataset(family: str, blocker_side: int, root: Path = ARTIFACTS_ROOT) -> dict:
    audit = audit_rows(family, blocker_side, root)
    raw = read_jsonl(rollouts_path(family, blocker_side, root))
    rel = read_jsonl(relabel_dir(family, blocker_side, root) / "rollouts.jsonl")
    inc = {a["scenario_id"]: a for a in audit if a["included"]}
    use_rel = {s for s, a in inc.items() if a["label_category"] == "relabeled"}
    rollouts = ([r for r in raw if r["scenario_id"] in inc and r["scenario_id"] not in use_rel]
                + [r for r in rel if r["scenario_id"] in use_rel])
    scenarios = []
    for sid, a in inc.items():
        k, n = (a["relabel"]["k"], a["relabel"]["n"]) if sid in use_rel else (a["k"], a["n"])
        scenarios.append({"scenario_id": sid, "family": family, "blocker_side": blocker_side, "mode": "robust",
                          "theta": a["theta"], "k": k, "n": n, "label_category": a["label_category"]})
    reasons = {}
    for a in audit:
        if not a["included"]:
            reasons[a["label_category"]] = reasons.get(a["label_category"], 0) + 1
    prov = {"audit_version": AUDIT_VERSION, "theta_bounds_version": CURRENT[family],
            "rollover_event_model": f"{RELABEL_EVENT_MODEL} (relabeled/stamped) + legacy_v1 rows with zero rollover "
                                    f"terminations (lower-bound evidence)",
            "n_rows_total": len(audit), "n_included": len(inc), "n_excluded": len(audit) - len(inc),
            "exclusion_reasons": reasons,
            "included_by_category": {c: sum(a["label_category"] == c for a in inc.values())
                                     for c in sorted({a["label_category"] for a in inc.values()})},
            "excluded_scenario_ids": sorted(a["scenario_id"] for a in audit if not a["included"])}
    return {"rollouts": rollouts, "scenarios": scenarios, "n_included": len(inc), "n_excluded": len(audit) - len(inc),
            "provenance": prov}


# ── surrogate admission validation ──────────────────────────────────────────
def validate_admission(family: str, blocker_side: int, root: Path = ARTIFACTS_ROOT, *, n_splits: int = 5,
                       n_bootstrap: int = 50, seed: int = 0, tau: float = 0.8) -> dict:
    """Grouped K-fold over scenarios: for each held-out scenario, fit the selected model family + its
    bootstrap ensemble on the OTHER folds only, apply the conservative admission rule (bootstrap
    2.5% bound >= tau -> admitted; 97.5% bound < tau -> rejected; else uncertain) and compare with the
    held-out PPO evidence. 'Demonstrated' = Wilson lower >= tau (the oracle's own rule).
      false admission  admitted, but the held-out evidence does NOT demonstrate solvability
      false rejection  rejected, but the held-out evidence DOES demonstrate it
    The model family is selected once on the full curated set (evaluate_candidates) -- a mild
    optimism, documented."""
    from sklearn.model_selection import GroupKFold
    from learning.feasibility_surrogate import (
        _bootstrap_ensemble, _build_training_rows, _candidate_models, ConstantClassifier, evaluate_candidates,
        normalize_theta,
    )
    cur = curated_dataset(family, blocker_side, root)
    bounds = BOUNDS_HISTORY[family][CURRENT[family]]
    reports = evaluate_candidates(cur["rollouts"], bounds, n_splits=n_splits)
    name = reports[0].name
    rows = _build_training_rows(cur["rollouts"], bounds)
    groups = np.array([r.scenario_id for r in rows])
    sids = sorted(set(groups))
    scen = {s["scenario_id"]: s for s in cur["scenarios"]}
    per = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=min(n_splits, len(sids))).split(groups, groups=groups)):
        trows = [rows[i] for i in tr]
        y = np.array([r.y for r in trows])
        model = ConstantClassifier() if (name == "constant" or len(set(y)) < 2) else _candidate_models()[name]
        model.fit(np.stack([r.theta_normalized for r in trows]), y)
        boot = _bootstrap_ensemble(name, trows, n_bootstrap, seed + fold)
        for sid in sorted(set(groups[te])):
            x = normalize_theta(scen[sid]["theta"], bounds).reshape(1, -1)
            p = float(model.predict_proba(x)[0, 1])
            bp = np.array([m.predict_proba(x)[0, 1] for m in boot]) if boot else np.array([p])
            lo, hi = float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))
            status = "admitted" if lo >= tau else ("rejected" if hi < tau and p < tau else "uncertain")
            k, n = scen[sid]["k"], scen[sid]["n"]
            per.append({"scenario_id": sid, "fold": fold, "p_hat": p, "lower": lo, "upper": hi, "status": status,
                        "k": k, "n": n, "empirical_rate": k / n, "demonstrated": demonstrated(k, n, tau),
                        "label_category": scen[sid]["label_category"]})
    n_dem = sum(r["demonstrated"] for r in per)
    adm = [r for r in per if r["status"] == "admitted"]
    rej = [r for r in per if r["status"] == "rejected"]
    bins = np.linspace(0, 1, 6)
    rel = []
    for lo_b, hi_b in zip(bins[:-1], bins[1:]):
        sel = [r for r in per if lo_b <= r["p_hat"] < hi_b or (hi_b == 1.0 and r["p_hat"] == 1.0)]
        if sel:
            rel.append({"bin": [float(lo_b), float(hi_b)], "n": len(sel), "mean_p_hat": float(np.mean([r["p_hat"] for r in sel])),
                        "mean_empirical": float(np.mean([r["empirical_rate"] for r in sel]))})
    return {
        "model_name": name, "candidate_reports": [{"name": r.name, "brier": r.brier, "log_loss": r.log_loss,
                                                    "roc_auc": r.roc_auc} for r in reports],
        "n_scenarios": len(per), "n_rollouts": len(rows), "n_demonstrated": n_dem, "tau": tau,
        "n_bootstrap": n_bootstrap, "provenance": cur["provenance"],
        "status_counts": {s: sum(r["status"] == s for r in per) for s in ("admitted", "uncertain", "rejected")},
        "false_admissions": len([r for r in adm if not r["demonstrated"]]),
        "false_admission_rate": (len([r for r in adm if not r["demonstrated"]]) / len(adm)) if adm else None,
        "false_admissions_empirical_below_tau": len([r for r in adm if r["empirical_rate"] < tau]),
        "false_rejections": len([r for r in rej if r["demonstrated"]]),
        "false_rejection_rate_of_demonstrated": (len([r for r in rej if r["demonstrated"]]) / n_dem) if n_dem else None,
        "demonstrated_not_admitted": sum(r["demonstrated"] and r["status"] != "admitted" for r in per),
        "scenario_brier": float(np.mean([(r["p_hat"] - r["empirical_rate"]) ** 2 for r in per])),
        "interval_coverage_of_empirical_rate": float(np.mean([r["lower"] <= r["empirical_rate"] <= r["upper"] for r in per])),
        "reliability": rel, "per_scenario": per,
        "note": "grouped out-of-fold; bootstrap intervals are an empirical spread over scenario resamples, not "
                "guaranteed confidence bounds; model family selected on the full curated set",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["audit", "relabel", "fit", "validate"])
    ap.add_argument("--scenario-family", choices=["cutin", "sandwich"], required=True)
    ap.add_argument("--blocker-side", type=int, choices=[-1, 1], required=True)
    ap.add_argument("--root", default=str(ARTIFACTS_ROOT))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--surrogate-out")
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--out", help="validate: JSON output path")
    a = ap.parse_args()
    root = Path(a.root)
    if a.mode == "audit":
        path, rows = write_audit(a.scenario_family, a.blocker_side, root)
        print(json.dumps(category_table(rows), indent=2))
        print(f"wrote {path}")
    elif a.mode == "relabel":
        for r in relabel(a.scenario_family, a.blocker_side, root, a.workers):
            print(f"{r['scenario_id']}: original k={r['original']['k']} (rollovers {r['original']['n_rollovers']}) -> "
                  f"legacy re-eval k={r['legacy_reevaluation']['k']} reproduced={r['legacy_reproduced']} -> "
                  f"physical_v2 k={r['k']} (rollovers {r['n_rollovers']}, collisions {r['n_collisions']}, "
                  f"timeouts {r['n_timeouts']}); successes over the friction circle: "
                  f"{r['n_success_with_friction_circle_exceeded']}")
        write_audit(a.scenario_family, a.blocker_side, root)
    elif a.mode == "fit":
        assert a.surrogate_out, "--surrogate-out is required"
        from learning.feasibility_pipeline import run_fit_only
        run_fit_only(a.scenario_family, a.blocker_side, a.surrogate_out, n_bootstrap=a.n_bootstrap, root=root)
    else:
        res = validate_admission(a.scenario_family, a.blocker_side, root, n_bootstrap=min(a.n_bootstrap, 50))
        out = Path(a.out or dataset_dir(a.scenario_family, a.blocker_side, root) / f"surrogate_validation_{AUDIT_VERSION}.json")
        _atomic_write_bytes(out, json.dumps(res, indent=2).encode())
        print(json.dumps({k: v for k, v in res.items() if k not in ("per_scenario",)}, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
