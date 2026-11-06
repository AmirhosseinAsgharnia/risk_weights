"""
Analysis of saved evaluation records (docs/description.md §23): no simulation here, and every number comes
from learning.evaluate_ppo's files. Figures, tables and reports all read these functions, so they cannot disagree.

Levels, kept apart on purpose:
  rollout    one binary outcome under one realization
  scenario   p_hat = k/N for one policy, from the test stream, with its Wilson interval and admission status
  stratum    a summary over scenarios. Rollouts of one scenario are not independent of each other as evidence
             about the stratum, so stratum-level intervals bootstrap over scenarios, not rollouts.

admission_status() is the one definition. It uses wilson_ci and learning.minimax_feasibility.demonstrated
unchanged and adds the matching upper-bound test: the existing oracle path treats everything undemonstrated as
UNCERTAIN (a candidate for refinement), while this reports Wilson upper < p0 as "rejected", meaning not admitted
by this evidence, never physically impossible. Exact-mode records have no statistical status at all.
"""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from learning.eval_batch import wilson_ci
from learning.feasibility_dataset import read_jsonl
from learning.minimax_feasibility import ADMITTED, REJECTED, UNCERTAIN, demonstrated
from learning.ppo_experiment import ADMISSION_P0, json_hash, read_json

NOT_APPLICABLE = "not_applicable"
_MISSING = "unrecorded"


def admission_status(k: int, n: int, p0: float = ADMISSION_P0, independent: bool = True) -> str:
    if not independent or n == 0:
        return NOT_APPLICABLE
    if demonstrated(k, n, p0):
        return ADMITTED
    return REJECTED if wilson_ci(k, n)[1] < p0 else UNCERTAIN


def load_records(exp: Path) -> list[dict]:
    return [read_json(p) for p in sorted(Path(exp).glob("evaluations/**/*.json"))]


def stratum_label(family: str, side: int) -> str:
    return f"{family} {'+1' if side == 1 else '-1'}"


def reward_key(policy: dict) -> str:
    """Only from the policy's metadata: a version plus a hash of its coefficients."""
    if not policy.get("reward_version"):
        return _MISSING
    return f"{policy['reward_version']}/{json_hash(policy.get('reward_config'))}"


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _peak(values) -> float | None:
    values = [v for v in values if v is not None]
    return max(values) if values else None


def _low(values) -> float | None:
    values = [v for v in values if v is not None]
    return min(values) if values else None


def scenario_row(rec: dict) -> dict:
    s, rolls, sc = rec["summary"], rec["rollouts"], rec["scenario"]
    independent = rec["independent"]
    if not independent:   # the repeats confirm one outcome: counting them would pretend otherwise
        rolls = rolls[:1]
    k, n = sum(bool(r["success"]) for r in rolls), len(rolls)
    count = lambda reason: sum(r["failure_reason"] == reason for r in rolls)   # noqa: E731
    phys = [r.get("physical") or {} for r in rolls]
    lo, hi = wilson_ci(k, n) if independent else (None, None)
    peaks = [max(v for v in (p.get("peak_eta_front"), p.get("peak_eta_rear")) if v is not None)
             for p in phys if p.get("peak_eta_front") is not None and p.get("peak_eta_rear") is not None]
    statuses = {p.get("status", "unavailable") for p in phys}
    return {"family": sc["family"], "blocker_side": sc["blocker_side"], "stratum": stratum_label(sc["family"], sc["blocker_side"]),
            "scenario_id": sc["scenario_id"], "training_scenario_id": sc["training_scenario_id"], "theta": sc["theta"],
            "policy_id": rec["policy"]["policy_id"], "ppo_seed": rec["policy"].get("ppo_seed"), "trained_mode": rec["policy"]["trained_mode"],
            "reward": reward_key(rec["policy"]), "eval_mode": rec["protocol"]["eval_mode"], "independent": independent,
            "k": k, "n": n, "p_hat": k / n if n else None, "wilson_lower": lo, "wilson_upper": hi,
            "status": admission_status(k, n, independent=independent),
            "selection_rate": rec["selection"]["k"] / rec["selection"]["n"] if rec["selection"]["n"] else None,
            "repeatable": rec.get("repeatable"), "n_rollouts": n, "n_collision": count("collision"), "n_off_road": count("off_road"),
            "n_rollover": count("rollover"), "n_timeout": sum(bool(r["timeout"]) for r in rolls),
            "mean_success_time_s": _mean(r["elapsed_s"] for r in rolls if r["success"]),
            "success_times_s": [r["elapsed_s"] for r in rolls if r["success"]], "progress": [r["episode_progress"] for r in rolls],
            "peak_tire_utilization": max(peaks) if peaks else None, "mean_peak_tire_utilization": _mean(peaks),
            "friction_violations": sum(p.get("friction_violations") or 0 for p in phys),
            "peak_abs_lat_accel": _peak(p.get("peak_abs_lat_accel") for p in phys), "peak_long_decel": _peak(p.get("peak_long_decel") for p in phys),
            "min_clearance_m": _low(p.get("min_clearance_m") for p in phys), "min_road_edge_clearance_m": _low(p.get("min_road_edge_clearance_m") for p in phys),
            "stability_loss_rollouts": sum(bool(p.get("stability_loss")) for p in phys),
            "physical_status": "violation" if "violation" in statuses else ("unavailable" if "unavailable" in statuses else "valid")}


def select_policies(rows: list[dict]) -> list[dict]:
    """One row per (scenario, evaluation mode, reward). The PPO seed is chosen on the selection stream only, so
    test rollouts are not used twice; a test-stream maximum over seeds would inflate p_hat."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["training_scenario_id"], r["eval_mode"], r["reward"]].append(r)
    picked = []
    for group in groups.values():
        best = max(group, key=lambda r: (r["selection_rate"] if r["selection_rate"] is not None else -1.0, -(r["ppo_seed"] or 0)))
        picked.append({**best, "n_seeds_considered": len(group)})
    return sorted(picked, key=lambda r: (r["stratum"], r["reward"], r["eval_mode"], r["scenario_id"]))


def scenario_table(records: list[dict]) -> list[dict]:
    return select_policies([scenario_row(r) for r in records])


def bootstrap_ci(values, seed: int = 0, n_boot: int = 2000) -> tuple[float, float] | None:
    """Percentile 95% interval of a mean, resampling scenarios."""
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if len(values) < 2:
        return None
    means = values[np.random.default_rng(seed).integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def table_a(rows: list[dict], scenarios: list[dict]) -> list[dict]:
    """Overall results per stratum, reward and evaluation mode. Scenario counts and rollout rates stay separate."""
    sampled = defaultdict(int)
    for s in scenarios:
        sampled[stratum_label(s["family"], s["blocker_side"])] += 1
    groups = defaultdict(list)
    for r in rows:
        groups[r["stratum"], r["reward"], r["eval_mode"]].append(r)
    out = []
    for (stratum, reward, mode), g in sorted(groups.items()):
        n_roll = sum(r["n_rollouts"] for r in g)
        counts = {s: sum(r["status"] == s for r in g) for s in (ADMITTED, UNCERTAIN, REJECTED)}
        stat = mode == "robust" or any(r["independent"] for r in g)
        times = [t for r in g for t in r["success_times_s"]]
        out.append({"stratum": stratum, "reward": reward, "eval_mode": mode, "scenarios_sampled": sampled.get(stratum),
                    "scenarios_evaluated": len(g), "admitted": counts[ADMITTED] if stat else None,
                    "uncertain": counts[UNCERTAIN] if stat else None, "not_admitted": counts[REJECTED] if stat else None,
                    "mean_scenario_success_rate": _mean(r["p_hat"] for r in g), "rollouts": n_roll,
                    "rollout_collision_rate": sum(r["n_collision"] for r in g) / n_roll if n_roll else None,
                    "rollout_road_departure_rate": sum(r["n_off_road"] for r in g) / n_roll if n_roll else None,
                    "rollout_rollover_rate": sum(r["n_rollover"] for r in g) / n_roll if n_roll else None,
                    "rollout_timeout_rate": sum(r["n_timeout"] for r in g) / n_roll if n_roll else None,
                    "mean_success_time_s": _mean(times)})
    return out


def table_b(rows: list[dict]) -> list[dict]:
    keys = ("stratum", "eval_mode", "scenario_id", "policy_id", "p_hat", "peak_tire_utilization", "friction_violations",
            "peak_abs_lat_accel", "peak_long_decel", "min_clearance_m", "stability_loss_rollouts", "physical_status")
    return [{k: r[k] for k in keys} for r in rows]


def comparison_issues(records: list[dict]) -> list[str]:
    """Why a set of evaluations may not be compared as equivalent; empty means no difference was detected."""
    issues = []
    policies = [r["policy"] for r in records]

    def differ(path, label):
        values = {str(path(r)) for r in records}
        if len(values) > 1:
            issues.append(f"{label} differ: {sorted(values)}")

    for key, label in (("vehicle_physics_version", "training vehicle dynamics"), ("rollover_event_model", "training rollover events"),
                       ("obs_dim", "observation sizes"), ("action_dim", "action sizes"), ("dt_s", "time steps")):
        differ(lambda r: r["policy"].get(key), label)
        if any(p.get(key) is None for p in policies):
            issues.append(f"{label}: not recorded for every policy")
    differ(lambda r: r["provenance"].get("vehicle_physics_version"), "evaluation vehicle dynamics")
    differ(lambda r: (r["protocol"]["test_episodes"], r["protocol"]["deterministic"], r["protocol"]["rollover_event"]), "evaluation protocols")
    if any(not p.get("reward_version") for p in policies):
        issues.append("reward configuration not recorded for every policy")
    return issues


def table_c(records: list[dict], rows: list[dict], metrics_final: dict | None = None) -> tuple[list[dict], list[str]]:
    """Reward comparison on the robust test stream, over scenarios every reward was evaluated on."""
    rows = [r for r in rows if r["eval_mode"] == "robust"]
    groups = defaultdict(list)
    for r in rows:
        groups[r["reward"]].append(r)
    issues = comparison_issues([r for r in records if r["protocol"]["eval_mode"] == "robust"])
    if len(groups) < 2:
        return [], ["only one reward configuration is present, so no comparison is possible"] if groups else []
    common = set.intersection(*[{r["training_scenario_id"] for r in g} for g in groups.values()])
    if any({r["training_scenario_id"] for r in g} != common for g in groups.values()):
        issues.append(f"scenario sets differ between rewards: compared on the {len(common)} they share")
    by_policy = defaultdict(list)
    for rec in records:
        by_policy[reward_key(rec["policy"])].append(rec["policy"].get("timesteps_actual"))
    out = []
    for reward, g in sorted(groups.items()):
        g = [r for r in g if r["training_scenario_id"] in common]
        n_roll = sum(r["n_rollouts"] for r in g)
        if not n_roll:
            continue
        times = [t for r in g for t in r["success_times_s"]]
        out.append({"reward": reward, "scenarios": len(g), "mean_training_timesteps": _mean(by_policy[reward]),
                    "final_validation_success_rate": (metrics_final or {}).get(reward), "heldout_success_rate": _mean(r["p_hat"] for r in g),
                    "rollout_collision_rate": sum(r["n_collision"] for r in g) / n_roll, "rollout_road_departure_rate": sum(r["n_off_road"] for r in g) / n_roll,
                    "rollout_timeout_rate": sum(r["n_timeout"] for r in g) / n_roll, "mean_success_time_s": _mean(times),
                    "peak_tire_utilization": _peak(r["peak_tire_utilization"] for r in g),
                    "admitted_percent": 100.0 * sum(r["status"] == ADMITTED for r in g) / len(g), "comparison": "non-equivalent" if issues else "equivalent"})
    return out, issues


def robustness_table(rows: list[dict]) -> list[dict]:
    """Same policy in both modes. Per scenario the exact result is a single deterministic outcome, so each mode
    contributes one rate per policy and the stratum interval bootstraps over those paired scenarios."""
    pairs = defaultdict(dict)
    for r in rows:
        if r["eval_mode"] in ("exact", "robust"):
            pairs[r["policy_id"]][r["eval_mode"]] = r
    groups = defaultdict(list)
    for p in pairs.values():
        if len(p) == 2:
            groups[p["robust"]["stratum"], p["robust"]["trained_mode"], p["robust"]["reward"]].append(p)
    out = []
    for (stratum, trained, reward), g in sorted(groups.items()):
        exact, robust = [p["exact"]["p_hat"] for p in g], [p["robust"]["p_hat"] for p in g]
        rate = lambda mode, key: sum(p[mode][key] for p in g) / sum(p[mode]["n_rollouts"] for p in g)   # noqa: E731
        out.append({"stratum": stratum, "trained_mode": trained, "reward": reward, "policies": len(g),
                    "exact_success": _mean(exact), "exact_ci": bootstrap_ci(exact), "robust_success": _mean(robust), "robust_ci": bootstrap_ci(robust),
                    "difference": _mean(robust) - _mean(exact), "difference_ci": bootstrap_ci([b - a for a, b in zip(exact, robust)]),
                    **{f"{m}_{name}_rate": rate(m, f"n_{key}") for m in ("exact", "robust")
                       for name, key in (("collision", "collision"), ("road_departure", "off_road"), ("rollover", "rollover"), ("timeout", "timeout"))},
                    "exact_success_time_s": _mean(t for p in g for t in p["exact"]["success_times_s"]),
                    "robust_success_time_s": _mean(t for p in g for t in p["robust"]["success_times_s"]),
                    "exact_progress": [v for p in g for v in p["exact"]["progress"]], "robust_progress": [v for p in g for v in p["robust"]["progress"]]})
    return out


def training_runs(exp: Path) -> list[dict]:
    """Every metrics.jsonl with its identity, read from the run's own metadata when it exists."""
    runs = []
    for path in sorted(Path(exp).glob("training/*/*/*/seed_*/metrics.jsonl")):
        meta_path = path.with_name("model.meta.json")
        meta = read_json(meta_path) if meta_path.exists() else {}
        rows = read_jsonl(path)
        runs.append({"path": str(path), "stratum": stratum_label(meta["scenario_family"], meta["blocker_side"]) if meta else _MISSING,
                     "reward": reward_key(meta), "scenario_id": meta.get("scenario_id"), "ppo_seed": meta.get("ppo_seed"),
                     "train": [r for r in rows if r["kind"] == "train"], "validation": [r for r in rows if r["kind"] == "validation"]})
    return runs


def final_validation_by_reward(runs: list[dict]) -> dict:
    out = defaultdict(list)
    for run in runs:
        if run["validation"]:
            out[run["reward"]].append(run["validation"][-1]["success_rate"])
    return {k: _mean(v) for k, v in out.items()}


def aligned_curves(runs: list[dict], stream: str, key: str, n_grid: int = 60) -> dict | None:
    """Mean, spread and run count across runs on a common step grid. Each run is interpolated only inside its
    own range: a run that stopped early drops out of the later steps rather than being extrapolated."""
    series = [(np.array([r["timesteps"] for r in run[stream] if r.get(key) is not None], dtype=float),
               np.array([r[key] for r in run[stream] if r.get(key) is not None], dtype=float)) for run in runs]
    series = [(x, y) for x, y in series if len(x)]
    if not series:
        return None
    grid = np.linspace(min(x[0] for x, _ in series), max(x[-1] for x, _ in series), n_grid if len(series) > 1 or len(series[0][0]) > n_grid else len(series[0][0]))
    stack = np.array([np.where((grid >= x[0]) & (grid <= x[-1]), np.interp(grid, x, y), np.nan) for x, y in series])
    count = np.sum(np.isfinite(stack), axis=0)
    valid = count > 0   # every grid step lies in some run's range, except between runs' disjoint ranges
    mean, std = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
    mean[valid] = np.nanmean(stack[:, valid], axis=0)
    std[valid] = np.nanstd(stack[:, valid], axis=0)
    return {"steps": grid, "mean": mean, "std": std, "count": count, "n_runs": len(series)}


def fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)
