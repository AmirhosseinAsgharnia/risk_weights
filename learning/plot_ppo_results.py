"""
Publication figures from saved PPO evaluation records (docs/description.md §23). Reads an experiment directory
and runs no simulation, so figures can be remade without rerunning PPO.

Every plotted value is a measurement from learning.evaluate_ppo or learning.ppo_metrics: no surrogate
prediction, no interpolated probability surface. A figure for which no real record exists is skipped and listed
in figures/index.json, never filled in. A data class other than "experiment" (manifest.json) puts a visible
notice on every figure.

Usage:
    python -m learning.plot_ppo_results --experiment results/ppo_feasibility/<name> [--width single|double]
        [--pair FAMILY:PARAM_X,PARAM_Y] [--fix PARAM=VALUE ...] [--rank p_hat|lower|difficulty|family]
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from initialization.traffic_init import CAR_LENGTH, CAR_WIDTH
from learning.ppo_analysis import (
    aligned_curves, final_validation_by_reward, load_records, robustness_table,
    scenario_table, stratum_label, table_c, training_runs,
)
from learning.minimax_feasibility import ADMITTED, REJECTED, UNCERTAIN
from learning.ppo_experiment import ADMISSION_P0, FAMILIES, read_json, write_json
from learning.trajectory_log import friction_utilization, load_trajectory
from model.road.road import Road

WIDTHS = {"single": 3.5, "double": 7.16}   # IEEE column widths [in]
# Okabe-Ito: colorblind-safe; every series also differs by marker and line style for grayscale.
SERIES = [("#0072B2", "o", "-"), ("#D55E00", "s", "--"), ("#009E73", "^", "-."), ("#CC79A7", "D", ":"), ("#E69F00", "v", (0, (5, 1)))]
STATUS_STYLE = {ADMITTED: ("#009E73", "o", "Admitted"), UNCERTAIN: ("#E69F00", "s", "Uncertain"), REJECTED: ("#D55E00", "X", "Not admitted")}
DEFAULT_PAIRS = {"cutin": [("road_mu", "cutter_gap_m"), ("road_mu", "cutter_relative_speed_mps"),
                           ("escape_gap_length_m", "cutter_gap_m"), ("road_kappa_max", "front_gap_m")],
                 "sandwich": [("road_mu", "front_gap_m"), ("front_gap_m", "rear_gap_m"),
                              ("escape_gap_length_m", "rear_relative_speed_mps"),
                              ("escape_gap_center_m", "blocker_relative_speed_mps")]}
PARAM_LABELS = {"road_mu": r"Road friction $\mu$", "road_kappa_max": r"Road curvature $\kappa_{\max}$ [1/m]",
                "s_cutin_m": "Cut-in station [m]", "s_stop_m": "Stop station [m]", "cutter_gap_m": "Cutter gap [m]",
                "cutter_relative_speed_mps": "Cutter relative speed [m/s]", "front_gap_m": "Front gap [m]",
                "front_relative_speed_mps": "Front relative speed [m/s]", "rear_gap_m": "Rear gap [m]",
                "rear_relative_speed_mps": "Rear relative speed [m/s]", "escape_gap_length_m": "Escape-gap length [m]",
                "escape_gap_center_m": "Escape-gap center [m]", "blocker_relative_speed_mps": "Blocker relative speed [m/s]"}
NOT_PROOF = "A policy's failure does not show a scenario is physically impossible."


def apply_style() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2, "lines.markersize": 4,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linewidth": 0.4, "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})


class Figures:
    """Saves PDF and PNG, adds the data-class notice, and records what was made and what could not be."""

    def __init__(self, exp: Path, width: str = "double"):
        self.exp, self.dir = Path(exp), Path(exp) / "figures"
        self.dir.mkdir(exist_ok=True)
        self.w = WIDTHS[width]
        self.data_class = read_json(self.exp / "manifest.json")["data_class"]
        self.made, self.skipped = [], []

    def save(self, fig, name: str, caption: str) -> None:
        if self.data_class != "experiment":
            fig.text(0.5, 1.0, f"DEMONSTRATION ({self.data_class}): not an experimental result", ha="center", va="bottom",
                     fontsize=7, color="#D55E00", fontweight="bold")
        for ext in ("pdf", "png"):
            fig.savefig(self.dir / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        self.made.append({"name": name, "caption": caption})

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append({"name": name, "reason": reason})

    def close(self) -> None:
        write_json(self.dir / "index.json", {"data_class": self.data_class, "figures": self.made, "skipped": self.skipped})


def _line(ax, x, y, i: int, label: str | None = None, every: int | None = None):
    c, m, ls = SERIES[i % len(SERIES)]
    ax.plot(x, y, color=c, marker=m, linestyle=ls, label=label, markevery=every or max(1, len(x) // 8))
    return c


# ── learning curves ─────────────────────────────────────────────────────────
def _curve_panel(ax, groups: dict, stream: str, key: str, ylabel: str, ylim=None) -> bool:
    drawn = False
    for i, (label, runs) in enumerate(sorted(groups.items())):
        cur = aligned_curves(runs, stream, key)
        if cur is None:
            continue
        drawn = True
        c = _line(ax, cur["steps"], cur["mean"], i, f"{label} ({cur['n_runs']} runs)")
        if cur["n_runs"] > 1:
            ax.fill_between(cur["steps"], cur["mean"] - cur["std"], cur["mean"] + cur["std"], color=c, alpha=0.18, linewidth=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Training steps")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    if ylim:
        ax.set_ylim(*ylim)
    if not drawn:
        ax.text(0.5, 0.5, "not recorded", transform=ax.transAxes, ha="center", va="center")
    return drawn


def learning_curves(f: Figures, runs: list[dict]) -> None:
    groups = {}
    for run in runs:
        groups.setdefault(f"{run['stratum']}", []).append(run)
    if not any(r["validation"] for r in runs):
        f.skip("learning_validation", "no validation metrics recorded")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(f.w, f.w * 0.62), constrained_layout=True)
        _curve_panel(axes[0, 0], groups, "validation", "success_rate", "Validation success rate", (-0.02, 1.02))
        _curve_panel(axes[0, 1], groups, "validation", "collision_rate", "Validation collision rate", (-0.02, 1.02))
        _curve_panel(axes[1, 0], groups, "validation", "ep_return_mean", "Validation mean return")
        _curve_panel(axes[1, 1], groups, "validation", "ep_length_mean", "Validation episode length [steps]")
        axes[0, 0].legend()
        f.save(fig, "learning_validation", "Independent validation: deterministic policy on fixed held-out realizations, which never "
                                           "update the policy. Line: mean across runs; band: one standard deviation. Runs that stop "
                                           "early leave the average, so late steps may cover fewer runs.")
    if not any(r["train"] for r in runs):
        f.skip("learning_training", "no training metrics recorded")
        return
    fig, axes = plt.subplots(1, 3, figsize=(f.w, f.w * 0.3), constrained_layout=True)
    _curve_panel(axes[0], groups, "train", "ep_return_mean", "Training return")
    _curve_panel(axes[1], groups, "train", "ep_length_mean", "Training length [steps]")
    _curve_panel(axes[2], groups, "train", "success_rate", "Training success rate", (-0.02, 1.02))
    axes[0].legend()
    f.save(fig, "learning_training", "PPO training metrics: a stochastic policy on the training realization. These are not evaluation results.")
    fig, axes = plt.subplots(2, 3, figsize=(f.w, f.w * 0.55), constrained_layout=True)
    for ax, (key, label) in zip(axes.flat, (("policy_gradient_loss", "Policy loss"), ("value_loss", "Value loss"), ("entropy_loss", "Entropy loss"),
                                            ("approx_kl", "Approximate KL"), ("explained_variance", "Explained variance"), ("learning_rate", "Learning rate"))):
        _curve_panel(ax, groups, "train", key, label)
    f.save(fig, "learning_diagnostics", "PPO optimization diagnostics reported by Stable-Baselines3.")


# ── feasibility maps and Wilson intervals ───────────────────────────────────
def select_slice(rows: list[dict], pair: tuple, fixed: dict | None) -> tuple[list[dict], str, bool]:
    """(rows, description, is_slice). With `fixed`, keeps scenarios at those values. Parameters outside the pair
    that are constant across the kept rows also define a slice; if any still varies, this is a projection, and
    says so: its points are different scenarios in the hidden dimensions."""
    if fixed:
        rows = [r for r in rows if all(math.isclose(r["theta"][k], v, rel_tol=1e-9, abs_tol=1e-9) for k, v in fixed.items() if k in r["theta"])]
    if not rows:
        return [], "", False
    others = [k for k in rows[0]["theta"] if k not in pair]
    varying = [k for k in others if len({r["theta"][k] for r in rows}) > 1]
    held = ", ".join(f"{k}={rows[0]['theta'][k]:g}" for k in others if k not in varying)
    if varying:
        return rows, f"Projection, not a slice: {len(varying)} other parameters vary ({', '.join(varying)})" + (f"; held: {held}" if held else ""), False
    return rows, f"Slice: {held}", True


def _is_full_grid(xs, ys) -> bool:
    ux, uy = sorted(set(xs)), sorted(set(ys))
    return len(ux) >= 2 and len(uy) >= 2 and len(set(zip(xs, ys))) == len(ux) * len(uy) == len(xs)


def _edges(values: list[float]) -> np.ndarray:
    v = np.array(sorted(set(values)))
    mid = (v[:-1] + v[1:]) / 2
    return np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])


def _bound_axes(ax, bounds: dict, pair: tuple) -> None:
    """The validated theta range plus a small margin, so points on a bound are not clipped."""
    for axis, name in zip((ax.set_xlim, ax.set_ylim), pair):
        lo, hi = bounds[name]
        axis(lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))


def feasibility_maps(f: Figures, rows: list[dict], pairs: dict, fixed: dict | None) -> None:
    rows = [r for r in rows if r["eval_mode"] == "robust" and r["independent"]]
    for family, side in sorted({(r["family"], r["blocker_side"]) for r in rows}):
        for pair in pairs.get(family, []):
            tag = f"{family}_{'pos' if side == 1 else 'neg'}1_{pair[0]}__{pair[1]}"
            subset, describe, _ = select_slice([r for r in rows if (r["family"], r["blocker_side"]) == (family, side)], pair, fixed)
            if not subset:
                f.skip(f"map_{tag}", "no evaluated scenario matches the requested slice")
                continue
            xs, ys = [r["theta"][pair[0]] for r in subset], [r["theta"][pair[1]] for r in subset]
            bounds = FAMILIES[family][2]

            fig, ax = plt.subplots(figsize=(f.w * 0.6, f.w * 0.42), constrained_layout=True)
            if _is_full_grid(xs, ys):
                lookup = {(x, y): r for x, y, r in zip(xs, ys, subset)}
                ux, uy = sorted(set(xs)), sorted(set(ys))
                z = np.array([[lookup[x, y]["p_hat"] for x in ux] for y in uy])
                mesh = ax.pcolormesh(_edges(xs), _edges(ys), z, cmap="cividis", vmin=0, vmax=1, edgecolors="white", linewidth=0.4)
                for (x, y), r in lookup.items():
                    ax.text(x, y, f"{r['k']}/{r['n']}", ha="center", va="center", fontsize=6, color="white" if r["p_hat"] < 0.6 else "black")
            else:
                mesh = ax.scatter(xs, ys, c=[r["p_hat"] for r in subset], cmap="cividis", vmin=0, vmax=1, s=22, edgecolors="black", linewidths=0.3)
                _bound_axes(ax, bounds, pair)
            fig.colorbar(mesh, ax=ax, label="Observed success rate $k/N$")
            ax.set_xlabel(PARAM_LABELS[pair[0]])
            ax.set_ylabel(PARAM_LABELS[pair[1]])
            ax.set_title(f"{stratum_label(family, side)}: observed rollouts ($N$={sorted({r['n'] for r in subset})})")
            f.save(fig, f"map_empirical_{tag}", f"Observed success rate per evaluated scenario, no interpolation. {describe}.")

            fig, ax = plt.subplots(figsize=(f.w * 0.6, f.w * 0.42), constrained_layout=True)
            for status, (color, marker, label) in STATUS_STYLE.items():
                g = [r for r in subset if r["status"] == status]
                ax.scatter([r["theta"][pair[0]] for r in g], [r["theta"][pair[1]] for r in g], c=color, marker=marker, s=26,
                           edgecolors="black", linewidths=0.3, label=f"{label} ({len(g)})")
            _bound_axes(ax, bounds, pair)
            ax.set_xlabel(PARAM_LABELS[pair[0]])
            ax.set_ylabel(PARAM_LABELS[pair[1]])
            ax.set_title(f"{stratum_label(family, side)}: statistical admission, $p_0$={ADMISSION_P0}")
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
            f.save(fig, f"map_admission_{tag}", f"Admission from each scenario's 95% Wilson interval against p0 = {ADMISSION_P0}: decided per "
                                                f"scenario, never read off a surface. {describe}. {NOT_PROOF}")


def rank_rows(rows: list[dict], rank: str) -> list[dict]:
    """difficulty is a measured proxy, the mean peak tire utilization over rollouts: nothing here predicts difficulty."""
    keys = {"p_hat": lambda r: r["p_hat"], "lower": lambda r: r["wilson_lower"], "family": lambda r: (r["stratum"], r["p_hat"]),
            "difficulty": lambda r: r["mean_peak_tire_utilization"] if r["mean_peak_tire_utilization"] is not None else math.inf}
    return sorted(rows, key=keys[rank])


def wilson_plot(f: Figures, rows: list[dict], rank: str) -> None:
    rows = rank_rows([r for r in rows if r["independent"] and r["n"]], rank)
    if not rows:
        f.skip("wilson_intervals", "no independent evaluation records")
        return
    fig, ax = plt.subplots(figsize=(f.w, f.w * 0.36), constrained_layout=True)
    for status, (color, marker, label) in STATUS_STYLE.items():
        idx = [i for i, r in enumerate(rows) if r["status"] == status]
        if idx:
            p = np.array([rows[i]["p_hat"] for i in idx])
            # clipped: at k = 0 or k = n the bound equals p_hat up to rounding, which errorbar rejects if negative
            err = [np.maximum(p - np.array([rows[i]["wilson_lower"] for i in idx]), 0.0),
                   np.maximum(np.array([rows[i]["wilson_upper"] for i in idx]) - p, 0.0)]
            ax.errorbar(idx, p, yerr=err, fmt=marker, color=color, ecolor=color, elinewidth=0.8, capsize=1.5, markersize=3.5, label=f"{label} ({len(idx)})")
    ax.axhline(ADMISSION_P0, color="black", linestyle="--", linewidth=0.9, label=f"$p_0$ = {ADMISSION_P0}")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel({"p_hat": "Scenarios ranked by observed success rate", "lower": "Scenarios ranked by Wilson lower bound",
                   "difficulty": "Scenarios ranked by mean peak tire utilization (measured proxy)", "family": "Scenarios grouped by stratum"}[rank])
    ax.set_ylabel(r"Success rate $\hat{p}$ with 95% Wilson interval")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    f.save(fig, f"wilson_intervals_{rank}", f"Observed policy success with two-sided 95% Wilson intervals from independent test rollouts of the policy selected "
                                            f"on a separate stream. This is evidence about PPO success. {NOT_PROOF}")


# ── trajectories ────────────────────────────────────────────────────────────
def build_road(meta: dict) -> Road:
    return Road(s_max=meta["s_max"], kappa_max=meta["kappa_max"], L_clothoid=meta["L_clothoid"], mu=meta["mu"], lane_num=meta["lane_num"])


def footprint(x: float, y: float, heading: float) -> np.ndarray:
    """Oriented CAR_LENGTH x CAR_WIDTH rectangle in world coordinates, the simulator's collision dimensions."""
    c, s, hl, hw = math.cos(heading), math.sin(heading), CAR_LENGTH / 2, CAR_WIDTH / 2
    return np.array([[x + a * hl * c - b * hw * s, y + a * hl * s + b * hw * c] for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def collision_location(traj) -> dict | None:
    """Ego's final pose and the nearest vehicle, from the simulator's own collision flag on the last step."""
    if not np.nan_to_num(traj.ego("collided"))[-1]:
        return None
    x, y = traj.ego("x")[-1], traj.ego("y")[-1]
    d = np.hypot(traj.surr("x")[-1] - x, traj.surr("y")[-1] - y)
    return {"x": float(x), "y": float(y), "car_id": int(traj.surr("car_id")[int(np.argmin(d))])}


def event_index(traj) -> int | None:
    flags = traj.ego("event_triggered")
    if flags is None:
        return None
    hits = np.nonzero(np.nan_to_num(flags) > 0)[0]
    return int(hits[0]) if len(hits) else None


def draw_road(ax, road: Road) -> None:
    half = road.lanes[-1].offset + road.lanes[-1].width / 2
    lo, hi = road.offset_curve(-half), road.offset_curve(half)
    ax.fill(np.concatenate([lo[0], hi[0][::-1]]), np.concatenate([lo[1], hi[1][::-1]]), color="#f2f2f2", zorder=0, linewidth=0)
    ax.plot(*lo, color="black", linewidth=0.9, zorder=1)
    ax.plot(*hi, color="black", linewidth=0.9, zorder=1)
    for a, b in zip(road.lanes[:-1], road.lanes[1:]):
        ax.plot(*road.offset_curve((a.offset + b.offset) / 2), color="gray", linewidth=0.5, linestyle=(0, (6, 6)), zorder=1)


def trajectory_plot(f: Figures, traj, name: str, title: str) -> None:
    road = build_road(traj.meta)
    x, y, h = traj.ego("x"), traj.ego("y"), traj.ego("heading")
    ev = event_index(traj)
    start = max(0, (ev or 0) - 40)
    x0, x1 = x[start:].min() - 25, x[start:].max() + 25
    y0, y1 = y[start:].min() - 12, y[start:].max() + 12
    height = float(np.clip(f.w * (y1 - y0) / (x1 - x0) + 0.75, 1.5, 6.0))
    fig, ax = plt.subplots(figsize=(f.w, height), layout="constrained")
    draw_road(ax, road)
    stride = max(1, int(round(1.0 / 0.05)))
    for j, cid in enumerate(traj.surr("car_id")):
        sx, sy, sh = traj.surr("x")[:, j], traj.surr("y")[:, j], traj.surr("heading")[:, j]
        ax.plot(sx[start:], sy[start:], color="#0072B2", linewidth=0.7, linestyle=":", zorder=2)
        for i, fc, a in ((start, "none", 0.9), (len(sx) - 1, "#0072B2", 0.75)):
            ax.add_patch(Polygon(footprint(sx[i], sy[i], sh[i]), closed=True, facecolor=fc, edgecolor="#0072B2", linewidth=0.6, alpha=a, zorder=3))
        ax.annotate(str(cid), (sx[-1], sy[-1]), fontsize=5, ha="center", va="center", zorder=6)
    ax.plot(x[start:], y[start:], color="black", linewidth=1.3, zorder=4, label="Ego path")
    for i in range(start, len(x), stride):
        ax.add_patch(Polygon(footprint(x[i], y[i], h[i]), closed=True, facecolor="none", edgecolor="#D55E00", linewidth=0.4, alpha=0.6, zorder=4))
    ax.add_patch(Polygon(footprint(x[start], y[start], h[start]), closed=True, facecolor="none", edgecolor="black", linewidth=0.9, zorder=5, label="Ego start of plot"))
    ax.add_patch(Polygon(footprint(x[-1], y[-1], h[-1]), closed=True, facecolor="#D55E00", edgecolor="black", linewidth=0.7, zorder=5, label="Ego final"))
    ax.plot([], [], color="#0072B2", linestyle=":", label="Traffic (hollow: start of plot; filled: final)")
    if ev is not None:
        ax.plot(x[ev], y[ev], marker="v", color="#009E73", markeredgecolor="black", markersize=6, linestyle="none", zorder=7, label="Event trigger")
    hit = collision_location(traj)
    if hit:
        ax.plot(hit["x"], hit["y"], marker="X", color="#CC79A7", markeredgecolor="black", markersize=7, linestyle="none", zorder=8, label=f"Collision (car {hit['car_id']})")
    goal = traj.meta["goal_s"]
    half = traj.meta["road_half_width_m"]
    gx0, gy0, _ = road.frenet_to_global(goal, -half)
    gx1, gy1, _ = road.frenet_to_global(goal, half)
    in_view = x0 <= (gx0 + gx1) / 2 <= x1 and y0 <= (gy0 + gy1) / 2 <= y1
    ax.plot([gx0, gx1], [gy0, gy1], color="#E69F00", linewidth=1.6, zorder=2, label="Goal" if in_view else f"Goal (s = {goal:.0f} m, outside this view)")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("World x [m]")
    ax.set_ylabel("World y [m]")
    ax.set_title(title)
    fig.legend(*ax.get_legend_handles_labels(), ncol=3, loc="outside lower center", fontsize=6)
    f.save(fig, name, f"{title}. World coordinates on the true road geometry; ego footprints every 1 s. The view starts 2 s before the event.")


def time_series_plot(f: Figures, traj, name: str, title: str) -> None:
    t = traj.ego("t")
    util = friction_utilization(traj)
    fig, axes = plt.subplots(4, 1, figsize=(f.w, f.w * 0.72), sharex=True, constrained_layout=True)
    if util is None:
        axes[0].text(0.5, 0.5, "tire forces not recorded", transform=axes[0].transAxes, ha="center")
    else:
        _line(axes[0], t, util["front"], 0, "Front axle")
        _line(axes[0], t, util["rear"], 1, "Rear axle")
        axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8, label=r"Friction limit $\eta$ = 1")
        axes[0].set_ylim(0, 1.3)
    axes[0].set_ylabel(r"Utilization $\eta$")
    _line(axes[1], t, traj.ego("a_x"), 0, r"Realized $a_x$")
    _line(axes[1], t, traj.ego("a_y"), 1, r"Realized $a_y$")
    _line(axes[1], t, traj.ego("accel_applied"), 2, "Applied command")
    axes[1].set_ylabel(r"Accel. [m/s$^2$]")
    _line(axes[2], t, traj.ego("delta_applied"), 0, "Applied")
    _line(axes[2], t, traj.ego("delta_requested"), 1, "Requested")
    axes[2].set_ylabel(r"Steering $\delta$ [rad]")
    _line(axes[3], t, traj.ego("r"), 0)
    axes[3].set_ylabel("Yaw rate [rad/s]")
    axes[3].set_xlabel("Time [s]")
    for ax in axes[:3]:
        ax.legend(ncol=3, loc="upper right", frameon=True, framealpha=0.85, edgecolor="none")
    ev = event_index(traj)
    if ev is not None:
        for ax in axes:
            ax.axvline(t[ev], color="#009E73", linewidth=0.8, linestyle=":")
    axes[0].set_title(title)
    f.save(fig, name, f"{title}. Simulator tire forces and states, sampled once per control step. "
                      f"Utilization at 1 is tire saturation, which is not by itself a failure.")


def representatives(records: list[dict]) -> tuple[dict, list[str]]:
    """Category -> (trajectory path, record, rollout), from real saved rollouts only; missing categories are returned."""
    wanted = {"cutin_success": lambda rec, r: rec["scenario"]["family"] == "cutin" and r["success"],
              "cutin_failure": lambda rec, r: rec["scenario"]["family"] == "cutin" and not r["success"],
              "sandwich_success": lambda rec, r: rec["scenario"]["family"] == "sandwich" and r["success"],
              "sandwich_failure": lambda rec, r: rec["scenario"]["family"] == "sandwich" and not r["success"],
              "high_friction": lambda rec, r: rec["scenario"]["theta"]["road_mu"] >= 0.8,
              "low_friction": lambda rec, r: rec["scenario"]["theta"]["road_mu"] <= 0.5}
    found = {}
    for name, test in wanted.items():
        for rec in records:
            hit = next((r for r in rec["rollouts"] if r.get("trajectory") and test(rec, r)), None)
            if hit:
                found[name] = (hit["trajectory"], rec, hit)
                break
    return found, [n for n in wanted if n not in found]


def rollout_title(rec: dict, r: dict) -> str:
    sc = rec["scenario"]
    side = "+1" if sc["blocker_side"] == 1 else "-1"
    outcome = "success" if r["success"] else (r["failure_reason"] or "timeout")
    return f"{sc['family']}, side {side}, $\\mu$={sc['theta']['road_mu']:.2f}, {rec['protocol']['eval_mode']} ep {r['episode']}: {outcome}"


def trajectory_figures(f: Figures, records: list[dict]) -> None:
    found, missing = representatives(records)
    for name in missing:
        f.skip(f"trajectory_{name}", "no saved rollout in this category")
    for name, (path, rec, r) in found.items():
        traj = load_trajectory(f.exp / path)
        trajectory_plot(f, traj, f"trajectory_{name}", rollout_title(rec, r))
        time_series_plot(f, traj, f"physics_{name}", rollout_title(rec, r))


# ── comparisons ─────────────────────────────────────────────────────────────
def robustness_figure(f: Figures, rows: list[dict]) -> None:
    table = robustness_table(rows)
    if not table:
        f.skip("robustness", "no policy was evaluated in both exact and robust modes")
        return
    fig, axes = plt.subplots(1, 2, figsize=(f.w, f.w * 0.36), constrained_layout=True)
    x = np.arange(len(table))
    for off, mode, hatch, c in ((-0.18, "exact", "", "#0072B2"), (0.18, "robust", "///", "#E69F00")):
        v = [t[f"{mode}_success"] for t in table]
        ci = [t[f"{mode}_ci"] or (a, a) for t, a in zip(table, v)]
        axes[0].bar(x + off, v, 0.34, color=c, hatch=hatch, edgecolor="black", linewidth=0.4, label=mode,
                    yerr=[[max(a - lo, 0) for a, (lo, _) in zip(v, ci)], [max(hi - a, 0) for a, (_, hi) in zip(v, ci)]], error_kw={"elinewidth": 0.8, "capsize": 2})
    axes[0].set_xticks(x, [t["stratum"] for t in table], rotation=15)
    axes[0].set_ylim(0, 1.03)
    axes[0].set_ylabel("Success rate (mean over scenarios)")
    axes[0].legend()
    positions, data = [], []
    for i, t in enumerate(table):
        positions += [i - 0.18, i + 0.18]
        data += [t["exact_progress"], t["robust_progress"]]
    axes[1].boxplot(data, positions=positions, widths=0.3, flierprops={"markersize": 1.5})
    axes[1].set_xticks(x, [t["stratum"] for t in table], rotation=15)
    axes[1].set_ylim(0, 1.03)
    axes[1].set_ylabel("Progress (fraction of goal)\nleft: exact, right: robust")
    trained = ", ".join(sorted({t["trained_mode"] for t in table}))
    f.save(fig, "robustness", f"The same policies in exact and robust evaluation (trained in {trained} mode on a single fixed realization). Each exact result is one "
                              f"deterministic outcome per policy, not repeated trials; intervals bootstrap over scenarios.")


def reward_figure(f: Figures, records: list[dict], rows: list[dict], runs: list[dict]) -> None:
    table, issues = table_c(records, rows, final_validation_by_reward(runs))
    if not table:
        f.skip("reward_comparison", issues[0] if issues else "fewer than two reward configurations are present")
        return
    metrics = (("heldout_success_rate", "Success"), ("rollout_collision_rate", "Collision"), ("rollout_road_departure_rate", "Road departure"), ("rollout_timeout_rate", "Timeout"))
    fig, ax = plt.subplots(figsize=(f.w, f.w * 0.34), constrained_layout=True)
    width = 0.8 / len(table)
    for i, row in enumerate(table):
        c, _, _ = SERIES[i % len(SERIES)]
        ax.bar(np.arange(len(metrics)) + i * width, [row[k] or 0 for k, _ in metrics], width, color=c, hatch=("", "///", "...", "xx")[i % 4], edgecolor="black", linewidth=0.4, label=row["reward"])
    ax.set_xticks(np.arange(len(metrics)) + 0.4 - width / 2, [m[1] for m in metrics])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Rate")
    ax.legend()
    if issues:
        ax.set_title("NON-EQUIVALENT COMPARISON: see report")
    f.save(fig, "reward_comparison", "Held-out robust evaluation of policies trained under each reward. " + ("Non-equivalent: " + "; ".join(issues) if issues else "No configuration difference was detected."))


def make_all(exp: Path, width: str = "double", rank: str = "p_hat", pairs: dict | None = None, fixed: dict | None = None) -> Figures:
    apply_style()
    exp = Path(exp)
    f = Figures(exp, width)
    records, runs = load_records(exp), training_runs(exp)
    rows = scenario_table(records)
    learning_curves(f, runs)
    feasibility_maps(f, rows, pairs or DEFAULT_PAIRS, fixed)
    wilson_plot(f, rows, rank)
    trajectory_figures(f, records)
    robustness_figure(f, rows)
    reward_figure(f, records, rows, runs)
    f.close()
    return f


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--width", choices=list(WIDTHS), default="double")
    parser.add_argument("--rank", choices=["p_hat", "lower", "difficulty", "family"], default="p_hat")
    parser.add_argument("--pair", action="append", default=None, help="FAMILY:PARAM_X,PARAM_Y; replaces the default pairs; repeatable")
    parser.add_argument("--fix", nargs="*", default=[], help="PARAM=VALUE for the parameters held fixed in a slice")
    args = parser.parse_args()
    pairs = None
    if args.pair:
        pairs = {}
        for spec in args.pair:
            family, names = spec.split(":")
            pairs.setdefault(family, []).append(tuple(names.split(",")))
    fixed = {k: float(v) for k, v in (kv.split("=") for kv in args.fix)} or None
    f = make_all(Path(args.experiment), args.width, args.rank, pairs, fixed)
    print(f"{len(f.made)} figures in {f.dir}; {len(f.skipped)} skipped (see index.json)")


if __name__ == "__main__":
    main()
