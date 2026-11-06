"""
Animated replay of saved evaluation rollouts (docs/description.md §23), from the saved arrays only: nothing is
re-simulated. Optional by design: nothing in the evaluation pipeline imports this module. Writes MP4 if ffmpeg
is available, otherwise GIF. To replay a rollout live from its checkpoint instead, see learning.animate_rollout.

Usage:
    python -m learning.animate_ppo_results --experiment results/ppo_feasibility/<name>      # representative rollouts
    python -m learning.animate_ppo_results --experiment <dir> --scenario <id> --policy <id> --eval-mode robust --episode 12
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from learning.env import DT
from learning.ppo_analysis import load_records
from learning.plot_ppo_results import apply_style, build_road, draw_road, footprint, representatives, rollout_title
from learning.ppo_experiment import read_json
from learning.trajectory_log import load_trajectory

_VIEW_M = 55.0


def find_rollout(records: list[dict], scenario: str, policy: str | None, eval_mode: str, episode: int) -> tuple:
    for rec in records:
        if rec["scenario"]["scenario_id"] != scenario and rec["scenario"]["training_scenario_id"] != scenario:
            continue
        if rec["protocol"]["eval_mode"] != eval_mode or (policy and rec["policy"]["policy_id"] != policy):
            continue
        for r in rec["rollouts"]:
            if r["episode"] == episode:
                if not r.get("trajectory"):
                    raise SystemExit(f"episode {episode} was evaluated but its trajectory was not kept "
                                     f"(seed {r['seed_info']['scenario_seed']}, rank {r['seed_info']['worker_rank']}).")
                return r["trajectory"], rec, r
    raise SystemExit(f"no such rollout: scenario {scenario}, policy {policy}, {eval_mode}, episode {episode}")


def animate(traj, out: Path, title: str, notice: str | None = None, stride: int = 2) -> Path:
    apply_style()
    t, x, y, h = traj.ego("t"), traj.ego("x"), traj.ego("y"), traj.ego("heading")
    sx, sy, sh, crashed = traj.surr("x"), traj.surr("y"), traj.surr("heading"), traj.surr("crashed")
    fig, ax = plt.subplots(figsize=(7.16, 3.0), constrained_layout=True)
    draw_road(ax, build_road(traj.meta))
    ax.set_aspect("equal")
    ax.set_xlabel("World x [m]")
    ax.set_ylabel("World y [m]")
    ax.set_title(title)
    if notice:
        fig.text(0.5, 0.02, notice, ha="center", fontsize=7, color="#D55E00", fontweight="bold")
    ego = ax.add_patch(Polygon(footprint(x[0], y[0], h[0]), closed=True, facecolor="#D55E00", edgecolor="black", zorder=5))
    cars = [ax.add_patch(Polygon(footprint(sx[0, j], sy[0, j], sh[0, j]), closed=True, facecolor="#0072B2", edgecolor="black", zorder=4)) for j in range(sx.shape[1])]
    text = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", fontsize=7, family="monospace", zorder=9)
    triggered, accel, delta = traj.ego("event_triggered"), traj.ego("accel_applied"), traj.ego("delta_applied")
    outcome = traj.meta["outcome"] if not traj.meta["failure_reason"] else traj.meta["failure_reason"]

    def update(i):
        ego.set_xy(footprint(x[i], y[i], h[i]))
        for j, car in enumerate(cars):
            car.set_xy(footprint(sx[i, j], sy[i, j], sh[i, j]))
            car.set_facecolor("#7f7f7f" if crashed[i, j] else "#0072B2")
        ax.set_xlim(x[i] - _VIEW_M * 0.45, x[i] + _VIEW_M)
        ax.set_ylim(y[i] - 18.0, y[i] + 18.0)
        event = "event active" if triggered[i] > 0 else "before event"
        a = "n/a" if accel[i] != accel[i] else f"{accel[i]:+.1f}"
        d = "n/a" if delta[i] != delta[i] else f"{delta[i]:+.3f}"
        text.set_text(f"t={t[i]:5.2f} s  v={traj.ego('v_x')[i]:5.1f} m/s  a_cmd={a} m/s2  delta={d} rad  {event}\nfinal outcome: {outcome}")
        return [ego, text, *cars]

    frames = range(0, len(t), stride)
    ani = FuncAnimation(fig, update, frames=frames, blit=False)
    fps = int(round(1.0 / (DT * stride)))
    out.parent.mkdir(parents=True, exist_ok=True)
    if FFMpegWriter.isAvailable():
        path = out.with_suffix(".mp4")
        ani.save(path, writer=FFMpegWriter(fps=fps), dpi=130)
    else:
        path = out.with_suffix(".gif")
        ani.save(path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--scenario", default=None, help="scenario id; omit to animate the representative rollouts")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--eval-mode", choices=["exact", "robust"], default="robust")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    exp = Path(args.experiment)
    records = load_records(exp)
    data_class = read_json(exp / "manifest.json")["data_class"]
    notice = None if data_class == "experiment" else f"DEMONSTRATION ({data_class}): not an experimental result"
    if args.scenario:
        jobs = {f"{args.scenario}_{args.eval_mode}_ep{args.episode}": find_rollout(records, args.scenario, args.policy, args.eval_mode, args.episode)}
    else:
        jobs, missing = representatives(records)
        print("no saved rollout for: " + (", ".join(missing) or "(none missing)"))
    for name, (path, rec, r) in jobs.items():
        print(animate(load_trajectory(exp / path), exp / "animations" / name, rollout_title(rec, r), notice))


if __name__ == "__main__":
    main()
