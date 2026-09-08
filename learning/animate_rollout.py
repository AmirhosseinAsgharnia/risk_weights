"""
Replay one specific evaluation-episode realization of a trained
feasibility-arena policy (cutin/sandwich) and animate it.

learning.eval can't do this: it hardcodes mode="fixed", which always
replays episode-index 0 for a given scenario, ignoring any --seed. The
final evaluation that produced <scenario_id>'s success_rate/rollouts.jsonl
records ran in mode="distribution" (worker_rank=0), stepping through
episode indices 0, 1, 2, ... in order via repeated reset() calls (see
learning.eval_batch.run_episodes) -- so to replay episode K exactly, this
reconstructs the same env and calls reset() K+1 times, discarding all but
the last, which reproduces derive_seed(scenario_seed, worker_rank=0,
episode_index=K) exactly.

Usage:
    python -m learning.animate_rollout \\
        --model artifacts/feasibility/policies/cutin/side_pos1/<scenario_id>/model \\
        --episode-index 37 [--stochastic] [--save out.mp4]

Find which episode indices actually failed (and why) for a given
scenario_id via its rollouts.jsonl record -- see docs/feasibility_pipeline.md.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

from stable_baselines3 import PPO

from learning.env import EgoTrafficEnv, DT, Outcome
from learning.eval_batch import _load_meta
from learning.feasibility_cutin import CutinConfig, CutinRuntime
from learning.feasibility_sandwich import SandwichConfig, SandwichRuntime
from learning.eval import car_corners, _BEHAVIOUR_COLOR, _CRASHED_COLOR, EGO_COLOR, _OUTCOME_TEXT

_CFG_CLASSES = {"cutin": CutinConfig, "sandwich": SandwichConfig}
_RUNTIME_CLASSES = {"cutin": CutinRuntime, "sandwich": SandwichRuntime}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="path to a feasibility-pipeline model (no .zip suffix)")
    parser.add_argument("--episode-index", type=int, required=True,
                         help="0-based -- must match a rollouts.jsonl 'episode' value for this scenario_id")
    parser.add_argument("--stochastic", action="store_true",
                         help="sample actions from the policy instead of using its deterministic mean")
    parser.add_argument("--save", type=str, default=None,
                         help="write an mp4/gif here instead of opening an interactive window "
                              "(required when running over SSH with no display, e.g. on abl-drivsim-iii)")
    args = parser.parse_args()

    meta = _load_meta(args.model)
    if meta is None or meta.get("scenario_family") is None:
        raise SystemExit(f"{args.model}.meta.json is missing or isn't a feasibility-arena model "
                          f"(no 'scenario_family' key) -- this script only supports cutin/sandwich models.")

    cfg_cls = _CFG_CLASSES[meta["scenario_family"]]
    runtime_cls = _RUNTIME_CLASSES[meta["scenario_family"]]
    cfg = cfg_cls(blocker_side=meta["blocker_side"], mode=meta["mode"], seed=meta["scenario_seed"],
                  goal_distance_m=meta["goal_distance_m"], max_episode_seconds=meta["max_episode_seconds"],
                  min_progress_m=meta["min_progress_m"], **meta["theta"])

    model = PPO.load(args.model, device="cpu")   # PPO.load()'s own device param defaults to "auto",
                                               # which would silently grab CUDA if available -- see
                                               # feasibility_train_one's identical fix.

    # mode="distribution", worker_rank=0 -- exactly what train_and_evaluate's own final evaluation used.
    # Burning through `episode_index` resets first reproduces run_episodes()'s episode-counter bookkeeping
    # exactly, landing on the same realization episode `episode_index` saw during the real evaluation.
    env = EgoTrafficEnv(arena=runtime_cls(cfg), mode="distribution", worker_rank=0)
    for _ in range(args.episode_index):
        env.reset()
    obs, info = env.reset()

    history = {agent.car.car_id: {"x": [], "y": [], "heading": [], "crashed": []} for agent in env.agents}
    ego_history = {"x": [], "y": [], "heading": []}

    def record():
        for agent in env.agents:
            h = history[agent.car.car_id]
            h["x"].append(agent.car.state.x)
            h["y"].append(agent.car.state.y)
            h["heading"].append(agent.car.state.heading)
            h["crashed"].append(agent.crashed)
        ego_history["x"].append(env.ego_car.state.x)
        ego_history["y"].append(env.ego_car.state.y)
        ego_history["heading"].append(env.ego_car.state.heading)

    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, reward, terminated, truncated, info = env.step(action)
        record()

    outcome = (info["failure_reason"] if info["outcome"] == Outcome.SAFETY_FAILURE.value
               else _OUTCOME_TEXT[info["outcome"]])
    n_frames = len(ego_history["x"])
    print(f"{args.model}  episode {args.episode_index}: {n_frames} steps ({n_frames * DT:.2f}s) -- "
          f"outcome: {outcome} -- success: {info['success']}")

    # ── Figure: road + surr cars + ego (same layout as learning.eval) ──────
    road = env.road
    lane_num = len(road.lanes)
    fig, ax = plt.subplots(figsize=(24, 8))

    first, last = road.lanes[0], road.lanes[-1]
    edge_low_x, edge_low_y = road.offset_curve(first.offset - first.width / 2)
    edge_high_x, edge_high_y = road.offset_curve(last.offset + last.width / 2)
    poly_x = np.concatenate([edge_low_x, edge_high_x[::-1]])
    poly_y = np.concatenate([edge_low_y, edge_high_y[::-1]])

    ax.set_facecolor("#e3efe0")
    ax.fill(poly_x, poly_y, color="white", zorder=1)
    ax.plot(edge_low_x, edge_low_y, color="dimgray", linewidth=1.2, zorder=2)
    ax.plot(edge_high_x, edge_high_y, color="dimgray", linewidth=1.2, zorder=2)
    for l in range(lane_num - 1):
        div_x = (road.lanes[l].x + road.lanes[l + 1].x) / 2
        div_y = (road.lanes[l].y + road.lanes[l + 1].y) / 2
        ax.plot(div_x, div_y, linestyle="dashed", color="dimgray", linewidth=1.5, zorder=3)
    for lane in road.lanes:
        ax.plot(lane.x, lane.y, linestyle="dotted", color="goldenrod", linewidth=1.5, zorder=4)

    ax.set_xlim(0, road.s_max)
    ax.set_ylim(-15, 15)
    ax.set_aspect("equal")
    ax.set_title(f"{args.model} ep {args.episode_index} -- green=conservative, blue=moderate, "
                 f"red=aggressive, {_CRASHED_COLOR}=crashed, {EGO_COLOR}=ego -- "
                 f"outcome: {outcome} (success={info['success']})")

    patches = {}
    for agent in env.agents:
        h = history[agent.car.car_id]
        color = _CRASHED_COLOR if h["crashed"][0] else _BEHAVIOUR_COLOR[agent.car.behaviour]
        patch = Polygon(car_corners(h["x"][0], h["y"][0], h["heading"][0]),
                         closed=True, facecolor=color, edgecolor="black", zorder=6)
        ax.add_patch(patch)
        patches[agent.car.car_id] = patch

    ego_patch = Polygon(car_corners(ego_history["x"][0], ego_history["y"][0], ego_history["heading"][0]),
                         closed=True, facecolor=EGO_COLOR, edgecolor="black", zorder=7)
    ax.add_patch(ego_patch)

    def update(i):
        for agent in env.agents:
            h = history[agent.car.car_id]
            patches[agent.car.car_id].set_xy(car_corners(h["x"][i], h["y"][i], h["heading"][i]))
            color = _CRASHED_COLOR if h["crashed"][i] else _BEHAVIOUR_COLOR[agent.car.behaviour]
            patches[agent.car.car_id].set_facecolor(color)
        ego_patch.set_xy(car_corners(ego_history["x"][i], ego_history["y"][i], ego_history["heading"][i]))
        return list(patches.values()) + [ego_patch]

    ani = FuncAnimation(fig, update, frames=n_frames, interval=DT * 1000, blit=False)
    plt.tight_layout()

    if args.save:
        ani.save(args.save, fps=int(round(1 / DT)), dpi=150)
        print(f"saved to {args.save}")
    else:
        # Maximize the window so the (aspect="equal") road -- ~500m long by 30m wide, a very wide/flat
        # box -- gets as much real screen width as possible rather than sitting in a small default
        # window. Backend-specific (Tk/Qt/... each expose this differently), so best-effort only: does
        # nothing (never errors) on a backend that doesn't support it.
        try:
            manager = plt.get_current_fig_manager()
            window = getattr(manager, "window", None)
            if window is not None and hasattr(window, "showMaximized"):        # Qt
                window.showMaximized()
            elif window is not None and hasattr(window, "state"):               # Tk
                window.state("zoomed")
            elif hasattr(manager, "full_screen_toggle"):                        # some GTK/other backends
                getattr(manager, "full_screen_toggle")()
        except Exception:
            pass
        plt.show()


if __name__ == "__main__":
    main()
