"""
Visualize a trained ego policy driving inside the same scenario
tests/traffic_test.py demos -- same road/curve, same surrounding IDM/MOBIL
traffic, animated the same way, but ego is now driven by a loaded PPO
policy each step (see env.py) instead of being a stationary placeholder.

Usage:
    python -m learning.eval [--model PATH] [--seed N] [--stochastic]
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

from stable_baselines3 import PPO

from learning.env import EgoTrafficEnv, DT
from initialization.traffic_init import CAR_LENGTH, CAR_WIDTH

EGO_COLOR = "black"
_BEHAVIOUR_COLOR = {1: "seagreen", 2: "steelblue", 3: "firebrick"}   # conservative/moderate/aggressive
_CRASHED_COLOR = "dimgray"


def car_corners(x: float, y: float, heading: float, length: float = CAR_LENGTH, width: float = CAR_WIDTH):
    """Same body-rectangle construction as tests/traffic_test.py's own
    car_corners -- kept as a local copy rather than a shared import since
    both scripts treat it as a small, self-contained plotting helper."""
    hl, hw = length / 2.0, width / 2.0
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    return [
        (x + hl * cos_h - hw * sin_h, y + hl * sin_h + hw * cos_h),
        (x + hl * cos_h + hw * sin_h, y + hl * sin_h - hw * cos_h),
        (x - hl * cos_h + hw * sin_h, y - hl * sin_h - hw * cos_h),
        (x - hl * cos_h - hw * sin_h, y - hl * sin_h + hw * cos_h),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default = "learning/ppo_ego")
    parser.add_argument("--seed", type = int, default = None)
    parser.add_argument("--stochastic", action = "store_true",
                         help = "sample actions from the policy instead of using its deterministic mean")
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = EgoTrafficEnv()
    obs, info = env.reset(seed = args.seed)

    # Per-agent (and ego) pose history, recorded straight off env internals
    # after every step -- same fields tests/traffic_test.py's `history`
    # dict tracks, so the plotting code below matches it closely.
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

    # No frame 0 here: right after reset(), surr cars' x/y/heading haven't
    # been computed yet (that only happens inside step_surr_agents, called
    # from env.step()) -- so the first recorded frame is post-first-step.
    terminated = truncated = False
    outcome = "reached the time horizon safely"
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic = not args.stochastic)
        obs, reward, terminated, truncated, info = env.step(action)
        record()
        if terminated:
            outcome = "rolled over" if info["rolled_over"] else "collided"

    n_frames = len(ego_history["x"])
    print(f"Episode ended after {n_frames} steps ({n_frames * DT:.2f}s) -- outcome: {outcome}")

    # ── Figure: road + surr cars + ego ──────────────────────────────────────
    road = env.road
    lane_num = len(road.lanes)
    fig, ax = plt.subplots(figsize = (18, 4))

    first, last = road.lanes[0], road.lanes[-1]
    # road.offset_curve, not a naive per-point (sin, cos) shift off first/last's
    # own centreline -- that drifts and permanently narrows the plotted road
    # after a curve (see Road.offset_curve's docstring).
    edge_low_x,  edge_low_y  = road.offset_curve(first.offset - first.width / 2)   # type: ignore
    edge_high_x, edge_high_y = road.offset_curve(last.offset  + last.width  / 2)   # type: ignore
    poly_x = np.concatenate([edge_low_x, edge_high_x[::-1]])
    poly_y = np.concatenate([edge_low_y, edge_high_y[::-1]])

    ax.set_facecolor("#e3efe0")
    ax.fill(poly_x, poly_y, color = "white", zorder = 1)
    ax.plot(edge_low_x,  edge_low_y,  color = "dimgray", linewidth = 1.2, zorder = 2)
    ax.plot(edge_high_x, edge_high_y, color = "dimgray", linewidth = 1.2, zorder = 2)
    for l in range(lane_num - 1):
        div_x = (road.lanes[l].x + road.lanes[l + 1].x) / 2   # type: ignore
        div_y = (road.lanes[l].y + road.lanes[l + 1].y) / 2   # type: ignore
        ax.plot(div_x, div_y, linestyle = "dashed", color = "dimgray", linewidth = 1.5, zorder = 3)
    for lane in road.lanes:
        ax.plot(lane.x, lane.y, linestyle = "dotted", color = "goldenrod", linewidth = 1.5, zorder = 4)   # type: ignore

    ax.set_xlim(0, road.s_max)
    ax.set_ylim(-15, 15)
    ax.set_aspect("equal")
    ax.set_title(f"Trained ego ({args.model}) -- green=conservative, blue=moderate, "
                 f"red=aggressive, {_CRASHED_COLOR}=crashed, {EGO_COLOR}=ego -- outcome: {outcome}")

    patches = {}
    for agent in env.agents:
        h = history[agent.car.car_id]
        color = _CRASHED_COLOR if h["crashed"][0] else _BEHAVIOUR_COLOR[agent.car.behaviour]
        patch = Polygon(car_corners(h["x"][0], h["y"][0], h["heading"][0]),
                         closed = True, facecolor = color, edgecolor = "black", zorder = 6)
        ax.add_patch(patch)
        patches[agent.car.car_id] = patch

    ego_patch = Polygon(car_corners(ego_history["x"][0], ego_history["y"][0], ego_history["heading"][0]),
                         closed = True, facecolor = EGO_COLOR, edgecolor = "black", zorder = 7)
    ax.add_patch(ego_patch)

    def update(i):
        for agent in env.agents:
            h = history[agent.car.car_id]
            patches[agent.car.car_id].set_xy(car_corners(h["x"][i], h["y"][i], h["heading"][i]))
            color = _CRASHED_COLOR if h["crashed"][i] else _BEHAVIOUR_COLOR[agent.car.behaviour]
            patches[agent.car.car_id].set_facecolor(color)
        ego_patch.set_xy(car_corners(ego_history["x"][i], ego_history["y"][i], ego_history["heading"][i]))
        return list(patches.values()) + [ego_patch]

    ani = FuncAnimation(fig, update, frames = n_frames, interval = DT * 1000, blit = False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
