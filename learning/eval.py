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

from learning.env import EgoTrafficEnv, DT, Outcome
from learning.eval_batch import _load_meta
from learning.scenario import ScenarioConfig
from learning.feasibility_cutin import CutinConfig, CutinRuntime
from learning.feasibility_sandwich import SandwichConfig, SandwichRuntime
from initialization.traffic_init import CAR_LENGTH, CAR_WIDTH

_ARENA_CONFIG_CLASSES = {"cutin": CutinConfig, "sandwich": SandwichConfig}
_ARENA_RUNTIME_CLASSES = {"cutin": CutinRuntime, "sandwich": SandwichRuntime}


def _env_from_meta(meta: dict | None) -> EgoTrafficEnv:
    """Reconstruct the exact scenario a model was trained on from its
    .meta.json -- covers both learning.train's ScenarioConfig models and
    learning.feasibility_train_one's arena (cutin/sandwich) models. Falls
    back to a fresh fully-random EgoTrafficEnv() only when no meta.json (or
    an unrecognized one) is found -- see _load_meta's own warning for that
    case. EgoTrafficEnv(mode="fixed") is used for both known cases so the
    animation always replays one reproducible realization, regardless of
    that scenario's own recorded training/realization mode."""
    if meta is not None and meta.get("scenario_family") is not None:
        cfg_cls = _ARENA_CONFIG_CLASSES[meta["scenario_family"]]
        cfg = cfg_cls(blocker_side=meta["blocker_side"], mode=meta["mode"], seed=meta["scenario_seed"],
                      goal_distance_m=meta["goal_distance_m"], max_episode_seconds=meta["max_episode_seconds"],
                      min_progress_m=meta["min_progress_m"], **meta["theta"])
        runtime = _ARENA_RUNTIME_CLASSES[meta["scenario_family"]](cfg)
        return EgoTrafficEnv(arena=runtime, mode="fixed", worker_rank=0)
    if meta is not None and meta.get("scenario_config") is not None:
        # Unchanged from before arena support existed: respects whatever scenario_mode this
        # ScenarioConfig model actually trained under (fixed or distribution), not hard-coded.
        scenario_config = ScenarioConfig.from_dict(meta["scenario_config"])
        return EgoTrafficEnv(scenario_config=scenario_config, mode=meta["scenario_mode"], worker_rank=0)
    return EgoTrafficEnv()

_OUTCOME_TEXT = {
    Outcome.FINISHED.value: "reached the scenario's goal",
    Outcome.SAFETY_FAILURE.value: "had a safety failure",   # refined to collision/rollover/off_road below
    Outcome.TIMEOUT.value: "reached the time horizon without resolving the scenario",
}

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

    model = PPO.load(args.model, device="cpu")   # see feasibility_train_one's own fix -- PPO.load()
                                                   # defaults device to "auto", which would silently
                                                   # grab CUDA if available regardless of this project's
                                                   # CPU-only-training rationale.

    # Reconstruct the exact scenario this model was trained on (see _env_from_meta -- covers both
    # learning.train's ScenarioConfig models and learning.feasibility_train_one's arena models).
    # EgoTrafficEnv() alone would silently visualize a DIFFERENT (fully-random-traffic) scenario than
    # whatever this policy actually learned to handle.
    env = _env_from_meta(_load_meta(args.model))
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
    info = {}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic = not args.stochastic)
        obs, reward, terminated, truncated, info = env.step(action)
        record()

    # Canonical outcome/success (see learning.env's module docstring) -- physical terminal conditions,
    # not re-derived from the legacy collided/rolled_over/off_road/finished keys.
    outcome = (info["failure_reason"] if info["outcome"] == Outcome.SAFETY_FAILURE.value
               else _OUTCOME_TEXT[info["outcome"]])
    success = info["success"]

    n_frames = len(ego_history["x"])
    print(f"Episode ended after {n_frames} steps ({n_frames * DT:.2f}s) -- "
          f"outcome: {outcome} -- success: {success}")

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
                 f"red=aggressive, {_CRASHED_COLOR}=crashed, {EGO_COLOR}=ego -- "
                 f"outcome: {outcome} (success={success})")

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
