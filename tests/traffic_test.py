import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

from initialization.traffic_init import CAR_LENGTH, CAR_WIDTH
from learning.scenario import ScenarioConfig, CriticalActorConfig
from learning.env import EgoTrafficEnv, LANE_NUM, DT, MAX_STEPS, ACCEL_MIN, ACCEL_MAX

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

# ── Options ───────────────────────────────────────────────────────────────
mode = "animation"   # "plot" (static figure) or "animation" (traffic driving live)
seed = 3   # scenario seed -- None draws a fresh scenario every run; set an
           # int (e.g. 0) to reproduce the same one. This is theta.seed
           # below (episode_index/worker_rank are fixed at 0 here -- this
           # script only ever renders one deterministic realization, same
           # idea as EgoTrafficEnv's "fixed" mode).

model_path = "learning/ppo_ego"   # trained PPO policy (see learning.train)
                                   # that actually drives ego through the
                                   # environment below. Set to None (or
                                   # leave a missing path) to fall back to
                                   # a simple hold-speed, straight-line
                                   # ego controller instead, so this script
                                   # still runs before anything's trained.

# ── Scenario theta -- the hand knob: compact global + critical-actor
# parameters (see learning.scenario.ScenarioConfig for full field docs).
# Edit these directly and rerun to see the effect immediately in the plot/
# animation below -- this is the same theta learning.train/eval_batch
# consume, just visualized instead of trained against. ──────────────────────
theta = ScenarioConfig(
    seed                   = seed if seed is not None else int(np.random.SeedSequence().generate_state(1)[0]),
    road_mu                = 1.0,     # [-] road friction
    road_kappa_max         = 0.005,   # [1/m] peak curve curvature
    background_mean_speed  = 20.0,    # [m/s]
    background_speed_std   = 2.5,     # [m/s]
    aggressive_fraction    = 0.2,     # P(background car is aggressive); rest split conservative/moderate
    lane_change_tendency   = 1.0,     # >1 => more background lane changes, <1 => fewer
    traffic_spread         = 100.0,    # [m] background cars placed within +/- this of ego_s
    # No ego_initial_speed knob: ego's speed is the road speed -- drawn
    # from Normal(background_mean_speed, background_speed_std) same as
    # every background car -- front/rear/blocker.relative_speed is
    # relative to that draw, printed below once theta is realized.
    front = CriticalActorConfig(role = "front", behavior = "moderate",
                                 relative_speed = 0.0, gap = 30.0),
    rear = CriticalActorConfig(role = "rear", behavior = "moderate",
                                relative_speed = 0.0, gap = 20.0),
    blocker = CriticalActorConfig(role = "blocker", behavior = "moderate",
                                   relative_speed = 0.0, lane_offset = 1, relative_s = 0.0),
)

car_width = CAR_WIDTH   # [m] (CAR_LENGTH/CAR_WIDTH come from initialization.traffic_init,
                        # shared with the gap calculations and collision test so plotted
                        # bodies match spacing)

EGO_COLOR = "black"
_BEHAVIOUR_COLOR = {1: "seagreen", 2: "steelblue", 3: "firebrick"}   # conservative/moderate/aggressive
_ROLE_COLOR = {"front": "orange", "rear": "purple", "blocker": "cyan"}   # theta's 3 critical actors
_CRASHED_COLOR = "dimgray"


def _color_for(agent, crashed: bool, role_by_car_id: dict) -> str:
    if crashed:
        return _CRASHED_COLOR
    role = role_by_car_id.get(agent.car.car_id)
    return _ROLE_COLOR.get(role, _BEHAVIOUR_COLOR[agent.car.behaviour])


def car_corners(x: float, y: float, heading: float, length: float = CAR_LENGTH, width: float = car_width):
    """4 corners of a car's body rectangle, centred on (x, y), long axis
    along `heading` (same (cos, sin) tangent convention as Road.cartesean_calc)."""
    hl, hw = length / 2.0, width / 2.0
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    return [
        (x + hl * cos_h - hw * sin_h, y + hl * sin_h + hw * cos_h),
        (x + hl * cos_h + hw * sin_h, y + hl * sin_h - hw * cos_h),
        (x - hl * cos_h + hw * sin_h, y - hl * sin_h - hw * cos_h),
        (x - hl * cos_h - hw * sin_h, y - hl * sin_h + hw * cos_h),
    ]


# ── Build the scenario via the real environment (not a manual sim loop) --
# theta is realized exactly the way learning.train/learning.eval_batch
# would realize it, and ego is stepped through EgoTrafficEnv.step() just
# like a trained policy would drive it, rather than sitting still. ─────────
env = EgoTrafficEnv(scenario_config = theta, mode = "fixed", worker_rank = 0)
obs, info = env.reset()
realized = env.get_realized_scenario()

# env.agents[0]/[1]/[2] are always front/rear/blocker (see generate_scenario)
# -- used to color theta's 3 critical actors distinctly from background traffic.
role_by_car_id = {env.agents[0].car.car_id: "front", env.agents[1].car.car_id: "rear",
                   env.agents[2].car.car_id: "blocker"}

print(f"theta realized: ego_v0={realized['ego']['v_x']:.1f} m/s | "
      f"front: gap={theta.front.gap}m, v={theta.front.relative_speed:+.1f} rel, {theta.front.behavior} | "
      f"rear: gap={theta.rear.gap}m, v={theta.rear.relative_speed:+.1f} rel, {theta.rear.behavior} | "
      f"blocker: lane{theta.blocker.lane_offset:+d}, ds={theta.blocker.relative_s:+.1f}m, "
      f"v={theta.blocker.relative_speed:+.1f} rel, {theta.blocker.behavior}")

model = None
if model_path is not None:
    if PPO is None:
        print("stable_baselines3 isn't installed -- falling back to a straight, constant-speed ego controller.")
    else:
        try:
            model = PPO.load(model_path)
            print(f"ego driven by trained policy: {model_path}")
        except FileNotFoundError:
            print(f"No trained model found at {model_path!r} -- falling back to a straight, constant-speed "
                  f"ego controller. Train one with `python -m learning.train --scenario-config ...`.")

# A constant action that unscales to (accel=0, delta=0) -- see
# learning.env._unscale_action's linear map -- i.e. "hold current speed,
# drive straight", used only when no trained policy is available.
_CRUISE_ACTION = np.array([2.0 * (0.0 - ACCEL_MIN) / (ACCEL_MAX - ACCEL_MIN) - 1.0, 0.0], dtype = np.float32)


def _ego_action(obs: np.ndarray) -> np.ndarray:
    if model is not None:
        action, _ = model.predict(obs, deterministic = True)
        return action
    return _CRUISE_ACTION


# ── Simulate: step ego through the environment (drives itself each frame,
# same as learning.eval.py) while surr traffic runs its usual IDM/MOBIL. ──
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


# No frame 0 here: right after reset(), surr cars' x/y/heading haven't been
# computed yet (that only happens inside step_surr_agents, called from
# env.step()) -- so the first recorded frame is post-first-step.
terminated = truncated = False
outcome = "reached the time horizon safely"
step_i = 0
while not (terminated or truncated) and step_i < MAX_STEPS:
    obs, reward, terminated, truncated, info = env.step(_ego_action(obs))
    record()
    step_i += 1
    if terminated:
        outcome = "rolled over" if info["rolled_over"] else "went off-road" if info["off_road"] else "collided"

n_frames = len(ego_history["x"])
print(f"Episode ended after {n_frames} steps ({n_frames * DT:.2f}s) -- outcome: {outcome}")

# ── Figure: road + all cars ─────────────────────────────────────────────────
road = env.road
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
for l in range(LANE_NUM - 1):
    div_x = (road.lanes[l].x + road.lanes[l + 1].x) / 2  # type: ignore
    div_y = (road.lanes[l].y + road.lanes[l + 1].y) / 2  # type: ignore
    ax.plot(div_x, div_y, linestyle = "dashed", color = "dimgray", linewidth = 1.5, zorder = 3)
for lane in road.lanes:
    ax.plot(lane.x, lane.y, linestyle = "dotted", color = "goldenrod", linewidth = 1.5, zorder = 4)  # type: ignore

ax.set_xlim(0, road.s_max)
ax.set_ylim(-15, 15)
ax.set_aspect('equal')
ego_driver = f"policy ({model_path})" if model is not None else "cruise (no trained policy)"
ax.set_title(f"Traffic ({len(env.agents)} cars + ego, theta.seed={theta.seed}, ego={ego_driver}) -- "
             f"green=conservative, blue=moderate, red=aggressive, "
             f"orange=front, purple=rear, cyan=blocker, {_CRASHED_COLOR}=crashed, {EGO_COLOR}=ego -- "
             f"outcome: {outcome}")

if mode == "plot":
    for agent in env.agents:
        h = history[agent.car.car_id]
        color = _color_for(agent, agent.crashed, role_by_car_id)
        ax.plot(h["x"], h["y"], color = color, linewidth = 1.0, alpha = 0.5, zorder = 5)
        corners = car_corners(h["x"][-1], h["y"][-1], h["heading"][-1])
        ax.add_patch(Polygon(corners, closed = True, facecolor = color,
                              edgecolor = "black", alpha = 0.85, zorder = 6))
    ax.plot(ego_history["x"], ego_history["y"], color = EGO_COLOR, linewidth = 1.0, alpha = 0.5, zorder = 5)
    ego_corners = car_corners(ego_history["x"][-1], ego_history["y"][-1], ego_history["heading"][-1])
    ax.add_patch(Polygon(ego_corners, closed = True, facecolor = EGO_COLOR,
                          edgecolor = "black", alpha = 0.85, zorder = 7))
    plt.tight_layout()
    plt.show()

elif mode == "animation":
    patches = {}
    for agent in env.agents:
        h = history[agent.car.car_id]
        color = _color_for(agent, h["crashed"][0], role_by_car_id)
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
            patches[agent.car.car_id].set_facecolor(_color_for(agent, h["crashed"][i], role_by_car_id))
        ego_patch.set_xy(car_corners(ego_history["x"][i], ego_history["y"][i], ego_history["heading"][i]))
        return list(patches.values()) + [ego_patch]

    ani = FuncAnimation(fig, update, frames = n_frames, interval = DT * 1000, blit = False)
    plt.tight_layout()
    plt.show()

else:
    raise ValueError(f"mode must be 'plot' or 'animation', got {mode!r}")
