import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

from model.road.road import Road
from controllers.mobil import MobilParams
from initialization.traffic_init import CAR_LENGTH, CAR_WIDTH
from model.traffic_step import step_surr_agents
from model.car.car import Car, CarState
from model.car.config import VehicleParameters
from learning.scenario import ScenarioConfig, CriticalActorConfig, generate_scenario, derive_seed

# ── Options ───────────────────────────────────────────────────────────────
mode = "animation"   # "plot" (static figure) or "animation" (traffic driving live)
seed = 3   # scenario seed -- None draws a fresh scenario every run; set an
           # int (e.g. 0) to reproduce the same one. This is theta.seed
           # below (episode_index/worker_rank are fixed at 0 here -- this
           # script only ever renders one deterministic realization, same
           # idea as EgoTrafficEnv's "fixed" mode).

# ── Scenario theta -- the hand knob: compact global + critical-actor
# parameters (see learning.scenario.ScenarioConfig for full field docs).
# Edit these directly and rerun to see the effect immediately in the plot/
# animation below -- this is the same theta learning.train/eval_batch
# consume, just visualized instead of trained against. ──────────────────────
ego_s = 50.0   # [m] where ego (and theta) is anchored -- also this script's
               # only non-theta scenario knob, since it's about where in
               # the road geometry to look, not the traffic itself.
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

lane_num = 3
dt       = 0.05   # [s]
duration = 20.0   # [s]

mobil_params = MobilParams(threshold = MobilParams().threshold / theta.lane_change_tendency)
MAX_BRAKING = 8.0   # [m/s^2] physical actuation limit clamped onto IDM's raw output
LANE_CHANGE_COOLDOWN = 1.0   # [s] a car may not start another lane change this
                             # soon after its last one committed -- damps MOBIL
                             # lane-hopping back and forth right after a merge.
CRASH_BLEED_K = 1.0   # [-] post-crash deceleration = k * road.mu * g -- see model.collision

car_width = CAR_WIDTH   # [m] (CAR_LENGTH/CAR_WIDTH come from initialization.traffic_init,
                        # shared with the gap calculations and collision test so plotted
                        # bodies match spacing)

_BEHAVIOUR_COLOR = {1: "seagreen", 2: "steelblue", 3: "firebrick"}   # conservative/moderate/aggressive
_ROLE_COLOR = {"front": "orange", "rear": "purple", "blocker": "cyan"}   # theta's 3 critical actors
_CRASHED_COLOR = "dimgray"


def _color_for(agent, crashed: bool, role_by_car_id: dict) -> str:
    if crashed:
        return _CRASHED_COLOR
    role = role_by_car_id.get(agent.car.car_id)
    return _ROLE_COLOR.get(role, _BEHAVIOUR_COLOR[agent.car.behaviour])


road = Road(s_max = 500, kappa_max = theta.road_kappa_max, L_clothoid = 60,
            mu = theta.road_mu, lane_num = lane_num)

ego_lane = lane_num // 2
scenario_seed = derive_seed(theta.seed, worker_rank = 0, episode_index = 0)
rng = np.random.default_rng(scenario_seed)
agents, ego_v0, realized = generate_scenario(theta, road, ego_s = ego_s, ego_lane = ego_lane, rng = rng,
                                              worker_rank = 0, episode_index = 0, scenario_seed = scenario_seed)
# agents[0]/[1]/[2] are always front/rear/blocker (see generate_scenario) --
# used to color theta's 3 critical actors distinctly from background traffic.
role_by_car_id = {agents[0].car.car_id: "front", agents[1].car.car_id: "rear", agents[2].car.car_id: "blocker"}

print(f"theta realized: ego_v0={ego_v0:.1f} m/s | "
      f"front: gap={theta.front.gap}m, v={theta.front.relative_speed:+.1f} rel, {theta.front.behavior} | "
      f"rear: gap={theta.rear.gap}m, v={theta.rear.relative_speed:+.1f} rel, {theta.rear.behavior} | "
      f"blocker: lane{theta.blocker.lane_offset:+d}, ds={theta.blocker.relative_s:+.1f}m, "
      f"v={theta.blocker.relative_speed:+.1f} rel, {theta.blocker.behavior}")

# ── Ego -- stationary placeholder, no controller yet: it never steps, so
# its (x, y, heading) are computed once here rather than every frame. Not
# a TrafficAgent -- it doesn't run IDM/MOBIL and isn't in `agents`, so it's
# not yet visible to surr cars' leader lookups or collision detection.
EGO_COLOR = "black"
ego_car = Car(state = CarState(s = ego_s, e_y = 0.0, e_psi = 0.0, v_x = ego_v0, lane = ego_lane),
              vehicle_params = VehicleParameters())
_ego_lane_obj = road.lanes[ego_car.state.lane]
_ego_backbone_e_y = -_ego_lane_obj.offset + ego_car.state.e_y   # type: ignore
ego_x, ego_y, ego_heading = road.frenet_to_global(ego_car.state.s, _ego_backbone_e_y, ego_car.state.e_psi)
ego_car.state.x, ego_car.state.y, ego_car.state.heading = ego_x, ego_y, ego_heading


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


# ── Simulate ──────────────────────────────────────────────────────────────
# EGO RULE (not yet reachable here): an ego/surr overlap must end the episode
# via model.collision.ego_overlaps_any, never route through
# resolve_surr_collisions -- see that function's docstring. Ego above is a
# stationary placeholder with no controller, so there's nothing driving it
# into traffic yet; wire this in once it has one (see learning/env.py).
history = {agent.car.car_id: {"t": [], "x": [], "y": [], "heading": [], "crashed": []} for agent in agents}

n_steps = int(duration / dt)
for step in range(n_steps):
    t = step * dt

    step_surr_agents(agents, road, t, dt, mobil_params = mobil_params, lane_num = lane_num,
                      max_braking = MAX_BRAKING, lane_change_cooldown = LANE_CHANGE_COOLDOWN,
                      crash_bleed_k = CRASH_BLEED_K)

    for agent in agents:
        h = history[agent.car.car_id]
        h["t"].append(t)
        h["x"].append(agent.car.state.x)
        h["y"].append(agent.car.state.y)
        h["heading"].append(agent.car.state.heading)
        h["crashed"].append(agent.crashed)

# ── Figure: road + all cars ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize = (18, 4))

first, last = road.lanes[0], road.lanes[-1]
edge_low_x  = first.x - (first.width / 2) * np.sin(first.heading)   # type: ignore
edge_low_y  = first.y - (first.width / 2) * np.cos(first.heading)   # type: ignore
edge_high_x = last.x  + (last.width  / 2) * np.sin(last.heading)    # type: ignore
edge_high_y = last.y  + (last.width  / 2) * np.cos(last.heading)    # type: ignore
poly_x = np.concatenate([edge_low_x, edge_high_x[::-1]])
poly_y = np.concatenate([edge_low_y, edge_high_y[::-1]])

ax.set_facecolor("#e3efe0")
ax.fill(poly_x, poly_y, color = "white", zorder = 1)
ax.plot(edge_low_x,  edge_low_y,  color = "dimgray", linewidth = 1.2, zorder = 2)
ax.plot(edge_high_x, edge_high_y, color = "dimgray", linewidth = 1.2, zorder = 2)
for l in range(lane_num - 1):
    div_x = (road.lanes[l].x + road.lanes[l + 1].x) / 2  # type: ignore
    div_y = (road.lanes[l].y + road.lanes[l + 1].y) / 2  # type: ignore
    ax.plot(div_x, div_y, linestyle = "dashed", color = "dimgray", linewidth = 1.5, zorder = 3)
for lane in road.lanes:
    ax.plot(lane.x, lane.y, linestyle = "dotted", color = "goldenrod", linewidth = 1.5, zorder = 4)  # type: ignore

s_vals = [agent.car.state.s for agent in agents]
ax.set_xlim(0, 500)
ax.set_ylim(-15, 15)
ax.set_aspect('equal')
ax.set_title(f"Traffic ({len(agents)} cars + stationary ego, theta.seed={theta.seed}) -- "
             f"green=conservative, blue=moderate, red=aggressive, "
             f"orange=front, purple=rear, cyan=blocker, {_CRASHED_COLOR}=crashed, {EGO_COLOR}=ego")

ego_patch = Polygon(car_corners(ego_x, ego_y, ego_heading), closed = True,
                     facecolor = EGO_COLOR, edgecolor = "black", zorder = 7)
ax.add_patch(ego_patch)

if mode == "plot":
    for agent in agents:
        h = history[agent.car.car_id]
        color = _color_for(agent, agent.crashed, role_by_car_id)
        ax.plot(h["x"], h["y"], color = color, linewidth = 1.0, alpha = 0.5, zorder = 5)
        corners = car_corners(h["x"][-1], h["y"][-1], h["heading"][-1])
        ax.add_patch(Polygon(corners, closed = True, facecolor = color,
                              edgecolor = "black", alpha = 0.85, zorder = 6))
    plt.tight_layout()
    plt.show()

elif mode == "animation":
    patches = {}
    for agent in agents:
        h = history[agent.car.car_id]
        color = _color_for(agent, h["crashed"][0], role_by_car_id)
        patch = Polygon(car_corners(h["x"][0], h["y"][0], h["heading"][0]),
                         closed = True, facecolor = color, edgecolor = "black", zorder = 6)
        ax.add_patch(patch)
        patches[agent.car.car_id] = patch

    def update(i):
        for agent in agents:
            h = history[agent.car.car_id]
            patches[agent.car.car_id].set_xy(car_corners(h["x"][i], h["y"][i], h["heading"][i]))
            patches[agent.car.car_id].set_facecolor(_color_for(agent, h["crashed"][i], role_by_car_id))
        return list(patches.values())

    ani = FuncAnimation(fig, update, frames = n_steps, interval = dt * 1000, blit = False)
    plt.tight_layout()
    plt.show()

else:
    raise ValueError(f"mode must be 'plot' or 'animation', got {mode!r}")
