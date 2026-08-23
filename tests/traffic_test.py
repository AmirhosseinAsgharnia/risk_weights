import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

from model.road.road import Road
from controllers.mobil import MobilParams
from initialization.traffic_init import generate_traffic, CAR_LENGTH, CAR_WIDTH
from model.traffic_step import step_surr_agents
from model.car.car import Car, CarState
from model.car.config import VehicleParameters

# ── Options ───────────────────────────────────────────────────────────────
mode = "animation"   # "plot" (static figure) or "animation" (traffic driving live)
seed = 2   # RNG seed for traffic generation -- None draws a fresh scenario
              # every run; set an int (e.g. 0) to reproduce the same one.

# ── Scenario ──────────────────────────────────────────────────────────────
N_c   = 15     # number of surrounding cars
ego_s = 50.0   # [m] reference point traffic is generated around -- there is
               # no ego car yet, this is just where a future one would sit.

lane_num = 3
dt       = 0.05   # [s]
duration = 20.0   # [s]

mobil_params = MobilParams()   # standard defaults: politeness=0.2, threshold=0.2, b_safe=4.0
MAX_BRAKING = 8.0   # [m/s^2] physical actuation limit clamped onto IDM's raw output
LANE_CHANGE_COOLDOWN = 1.0   # [s] a car may not start another lane change this
                             # soon after its last one committed -- damps MOBIL
                             # lane-hopping back and forth right after a merge.
CRASH_BLEED_K = 1.0   # [-] post-crash deceleration = k * road.mu * g -- see model.collision

car_width = CAR_WIDTH   # [m] (CAR_LENGTH/CAR_WIDTH come from initialization.traffic_init,
                        # shared with the gap calculations and collision test so plotted
                        # bodies match spacing)

_BEHAVIOUR_COLOR = {1: "seagreen", 2: "steelblue", 3: "firebrick"}   # conservative/moderate/aggressive
_CRASHED_COLOR = "dimgray"


def _agent_color(agent) -> str:
    return _CRASHED_COLOR if agent.crashed else _BEHAVIOUR_COLOR[agent.car.behaviour]

road = Road(s_max = 500, kappa_max = 0.005, L_clothoid = 60,
            mu = 1.0, lane_num = lane_num)

rng = np.random.default_rng(seed)
# ego_v0 is drawn by generate_traffic exactly like a surr car's v0 (ego is
# placed/spaced like a vehicle there too) -- used below for ego_car's
# initial v_x, even though this script's ego still has no controller (see
# next comment) and so never actually moves regardless of its v_x.
agents, ego_v0 = generate_traffic(N_c, ego_s = ego_s, lanes = tuple(range(lane_num)), rng = rng)

# ── Ego -- stationary placeholder, no controller yet: it never steps, so
# its (x, y, heading) are computed once here rather than every frame. Not
# a TrafficAgent -- it doesn't run IDM/MOBIL and isn't in `agents`, so it's
# not yet visible to surr cars' leader lookups or collision detection.
EGO_COLOR = "black"
ego_lane = lane_num // 2
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
ax.set_title(f"Traffic ({N_c} cars + stationary ego) -- green=conservative, blue=moderate, "
             f"red=aggressive, {_CRASHED_COLOR}=crashed, {EGO_COLOR}=ego")

ego_patch = Polygon(car_corners(ego_x, ego_y, ego_heading), closed = True,
                     facecolor = EGO_COLOR, edgecolor = "black", zorder = 7)
ax.add_patch(ego_patch)

if mode == "plot":
    for agent in agents:
        h = history[agent.car.car_id]
        color = _agent_color(agent)
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
        color = _CRASHED_COLOR if h["crashed"][0] else _BEHAVIOUR_COLOR[agent.car.behaviour]
        patch = Polygon(car_corners(h["x"][0], h["y"][0], h["heading"][0]),
                         closed = True, facecolor = color, edgecolor = "black", zorder = 6)
        ax.add_patch(patch)
        patches[agent.car.car_id] = patch

    def update(i):
        for agent in agents:
            h = history[agent.car.car_id]
            patches[agent.car.car_id].set_xy(car_corners(h["x"][i], h["y"][i], h["heading"][i]))
            crashed_color = _CRASHED_COLOR if h["crashed"][i] else _BEHAVIOUR_COLOR[agent.car.behaviour]
            patches[agent.car.car_id].set_facecolor(crashed_color)
        return list(patches.values())

    ani = FuncAnimation(fig, update, frames = n_steps, interval = dt * 1000, blit = False)
    plt.tight_layout()
    plt.show()

else:
    raise ValueError(f"mode must be 'plot' or 'animation', got {mode!r}")
