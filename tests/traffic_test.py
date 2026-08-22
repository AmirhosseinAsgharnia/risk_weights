import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

from model.road.road import Road
from controllers.idm import idm_accel, IDM_PRESETS
from controllers.far_near import (
    far_near_steering, far_near_lookahead_offset, clip_steering_rate,
    far_near_curvature_feedforward, FAR_NEAR_PRESETS,
)
from controllers.mobil import mobil_decision, MobilParams
from initialization.traffic_init import generate_traffic, CAR_LENGTH, CAR_WIDTH
from model.collision import resolve_surr_collisions, bleed_crashed
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
agents = generate_traffic(N_c, ego_s = ego_s, lanes = tuple(range(lane_num)), rng = rng)

# ── Ego -- stationary placeholder, no controller yet: it never steps, so
# its (x, y, heading) are computed once here rather than every frame. Not
# a TrafficAgent -- it doesn't run IDM/MOBIL and isn't in `agents`, so it's
# not yet visible to surr cars' leader lookups or collision detection.
EGO_COLOR = "black"
ego_lane = lane_num // 2
ego_car = Car(state = CarState(s = ego_s, e_y = 0.0, e_psi = 0.0, v_x = 0.0, lane = ego_lane),
              vehicle_params = VehicleParameters())
_ego_lane_obj = road.lanes[ego_car.state.lane]
_ego_backbone_e_y = -_ego_lane_obj.offset + ego_car.state.e_y   # type: ignore
ego_x, ego_y, ego_heading = road.frenet_to_global(ego_car.state.s, _ego_backbone_e_y, ego_car.state.e_psi)
ego_car.state.x, ego_car.state.y = ego_x, ego_y


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


# ── Traffic-snapshot helpers (nearest leader/follower in a lane, by current
# -- not target -- lane: a car mid-lane-change is still treated as part of
# its old lane's stream for other cars' purposes until it commits) ─────────

def find_leader(agents, lane, s, exclude_id):
    # This module finds the leader.
    ahead = [a for a in agents if a.car.state.lane == lane and a.car.car_id != exclude_id and a.car.state.s > s]
    return min(ahead, key = lambda a: a.car.state.s) if ahead else None


def find_follower(agents, lane, s, exclude_id):
    behind = [a for a in agents if a.car.state.lane == lane and a.car.car_id != exclude_id and a.car.state.s < s]
    return max(behind, key = lambda a: a.car.state.s) if behind else None


def _idm_accel_of(follower_state, v0, idm_params, leader_agent):
    """leader_agent may be any object with .car.state.s/.v_x (a TrafficAgent
    -- including, for MOBIL's "what if ego became your leader" case, ego's
    own agent), or None for free-road."""
    if leader_agent is None:
        gap, dv = float("inf"), 0.0
    else:
        gap = leader_agent.car.state.s - follower_state.s - CAR_LENGTH
        dv = follower_state.v_x - leader_agent.car.state.v_x
    return idm_accel(follower_state.v_x, gap, dv, v0, idm_params.a_max, idm_params.b,
                      idm_params.s0, idm_params.T, idm_params.delta)


def evaluate_mobil(agents, agent, candidate_lane):
    """MOBIL incentive/safety for `agent` moving from its current lane to
    candidate_lane, given the current traffic snapshot."""
    car = agent.car
    ego_lane = car.state.lane
    ego_idm = IDM_PRESETS[car.behaviour]

    old_leader   = find_leader(agents, ego_lane, car.state.s, car.car_id)
    old_follower = find_follower(agents, ego_lane, car.state.s, car.car_id)
    new_leader   = find_leader(agents, candidate_lane, car.state.s, car.car_id)
    new_follower = find_follower(agents, candidate_lane, car.state.s, car.car_id)

    a_ego_before = _idm_accel_of(car.state, agent.v0, ego_idm, old_leader)
    a_ego_after  = _idm_accel_of(car.state, agent.v0, ego_idm, new_leader)

    if old_follower is None:
        a_old_follower_before = a_old_follower_after = 0.0
    else:
        of_idm = IDM_PRESETS[old_follower.car.behaviour]
        a_old_follower_before = _idm_accel_of(old_follower.car.state, old_follower.v0, of_idm, agent)
        a_old_follower_after  = _idm_accel_of(old_follower.car.state, old_follower.v0, of_idm, old_leader)

    if new_follower is None:
        a_new_follower_before = a_new_follower_after = 0.0
    else:
        nf_idm = IDM_PRESETS[new_follower.car.behaviour]
        a_new_follower_before = _idm_accel_of(new_follower.car.state, new_follower.v0, nf_idm, new_leader)
        a_new_follower_after  = _idm_accel_of(new_follower.car.state, new_follower.v0, nf_idm, agent)

    return mobil_decision(a_ego_before, a_ego_after,
                           a_old_follower_before, a_old_follower_after,
                           a_new_follower_before, a_new_follower_after,
                           mobil_params)


# ── Simulate ──────────────────────────────────────────────────────────────
# EGO RULE (not yet reachable here): an ego/surr overlap must end the episode
# via model.collision.ego_overlaps_any, never route through
# resolve_surr_collisions -- see that function's docstring. This script has
# no ego vehicle instantiated (ego_s above is only a reference point), so
# there's nothing for that hook to check yet; wire it in once one exists.
history = {agent.car.car_id: {"t": [], "x": [], "y": [], "heading": [], "crashed": []} for agent in agents}

n_steps = int(duration / dt)
for step in range(n_steps):
    t = step * dt

    for agent in agents:
        car = agent.car

        if agent.crashed:
            # Passive obstacle: no IDM, no MOBIL, no steering -- just bleed
            # off the momentum from its crash (see model.collision).
            bleed_crashed(agent, road, dt, k = CRASH_BLEED_K)
        else:
            # 1. MOBIL: only consider a new lane change once any active one has
            # committed, and not within LANE_CHANGE_COOLDOWN of the last one.
            on_cooldown = (agent.last_lane_change_t is not None
                           and (t - agent.last_lane_change_t) < LANE_CHANGE_COOLDOWN)
            if agent.lane_change_t0 is None and not on_cooldown:
                best_lane, best_incentive = None, mobil_params.threshold
                for candidate in (car.state.lane - 1, car.state.lane + 1):
                    if not (0 <= candidate < lane_num):
                        continue
                    should_change, incentive = evaluate_mobil(agents, agent, candidate)
                    if should_change and incentive > best_incentive:
                        best_lane, best_incentive = candidate, incentive
                if best_lane is not None:
                    agent.target_lane = best_lane
                    agent.lane_change_t0 = t

            # 2. IDM: follow the target lane's leader (ego commits to the new
            # lane's traffic stream as soon as a change starts, not just once
            # it completes). find_leader doesn't filter by crashed status, so
            # a stopped wreck is automatically a valid leader here, with
            # whatever v_x it currently has (0 once it's finished bleeding).
            idm_p = IDM_PRESETS[car.behaviour]
            leader = find_leader(agents, agent.target_lane, car.state.s, car.car_id)
            accel = _idm_accel_of(car.state, agent.v0, idm_p, leader)
            # idm_accel is deliberately unclamped (see its own docstring) -- a
            # near-zero gap sends (s_star/gap)^2, and so accel, toward -inf.
            # Physical actuation limit, not part of the IDM formula itself.
            accel = max(accel, -MAX_BRAKING)

            # 3. Far-near steering, target ramped from the old lane's centreline
            # to the new one over this car's lane_change_duration.
            fn_p = FAR_NEAR_PRESETS[car.behaviour]
            if agent.lane_change_t0 is not None:
                old_lane_obj = road.lanes[car.state.lane]
                new_lane_obj = road.lanes[agent.target_lane]
                full_shift = old_lane_obj.offset - new_lane_obj.offset   # type: ignore
                progress = min(1.0, (t - agent.lane_change_t0) / fn_p.lane_change_duration)
                e_y_ref = car.state.e_y - progress * full_shift
            else:
                e_y_ref = car.state.e_y

            e_y_near = far_near_lookahead_offset(e_y_ref, car.state.e_psi, fn_p.L_n)
            d_far    = car.state.v_x * fn_p.T_f
            e_y_far  = far_near_lookahead_offset(e_y_ref, car.state.e_psi, d_far)
            delta_cmd = far_near_steering(e_y_near, e_y_far, car.state.v_x, fn_p.k_n, fn_p.k_f)

            # Curvature-preview feedforward: look up the road's curvature at
            # the far lookahead point (not the car's current s) so delta
            # starts ramping toward what the upcoming curve needs before the
            # car geometrically reaches it -- pure e_y/e_psi feedback always
            # reacts after the fact (see far_near_curvature_feedforward).
            idx_preview = road.index_at(car.state.s + d_far)
            kappa_preview = -road.lanes[car.state.lane].kappa[idx_preview]  # type: ignore
            delta_cmd += far_near_curvature_feedforward(kappa_preview, car.vehicle_params.L)

            delta = clip_steering_rate(delta_cmd, agent.prev_delta, fn_p.steer_rate_limit, dt)
            agent.prev_delta = delta

            # 4. Road curvature/friction at this car's current position, using
            # its nominal (not target) lane, matching the leader lookups above.
            idx = road.index_at(car.state.s)
            kappa = -road.lanes[car.state.lane].kappa[idx]  # type: ignore
            mu = road.mu

            car.step(accel, delta, kappa, mu, dt)

            # 5. Commit the lane change once its duration has elapsed.
            if agent.lane_change_t0 is not None and (t - agent.lane_change_t0) >= fn_p.lane_change_duration:
                old_lane_obj = road.lanes[car.state.lane]
                new_lane_obj = road.lanes[agent.target_lane]
                car.state.e_y = car.state.e_y + (new_lane_obj.offset - old_lane_obj.offset)  # type: ignore
                car.state.lane = agent.target_lane
                agent.lane_change_t0 = None
                agent.last_lane_change_t = t

        # 6. Global position for plotting (common to both branches).
        lane_obj = road.lanes[car.state.lane]
        backbone_e_y = -lane_obj.offset + car.state.e_y  # type: ignore
        x, y, heading = road.frenet_to_global(car.state.s, backbone_e_y, car.state.e_psi)
        car.state.x, car.state.y = x, y

        h = history[car.car_id]
        h["t"].append(t)
        h["x"].append(x)
        h["y"].append(y)
        h["heading"].append(heading)
        h["crashed"].append(agent.crashed)

    # 7. Surr-surr collision detection + plastic response, once per step
    # after everyone (crashed or not) has moved -- see model.collision.
    resolve_surr_collisions(agents, road, t)

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
