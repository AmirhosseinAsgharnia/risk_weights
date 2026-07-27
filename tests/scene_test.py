import numpy as np
from dataclasses import replace
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle

from model.road import RoadScenario
from model.car_model import EgoModel
from model.car_init import init_ego, init_surr, frenet_to_global
from model.IDM_MOBIL import (
    IDMParams, MOBILParams, LaneChangeParams,
    idm_accel, gap_to_lead, mobil_decision, steering_cmd,
)

# =====================================================================
# 1) initialize the scenario
# =====================================================================

# Adjustable road parameters: [kappa_max, L_s, mu_patch, patch_location]
Theta_road = [0.01, 50, 0.5, 225.0]
kappa_max, L_s, mu_patch, patch_location = Theta_road
road = RoadScenario(kappa_max=kappa_max, L_s=L_s, mu_patch=mu_patch, patch_location=patch_location)

EGO_LANE = 1
EGO_SPEED = 15.0

ego_agent = init_ego(road, lane=EGO_LANE, speed=EGO_SPEED)
# The ego has no speed/lane controller of its own yet, so it just cruises
# at EGO_SPEED (simple P speed-hold, see the sim loop) and stays in lane
# -- no IDM/MOBIL applied to it. It still acts as a normal lane occupant
# that surrounding traffic reacts to, and it's driven by the same dynamic
# bicycle model (EgoModel) as everyone else.

# other cars: [N (lane), dS (offset from ego), dV (desired-speed offset from ego)]
# 9 surrounding cars + the ego = 10 actors total.
surr_cars = [
    (0, -40.0, 1.0), (0, -5.0, 5.0), (0, 35.0, -5.0), (0, 80.0, 4.0),
    (1, -25.0, 1.0), (1, 50.0, -2.0),
    (2, -45.0, 3.0), (2, 10.0, -4.0), (2, 55.0, 0.0),
]
surr = init_surr(road, ego_speed=EGO_SPEED, cars=surr_cars)

agents = [ego_agent] + surr  # index 0 is always the ego

car_model = EgoModel()  # shared dynamic bicycle model + vehicle params for every actor
idm_p = IDMParams()  # template: shared T, a_max, b, delta, s0; v0 is overridden per car
# noise_std = 0: lane changes are deterministic for now (MOBIL incentive +
# safety only), so they can be traced back to real traffic pressure --
# each car's own v0 (see car_init.init_surr) is what creates that pressure.
mobil_p = MOBILParams(a_thr=0.1, noise_std=0.0)
lc_p = LaneChangeParams()
rng = np.random.default_rng(0)

EGO_K_CRUISE = 0.5  # [1/s] proportional gain for the ego's speed-hold

DT = 0.1
T_SIM = 20.0
N_STEPS = int(T_SIM / DT)
LANE_CHANGE_COOLDOWN = 3.0  # [s] don't re-evaluate MOBIL right after a change


def kappa_at(s):
    idx = min(max(int(round(s / road.ds)), 0), len(road.s) - 1)
    return road.kappa[idx]


def mu_at(s):
    idx = min(max(int(round(s / road.ds)), 0), len(road.s) - 1)
    return road.mu[idx]


def drag_compensated(a_net, vx):
    """a_x input that makes EgoModel's net body-frame accel equal a_net
    (EgoModel subtracts aero drag from a_x internally)."""
    return a_net + (car_model.p.C_d / car_model.p.m) * vx**2


def neighbors_in_lane(agents, s, lane, exclude):
    """Nearest (leader_idx, follower_idx) to arclength s in `lane`, excluding agent `exclude`."""
    leader_idx = follower_idx = None
    leader_ds = follower_ds = np.inf
    for j, a in enumerate(agents):
        if j == exclude or a.lane != lane:
            continue
        ds = a.state[0] - s
        if ds > 0 and ds < leader_ds:
            leader_ds, leader_idx = ds, j
        elif ds <= 0 and -ds < follower_ds:
            follower_ds, follower_idx = -ds, j
    return leader_idx, follower_idx


def accel_with_leader(agents, idx, leader_idx):
    """IDM accel of agents[idx], following agents[leader_idx] (or free-flow if None)."""
    if idx is None:
        return 0.0
    me = agents[idx]
    if leader_idx is None:
        gap, dv = np.inf, 0.0
    else:
        lead = agents[leader_idx]
        gap = gap_to_lead(me.state[0], lead.state[0], me.length, lead.length)
        dv = me.state[3] - lead.state[3]
    p = replace(idm_p, v0=me.v0) if me.v0 is not None else idm_p
    return idm_accel(me.state[3], gap, dv, p)


def lane_change_incentive(agents, i, lane_new):
    """MOBIL incentive/safety for agents[i] moving from its current lane to lane_new."""
    me = agents[i]
    s_i, lane_old = me.state[0], me.lane

    leader_old, follower_old = neighbors_in_lane(agents, s_i, lane_old, exclude=i)
    leader_new, follower_new = neighbors_in_lane(agents, s_i, lane_new, exclude=i)

    a_ego_before = accel_with_leader(agents, i, leader_old)
    a_ego_after = accel_with_leader(agents, i, leader_new)

    a_old_follower_before = accel_with_leader(agents, follower_old, i)
    a_old_follower_after = accel_with_leader(agents, follower_old, leader_old)

    a_new_follower_before = accel_with_leader(agents, follower_new, leader_new)
    a_new_follower_after = accel_with_leader(agents, follower_new, i)

    return mobil_decision(
        a_ego_after, a_ego_before,
        a_new_follower_after, a_new_follower_before,
        a_old_follower_after, a_old_follower_before,
        mobil_p,
        rng,
    )


# =====================================================================
# 2) run the controller for the surrounding vehicles
# =====================================================================

last_change_t = {id(c): -1e9 for c in surr}
history = [tuple(frenet_to_global(road, a.state[0], a.state[1], a.state[2]) for a in agents)]

for step in range(N_STEPS):
    t = step * DT

    for i in range(1, len(agents)):  # skip the ego (index 0): no IDM/MOBIL for it
        me = agents[i]
        s_i, lane_i = me.state[0], me.lane

        # --- MOBIL: consider changing to either adjacent lane ---
        if t - last_change_t[id(me)] > LANE_CHANGE_COOLDOWN:
            for lane_new in (lane_i - 1, lane_i + 1):
                if 0 <= lane_new < road.lane_num and lane_change_incentive(agents, i, lane_new):
                    me.lane = lane_new
                    last_change_t[id(me)] = t
                    break

        # --- IDM: longitudinal acceleration from the (possibly new) lane's leader ---
        leader_idx, _ = neighbors_in_lane(agents, s_i, me.lane, exclude=i)
        a_cmd = accel_with_leader(agents, i, leader_idx)

        # --- steering: follow the curve, steer smoothly toward the target lane ---
        kappa_s = kappa_at(s_i)
        mu_s = mu_at(s_i)
        delta_cmd = steering_cmd(me.state[1], me.state[2], kappa_s,
                                  road.d_c[me.lane], car_model.p.L, lc_p)

        u = (drag_compensated(a_cmd, me.state[3]), delta_cmd)
        me.state = car_model.step(me.state, u, kappa_s, mu_s, DT)
        me.state[3] = max(me.state[3], 0.0)  # no reversing

    # --- ego: hold speed, stay in lane, just follow the curve ---
    kappa_ego = kappa_at(ego_agent.state[0])
    mu_ego = mu_at(ego_agent.state[0])
    a_cmd_ego = EGO_K_CRUISE * (EGO_SPEED - ego_agent.state[3])
    delta_ego = steering_cmd(ego_agent.state[1], ego_agent.state[2], kappa_ego,
                              road.d_c[ego_agent.lane], car_model.p.L, lc_p)

    u_ego = (drag_compensated(a_cmd_ego, ego_agent.state[3]), delta_ego)
    ego_agent.state = car_model.step(ego_agent.state, u_ego, kappa_ego, mu_ego, DT)
    ego_agent.state[3] = max(ego_agent.state[3], 0.0)

    history.append(tuple(frenet_to_global(road, a.state[0], a.state[1], a.state[2]) for a in agents))


# =====================================================================
# 3) show the animation
# =====================================================================

N = road.lane_num
n_x = np.sin(road.heading)
n_y = -np.cos(road.heading)
d_b = (N / 2.0 - np.arange(N + 1, dtype=np.float64)) * road.l_w
x_bounds = road.x[None, :] + d_b[:, None] * n_x[None, :]
y_bounds = road.y[None, :] + d_b[:, None] * n_y[None, :]

patch_mask = (road.s >= road.patch_location) & (road.s < road.patch_location + road.L_patch)

fig, ax = plt.subplots(figsize=(10, 6))

for k in range(N + 1):
    ax.plot(x_bounds[k, :], y_bounds[k, :], color="k", linewidth=1)
for n in range(N):
    ax.plot(road.x_lanes[n, :], road.y_lanes[n, :], linestyle="--", color="C0", linewidth=0.8)

patch_poly_x = np.concatenate([x_bounds[0, patch_mask], x_bounds[-1, patch_mask][::-1]])
patch_poly_y = np.concatenate([y_bounds[0, patch_mask], y_bounds[-1, patch_mask][::-1]])
ax.fill(patch_poly_x, patch_poly_y, color="r", alpha=0.3,
        label=f"friction patch ($\\mu$={road.mu_patch})")

ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.legend(loc="upper right")

colors = ["green"] + ["black"] * len(surr)
patches = []
for a, color in zip(agents, colors):
    rect = Rectangle((0, 0), a.length, a.width, facecolor=color, edgecolor="w", zorder=5)
    ax.add_patch(rect)
    patches.append(rect)

HALF_X, HALF_Y = 100.0, 50.0  # [m] chase-cam window around the ego


def update(frame):
    poses = history[frame]
    for rect, (x, y, psi), a in zip(patches, poses, agents):
        rect.set_xy((x - a.length / 2, y - a.width / 2))
        rect.set_transform(mtransforms.Affine2D().rotate_around(x, y, psi) + ax.transData)

    ego_x, ego_y, _ = poses[0]
    ax.set_xlim(ego_x - HALF_X, ego_x + HALF_X)
    ax.set_ylim(ego_y - HALF_Y, ego_y + HALF_Y)
    return patches


anim = animation.FuncAnimation(fig, update, frames=len(history), interval=DT * 1000, blit=False)
plt.show()
