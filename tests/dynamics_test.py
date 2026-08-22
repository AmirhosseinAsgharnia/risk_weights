import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

from model.road.road import Road
from model.car.config import VehicleParameters
from model.car.car import Car, CarState
from controllers.far_near import far_near_steering, far_near_lookahead_offset
from controllers.idm import idm_accel

# ── Options ───────────────────────────────────────────────────────────────
mode = "animation"   # "plot" (static figure) or "animation" (car driving live)

# ── Scenario ──────────────────────────────────────────────────────────────
lane_num        = 3
start_lane      = 2
destination_lane = 0
lane_change_s   = 100.0   # [m] arclength at which the lane change triggers

dt        = 0.05   # [s]
duration  = 20.0   # [s]

d_near = 1.0    # [m] near look-ahead distance
d_far  = 15.0   # [m] far look-ahead distance
k_near = 1.0    # [-] near-term gain
k_far  = 3.0    # [-] far-term gain

v0    = 20.0   # [m/s] IDM desired speed
a_max = 1.5    # [m/s^2]
b     = 2.0    # [m/s^2]

car_length = 4.0   # [m]
car_width  = 2.0   # [m]

road = Road(s_max = 500, kappa_max = 0.000, L_clothoid = 60,
            mu = 1.0, lane_num = lane_num)

car = Car(state = CarState(s = 0.0, e_y = 0.0, e_psi = 0.0, v_x = v0, lane = start_lane),
          vehicle_params = VehicleParameters())


def car_corners(x: float, y: float, heading: float, length: float = car_length, width: float = car_width):
    """4 corners of the car's body rectangle, centred on (x, y), long axis
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
current_lane = start_lane
switched     = False

t_hist, s_hist, e_y_hist, v_hist, delta_hist, x_hist, y_hist, heading_hist = [], [], [], [], [], [], [], []

for step in range(int(duration / dt)):
    t = step * dt

    lane_now = road.lanes[current_lane]
    idx      = road.index_at(car.state.s)
    # Negated: Road.kappa is positive for a left (counter-clockwise) turn,
    # matching its (cos, sin) heading convention -- CarDynamics' own kappa
    # is positive for a right (clockwise, SAE) turn, confirmed via its
    # e_psi ODE (kappa=+, delta=0 drives e_psi negative, i.e. the vehicle
    # ends up left of a *rightward*-curving reference).
    kappa    = -lane_now.kappa[idx]  # type: ignore
    mu       = road.mu

    e_y_near = far_near_lookahead_offset(car.state.e_y, car.state.e_psi, d_near)
    e_y_far  = far_near_lookahead_offset(car.state.e_y, car.state.e_psi, d_far)
    delta = far_near_steering(e_y_near, e_y_far, car.state.v_x, k_near, k_far)
    accel = idm_accel(car.state.v_x, math.inf, 0.0, v0, a_max, b)

    car.step(accel, delta, kappa, mu, dt)

    # One-time lane-change: switch which lane e_y/kappa are measured
    # against, at the instant s crosses lane_change_s. The car's physical
    # position doesn't jump -- only the reference lane does -- so e_y is
    # re-expressed relative to the new lane's centreline (old and new
    # lanes both being offsets from the same road backbone) rather than
    # left as-is, which would otherwise be silently reinterpreted as an
    # offset from the wrong lane. Lane.offset is +left while e_y is +right
    # (SAE), hence (new - old), not (old - new) -- verified against a
    # worked example (see the discussion this fix came out of).
    if not switched and car.state.s >= lane_change_s:
        old_lane = road.lanes[current_lane]
        new_lane = road.lanes[destination_lane]
        car.state.e_y = car.state.e_y + (new_lane.offset - old_lane.offset)  # type: ignore
        car.state.lane = destination_lane
        current_lane = destination_lane
        switched = True

    lane_now = road.lanes[current_lane]
    # -lane_now.offset: converts Lane.offset's +left convention to the
    # +right (SAE) convention frenet_to_global's e_y expects.
    backbone_e_y = -lane_now.offset + car.state.e_y  # type: ignore
    x, y, heading = road.frenet_to_global(car.state.s, backbone_e_y, car.state.e_psi)
    car.state.x, car.state.y = x, y

    t_hist.append(t)
    s_hist.append(car.state.s)
    e_y_hist.append(car.state.e_y)
    v_hist.append(car.state.v_x)
    delta_hist.append(delta)
    x_hist.append(x)
    y_hist.append(y)
    heading_hist.append(heading)

    if car.state.s >= road.s_max - 1.0:
        break

# ── Figure: road + trajectory, cross-track error, speed, steering ──────────
fig, axe = plt.subplots(2, 2, figsize = (11, 9))

# Road (asphalt band + lane dividers + per-lane centrelines).
first, last = road.lanes[0], road.lanes[-1]
edge_low_x  = first.x - (first.width / 2) * np.sin(first.heading)   # type: ignore
edge_low_y  = first.y - (first.width / 2) * np.cos(first.heading)   # type: ignore
edge_high_x = last.x  + (last.width  / 2) * np.sin(last.heading)    # type: ignore
edge_high_y = last.y  + (last.width  / 2) * np.cos(last.heading)    # type: ignore

poly_x = np.concatenate([edge_low_x, edge_high_x[::-1]])
poly_y = np.concatenate([edge_low_y, edge_high_y[::-1]])

axe[0][0].set_facecolor("#e3efe0")
axe[0][0].fill(poly_x, poly_y, color = "white", zorder = 1)
axe[0][0].plot(edge_low_x,  edge_low_y,  color = "dimgray", linewidth = 1.2, zorder = 2)
axe[0][0].plot(edge_high_x, edge_high_y, color = "dimgray", linewidth = 1.2, zorder = 2)
for l in range(lane_num - 1):
    div_x = (road.lanes[l].x + road.lanes[l + 1].x) / 2  # type: ignore
    div_y = (road.lanes[l].y + road.lanes[l + 1].y) / 2  # type: ignore
    axe[0][0].plot(div_x, div_y, linestyle = "dashed", color = "dimgray", linewidth = 1.5, zorder = 3)
for lane in road.lanes:
    axe[0][0].plot(lane.x, lane.y, linestyle = "dotted", color = "goldenrod", linewidth = 2.0, zorder = 4)  # type: ignore

axe[0][0].plot(x_hist, y_hist, color = "red", linewidth = 2.0, zorder = 5)
axe[0][0].set_aspect('equal')
axe[0][0].set_title("Trajectory")

axe[0][1].plot(s_hist, e_y_hist, color = "red")
axe[0][1].axvline(lane_change_s, color = "gray", linestyle = "dashed")
axe[0][1].set_xlabel("s [m]")
axe[0][1].set_ylabel("e_y (from target lane) [m]")
axe[0][1].set_title("Cross-track error")

axe[1][0].plot(t_hist, v_hist, color = "red")
axe[1][0].axhline(v0, color = "gray", linestyle = "dashed")
axe[1][0].set_xlabel("t [s]")
axe[1][0].set_ylabel("v_x [m/s]")
axe[1][0].set_title("Speed (IDM)")

axe[1][1].plot(t_hist, delta_hist, color = "red")
axe[1][1].set_xlabel("t [s]")
axe[1][1].set_ylabel("delta [rad]")
axe[1][1].set_title("Steering (far-near)")

plt.tight_layout()

if mode == "plot":
    # A handful of snapshots of the car's 4x2 body along the path.
    n_snapshots = 6
    snap_idx = np.linspace(0, len(t_hist) - 1, n_snapshots).astype(int)
    for i in snap_idx:
        corners = car_corners(x_hist[i], y_hist[i], heading_hist[i])
        axe[0][0].add_patch(Polygon(corners, closed = True, facecolor = "steelblue",
                                     edgecolor = "black", alpha = 0.6, zorder = 6))
    plt.show()

elif mode == "animation":
    car_patch = Polygon(car_corners(x_hist[0], y_hist[0], heading_hist[0]),
                         closed = True, facecolor = "steelblue", edgecolor = "black", zorder = 6)
    axe[0][0].add_patch(car_patch)

    ey_dot,    = axe[0][1].plot([s_hist[0]], [e_y_hist[0]], "o", color = "black", zorder = 6)
    v_dot,     = axe[1][0].plot([t_hist[0]], [v_hist[0]],   "o", color = "black", zorder = 6)
    delta_dot, = axe[1][1].plot([t_hist[0]], [delta_hist[0]], "o", color = "black", zorder = 6)

    def update(i):
        car_patch.set_xy(car_corners(x_hist[i], y_hist[i], heading_hist[i]))
        ey_dot.set_data([s_hist[i]], [e_y_hist[i]])
        v_dot.set_data([t_hist[i]], [v_hist[i]])
        delta_dot.set_data([t_hist[i]], [delta_hist[i]])
        return car_patch, ey_dot, v_dot, delta_dot

    ani = FuncAnimation(fig, update, frames = len(t_hist), interval = dt * 1000, blit = False)
    plt.show()

else:
    raise ValueError(f"mode must be 'plot' or 'animation', got {mode!r}")
