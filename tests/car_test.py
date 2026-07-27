import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle

from model.road import RoadScenario
from model.car_init import init_ego, init_surr, frenet_to_global

r = RoadScenario()

ego = init_ego(r, lane=1, speed=25.0)
ego_x, ego_y, ego_psi = frenet_to_global(r, ego.state[0], ego.state[1], ego.state[2])

# other cars: [N (lane), dS (offset from ego), dV (speed offset from ego)]
surr_cars = [(0, 20.0, -3.0), (2, -15.0, 2.0), (1, 45.0, -5.0)]
surr = init_surr(r, ego_speed=ego.state[3], cars=surr_cars)


def draw_car(ax, x, y, psi, length, width, color):
    rect = Rectangle((x - length / 2, y - width / 2), length, width,
                      facecolor=color, edgecolor="k", zorder=5)
    t = mtransforms.Affine2D().rotate_around(x, y, psi) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)


# --- road (lane boundary lines + centrelines + friction patch) ---

N = r.lane_num
n_x = np.sin(r.heading)
n_y = -np.cos(r.heading)
d_b = (N / 2.0 - np.arange(N + 1, dtype=np.float64)) * r.l_w
x_bounds = r.x[None, :] + d_b[:, None] * n_x[None, :]
y_bounds = r.y[None, :] + d_b[:, None] * n_y[None, :]

patch_mask = (r.s >= r.patch_location) & (r.s < r.patch_location + r.L_patch)

fig, ax = plt.subplots(figsize=(10, 6))

for k in range(N + 1):
    ax.plot(x_bounds[k, :], y_bounds[k, :], color="k", linewidth=1)

for n in range(N):
    ax.plot(r.x_lanes[n, :], r.y_lanes[n, :], linestyle="--", color="C0", linewidth=0.8)

patch_poly_x = np.concatenate([x_bounds[0, patch_mask], x_bounds[-1, patch_mask][::-1]])
patch_poly_y = np.concatenate([y_bounds[0, patch_mask], y_bounds[-1, patch_mask][::-1]])
ax.fill(patch_poly_x, patch_poly_y, color="r", alpha=0.3,
        label=f"friction patch ($\\mu$={r.mu_patch})")

# --- cars: ego green, everyone else black ---

draw_car(ax, ego_x, ego_y, ego_psi, ego.length, ego.width, "green")

surr_global = []
for c in surr:
    s, e_y, e_psi, v, omega = c.state
    x, y, psi = frenet_to_global(r, s, e_y, e_psi)
    surr_global.append((x, y, psi))
    draw_car(ax, x, y, psi, c.length, c.width, "black")

# zoom on the cars, not the whole 500 m road
xs = [ego_x] + [g[0] for g in surr_global]
ys = [ego_y] + [g[1] for g in surr_global]
ax.set_xlim(min(xs) - 20, max(xs) + 20)
ax.set_ylim(min(ys) - 10, max(ys) + 10)

ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.legend()

plt.show()
