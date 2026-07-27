import numpy as np
from model.road import RoadScenario
import matplotlib.pyplot as plt

# Adjustable road parameters: [kappa_max, L_s, mu_patch, patch_location]
Theta_road = [0.02, 50, 0.3, 225.0]

kappa_max, L_s, mu_patch, patch_location = Theta_road

r = RoadScenario(kappa_max=kappa_max, L_s=L_s, mu_patch=mu_patch, patch_location=patch_location)

# x_lanes / y_lanes are lane CENTRELINES, not the physical lane boundary
# lines. Derive the actual boundary lines by offsetting the reference
# centreline (r.x, r.y) by +-l_w/2 steps, same way _lane_centre() does
# for the lane centres.
N = r.lane_num
n_x = np.sin(r.heading)
n_y = -np.cos(r.heading)
d_b = (N / 2.0 - np.arange(N + 1, dtype=np.float64)) * r.l_w
x_bounds = r.x[None, :] + d_b[:, None] * n_x[None, :]
y_bounds = r.y[None, :] + d_b[:, None] * n_y[None, :]

# arc-length mask of the friction patch
patch_mask = (r.s >= r.patch_location) & (r.s < r.patch_location + r.L_patch)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# lane boundary lines (the actual road)
for k in range(N + 1):
    ax[0].plot(x_bounds[k, :], y_bounds[k, :], color="k", linewidth=1)

# lane centrelines (reference/control targets, not physical lines)
for n in range(N):
    ax[0].plot(r.x_lanes[n, :], r.y_lanes[n, :], linestyle="--", color="C0", linewidth=0.8)

# friction patch, shaded across the full road width
patch_poly_x = np.concatenate([x_bounds[0, patch_mask], x_bounds[-1, patch_mask][::-1]])
patch_poly_y = np.concatenate([y_bounds[0, patch_mask], y_bounds[-1, patch_mask][::-1]])
ax[0].fill(patch_poly_x, patch_poly_y, color="r", alpha=0.3,
           label=f"friction patch ($\\mu$={r.mu_patch})")

ax[0].set_aspect("equal")
ax[0].set_xlabel("x [m]")
ax[0].set_ylabel("y [m]")
ax[0].legend()

# friction coefficient along the road
ax[1].plot(r.s, r.mu)
ax[1].axvspan(r.patch_location, r.patch_location + r.L_patch, color="r", alpha=0.3)
ax[1].set_xlabel("s [m]")
ax[1].set_ylabel(r"$\mu$")

plt.show()
