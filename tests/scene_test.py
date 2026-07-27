import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle

from model.car_init import frenet_to_global
from model.simulation import build_demo_scenario, Simulation

# =====================================================================
# 1) initialize the scenario (all logic lives under model/, nothing here
#    affects the simulation -- this file only configures and renders it)
# =====================================================================

# Adjustable road parameters: [kappa_max, L_s, mu_patch, patch_location]
Theta_road = [0.01, 50, 0.5, 225.0]

EGO_LANE = 1
EGO_SPEED = 15.0

# other cars: [N (lane), dS (offset from ego), dV (desired-speed offset from ego)]
# 9 surrounding cars + the ego = 10 actors total.
surr_cars = [
    (0, -40.0, 1.0), (0, -5.0, 5.0), (0, 35.0, -5.0), (0, 80.0, 4.0),
    (1, -25.0, 1.0), (1, 50.0, -2.0),
    (2, -45.0, 3.0), (2, 10.0, -4.0), (2, 55.0, 0.0),
]

T_SIM = 20.0

scenario = build_demo_scenario(Theta_road, EGO_LANE, EGO_SPEED, surr_cars)
road = scenario.road
agents = [scenario.ego] + scenario.traffic  # for id/length/width/colour only; never mutated here

# =====================================================================
# 2) run the simulation (model.simulation.Simulation owns every decision;
#    this script only calls run() and reads back the result)
# =====================================================================

sim = Simulation(scenario)
result = sim.run(n_steps=int(T_SIM / scenario.dt))

print(
    f"completed={result.completed} collision={result.collision} "
    f"road_departure={result.road_departure} numerical_failure={result.numerical_failure} "
    f"loss_of_control={result.loss_of_control} lane_changes={result.lane_changes} "
    f"min_gap={result.min_gap:.2f} min_ttc={result.min_time_to_collision:.2f}"
)
for event in result.events:
    print(" ", event)

# frame 0 = the true initial condition, before any step; result.history is
# entirely post-step records, so this is prepended to animate from t=0.
initial_poses = {
    a.id: frenet_to_global(road, a.state[0], a.state[1], a.state[2]) for a in agents
}
frames = [initial_poses] + [
    {log.actor_id: log.pose for log in record.actors} for record in result.history
]

# =====================================================================
# 3) show the animation (matplotlib only ever reads `frames`, built above
#    entirely from the simulation's returned history -- rendering cannot
#    feed back into or alter the simulation result)
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

order = [scenario.ego.id] + [c.id for c in scenario.traffic]
dims = {a.id: (a.length, a.width) for a in agents}
colors = {scenario.ego.id: "green"}
colors.update({c.id: "black" for c in scenario.traffic})

patches = {}
for aid in order:
    length, width = dims[aid]
    rect = Rectangle((0, 0), length, width, facecolor=colors[aid], edgecolor="w", zorder=5)
    ax.add_patch(rect)
    patches[aid] = rect

HALF_X, HALF_Y = 100.0, 50.0  # [m] chase-cam window around the ego


def update(frame_idx):
    poses = frames[frame_idx]
    for aid in order:
        x, y, psi = poses[aid]
        length, width = dims[aid]
        patches[aid].set_xy((x - length / 2, y - width / 2))
        patches[aid].set_transform(mtransforms.Affine2D().rotate_around(x, y, psi) + ax.transData)

    ego_x, ego_y, _ = poses[scenario.ego.id]
    ax.set_xlim(ego_x - HALF_X, ego_x + HALF_X)
    ax.set_ylim(ego_y - HALF_Y, ego_y + HALF_Y)
    return list(patches.values())


anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=scenario.dt * 1000, blit=False)
plt.show()
