import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle

from model.car_init import frenet_to_global
from model.simulation import build_demo_scenario, Simulation
from model.risk import (
    RiskParams, rollover_risk, slip_risk, lane_departure_risk, road_departure_risk, car_to_car_risk,
    EKFPropagator, propagate_ego_trajectory,
)

# =====================================================================
# 1) initialize the scenario (all logic lives under model/, nothing here
#    affects the simulation -- this file only configures and renders it)
# =====================================================================

# Adjustable road parameters: [kappa_max, L_s, mu_patch, patch_location]
Theta_road = [0.005, 60, 0.5, 250.0]

EGO_LANE = 1
EGO_SPEED = 20.0

# other cars: [N (lane), dS (offset from ego), dV (desired-speed offset from ego)]
# 9 surrounding cars + the ego = 10 actors total.
surr_cars = [
    (0, -25.0, 2.0), (0, -10.0, 5.0), (0, -15.0, 5.0), (0, 80.0, 4.0),
    (1, -30.0, 2.0), (1, 70.0, 2.0),
    (2, -45.0, 3.0), (2, 10.0, 4.0), (2, 55.0, 0.0),
]

T_SIM = 20.0

scenario = build_demo_scenario(Theta_road, EGO_LANE, EGO_SPEED, surr_cars)
road = scenario.road
agents = [scenario.ego] + scenario.traffic  # for id/length/width/colour only; never mutated here

# Simulation.step() mutates Car.state (and .current_lane) in place, so this
# MUST be captured before sim.run() below -- reading it afterward (as this
# file used to) silently gives the FINAL state instead of the initial one.
initial_states = {a.id: a.state.copy() for a in agents}
initial_lanes = {a.id: a.current_lane for a in agents}

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
    aid: frenet_to_global(road, s[0], s[1], s[2]) for aid, s in initial_states.items()
}
frames = [initial_poses] + [
    {log.actor_id: log.pose for log in record.actors} for record in result.history
]

dims = {a.id: (a.length, a.width) for a in agents}

# =====================================================================
# 2b) an INSTANTANEOUS (N=0) risk snapshot for the ego at every recorded
#     frame, from model.risk -- a pure function of state. No MPC candidate
#     rollout exists in this codebase, so this uses the spec's explicitly
#     supported N=0 case: each frame's actual logged state stands in for a
#     length-1 predicted trajectory, with RiskParams' seed covariance
#     (R_ego/R_ctrv) as "how uncertain is this recorded state" rather than a
#     multi-step forecast. Computed once here, like `frames` above --
#     rendering below only ever reads risk_frames, never risk.py directly.
# =====================================================================

risk_params = RiskParams()
RISK_LABELS = ["slip", "roll", "lane", "road", "c2c"]


def _actor_states_at_frame(frame_idx):
    """(ego_state, ego_lane, ego_delta, [(neighbor_state, neighbor_pose, neighbor_dims), ...])."""
    if frame_idx == 0:
        ego_state = initial_states[scenario.ego.id]
        ego_lane = initial_lanes[scenario.ego.id]
        ego_delta = 0.0  # no steering command has been applied yet at the true initial condition
        neighbors = [(initial_states[c.id], frames[0][c.id], dims[c.id]) for c in scenario.traffic]
    else:
        logs_by_id = {log.actor_id: log for log in result.history[frame_idx - 1].actors}
        ego_log = logs_by_id[scenario.ego.id]
        ego_state, ego_lane, ego_delta = ego_log.state, ego_log.current_lane, ego_log.steering_cmd
        neighbors = [(logs_by_id[c.id].state, logs_by_id[c.id].pose, dims[c.id]) for c in scenario.traffic]
    return ego_state, ego_lane, ego_delta, neighbors


def _risk_bar_values(frame_idx):
    ego_state, ego_lane, ego_delta, neighbors = _actor_states_at_frame(frame_idx)
    # instantaneous snapshot: no additional commanded a_x is known/assumed, only
    # the steering actually applied -- a_x mainly affects tire load transfer,
    # a secondary effect for this diagnostic display.
    u = (0.0, ego_delta)
    ego_dims = dims[scenario.ego.id]

    p_roll, _, _ = rollover_risk(ego_state, risk_params.R_ego, scenario.vehicle_params, road, u, risk_params)
    p_slip, _, _ = slip_risk(ego_state, risk_params.R_ego, scenario.vehicle_params, road, u, risk_params)
    p_lane, _, _ = lane_departure_risk(ego_state, risk_params.R_ego, road, ego_dims[1], ego_lane, risk_params)
    p_road, _, _ = road_departure_risk(ego_state, risk_params.R_ego, road, ego_dims[1], risk_params)

    # neighbours' global CTRV state [x,y,psi,v,psi_dot], built from the logged
    # Frenet state (vx as speed, r as global yaw rate -- exact at zero
    # sideslip, a good approximation at this scenario's small vy/slip).
    n_means = [np.array([pose[0], pose[1], pose[2], state[3], state[5]]) for state, pose, _ in neighbors]
    n_dims_list = [nd for _, _, nd in neighbors]
    n_covs = [risk_params.R_ctrv] * len(n_means)
    try:
        p_c2c, _ = car_to_car_risk(ego_state, risk_params.R_ego, road, ego_dims,
                                    n_means, n_covs, n_dims_list, risk_params)
    except ValueError:
        # Frenet singularity guard tripped (e.g. mid excursion off the road)
        # -- display-only fallback, treat as maximal risk rather than crash
        # the animation; risk.py's own guard still raises for real callers.
        p_c2c = 1.0

    return [p_slip, p_roll, p_lane, p_road, p_c2c]


risk_frames = [_risk_bar_values(i) for i in range(len(frames))]

# =====================================================================
# 2c) a forward-predicted ego path at every recorded frame, via
#     model.risk.propagate_ego_trajectory -- same EKF used for the c2c risk
#     forecast, just read out as (x, y) means instead of feeding a risk
#     evaluator. No MPC control sequence exists in this codebase, so the
#     control is held constant (steering = last applied command, a_x = 0,
#     matching 2b's instantaneous-risk assumption) over the horizon; it's a
#     "coast on the current curve" preview, not a claim about future intent.
# =====================================================================

PREDICT_HORIZON_S = 2.5  # [s]
_predict_propagator = EKFPropagator(Q=risk_params.Q_ego)


def _predicted_path(frame_idx):
    ego_state, _, ego_delta, _ = _actor_states_at_frame(frame_idx)
    n_steps = int(round(PREDICT_HORIZON_S / scenario.dt))
    controls = [(0.0, ego_delta)] * n_steps
    trajectory = propagate_ego_trajectory(
        ego_state, risk_params.R_ego, controls, road, scenario.vehicle_params, _predict_propagator, scenario.dt,
    )
    xs, ys = [], []
    for mean, _ in trajectory:
        x, y, _ = frenet_to_global(road, mean[0], mean[1], mean[2])
        xs.append(x)
        ys.append(y)
    return xs, ys


predicted_frames = [_predicted_path(i) for i in range(len(frames))]

# =====================================================================
# 3) show the animation (matplotlib only ever reads `frames`/`risk_frames`,
#    built above entirely from the simulation's returned history --
#    rendering cannot feed back into or alter the simulation result)
# =====================================================================

N = road.lane_num
n_x = np.sin(road.heading)
n_y = -np.cos(road.heading)
d_b = (N / 2.0 - np.arange(N + 1, dtype=np.float64)) * road.l_w
x_bounds = road.x[None, :] + d_b[:, None] * n_x[None, :]
y_bounds = road.y[None, :] + d_b[:, None] * n_y[None, :]

patch_mask = (road.s >= road.patch_location) & (road.s < road.patch_location + road.L_patch)

fig, (ax, ax_risk) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]})

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
colors = {scenario.ego.id: "green"}
colors.update({c.id: "black" for c in scenario.traffic})

risk_bars = ax_risk.bar(RISK_LABELS, [0.0] * 5, color=["#4C72B0"] * 5)
ax_risk.set_ylim(0.0, 1.0)
ax_risk.set_ylabel("risk probability (ego, instantaneous)")
ax_risk.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")

patches = {}
for aid in order:
    length, width = dims[aid]
    rect = Rectangle((0, 0), length, width, facecolor=colors[aid], edgecolor="w", zorder=5)
    ax.add_patch(rect)
    patches[aid] = rect

predicted_line, = ax.plot([], [], linestyle="--", color="green", linewidth=1.5, alpha=0.7,
                           label=f"ego predicted path ({PREDICT_HORIZON_S:.1f}s)", zorder=4)
ax.legend(loc="upper right")

HALF_X, HALF_Y = 150.0, 100.0  # [m] chase-cam window around the ego


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

    pred_xs, pred_ys = predicted_frames[frame_idx]
    predicted_line.set_data(pred_xs, pred_ys)

    for bar, value in zip(risk_bars, risk_frames[frame_idx]):
        bar.set_height(value)
        bar.set_color("#C44E52" if value >= 0.5 else "#4C72B0")

    return list(patches.values()) + list(risk_bars) + [predicted_line]


anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=scenario.dt * 1000, blit=False)
plt.show()
