"""
The five PARALO risk channels evaluated on the LIVE simulator state, for
J_eval (learning.closed_loop). The risk models themselves are called
unchanged from risks/; this module only builds their inputs.

Predictive, as PARALO intends: at each sim step, the ego and every surr car
are extrapolated over a short horizon t_n = n * dt, n = 0..N, with the
ego's CURRENT applied inputs held constant and surr cars at constant
velocity -- all in the road's Frenet backbone frame (x = s forward, y =
backbone e_y, SAE +right; the frame model.collision already uses):

    ego:  v_n = max(0, v + a t_n),  s_n = s + sum v dt,  y_n = y + vy t_n,  heading e_psi
    surr: s_j + v_j t_n,  y_j + vy_j t_n,  heading e_psi_j        (vy = v_x sin e_psi + v_y cos e_psi)

Channel inputs (simulation ground truth -> exact maps, no perception):
    c2c            ego + 6 surr trajectories; footprints = the sim's CAR_LENGTH x CAR_WIDTH boxes;
                   position std sigma_ego / sigma_obj (sigma -> 0 reproduces model.collision's overlap test)
    rollover       (v_n, delta)
    slippage       (v_n, delta, a), friction mu_max = this scenario's road mu
    lane_departure lateral position vs the NEAREST lane's centre (fixed over the horizon), lane width
                   = road.l_w, vehicle width = CAR_WIDTH, centreline covariance 0 (exact map)
    road_departure same model against the whole paved road (centre = mean lane centre,
                   width = lane_num * l_w)

Each channel returns PARALO's (prob, score). risk_measure "prob" uses prob =
max over the horizon (dimensionless, in [0, 1]); "score" uses the PARALO
cumulative risk R (eq. 5.51) divided by S_ref = 0.5 m v_ref^2.
"""

import math

import numpy as np

from initialization.traffic_init import CAR_LENGTH, CAR_WIDTH
from risks.c2c import c2c, EgoStep, ObjectStep, ObjectTrajectory, C2CParams
from risks.departure import departure, TrajectoryStep as DepStep, LaneCenterline, DepartureParams
from risks.rollover import rollover, TrajectoryStep as RollStep, RolloverParams
from risks.slippage import slippage, TrajectoryStep as SlipStep, SlippageParams

CHANNELS = ("c2c", "rollover", "slippage", "lane_departure", "road_departure")
_ZERO_COV = np.zeros((3, 3))


def _lateral_velocity(state) -> float:
    return state.v_x * math.sin(state.e_psi) + state.v_y * math.cos(state.e_psi)


def predicted_risks(env, cost) -> dict[str, float]:
    """{channel: value} for the env's current state. `cost` is an
    EvalCostConfig (horizon, sigmas, risk measure, uncertainties)."""
    road, ego = env.road, env.ego_car
    st, vp = ego.state, ego.vehicle_params
    a, delta = env._prev_accel, st.delta

    n = max(0, int(round(cost.risk_horizon_s / cost.risk_dt_s)))
    t = cost.risk_dt_s * np.arange(n + 1)
    v = np.maximum(0.0, st.v_x + a * t)
    s = st.s + np.concatenate([[0.0], np.cumsum(v[:-1]) * cost.risk_dt_s])
    y0 = -road.lanes[st.lane].offset + st.e_y
    y = y0 + _lateral_velocity(st) * t

    var_e = cost.risk_sigma_ego_m ** 2
    ego_traj = [EgoStep(x=float(s[i]), y=float(y[i]), v=float(v[i]), heading=st.e_psi, var_x=var_e, var_y=var_e)
                for i in range(n + 1)]
    objects = []
    for ag in env.agents:
        cs = ag.car.state
        yj = -road.lanes[cs.lane].offset + cs.e_y
        vyj = _lateral_velocity(cs)
        objects.append(ObjectTrajectory(track_id=ag.car.car_id, length_m=CAR_LENGTH, width_m=CAR_WIDTH, steps=[
            ObjectStep(x=cs.s + cs.v_x * ti, y=yj + vyj * ti, v=cs.v_x, heading=cs.e_psi,
                       sigma_x=cost.risk_sigma_obj_m, sigma_y=cost.risk_sigma_obj_m) for ti in t]))
    out = {"c2c": c2c(ego_traj, objects, vp, C2CParams(ego_length_m=CAR_LENGTH, ego_width_m=CAR_WIDTH,
                                                       default_sigma_m=cost.risk_sigma_ego_m))}
    out["rollover"] = rollover([RollStep(v=float(vi), delta=delta) for vi in v], vp,
                               RolloverParams(U_H=cost.rollover_U_H_m))
    out["slippage"] = slippage([SlipStep(v=float(vi), delta=delta, a=a) for vi in v], vp,
                               SlippageParams(mu_max=road.mu, mu_uncertainty=cost.slip_mu_uncertainty))

    centers = np.array([-lane.offset for lane in road.lanes])
    lane_c = float(centers[int(np.argmin(np.abs(centers - y0)))])
    dep_traj = [DepStep(x=float(s[i] - st.s), y=float(y[i]), v=float(v[i]), var_x=var_e, var_y=var_e)
                for i in range(n + 1)]
    out["lane_departure"] = departure(dep_traj, LaneCenterline((0.0, 0.0, lane_c), _ZERO_COV), vp,
                                      DepartureParams(lane_width_m=road.l_w, vehicle_width_m=CAR_WIDTH,
                                                      default_sigma_lateral_m=cost.risk_sigma_ego_m))
    out["road_departure"] = departure(dep_traj, LaneCenterline((0.0, 0.0, float(centers.mean())), _ZERO_COV), vp,
                                      DepartureParams(lane_width_m=road.l_w * len(road.lanes),
                                                      vehicle_width_m=CAR_WIDTH,
                                                      default_sigma_lateral_m=cost.risk_sigma_ego_m))
    if cost.risk_measure == "prob":
        return {k: float(p) for k, (p, _) in out.items()}
    s_ref = 0.5 * vp.m * cost.v_ref_mps ** 2
    return {k: float(score) / s_ref for k, (_, score) in out.items()}
