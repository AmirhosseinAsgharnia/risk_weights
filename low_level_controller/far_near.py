"""
Far-near (two-point) steering controller — lane-keeping for a bicycle-model
vehicle, blending a short look-ahead cross-track correction with a longer
one that implicitly captures heading/curvature anticipation instead of an
explicit e_psi term (cf. stanley_steering).

For surrounding traffic only, not the ego vehicle (ego will be driven by an
MPC, designed later).
"""

import math
from typing import NamedTuple


def far_near_lookahead_offset(e_y: float, e_psi: float, d: float) -> float:
    """
    First-order estimate of the lateral offset from the target path at a
    point d metres ahead of the car along its current heading, given the
    car's own e_y/e_psi at its current position:
        e_y(s + d) ~= e_y + d * sin(e_psi)

    Same linearization as CarDynamics' own e_y ODE
    (d(e_y)/dt = v_x*sin(e_psi) + v_y*cos(e_psi), with d(s)/dt ~= v_x),
    just integrated over a lookahead distance instead of time -- so it
    shares that ODE's sign convention exactly (SAE J670, e_y +right,
    e_psi = psi_vehicle - psi_path). Valid for small e_psi and a d short
    enough that the target path's own curvature over that span is
    negligible -- no path-curvature term is applied.

    Inputs:
      e_y:   [m] current lateral offset from the target path, +right.
      e_psi: [rad] current heading error, psi_vehicle - psi_path.
      d:     [m] look-ahead distance, measured along the car's own heading.

    Returns:
      e_y_d: [m] estimated lateral offset at the look-ahead point, +right.
    """
    return e_y + d * math.sin(e_psi)


def far_near_steering(e_y_near: float, e_y_far: float, v: float,
                       k_near: float, k_far: float, eps: float = 1e-3) -> float:
    """
    Two-point steering command, blending a near and a far look-ahead
    cross-track correction:
        delta = -k_near * atan(e_y_near / (v + eps)) - k_far * atan(e_y_far / (v + eps))

    A short d_near keeps e_y_near close to the car's raw e_y (position
    correction, like stanley_steering's cross-track term); a longer d_far
    makes e_y_far increasingly dominated by d_far * sin(e_psi) (see
    far_near_lookahead_offset), so the far term acts as an implicit heading/
    anticipation correction without needing e_psi directly in this formula.

    Sign convention (SAE J670, +right — CarState's convention, same as
    stanley_steering): a look-ahead point to the right of the target path
    (e_y_near/e_y_far > 0) must pull delta negative (steer left) to
    correct — hence the minus signs. Verified against the same closed-loop
    instability stanley_steering's unflipped cross-track term produced
    before its fix.

    Inputs:
      e_y_near, e_y_far: [m] lateral offset of the near/far look-ahead
                          points from the target path, +right — see
                          far_near_lookahead_offset.
      v:                 [m/s] longitudinal speed (>= 0 expected).
      k_near, k_far:      [-] nonnegative gains on the near/far terms.
      eps:                [m/s] speed floor, as in stanley_steering.

    Returns:
      delta: [rad] commanded front-wheel steering angle.
    """
    return (-k_near * math.atan(e_y_near / (v + eps))
            - k_far * math.atan(e_y_far / (v + eps)))


def far_near_curvature_feedforward(kappa: float, wheelbase: float) -> float:
    """
    Kinematic (Ackermann) steering feedforward for the reference path's
    curvature: delta_ff = atan(kappa * wheelbase).

    far_near_steering above is pure feedback -- it only reacts to e_y/e_psi
    error that has *already* appeared, so on a curve it structurally lags:
    e_psi only starts drifting once the path curves (CarDynamics' e_psi_dot
    = r - kappa*s_dot), and only then does the far term react. Evaluating
    kappa here at a preview point ahead of the car (e.g. the far lookahead
    distance used for e_y_far) rather than at the car's current position
    lets delta start ramping up before the car geometrically reaches the
    curve, substantially cutting the entry-transient lane departure a
    pure-feedback controller otherwise shows on a clothoid.

    Uses the same sign convention as the kappa passed into Car.step (i.e.
    already flipped from the road/Lane's own kappa array -- see the SAE-
    vs-road heading convention note in Road.frenet_to_global) -- pass that
    same value in.
    """
    return math.atan(kappa * wheelbase)


def clip_steering_rate(delta_cmd: float, prev_delta: float, rate_limit: float, dt: float) -> float:
    """
    Limit how far delta_cmd can move from prev_delta in one step of size dt,
    to at most rate_limit [rad/s] -- models actuator/comfort limits on how
    fast the steering wheel can actually turn.
    """
    max_step = rate_limit * dt
    return prev_delta + max(-max_step, min(max_step, delta_cmd - prev_delta))


class FarNearBehaviorParams(NamedTuple):
    """Far-near tuning for one driving style.

    L_n:               [m] near look-ahead distance (fixed).
    T_f:                [s] far preview time -- far look-ahead distance is
                        L_f = v * T_f (see far_near_lookahead_offset), so it
                        grows with speed rather than being a fixed distance.
    k_n, k_f:            [-] near/far gains.
    steer_rate_limit:    [rad/s] see clip_steering_rate.
    lane_change_duration: [s] how long a commanded lane change takes to
                        blend the steering target from the old lane to the
                        new one (used by whatever drives the target-lane
                        transition, e.g. a MOBIL-triggered change -- not
                        used by far_near_steering itself).
    """
    L_n: float
    T_f: float
    k_n: float
    k_f: float
    steer_rate_limit: float
    lane_change_duration: float


FAR_NEAR_PRESETS: dict[int, FarNearBehaviorParams] = {
    # Retuned against tests/traffic_test.py's clothoid curve (kappa_max=0.005,
    # L_clothoid=60) together with far_near_curvature_feedforward: a grid
    # search over each behaviour's own desired-speed range (see
    # initialization.traffic_init.SPEED_RANGES) found L_n/T_f/k_n/k_f
    # converging to the same point for all three behaviours before hitting
    # closed-loop instability -- steer_rate_limit (still ordered
    # conservative < moderate < aggressive) is what's left to differentiate
    # reaction speed. Cut peak lane-departure on the curve from ~1.4-2.1 m
    # to ~0.9-1.8 m across the three behaviours.
    1: FarNearBehaviorParams(L_n=5.0, T_f=1.0, k_n=2.5, k_f=2.5, steer_rate_limit=0.6, lane_change_duration=5.5),  # conservative
    2: FarNearBehaviorParams(L_n=5.0, T_f=1.0, k_n=2.5, k_f=2.5, steer_rate_limit=0.8, lane_change_duration=4.0),  # moderate
    3: FarNearBehaviorParams(L_n=5.0, T_f=1.0, k_n=2.5, k_f=2.5, steer_rate_limit=1.0, lane_change_duration=2.5),  # aggressive
}
