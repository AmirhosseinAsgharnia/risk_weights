"""
IDM (longitudinal) / MOBIL (lane-change incentive) / steering controller.

Pure control logic only: given the current traffic state it returns acceleration
and steering *commands*. It knows nothing about EgoModel, RoadScenario, or how
state is integrated, and nothing about *which* lane a car should move to or when
a manoeuvre starts/ends (that's model.traffic and model.lane_change) -- this
module only scores a single before/after pair (mobil_incentive) and tracks
whatever lateral reference it's given (steering_cmd).
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class IDMParams:
    v0: float = 30.0     # [m/s] desired (free-flow) speed
    T: float = 1.5       # [s] desired time headway
    a_max: float = 1.5   # [m/s^2] max acceleration
    a_min: float = -6.0  # [m/s^2] hard braking bound (distinct from the "comfortable" b below)
    b: float = 2.0       # [m/s^2] comfortable braking deceleration (used inside the IDM formula)
    delta: float = 4.0   # [-] acceleration exponent
    s0: float = 2.0      # [m] minimum (jam) gap


@dataclass(frozen=True)
class MOBILParams:
    politeness: float = 0.3  # [-] weight given to neighbours' induced accelerations
    a_thr: float = 0.2       # [m/s^2] minimum incentive to bother changing lane
    b_safe: float = 4.0      # [m/s^2] max deceleration a change may impose on the new follower
    noise_std: float = 0.0   # [m/s^2] std-dev of random "whim" added to the incentive


@dataclass(frozen=True)
class LaneChangeParams:
    # PID gains on the lateral tracking error (e_y - e_y_ref):
    k_y: float = 1.6                      # [1/s]    proportional
    k_i: float = 0.05                     # [1/s^2]  integral
    k_d: float = 0.8                      # [-]      derivative (acts on d(error)/dt)
    integral_max: float = 10.0            # [m*s] anti-windup clamp on the raw integral accumulator
    integral_error_max: float = 1.0       # [m] only accumulate the integral below this |error| --
                                           # integral trims small residual bias; feeding it a large,
                                           # multi-second transient (e.g. curve-entry drift) just adds
                                           # lag on top of the transient instead of helping it settle

    k_psi: float = 3.0                    # [-] heading-error -> steering gain
    k_r_damp: float = 0.0                 # [s] optional yaw-rate damping term in delta
    e_psi_max: float = np.deg2rad(15.0)   # [rad] cap on desired heading error
    delta_max: float = np.deg2rad(30.0)   # [rad] steering angle limit
    delta_rate_max: float | None = None   # [rad/s] optional steering-rate limit
    v_min: float = 1.0                    # [m/s] low-speed guard for e_psi_des's 1/vx term


# ---------- IDM (longitudinal) ----------

def gap_to_lead(s_ego, s_lead, len_ego, len_lead):
    """Bumper-to-bumper gap [m] to the lead vehicle (may be negative if bodies overlap)."""
    return (s_lead - len_lead / 2.0) - (s_ego + len_ego / 2.0)


def idm_accel(v, gap, dv, p: IDMParams | None = None, clip: bool = True) -> float:
    """
    IDM longitudinal acceleration command.

    v    [m/s] own speed
    gap  [m]   bumper-to-bumper gap to the lead vehicle (np.inf if none).
               gap <= 0 (overlapping bodies) is clamped to a small positive value
               so this never divides by zero or a negative number -- it instead
               commands maximum braking, which is the correct IDM response to an
               (already-happened) overlap. It does NOT detect the overlap itself;
               that's model.collision's job.
    dv   [m/s] closing speed, v - v_lead (positive = approaching)
    clip [bool] clip the result to [p.a_min, p.a_max] -- the right thing to do
               before actually commanding a vehicle, but NOT before using this
               value to score a MOBIL candidate: two "after" gaps that are both
               merely very bad and catastrophically bad can clip to the exact
               same a_min, making the incentive unable to tell them apart and
               scoring a genuinely unsafe merge as no worse than a mediocre one.
               model.traffic scores candidates with clip=False for this reason.
    """
    p = p or IDMParams()
    gap_safe = max(gap, 1e-3)

    s_star = p.s0 + max(0.0, v * p.T + v * dv / (2.0 * np.sqrt(p.a_max * p.b)))
    a = p.a_max * (1.0 - (v / p.v0) ** p.delta - (s_star / gap_safe) ** 2)
    return float(np.clip(a, p.a_min, p.a_max)) if clip else float(a)


# ---------- MOBIL (lane-change incentive) ----------

def mobil_incentive(a_ego_after, a_ego_before,
                     a_new_follower_after, a_new_follower_before,
                     a_old_follower_after, a_old_follower_before,
                     p: MOBILParams | None = None) -> float:
    """MOBIL incentive value; a change is worth taking if this exceeds p.a_thr."""
    p = p or MOBILParams()
    return (
        (a_ego_after - a_ego_before)
        + p.politeness * (
            (a_new_follower_after - a_new_follower_before)
            + (a_old_follower_after - a_old_follower_before)
        )
    )


def mobil_decision(a_ego_after, a_ego_before,
                    a_new_follower_after, a_new_follower_before,
                    a_old_follower_after, a_old_follower_before,
                    p: MOBILParams | None = None,
                    rng: np.random.Generator | None = None) -> bool:
    """
    True if a single before/after pair is both safe and worthwhile. Kept for
    simple/single-candidate use and tests; model.traffic's best-incentive
    selection scores every candidate lane itself via mobil_incentive directly.

    If `rng` is given and p.noise_std > 0, a random "whim" term is added to the
    incentive before comparing to p.a_thr. Safety is never relaxed by the noise.
    """
    p = p or MOBILParams()

    safe = a_new_follower_after >= -p.b_safe
    incentive = mobil_incentive(
        a_ego_after, a_ego_before,
        a_new_follower_after, a_new_follower_before,
        a_old_follower_after, a_old_follower_before,
        p,
    )
    if rng is not None and p.noise_std > 0:
        incentive = incentive + rng.normal(0.0, p.noise_std)

    return safe and incentive > p.a_thr


# ---------- steering: PID on the lateral reference ----------

def steering_cmd(state, kappa_s, e_y_ref, e_y_ref_rate, wheelbase,
                  p: LaneChangeParams | None = None,
                  integral: float = 0.0, integrate: bool = True,
                  prev_delta: float | None = None, dt: float | None = None) -> tuple[float, float]:
    """
    Steering angle command (EgoModel's `delta` input), tracking a lateral
    reference (e_y_ref, e_y_ref_rate) -- the current lane centre when holding
    lane, or model.lane_change's quintic trajectory during a manoeuvre. Both
    cases go through this one function.

    Returns (delta, new_integral) -- new_integral is the updated accumulator
    and MUST be persisted by the caller (e.g. on Car) and passed back in as
    `integral` next step; this function stays a pure/stateless computation.

    Derivation (right-positive e_y, per EgoModel.state_space):
        de_y/dt = -(vx*sin(e_psi) + vy*cos(e_psi))   (exact, no small-angle approx needed)
    Want closed-loop  de_y/dt = e_y_ref_rate - PID(error)  where error = e_y - e_y_ref
    and PID(error) = k_y*error + k_i*integral(error) + k_d*d(error)/dt. The D
    term uses the EXACT de_y/dt above (not a finite difference) minus
    e_y_ref_rate, so it's noise-free. Substituting into the closed-loop target:
        -vx*e_psi_des = e_y_ref_rate - PID(error)
        e_psi_des = (PID(error) - e_y_ref_rate) / vx
    With k_i = k_d = 0 this is exactly the original P-only controller (same
    validated sign).

    integrate=False freezes the accumulator (still returned unchanged) without
    touching dt-gated steering-rate limiting below -- pass this during an
    active lane change. Integral action fighting an already-planned quintic
    transient adds pure lag: in testing it turned a 3s nominal lane change
    into a 15s one with a much bigger overshoot, instead of helping. The
    accumulator also only accumulates while |error| <= p.integral_error_max
    (same reasoning: a large multi-second transient, e.g. curve-entry drift
    while holding a lane, wound the integral up to its clamp in testing
    without actually speeding up recovery -- integral is for trimming small
    residual bias once near the reference, not chasing a big transient. It
    resets to 0 when a lane change starts (model.lane_change.start_lane_change)
    and resumes accumulating once back to steady, close-to-target holding, and
    is clamped (p.integral_max) against windup even then.
    """
    p = p or LaneChangeParams()
    _, e_y, e_psi, vx, vy, r = state

    vx_safe = max(vx, p.v_min)
    error = e_y - e_y_ref
    e_y_dot = -(vx * np.sin(e_psi) + vy * np.cos(e_psi))
    error_dot = e_y_dot - e_y_ref_rate

    new_integral = integral
    if integrate and dt is not None and dt > 0 and abs(error) <= p.integral_error_max:
        new_integral = float(np.clip(integral + error * dt, -p.integral_max, p.integral_max))

    pid_output = p.k_y * error + p.k_i * new_integral + p.k_d * error_dot

    e_psi_des = (pid_output - e_y_ref_rate) / vx_safe
    e_psi_des = np.clip(e_psi_des, -p.e_psi_max, p.e_psi_max)

    delta_ff = wheelbase * kappa_s
    delta = delta_ff + p.k_psi * (e_psi_des - e_psi) - p.k_r_damp * r
    delta = np.clip(delta, -p.delta_max, p.delta_max)

    if p.delta_rate_max is not None and prev_delta is not None and dt is not None and dt > 0:
        max_step = p.delta_rate_max * dt
        delta = np.clip(delta, prev_delta - max_step, prev_delta + max_step)

    return float(delta), new_integral
