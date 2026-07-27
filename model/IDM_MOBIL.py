"""
IDM (longitudinal) / MOBIL (lane-change decision) controller.

Pure control logic only: given the current traffic state it returns
acceleration and steering *commands*. It knows nothing about EgoModel,
RoadScenario, or how state is integrated -- that hookup (reading neighbour
states off the road, feeding these commands into EgoModel.step, etc.)
lives in a separate module.
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class IDMParams:
    v0: float = 30.0     # [m/s] desired (free-flow) speed
    T: float = 1.5       # [s] desired time headway
    a_max: float = 1.5   # [m/s^2] max acceleration
    b: float = 2.0       # [m/s^2] comfortable braking deceleration
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
    k_y: float = 0.05                     # [1/s] lateral-error -> desired heading-error gain
    k_psi: float = 3.0                    # [-] heading-error -> steering gain
    e_psi_max: float = np.deg2rad(15.0)   # [rad] cap on desired heading error


# ---------- IDM (longitudinal) ----------

def gap_to_lead(s_ego, s_lead, len_ego, len_lead):
    """Bumper-to-bumper gap [m] to the lead vehicle."""
    return (s_lead - len_lead / 2.0) - (s_ego + len_ego / 2.0)


def idm_accel(v, gap, dv, p: IDMParams | None = None) -> float:
    """
    IDM longitudinal acceleration command.

    v    [m/s] own speed
    gap  [m]   bumper-to-bumper gap to the lead vehicle (np.inf if none)
    dv   [m/s] closing speed, v - v_lead (positive = approaching)
    """
    p = p or IDMParams()
    gap = max(gap, 1e-3)

    s_star = p.s0 + max(0.0, v * p.T + v * dv / (2.0 * np.sqrt(p.a_max * p.b)))
    return p.a_max * (1.0 - (v / p.v0) ** p.delta - (s_star / gap) ** 2)


# ---------- MOBIL (lane-change decision) ----------

def mobil_incentive(a_ego_after, a_ego_before,
                     a_new_follower_after, a_new_follower_before,
                     a_old_follower_after, a_old_follower_before,
                     p: MOBILParams | None = None) -> float:
    """MOBIL incentive value; the change is worth taking if this exceeds p.a_thr."""
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
    True if the lane change is both safe and worthwhile.

    If `rng` is given and p.noise_std > 0, a random "whim" term is added to
    the incentive before comparing to p.a_thr, so lane changes that aren't
    strictly necessary still happen occasionally (real drivers do this too).
    Safety (the new follower's imposed deceleration) is never relaxed by
    the noise.
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


# ---------- steering command: curve following + smooth lane change ----------

def steering_cmd(e_y, e_psi, kappa_s, e_y_target, wheelbase, p: LaneChangeParams | None = None) -> float:
    """
    Steering angle command (EgoModel's `delta` input) in Frenet coordinates
    (state = [s, e_y, e_psi, vx, vy, r]).

    delta = wheelbase*kappa(s), the steady kinematic turn that follows the
    road curvature with e_y, e_psi held at zero, plus proportional feedback
    on the heading error relative to a bounded, e_y-driven desired heading
    e_psi_des -- steering smoothly toward e_y_target (the target lane's
    centreline offset, road.d_c[lane]). The e_psi_max cap keeps a lane
    change a smooth, continuous manoeuvre rather than a jerky heading snap.

    This closes the loop directly on delta rather than cascading through a
    desired yaw rate: commanding a specific r via steering feedback fights
    a real tire-force-driven vehicle's own sideslip dynamics (the yaw and
    lateral-velocity states are coupled) and tends to oscillate or diverge.
    Driving delta straight off the path errors is the standard, stable
    approach for a steering-input vehicle model.
    """
    p = p or LaneChangeParams()

    # e_y is RIGHT-positive: to shed a positive (e_y - e_y_target) error the
    # nose must turn toward +e_psi (LEFT, see EgoModel.state_space), hence
    # the plus sign here.
    e_psi_des = np.clip(p.k_y * (e_y - e_y_target), -p.e_psi_max, p.e_psi_max)
    return wheelbase * kappa_s + p.k_psi * (e_psi_des - e_psi)
