"""
Rollover risk — vehicle rollover probability along a predicted ego trajectory.

Pure-function port of the former ROS `rollover_node`: same physics (PARALO
§5.2.2 "Rollover Model", eqs. 5.2-5.6) and the same §5.2.5 "Cumulative Risk
Calculation" reduction (eq. 5.51), but with ROS messaging/QoS/visualisation
stripped out and the interface swapped from pub/sub callbacks to a plain
function over an explicit trajectory. The node was a suggestion for the
calculation, not something to copy wholesale.

Weighting (w_rollover) is NOT applied here — rollover() returns the
*unweighted* (weight=1) probability/score pair; the caller multiplies in its
own w_rollover.
"""

import math
from typing import NamedTuple, Sequence

from model.car.config import VehicleParameters

_EPS = 1e-9        # near-zero floor for denominators
_PROB_FLOOR = 1e-6  # minimum Δv/3 to avoid division by zero in CDF
_R_MAX = 1.0e6      # [m] effective infinite turning radius (straight line)


class TrajectoryStep(NamedTuple):
    """One predicted step of the ego trajectory, SAE J670 body frame.

    v:     [m/s] longitudinal speed.
    delta: [rad] front-wheel steering angle.
    """
    v: float
    delta: float


class RolloverParams(NamedTuple):
    """Tuning independent of vehicle mass/geometry (see VehicleParameters
    for h_CG, W_c, m, l_r, L). PARALO §5.2.2 eqs. (5.2)-(5.6).

    U_H: [m] CG-height uncertainty. Paper range: 0.07-0.18 m for passenger
         vehicles.
    g:   [m/s^2] gravitational acceleration.
    """
    U_H: float = 0.1
    g: float = 9.81


# ──────────────────────────────────────────────────────────────────────────────
# Pure physics helpers
# ──────────────────────────────────────────────────────────────────────────────

def normal_cdf(z: float) -> float:
    """Φ(z) = 0.5 · (1 + erf(z / √2))"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_beta(delta: float, lr: float, wheelbase_L: float) -> float:
    """Body slip angle: β = atan(lr · tan(δ) / L)"""
    return math.atan(lr * math.tan(delta) / (wheelbase_L + _EPS))


def compute_radius(beta: float, lr: float) -> float:
    """Turning radius: R = |lr / sin(β)|; large value when β ≈ 0."""
    s = math.sin(beta)
    return abs(lr / s) if abs(s) > _EPS else _R_MAX


def compute_v_safe_rollover(g: float, R: float, w_c: float, h_cg: float) -> float:
    """
    Rollover threshold speed.
      v_safe = sqrt(g · R · W_c / (2 · h_cg))
    Derived from the condition that lateral centripetal force equals the
    restoring moment: m·v²/R · h_cg = m·g · W_c/2.
    """
    return math.sqrt(max(0.0, g * R * w_c / (2.0 * h_cg)))


def compute_delta_v(g: float, R: float, w_c: float, h_cg: float, U_H: float) -> float:
    """
    v_safe uncertainty from CG-height uncertainty U_H — PARALO §5.2.2 eq. (5.5):
      Δv = sqrt(g · R · W_c / (4 · h_cg³)) · U_H
    This is |∂v_safe/∂h_cg| · U_H — the sensitivity of the rollover
    threshold to how precisely h_cg is known.
    Units: [1/s] · [m] = [m/s].
    """
    return math.sqrt(max(0.0, g * R * w_c / (4.0 * h_cg**3))) * U_H


def compute_p_rollover(v: float, v_safe: float, delta_v: float) -> float:
    """
    Rollover probability — PARALO §5.2.2 eq. (5.6):
      P_roll = Φ((v − v_safe) / (Δv / 3))
    The only uncertainty source is Δv (eq. 5.5, CG-height-uncertainty-driven)
    — no EKF/trajectory-covariance term belongs here (see module docstring).
    """
    sigma = max(delta_v / 3.0, _PROB_FLOOR)
    return _clamp(normal_cdf((v - v_safe) / sigma), 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────────────

def rollover(
        trajectory: Sequence[TrajectoryStep],
        vehicle_params: VehicleParameters | None = None,
        params: RolloverParams | None = None,
) -> tuple[float, float]:
    """
    Evaluate rollover risk over a predicted ego trajectory.

    Inputs:
      trajectory:     sequence of TrajectoryStep(v, delta), one per
                       predicted step t = 0..N, in chronological order.
      vehicle_params: mass/geometry — VehicleParameters.m, .l_r, .L, .h_CG,
                       .W_c; defaults to VehicleParameters() if None.
      params:         rollover-model tuning — RolloverParams; defaults to
                       RolloverParams() if None.

    Returns (prob, score):
      prob  = max_t P_roll(t) over the trajectory — PARALO eq. (5.6).
      score = cumulative rollover risk R — PARALO §5.2.5 eq. (5.51), with
              the severity weight w_rollover implicitly 1.0 (multiply by
              the real w_rollover yourself):
                S(t)  = 0.5 · m · v(t)²
                S_max = max_t S(t)
                R     = S_max · (1 − Π_t (1 − (S(t)/S_max) · P_roll(t)))
    """
    vp = vehicle_params or VehicleParameters()
    p = params or RolloverParams()

    p_list: list[float] = []   # P_roll(t) — PARALO eq. (5.6)
    s_list: list[float] = []   # S(t)      — kinetic-energy severity, w_rollover=1

    for step in trajectory:
        v = max(0.0, step.v)

        # ── Rollover probability P(t) — PARALO §5.2.2 eqs. (5.2)-(5.6) ──────
        beta = compute_beta(step.delta, vp.l_r, vp.L)
        R = compute_radius(beta, vp.l_r)
        v_safe = compute_v_safe_rollover(p.g, R, vp.W_c, vp.h_CG)
        delta_v = compute_delta_v(p.g, R, vp.W_c, vp.h_CG, p.U_H)
        p_roll = compute_p_rollover(v, v_safe, delta_v)

        p_list.append(p_roll)
        s_list.append(0.5 * vp.m * v * v)

    # ── Cumulative risk R — PARALO §5.2.5 eq. (5.51a-b) ─────────────────────
    max_p = max(p_list, default=0.0)
    S_max = max(s_list, default=0.0)

    if S_max > _EPS:
        survival = 1.0
        for S_t, P_t in zip(s_list, p_list):
            survival *= 1.0 - (S_t / S_max) * P_t
        score = S_max * (1.0 - survival)
    else:
        score = 0.0   # stationary trajectory — zero severity, zero risk

    return max_p, score
