"""
Slippage risk — passive tire slippage probability along a predicted ego
trajectory.

Pure-function port of the former ROS `slippage_node`: same physics (PARALO
§5.2.2 "Slippage Model", eqs. 5.9-5.19) and the same §5.2.5 "Cumulative Risk
Calculation" reduction (eq. 5.51), but with ROS messaging/QoS/visualisation
stripped out and the interface swapped from pub/sub callbacks to a plain
function over an explicit trajectory. The node was a suggestion for the
calculation, not something to copy wholesale.

Weighting (w_slippage) is NOT applied here — slippage() returns the
*unweighted* (weight=1) probability/score pair; the caller multiplies in its
own w_slippage.
"""

import math
from typing import NamedTuple, Sequence

from model.car.config import VehicleParameters

_EPS = 1e-9        # near-zero floor for denominators
_PROB_FLOOR = 1e-6  # minimum U/3 to avoid division by zero in CDF


class TrajectoryStep(NamedTuple):
    """One predicted step of the ego trajectory, SAE J670 body frame.

    v:     [m/s] longitudinal speed.
    delta: [rad] front-wheel steering angle.
    a:     [m/s^2] longitudinal acceleration (signed; negative = braking).
    """
    v: float
    delta: float
    a: float


class SlippageParams(NamedTuple):
    """Friction-model tuning — independent of vehicle mass/geometry (see
    VehicleParameters for those). PARALO §5.2.2 eqs. (5.9)-(5.19).

    mu_max:                 [-] dry-road friction coefficient upper bound.
    mu_uncertainty:         [-] road-friction uncertainty U_mu.
    friction_speed_decay_k: [-] decay rate k_f in mu = mu_max * exp(-k_f*|v|).
    alpha_max_deg_at_mu_1:  [deg] max tire slip angle at mu=1, scaled by mu.
    g:                      [m/s^2] gravitational acceleration.
    """
    mu_max: float = 0.7
    mu_uncertainty: float = 0.05
    friction_speed_decay_k: float = 0.0
    alpha_max_deg_at_mu_1: float = 8.6
    g: float = 9.81


# ──────────────────────────────────────────────────────────────────────────────
# Pure helper functions
# ──────────────────────────────────────────────────────────────────────────────

def normal_cdf(z: float) -> float:
    """Φ(z) = 0.5 · (1 + erf(z / √2))"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ──────────────────────────────────────────────────────────────────────────────
# Physics helpers — each takes explicit parameter arguments for testability
# ──────────────────────────────────────────────────────────────────────────────

def compute_mu(v: float,
               mu_max: float,
               friction_speed_decay_k: float) -> float:
    """
    Speed-dependent friction coefficient — PARALO §5.2.2 eq. (5.9):
    μ_k = μ_max · exp(−k_f · |v|).
    Unclamped, matching the paper exactly: the exponential is already in
    (0, μ_max] for any finite v, so a floor/ceiling would be a no-op except
    for artificially propping up μ at very high speed.
    """
    return mu_max * math.exp(-friction_speed_decay_k * abs(v))


def compute_beta(delta: float, lr: float, wheelbase_L: float) -> float:
    """
    Body slip angle from steering angle (kinematic bicycle model).
    β = atan(l_r · tan(δ) / L)
    """
    return math.atan(lr * math.tan(delta) / (wheelbase_L + _EPS))


def compute_radius(delta: float, beta: float, lr: float) -> float:
    """
    Turning radius from slip angle.
    R = |l_r / sin(β)|  (large value when δ ≈ 0).
    """
    sin_beta = math.sin(beta)
    if abs(sin_beta) < _EPS:
        return 1.0e6          # effectively infinite radius
    return abs(lr / sin_beta)


def compute_v_safe(delta: float,
                   mu: float,
                   R: float,
                   g: float,
                   alpha_max_rad: float,
                   Cf: float,
                   Cr: float,
                   mass: float,
                   lf: float,
                   lr: float,
                   wheelbase_L: float) -> float:
    """
    Minimum safe speed from three limits:
      v_SS  = √(μ·g·R)                             skid-steer (lateral g limit)
      v_US  = √(α_max·R·Cf·L / (m·lr))             understeer
      v_OS  = √(α_max·R·Cr·L / (m·lf))             oversteer
    Returns min(v_SS, v_US, v_OS), clamped ≥ 0.
    """
    v_SS = math.sqrt(max(0.0, mu * g * R))
    v_US = math.sqrt(max(0.0, alpha_max_rad * R * Cf * wheelbase_L / (mass * lr + _EPS)))
    v_OS = math.sqrt(max(0.0, alpha_max_rad * R * Cr * wheelbase_L / (mass * lf + _EPS)))
    return min(v_SS, v_US, v_OS)


def compute_a_min(beta: float, mu: float, g: float) -> float:
    """
    Minimum (most negative) deceleration before braking slip.
    a_min = −μ·g·cos(β)  — PARALO §5.2.2 eq. (5.16).
    """
    return -mu * g * math.cos(beta)


def compute_delta_v(R: float,
                    mu: float,
                    alpha_max_rad: float,
                    U_alpha_rad: float,
                    mu_uncertainty: float,
                    Cf: float,
                    Cr: float,
                    mass: float,
                    lf: float,
                    lr: float,
                    wheelbase_L: float,
                    g: float) -> float:
    """
    Cornering-slip safe-speed uncertainty — PARALO §5.2.2 eq. (5.14):
      Δv = max( √(R·Cf·L/(4·m·lr·αmax))·Uα,
                √(R·Cr·L/(4·m·lf·αmax))·Uα,
                √(g·R/(4·μ))·Uμ )
    One term per v_safe candidate (US, OS, SS in that order).
    """
    denom_us = max(4.0 * mass * lr * alpha_max_rad, _EPS)
    denom_os = max(4.0 * mass * lf * alpha_max_rad, _EPS)
    denom_ss = max(4.0 * mu, _EPS)
    term_us = math.sqrt(max(0.0, R * Cf * wheelbase_L / denom_us)) * U_alpha_rad
    term_os = math.sqrt(max(0.0, R * Cr * wheelbase_L / denom_os)) * U_alpha_rad
    term_ss = math.sqrt(max(0.0, g * R / denom_ss)) * mu_uncertainty
    return max(term_us, term_os, term_ss)


def compute_slip_probability(
        v: float,
        delta: float,
        a: float,
        mu_max: float,
        mu_uncertainty: float,
        friction_speed_decay_k: float,
        alpha_max_deg_at_mu_1: float,
        Cf: float,
        Cr: float,
        mass: float,
        lf: float,
        lr: float,
        wheelbase_L: float,
        g: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Compute cornering-slip probability P_CS, braking-slip probability P_BS,
    combined P_slip, and the intermediate physics values (mu, v_safe, a_min),
    following PARALO §5.2.2 "Slippage Model" (eqs. 5.9-5.19) exactly: the
    only uncertainty source is the road-friction uncertainty Uμ
    (mu_uncertainty), propagated through Δv (5.14) and Uamin (5.17) — there
    is no dependence on trajectory-covariance state here.

    Returns: (P_CS, P_BS, P_slip, mu, v_safe, a_min)
    """
    # ── eq. (5.9): friction ───────────────────────────────────────────────────
    mu = compute_mu(v, mu_max, friction_speed_decay_k)

    # ── eqs. (5.3)/(5.4): slip angle and turning radius ───────────────────────
    beta = compute_beta(delta, lr, wheelbase_L)
    R = compute_radius(delta, beta, lr)

    # ── maximum tire slip angle and its uncertainty [rad] ─────────────────────
    # αmax = μ·(base angle)·π/180 ;  Uα = Uμ·(base angle)·π/180 (same linear
    # form, Uμ substituted for μ — per the paragraph following eq. 5.13)
    deg_to_rad = alpha_max_deg_at_mu_1 * math.pi / 180.0
    alpha_max_rad = mu * deg_to_rad
    U_alpha_rad   = mu_uncertainty * deg_to_rad

    # ── eqs. (5.10)-(5.13): safe speed ─────────────────────────────────────────
    v_safe = compute_v_safe(delta, mu, R, g, alpha_max_rad,
                             Cf, Cr, mass, lf, lr, wheelbase_L)

    # ── eq. (5.14): cornering-slip safe-speed uncertainty ─────────────────────
    delta_v = compute_delta_v(R, mu, alpha_max_rad, U_alpha_rad, mu_uncertainty,
                               Cf, Cr, mass, lf, lr, wheelbase_L, g)
    U_CS = max(delta_v, _PROB_FLOOR)

    # ── eq. (5.15): cornering slip probability ─────────────────────────────────
    # P_CS = Φ((v − v_safe) / (U_CS / 3))
    P_CS = _clamp(normal_cdf((v - v_safe) / (U_CS / 3.0)), 0.0, 1.0)

    # ── eq. (5.16): braking slip ────────────────────────────────────────────────
    a_min = compute_a_min(beta, mu, g)

    # ── eq. (5.17): braking-slip deceleration uncertainty — Uamin = g·Uμ ───────
    U_BS = max(g * mu_uncertainty, _PROB_FLOOR)

    # ── eq. (5.18): braking slip probability ────────────────────────────────────
    # P_BS = 1 − Φ((a − a_min) / (U_BS / 3))
    P_BS = _clamp(1.0 - normal_cdf((a - a_min) / (U_BS / 3.0)), 0.0, 1.0)

    # ── eq. (5.19): combined slip probability ───────────────────────────────────
    P_slip = _clamp(1.0 - (1.0 - P_CS) * (1.0 - P_BS), 0.0, 1.0)

    return P_CS, P_BS, P_slip, mu, v_safe, a_min


# ──────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────────────

def slippage(
        trajectory: Sequence[TrajectoryStep],
        vehicle_params: VehicleParameters | None = None,
        params: SlippageParams | None = None,
) -> tuple[float, float]:
    """
    Evaluate passive tire-slippage risk over a predicted ego trajectory.

    Inputs:
      trajectory:     sequence of TrajectoryStep(v, delta, a), one per
                       predicted step t = 0..N, in chronological order.
      vehicle_params: mass/geometry — VehicleParameters.m, .l_f, .l_r, .L,
                       .C_f, .C_r; defaults to VehicleParameters() if None.
      params:         friction-model tuning — SlippageParams; defaults to
                       SlippageParams() if None.

    Returns (prob, score):
      prob  = max_t P_slip(t) over the trajectory — PARALO eq. (5.19).
      score = cumulative slippage risk R — PARALO §5.2.5 eq. (5.51), with
              the severity weight w_slippage implicitly 1.0 (multiply by
              the real w_slippage yourself):
                S(t)  = 0.5 · m · v(t)²
                S_max = max_t S(t)
                R     = S_max · (1 − Π_t (1 − (S(t)/S_max) · P_slip(t)))
    """
    vp = vehicle_params or VehicleParameters()
    p = params or SlippageParams()

    p_list: list[float] = []   # P_slip(t) — PARALO eq. (5.19)
    s_list: list[float] = []   # S(t)      — kinetic-energy severity, w_slippage=1

    for step in trajectory:
        v = max(0.0, step.v)
        _, _, P_slip, _, _, _ = compute_slip_probability(
            v=v,
            delta=step.delta,
            a=step.a,
            mu_max=p.mu_max,
            mu_uncertainty=p.mu_uncertainty,
            friction_speed_decay_k=p.friction_speed_decay_k,
            alpha_max_deg_at_mu_1=p.alpha_max_deg_at_mu_1,
            Cf=vp.C_f,
            Cr=vp.C_r,
            mass=vp.m,
            lf=vp.l_f,
            lr=vp.l_r,
            wheelbase_L=vp.L,
            g=p.g,
        )
        p_list.append(P_slip)
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
