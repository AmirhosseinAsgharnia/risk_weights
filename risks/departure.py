"""
Lane-departure risk — predictive lane departure probability along a
predicted ego trajectory.

Pure-function port of the former ROS `departure_node`: same physics (Gaussian
clearance model against a quadratic lane centerline) and the same PARALO
§5.2.5 "Cumulative Risk Calculation" reduction (eq. 5.51, kinetic-energy
severity only — no per-channel VF, matching the node), but with ROS
messaging/QoS/visualisation and dead (declared-but-unused) parameters
stripped out, and the interface swapped from pub/sub callbacks to a plain
function over an explicit trajectory + lane model. The node was a suggestion
for the calculation, not something to copy wholesale.

There is no weight applied here, same as the node itself (see module
docstring there: "no per-channel VF, kinetic-energy severity only") — the
caller multiplies in its own w_departure.
"""

import math
from typing import NamedTuple, Sequence

import numpy as np

from model.car.config import VehicleParameters

_EPS = 1e-9


class TrajectoryStep(NamedTuple):
    """One predicted step of the ego trajectory, ego frame, SAE J670
    (+forward, +right).

    x:      [m] forward position.
    y:      [m] lateral position, +right.
    v:      [m/s] speed.
    var_x:  [m^2] forward-position variance at this step, or None to fall
            back to DepartureParams.default_sigma_lateral_m^2 (isotropic).
    var_y:  [m^2] lateral-position variance at this step, or None for the
            same fallback.
    cov_xy: [m^2] forward/lateral position covariance (cross term).
    """
    x: float
    y: float
    v: float
    var_x: float | None = None
    var_y: float | None = None
    cov_xy: float = 0.0


class LaneCenterline(NamedTuple):
    """Quadratic lane centerline fit y = a*x^2 + b*x + c, ego frame, plus its
    3x3 coefficient covariance Sigma_theta (from the perception pipeline's
    lane-fit estimator). Pass None as the trajectory's lane model when no
    valid centerline is available -- see departure()'s early return.

    coeffs: (a, b, c).
    cov:    3x3 covariance of (a, b, c).
    """
    coeffs: tuple[float, float, float]
    cov: np.ndarray


class DepartureParams(NamedTuple):
    """Lane/vehicle geometry and fallback uncertainty -- independent of
    vehicle mass (see VehicleParameters.m for that).

    lane_width_m:             [m] lane width.
    vehicle_width_m:          [m] ego vehicle width.
    default_sigma_lateral_m:  [m] fallback isotropic position std used for a
                               trajectory step whose var_x/var_y isn't given.
    min_sigma_y_m:            [m] floor on the clearance std sigma_D.
    """
    lane_width_m: float = 3.8
    vehicle_width_m: float = 1.85
    default_sigma_lateral_m: float = 0.30
    min_sigma_y_m: float = 0.05


# ──────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ──────────────────────────────────────────────────────────────────────────────

def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def normal_cdf(z: float) -> float:
    """Φ(z) via erfc for numerical stability at large |z|."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def eval_quad(coeffs: tuple[float, float, float], x: float) -> float:
    """Evaluate parabolic lane boundary y = a*x^2 + b*x + c."""
    a, b, c = coeffs
    return a * x * x + b * x + c


def eval_slope(coeffs: tuple[float, float, float], x: float) -> float:
    """Centerline slope dy/dx = 2*a*x + b for y = a*x^2 + b*x + c."""
    a, b, _ = coeffs
    return 2.0 * a * x + b


def propagate_poly_variance(cov3: np.ndarray, x: float) -> float:
    """
    Propagate the centerline coefficient covariance to boundary lateral
    variance at distance x: sigma^2 = phi @ P @ phi, phi = [x^2, x, 1].
    """
    phi = np.array([x * x, x, 1.0], dtype=float)
    return max(float(phi @ cov3 @ phi), 0.0)


def sanitize_cov3(cov3: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Validate/repair the 3x3 centerline coefficient covariance Sigma_theta.

    Applies only minimal numerical repair -- symmetrization and clipping of
    tiny negative eigenvalues caused by floating-point error. A non-finite
    or materially non-PSD matrix (a negative eigenvalue beyond that
    tolerance) is reported invalid rather than silently masked.
    Returns (repaired_3x3_matrix, valid_flag).
    """
    P = np.asarray(cov3, dtype=float)
    if not np.all(np.isfinite(P)):
        return np.zeros((3, 3)), False

    P_sym = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P_sym)
    tol = max(1e-9, 1e-6 * float(np.max(np.abs(eigvals))))
    if float(np.min(eigvals)) < -tol:
        return P_sym, False

    eigvals_clipped = np.clip(eigvals, 0.0, None)
    P_repaired = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return 0.5 * (P_repaired + P_repaired.T), True


def _cumulative_risk(s_list: list[float], p_list: list[float]) -> float:
    """
    PARALO §5.2.5 "Cumulative Risk Calculation" eq. (5.51), discrete form:
        q_k = clamp((S_k / S_max) * P_k, 0, 1)
        R   = S_max * (1 - exp(sum_k log1p(-q_k)))
    Equivalent to S_max * (1 - prod_k (1 - q_k)) but computed in log-space
    via log1p for numerical stability. q_k == 1 is handled explicitly so the
    result becomes S_max without an invalid log(0).
    """
    if not s_list:
        return 0.0
    S_max = max(s_list)
    if S_max <= _EPS:
        return 0.0

    log_survival = 0.0
    for S_k, P_k in zip(s_list, p_list):
        q_k = clamp((S_k / S_max) * P_k, 0.0, 1.0)
        if q_k >= 1.0:
            return S_max
        log_survival += math.log1p(-q_k)
    return S_max * (1.0 - math.exp(log_survival))


# ──────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────────────

def departure(
        trajectory: Sequence[TrajectoryStep],
        lane: LaneCenterline | None,
        vehicle_params: VehicleParameters | None = None,
        params: DepartureParams | None = None,
) -> tuple[float, float]:
    """
    Evaluate lane-departure risk over a predicted ego trajectory.

    Inputs:
      trajectory:     sequence of TrajectoryStep(x, y, v, var_x, var_y,
                       cov_xy), one per predicted step t = 0..N, in
                       chronological order.
      lane:           LaneCenterline(coeffs, cov) fit against which
                       departure is measured, or None if no valid centerline
                       is currently available -- returns (0.0, 0.0), same as
                       the node's "invalid centerline forces P_LD=0, R=0".
      vehicle_params: mass -- VehicleParameters.m; defaults to
                       VehicleParameters() if None.
      params:         lane/vehicle geometry + fallback uncertainty --
                       DepartureParams; defaults to DepartureParams() if
                       None.

    Returns (prob, score):
      prob  = max_t P_LD(t) over the trajectory, where
              D(t) = y(t) - y_C(x(t)) is the ego's lateral clearance from
              the centerline and
                P_LD(t) = 1 - [Φ((h-mu_D)/sigma_D) - Φ((-h-mu_D)/sigma_D)],
                h = (lane_width_m - vehicle_width_m) / 2.
      score = cumulative lane-departure risk R -- PARALO §5.2.5 eq. (5.51),
              kinetic-energy severity only (no weight applied, same as the
              node -- multiply by the real w_departure yourself):
                S(t)  = 0.5 * m * v(t)^2
                S_max = max_t S(t)
                R     = S_max * (1 - prod_t (1 - (S(t)/S_max) * P_LD(t)))
    """
    if lane is None:
        return 0.0, 0.0

    vp = vehicle_params or VehicleParameters()
    p = params or DepartureParams()

    half_lane = p.lane_width_m / 2.0
    half_veh = p.vehicle_width_m / 2.0
    h = half_lane - half_veh
    assert h > 0.0, (
        f"invalid lane geometry: lane_width_m={p.lane_width_m} <= "
        f"vehicle_width_m={p.vehicle_width_m}"
    )

    Sigma_theta, cov_ok = sanitize_cov3(lane.cov)
    if not cov_ok:
        return 0.0, 0.0

    fallback_var = p.default_sigma_lateral_m ** 2
    sigma_floor2 = p.min_sigma_y_m ** 2

    p_list: list[float] = []   # P_LD(t)
    s_list: list[float] = []   # S(t) -- kinetic-energy severity, no weight

    for step in trajectory:
        var_x = step.var_x if step.var_x is not None else fallback_var
        var_y = step.var_y if step.var_y is not None else fallback_var

        # ── D(t) = y(t) - y_C(x(t)), variance first-order propagated from
        # the centerline fit (Sigma_theta) and this step's position
        # covariance, assumed independent of each other ──────────────────
        var_theta = propagate_poly_variance(Sigma_theta, step.x)
        g_x = eval_slope(lane.coeffs, step.x)
        mu_D = step.y - eval_quad(lane.coeffs, step.x)
        var_D = var_theta + g_x * g_x * var_x + var_y - 2.0 * g_x * step.cov_xy
        sigma_D = math.sqrt(max(var_D, sigma_floor2))

        p_inside = normal_cdf((h - mu_D) / sigma_D) - normal_cdf((-h - mu_D) / sigma_D)
        p_ld = clamp(1.0 - p_inside, 0.0, 1.0)

        p_list.append(p_ld)
        s_list.append(0.5 * vp.m * step.v * step.v)

    max_p = max(p_list, default=0.0)
    score = _cumulative_risk(s_list, p_list)
    return max_p, score
