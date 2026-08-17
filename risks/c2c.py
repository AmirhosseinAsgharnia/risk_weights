"""
Car-to-car (C2C) risk — predictive collision probability between the ego
vehicle and a set of surrounding tracked objects, over predicted
trajectories.

Pure-function port of the former ROS `c2c_node`: same physics (independent-
axis AABB rectangle-overlap probability, PARALO kinetic-energy-of-closure
severity) and the same cumulative-risk reduction generalized over every
(object, step) hypothesis, but with ROS messaging/QoS/visualisation and the
ROS-message-absence heuristics (time-based ego uncertainty growth model,
raw-array covariance parsing/repair) stripped out, and the interface swapped
from pub/sub callbacks to a plain function over explicit trajectories. The
node was a suggestion for the calculation, not something to copy wholesale.

Weighting (w_c2c) is NOT applied here — c2c() returns the *unweighted*
(weight=1) probability/score pair; the caller multiplies in its own w_c2c
(the node itself did apply w_c2c to score, unlike departure_node -- this is
the one channel where the caller now takes over work the node used to do).

Coordinate frame — SAE J670 throughout (+x forward, +y right).
"""

import math
from typing import NamedTuple, Sequence

from model.car.config import VehicleParameters

_EPS = 1e-9

_DEFAULT_OBJ_LENGTH_M = 4.5
_DEFAULT_OBJ_WIDTH_M = 1.85


class EgoStep(NamedTuple):
    """One predicted step of the ego trajectory, ego frame, SAE J670.

    x:       [m] forward position.
    y:       [m] lateral position, +right.
    v:       [m/s] speed.
    heading: [rad] heading, SAE (+right/CW); 0.0 fallback when unknown.
    var_x:   [m^2] forward-position variance, or None to fall back to
             C2CParams.default_sigma_m^2 (isotropic).
    var_y:   [m^2] lateral-position variance, or None for the same fallback.
    """
    x: float
    y: float
    v: float
    heading: float = 0.0
    var_x: float | None = None
    var_y: float | None = None


class ObjectStep(NamedTuple):
    """One predicted step of a tracked object's trajectory, ego frame.

    x:       [m] forward position.
    y:       [m] lateral position, +right.
    v:       [m/s] speed.
    heading: [rad] heading, SAE (+right/CW); 0.0 fallback (same-direction)
             when unknown.
    sigma_x: [m] 1-sigma forward position std (NOT variance); 0.0 (treated
             as deterministic) when unknown.
    sigma_y: [m] 1-sigma lateral position std; 0.0 when unknown.
    """
    x: float
    y: float
    v: float
    heading: float = 0.0
    sigma_x: float = 0.0
    sigma_y: float = 0.0


class ObjectTrajectory(NamedTuple):
    """A tracked object's predicted trajectory and footprint.

    track_id:  identifies the object (diagnostic only, not used in the math).
    length_m:  [m] object length; <= 0 falls back to _DEFAULT_OBJ_LENGTH_M.
    width_m:   [m] object width; <= 0 falls back to _DEFAULT_OBJ_WIDTH_M.
    steps:     sequence of ObjectStep, one per predicted step, aligned in
               time with the ego trajectory (same indexing, t = 0..N).
    """
    track_id: int
    length_m: float
    width_m: float
    steps: Sequence[ObjectStep]


class C2CParams(NamedTuple):
    """Tuning independent of vehicle mass (see VehicleParameters.m).

    ego_length_m:    [m] ego bounding-box length used for the AABB.
    ego_width_m:     [m] ego bounding-box width used for the AABB.
    default_sigma_m: [m] fallback isotropic position std used for an ego
                     step whose var_x/var_y isn't given.
    var_floor:       [m^2] variance floor below which an axis is treated as
                     deterministic instead of applying the Gaussian interval
                     formula with an artificially inflated sigma.
    """
    ego_length_m: float = 4.5
    ego_width_m: float = 1.85
    default_sigma_m: float = 0.30
    var_floor: float = 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ──────────────────────────────────────────────────────────────────────────────

def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def normal_cdf(z: float) -> float:
    """Φ(z) via erfc for numerical stability at large |z|."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def axis_interval_probability(mu: float, var: float, h: float, var_floor: float) -> float:
    """
    P(-h <= X <= h) for X ~ N(mu, var), one axis of the relative-position
    Gaussian. Near-zero variance (<= var_floor) uses the exact deterministic
    step function instead of letting the numerical floor fabricate spurious
    probability:
        P = 1 if -h <= mu <= h else 0.
    Ordinary nonzero variance uses the Gaussian interval formula. Clamped
    only against floating-point drift.
    """
    if var <= var_floor:
        return 1.0 if -h <= mu <= h else 0.0
    sigma = math.sqrt(var)
    p_inside = normal_cdf((h - mu) / sigma) - normal_cdf((-h - mu) / sigma)
    return clamp(p_inside, 0.0, 1.0)


def rectangle_overlap_probability(
    mu_rx: float, var_rx: float,
    mu_ry: float, var_ry: float,
    hx: float, hy: float,
    var_floor: float,
) -> tuple[float, float, float]:
    """
    Independent-axis AABB rectangle-overlap probability — the probability
    that the relative-position random variable R=(Rx,Ry) lies inside the
    [-hx,hx] x [-hy,hy] collision rectangle, treating Rx and Ry as
    independent (off-diagonal relative covariance is ignored — no bivariate
    normal CDF is used here). Returns (p_long, p_lat, p_collision).
    """
    p_long = axis_interval_probability(mu_rx, var_rx, hx, var_floor)
    p_lat = axis_interval_probability(mu_ry, var_ry, hy, var_floor)
    return p_long, p_lat, clamp(p_long * p_lat, 0.0, 1.0)


def paralo_severity(m_e: float, v_e: float, v_j: float, psi_e: float, psi_j: float) -> float:
    """
    PARALO C2C severity surrogate:
        S = 0.5 * m_e * (v_e - v_j * cos(psi_j - psi_e))^2
    Always nonnegative (squared). No additional vulnerability factor.
    """
    dv = v_e - v_j * math.cos(psi_j - psi_e)
    return 0.5 * m_e * dv * dv


def _cumulative_risk(s_list: list[float], p_list: list[float]) -> float:
    """
    PARALO cumulative C2C risk across every (object, step) hypothesis:
        S_max = max_{j,k} S_j,k
        q_j,k = clamp((S_j,k / S_max) * P_j,k, 0, 1)
        R     = S_max * (1 - prod_{j,k} (1 - q_j,k))
    computed in log-space via log1p for numerical stability; q=1 is handled
    explicitly so the result becomes S_max without an invalid log(0). This
    product treats adjacent prediction steps and different vehicle
    hypotheses as independent even though the underlying predicted states
    are physically correlated over time — documented per spec, no
    additional correction is applied.
    """
    if not s_list:
        return 0.0
    S_max = max(s_list)
    if S_max <= _EPS:
        return 0.0

    log_survival = 0.0
    for S, P in zip(s_list, p_list):
        q = clamp((S / S_max) * P, 0.0, 1.0)
        if q >= 1.0:
            return S_max
        log_survival += math.log1p(-q)
    return S_max * (1.0 - math.exp(log_survival))


# ──────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────────────

def c2c(
        ego_trajectory: Sequence[EgoStep],
        objects: Sequence[ObjectTrajectory],
        vehicle_params: VehicleParameters | None = None,
        params: C2CParams | None = None,
) -> tuple[float, float]:
    """
    Evaluate car-to-car collision risk between the ego vehicle and a set of
    surrounding tracked objects, over predicted trajectories.

    Inputs:
      ego_trajectory: sequence of EgoStep(x, y, v, heading, var_x, var_y),
                       one per predicted step t = 0..N, in chronological
                       order.
      objects:         sequence of ObjectTrajectory, one per tracked object,
                       each with its own .steps sequence aligned in time
                       with ego_trajectory (indexed the same way; a shorter
                       object trajectory is truncated to the common length).
      vehicle_params:  mass — VehicleParameters.m; defaults to
                       VehicleParameters() if None.
      params:          ego footprint + fallback uncertainty — C2CParams;
                       defaults to C2CParams() if None.

    Returns (prob, score):
      prob  = max_{j,k} P_C2C,j,k over every object and predicted step.
      score = cumulative C2C risk R, generalizing PARALO §5.2.5 eq. (5.51)
              from a single trajectory to every (object, step) hypothesis,
              with no weight applied (multiply by the real w_c2c yourself):
                S_max = max_{j,k} S_j,k
                R     = S_max * (1 - prod_{j,k} (1 - (S_j,k/S_max) * P_j,k))
              S_j,k is the kinetic-energy-of-closure severity
              (paralo_severity) and P_j,k the rectangle-overlap probability
              (rectangle_overlap_probability) at object j, step k.
    """
    vp = vehicle_params or VehicleParameters()
    p = params or C2CParams()

    ego_half_length = p.ego_length_m / 2.0
    ego_half_width = p.ego_width_m / 2.0

    all_s: list[float] = []
    all_p: list[float] = []
    per_object_max: list[float] = []

    for obj in objects:
        obj_length_m = obj.length_m if obj.length_m > _EPS else _DEFAULT_OBJ_LENGTH_M
        obj_width_m = obj.width_m if obj.width_m > _EPS else _DEFAULT_OBJ_WIDTH_M
        hx = ego_half_length + obj_length_m / 2.0
        hy = ego_half_width + obj_width_m / 2.0

        n = min(len(ego_trajectory), len(obj.steps))
        obj_p_list: list[float] = []

        for k in range(n):
            e = ego_trajectory[k]
            o = obj.steps[k]

            var_ex = e.var_x if e.var_x is not None else p.default_sigma_m ** 2
            var_ey = e.var_y if e.var_y is not None else p.default_sigma_m ** 2

            # Relative position distribution: independent ego/object
            # estimators (no cross-covariance is available), so variances
            # add. sigma_x/sigma_y are 1-sigma std, squared into variance.
            mu_rx = o.x - e.x
            mu_ry = o.y - e.y
            var_rx = var_ex + o.sigma_x ** 2
            var_ry = var_ey + o.sigma_y ** 2

            _, _, p_collision = rectangle_overlap_probability(
                mu_rx, var_rx, mu_ry, var_ry, hx, hy, p.var_floor,
            )
            S = paralo_severity(vp.m, max(0.0, e.v), max(0.0, o.v), e.heading, o.heading)

            obj_p_list.append(p_collision)
            all_p.append(p_collision)
            all_s.append(S)

        if obj_p_list:
            per_object_max.append(max(obj_p_list))

    max_p = max(per_object_max, default=0.0)
    score = _cumulative_risk(all_s, all_p)
    return max_p, score
