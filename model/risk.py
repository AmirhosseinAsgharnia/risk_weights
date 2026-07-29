"""
Continuous, non-terminating risk evaluation over a PREDICTED trajectory.

Pure and side-effect-free: given predicted trajectories + covariances, this
module returns a per-step risk profile. It never terminates the simulation,
mutates an actor, gates a controller, or samples (no RNG anywhere). That's
deliberate separation of concerns: hard/binary safety events (collision,
road departure as a stop condition) live in model.collision; this module is
the graded, differentiable cost an MPC loop would integrate over a candidate
control rollout to compare "how risky was this trajectory", not a checker
that stops anything. The instantaneous (single-state) case is just N = 0 --
a length-1 trajectory.

RISK SET (all graded, all non-terminating): slip, rollover, lane departure,
road departure, car-to-car. Every probability is P = Phi(g/sigma), where g is
a signed margin (positive = safe) and sigma is DERIVED from the propagated
covariance via the delta method (sigma_g = sqrt(grad_g . P . grad_g^T)) --
never a free/tuned knob. The only tunables are the process/seed noise
matrices in RiskParams and a small sigma_min floor per risk (avoids
divide-by-zero at t=0, where the covariance may be ~0).
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.special import erf, logsumexp

from model.road import RoadScenario
from model.car.config import VehicleParameters
from model.car.model import EgoModel, Propagator
from model.car_init import frenet_to_global
from model.collision import rectangle_corners, min_separation, check_singularity_risk

G = 9.81
_ABS_SMOOTH_EPS = 1e-9  # [m/s^2] smooth-abs regularizer, see rollover_risk. Must
# stay well below the a_y perturbation scale numerical_jacobian's finite-difference
# step actually probes (~1e-6 here), or the smoothing swamps the gradient signal
# everywhere instead of only exactly at the a_y=0 kink it's meant to round off.


# ---------- numerical Jacobian (shared utility) ----------

def numerical_jacobian(f, x0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Central-difference Jacobian of f at x0. Used instead of hand-derived
    analytic partials because EgoModel's dynamics (saturated tire forces,
    RK4 integration) and frenet_to_global (table lookup) aren't worth -- or
    in the lookup case, aren't cleanly possible -- to differentiate by hand.
    Standard practice for EKFs over complex nonlinear maps.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    f0 = np.atleast_1d(np.asarray(f(x0), dtype=np.float64))
    n = x0.shape[0]
    m = f0.shape[0]
    J = np.zeros((m, n), dtype=np.float64)
    for i in range(n):
        dx = np.zeros(n, dtype=np.float64)
        dx[i] = eps
        f_plus = np.atleast_1d(np.asarray(f(x0 + dx), dtype=np.float64))
        f_minus = np.atleast_1d(np.asarray(f(x0 - dx), dtype=np.float64))
        J[:, i] = (f_plus - f_minus) / (2.0 * eps)
    return J


@dataclass
class EKFPropagator:
    """EKF time-update ONLY (no measurement update -- there's no measurement
    over a forecast): P(t+1) = F P(t) F^T + Q, F = numerical Jacobian of
    step_fn at the current mean. Covariance grows monotonically with
    horizon, as it should for a pure forecast."""

    Q: np.ndarray

    def propagate(self, mean: np.ndarray, cov: np.ndarray, step_fn) -> tuple[np.ndarray, np.ndarray]:
        F = numerical_jacobian(step_fn, mean)
        new_mean = np.asarray(step_fn(mean), dtype=np.float64)
        new_cov = F @ cov @ F.T + self.Q
        return new_mean, new_cov


# ---------- trajectory builders ----------

def _kappa_at(road: RoadScenario, s: float) -> float:
    idx = min(max(int(round(s / road.ds)), 0), len(road.s) - 1)
    return float(road.kappa[idx])


def _mu_at(road: RoadScenario, s: float) -> float:
    idx = min(max(int(round(s / road.ds)), 0), len(road.s) - 1)
    return float(road.mu[idx])


def propagate_ego_trajectory(x0, P0, controls, road: RoadScenario, vehicle_params: VehicleParameters,
                              propagator: Propagator, dt: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Roll the ego mean/covariance forward under a candidate control sequence
    `controls` (list of (a_x, delta), length N). kappa(s)/mu(s) are sampled
    at the start of each step and held constant over that step (same
    simplification model.simulation's real step loop already makes).

    Returns [(mean, cov)] for t = 0..N (N+1 entries, including the t=0 seed
    -- so N=0 (no controls) is just [(x0, P0)]).
    """
    model = EgoModel(vehicle_params)
    mean = np.asarray(x0, dtype=np.float64)
    cov = np.asarray(P0, dtype=np.float64)
    trajectory = [(mean, cov)]

    for u in controls:
        s = mean[EgoModel.IDX_S]
        kappa_s = _kappa_at(road, s)
        mu_s = _mu_at(road, s)

        def step_fn(x, u=u, kappa_s=kappa_s, mu_s=mu_s):
            return model.step(x, u, kappa_s, mu_s, dt)

        mean, cov = propagator.propagate(mean, cov, step_fn)
        trajectory.append((mean, cov))

    return trajectory


# ---------- frame handling ----------

def _frenet_to_global_fn(road: RoadScenario):
    def f(x):
        gx, gy, gpsi = frenet_to_global(road, x[0], x[1], x[2])
        return np.array([gx, gy, gpsi], dtype=np.float64)
    return f


def frenet_to_global_with_cov(ego_mean: np.ndarray, ego_cov: np.ndarray, road: RoadScenario,
                               singularity_threshold: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """
    (global_mean [x,y,psi], global_cov 3x3), covariance carried through via
    P_global = J P J^T, J = Jacobian of frenet_to_global at ego_mean.
    Skipping J would make the ego uncertainty ellipse wrong on curves --
    exactly where car-to-car risk matters most.

    Guards the Frenet singularity (1 - kappa*e_y -> 0, same term as
    collision.check_singularity_risk) with a HARD raise rather than silently
    emitting a garbage Jacobian -- note this guard threshold (0.2, "getting
    numerically dangerous") is deliberately tighter than
    check_singularity_risk's own default (0.8, an early-warning log
    threshold used elsewhere) since a healthy curve with denom=0.8 is nowhere
    near actually singular.
    """
    s, e_y = float(ego_mean[0]), float(ego_mean[1])
    kappa_s = _kappa_at(road, s)
    if check_singularity_risk(kappa_s, e_y, threshold=singularity_threshold):
        raise ValueError(
            f"Frenet singularity guard triggered at s={s}, e_y={e_y}, kappa={kappa_s}: "
            f"|1 - kappa*e_y| is within {singularity_threshold} of 0 -- refusing to map to "
            "global coordinates rather than emit a garbage Jacobian."
        )
    f = _frenet_to_global_fn(road)
    mean_global = f(ego_mean)
    J = numerical_jacobian(f, ego_mean)  # 3x6
    cov_global = J @ ego_cov @ J.T
    return mean_global, cov_global


# ---------- Phi ----------

def _phi(x):
    """Standard normal CDF via erf, vectorized."""
    return 0.5 * (1.0 + erf(np.asarray(x, dtype=np.float64) / np.sqrt(2.0)))


# ---------- params / output ----------

def _default_Q_ego():
    return np.diag([0.02, 0.01, 0.002, 0.05, 0.05, 0.01]).astype(np.float64)


def _default_R_ego():
    return np.diag([0.1, 0.05, 0.01, 0.2, 0.1, 0.02]).astype(np.float64)


def _default_Q_ctrv():
    return np.diag([0.05, 0.05, 0.01, 0.1, 0.02]).astype(np.float64)


def _default_R_ctrv():
    return np.diag([0.2, 0.2, 0.02, 0.2, 0.02]).astype(np.float64)


@dataclass(frozen=True)
class RiskParams:
    # Ego EKF (6x6, Frenet state order): time-update-only, so there's no
    # classical measurement noise here. R_ego is read as P(0), the seed
    # covariance (e.g. current localization/state-estimation uncertainty
    # handed in from outside); Q_ego is the per-step process noise added
    # during propagate_ego_trajectory. Same convention for neighbours (5x5,
    # CTRV state order), per neighbour.
    Q_ego: np.ndarray = field(default_factory=_default_Q_ego)
    R_ego: np.ndarray = field(default_factory=_default_R_ego)
    Q_ctrv: np.ndarray = field(default_factory=_default_Q_ctrv)
    R_ctrv: np.ndarray = field(default_factory=_default_R_ctrv)

    sigma_min_slip: float = 0.02
    sigma_min_roll: float = 0.05
    sigma_min_lane: float = 0.02
    sigma_min_road: float = 0.02
    sigma_min_c2c: float = 0.1

    d_safe: float = 2.0     # [m] default: half an ego length
    lse_beta: float = 8.0   # [-] LSE smooth-max temperature for c2c neighbour aggregation
    singularity_threshold: float = 0.2  # see frenet_to_global_with_cov


@dataclass(frozen=True)
class RiskProfile:
    p_slip: np.ndarray
    p_roll: np.ndarray
    p_lane: np.ndarray
    p_road: np.ndarray
    p_c2c: np.ndarray

    g_slip: np.ndarray
    g_roll: np.ndarray
    g_lane: np.ndarray
    g_road: np.ndarray
    g_c2c: np.ndarray

    sigma_slip: np.ndarray
    sigma_roll: np.ndarray
    sigma_lane: np.ndarray
    sigma_road: np.ndarray
    sigma_c2c: np.ndarray


# ---------- shared delta-method pattern (rollover, slip) ----------

def _delta_method_risk(g_fn, mean: np.ndarray, cov: np.ndarray, sigma_min: float) -> tuple[float, float, float]:
    g = float(np.atleast_1d(np.asarray(g_fn(mean), dtype=np.float64))[0])
    grad = numerical_jacobian(g_fn, mean)[0]
    variance = float(grad @ cov @ grad)
    sigma = max(float(np.sqrt(max(variance, 0.0))), sigma_min)
    prob = float(np.clip(_phi(g / sigma), 0.0, 1.0))
    return prob, g, sigma


# ---------- per-risk evaluators (single predicted step) ----------

def rollover_risk(ego_mean, ego_cov, vehicle_params: VehicleParameters, road: RoadScenario,
                   u, p: RiskParams | None = None) -> tuple[float, float, float]:
    """
    g_roll = |a_y| - SSF*g, a_y = (Fyf+Fyr)/m -- the same lateral specific
    force the tire model balances (NOT v^2*kappa; that would silently ignore
    slip/sideslip and understate risk exactly when it matters).
    p_roll = Phi(g_roll/sigma_roll). On a straight road a_y -> 0 so
    p_roll -> ~0 (not exactly 0, since sigma_min floors it slightly above).

    Uses a smooth-abs sqrt(a_y**2 + eps**2) instead of a bare abs(a_y):
    |x| has a kink exactly at x=0, where its central-difference derivative
    is identically 0 regardless of the true sensitivity either side of it --
    for a car driving perfectly straight (a_y=0 exactly) that silently zeroed
    out sigma_roll's dependence on the covariance entirely, contradicting the
    "no discontinuity in any probability" requirement. eps is tiny (1e-3
    m/s^2) relative to any real lateral acceleration.
    """
    p = p or RiskParams()
    s = float(ego_mean[EgoModel.IDX_S])
    kappa_s = _kappa_at(road, s)
    mu_s = _mu_at(road, s)
    model = EgoModel(vehicle_params)

    def g_fn(x):
        diag = model.tire_diagnostics(x, u, kappa_s, mu_s)
        a_y = (diag.Fyf + diag.Fyr) / vehicle_params.m
        a_y_smooth_abs = np.sqrt(a_y**2 + _ABS_SMOOTH_EPS**2)
        return np.array([a_y_smooth_abs - vehicle_params.SSF * G])

    return _delta_method_risk(g_fn, ego_mean, ego_cov, p.sigma_min_roll)


def slip_risk(ego_mean, ego_cov, vehicle_params: VehicleParameters, road: RoadScenario,
              u, p: RiskParams | None = None) -> tuple[float, float, float]:
    """
    g_slip = rho_max - 1, rho_max = max(front, rear) axle saturation ratio
    from EgoModel.tire_diagnostics (the same saturated-linear tire model
    state_space actually uses -- not a separate/simplified slip estimate).
    """
    p = p or RiskParams()
    s = float(ego_mean[EgoModel.IDX_S])
    kappa_s = _kappa_at(road, s)
    mu_s = _mu_at(road, s)
    model = EgoModel(vehicle_params)

    def g_fn(x):
        diag = model.tire_diagnostics(x, u, kappa_s, mu_s)
        return np.array([diag.saturation_ratio - 1.0])

    return _delta_method_risk(g_fn, ego_mean, ego_cov, p.sigma_min_slip)


def _two_sided_boundary_risk(e_y_mean: float, sigma_ey: float, left_edge: float, right_edge: float,
                              width: float, sigma_min: float) -> tuple[float, float, float]:
    """
    p = 1 - P(body fully inside [left_edge, right_edge]), body = interval of
    width `width` centred on the (Gaussian, mean e_y_mean, std sigma_ey)
    lateral position. Only e_y is random; the body width is deterministic.

        g_R = (right_edge - width/2) - e_y_mean   (margin to the right edge)
        g_L = e_y_mean - (left_edge + width/2)    (margin to the left edge)
        P(fully inside) = Phi(g_R/sigma) + Phi(g_L/sigma) - 1
        p = 2 - Phi(g_R/sigma) - Phi(g_L/sigma)

    Check: body's right edge exactly on the boundary -> g_R=0, g_L large ->
    p = 2 - 0.5 - 1 = 0.5, as required.
    """
    sigma = max(sigma_ey, sigma_min)
    g_R = (right_edge - width / 2.0) - e_y_mean
    g_L = e_y_mean - (left_edge + width / 2.0)
    p_inside = _phi(g_R / sigma) + _phi(g_L / sigma) - 1.0
    prob = float(np.clip(1.0 - p_inside, 0.0, 1.0))
    g = float(min(g_R, g_L))  # the binding (worse) margin, for diagnostics
    return prob, g, float(sigma)


def lane_departure_risk(ego_mean, ego_cov, road: RoadScenario, width: float, lane_index: int,
                         p: RiskParams | None = None) -> tuple[float, float, float]:
    """Two-sided, non-terminating -- crossing a lane line is a risk, not a
    stop condition (that distinction lives entirely in model.collision)."""
    p = p or RiskParams()
    e_y = float(ego_mean[EgoModel.IDX_EY])
    sigma_ey = float(np.sqrt(max(ego_cov[EgoModel.IDX_EY, EgoModel.IDX_EY], 0.0)))
    left_edge = road.d_c[lane_index] - road.l_w / 2.0
    right_edge = road.d_c[lane_index] + road.l_w / 2.0
    return _two_sided_boundary_risk(e_y, sigma_ey, left_edge, right_edge, width, p.sigma_min_lane)


def road_departure_risk(ego_mean, ego_cov, road: RoadScenario, width: float,
                         p: RiskParams | None = None) -> tuple[float, float, float]:
    """Same form as lane_departure_risk, road (outermost) boundaries. Kept as
    a separate function per the spec even though p_lane and p_road are both
    e_y-driven and will be hard to separate in an identifiability sense --
    that's flagged, not resolved, here."""
    p = p or RiskParams()
    e_y = float(ego_mean[EgoModel.IDX_EY])
    sigma_ey = float(np.sqrt(max(ego_cov[EgoModel.IDX_EY, EgoModel.IDX_EY], 0.0)))
    right_edge = road.d_c[0] + road.l_w / 2.0    # lane 0 = rightmost (road.py convention)
    left_edge = road.d_c[-1] - road.l_w / 2.0    # lane N-1 = leftmost
    return _two_sided_boundary_risk(e_y, sigma_ey, left_edge, right_edge, width, p.sigma_min_road)


def car_to_car_risk(ego_mean, ego_cov, road: RoadScenario, ego_dims: tuple[float, float],
                     neighbor_means, neighbor_covs, neighbor_dims,
                     p: RiskParams | None = None) -> tuple[float, list[tuple[float, float, float]]]:
    """
    Per neighbour: predicted min gap (collision.min_separation, SAT-based,
    reused not reimplemented) between the ego (projected to global) and the
    neighbour's oriented box; combined position uncertainty along the
    closing axis via the delta method; p_i = Phi((d_safe - g_min_i)/sigma_i).
    Neighbours aggregated with a smooth (LSE) max -- never a hard max, so
    the result stays differentiable in every input.

    Returns (p_aggregate, [(p_i, g_min_i, sigma_i), ...]) in input order.
    """
    p = p or RiskParams()
    ego_global_mean, ego_global_cov = frenet_to_global_with_cov(ego_mean, ego_cov, road, p.singularity_threshold)
    ego_pos_cov = ego_global_cov[:2, :2]
    ego_corners = rectangle_corners(ego_global_mean[0], ego_global_mean[1], ego_global_mean[2],
                                     ego_dims[0], ego_dims[1])

    per_neighbor = []
    for n_mean, n_cov, n_dims in zip(neighbor_means, neighbor_covs, neighbor_dims):
        n_corners = rectangle_corners(n_mean[0], n_mean[1], n_mean[2], n_dims[0], n_dims[1])
        g_min, axis = min_separation(ego_corners, n_corners)

        n_pos_cov = np.asarray(n_cov)[:2, :2]
        combined = ego_pos_cov + n_pos_cov
        variance = float(axis @ combined @ axis)
        sigma = max(float(np.sqrt(max(variance, 0.0))), p.sigma_min_c2c)

        prob = float(np.clip(_phi((p.d_safe - g_min) / sigma), 0.0, 1.0))
        per_neighbor.append((prob, float(g_min), sigma))

    if not per_neighbor:
        return 0.0, []

    probs = np.array([pi for pi, _, _ in per_neighbor], dtype=np.float64)
    agg = float(np.clip(logsumexp(p.lse_beta * probs) / p.lse_beta, 0.0, 1.0))
    return agg, per_neighbor


# ---------- top level ----------

def evaluate_risk_profile(
    ego_trajectory: list[tuple[np.ndarray, np.ndarray]],
    controls,
    neighbor_trajectories: list[list[tuple[np.ndarray, np.ndarray]]],
    neighbor_dims: list[tuple[float, float]],
    road: RoadScenario,
    vehicle_params: VehicleParameters,
    ego_dims: tuple[float, float],
    lane_index: int,
    p: RiskParams | None = None,
) -> RiskProfile:
    """
    ego_trajectory: [(mean(6,), cov(6,6))] for t=0..N, from
        propagate_ego_trajectory (or any source satisfying its contract).
    controls: [(a_x, delta)], SAME length as ego_trajectory -- the control
        associated with/evaluated at each predicted state (used by
        slip/rollover's tire-force evaluation). N=0 is just a length-1 list.
    neighbor_trajectories: one [(mean(5,), cov(5,5))] list per neighbour,
        each the same length as ego_trajectory (e.g. from
        propagate_ctrv_trajectory).
    lane_index: which lane's boundaries lane_departure_risk measures against
        (the road's outer boundaries, used for road_departure_risk, are
        unambiguous and need no such argument).
    """
    p = p or RiskParams()
    n_steps = len(ego_trajectory)

    p_slip = np.zeros(n_steps); g_slip = np.zeros(n_steps); sigma_slip = np.zeros(n_steps)
    p_roll = np.zeros(n_steps); g_roll = np.zeros(n_steps); sigma_roll = np.zeros(n_steps)
    p_lane = np.zeros(n_steps); g_lane = np.zeros(n_steps); sigma_lane = np.zeros(n_steps)
    p_road = np.zeros(n_steps); g_road = np.zeros(n_steps); sigma_road = np.zeros(n_steps)
    p_c2c = np.zeros(n_steps); g_c2c = np.zeros(n_steps); sigma_c2c = np.zeros(n_steps)

    for t in range(n_steps):
        ego_mean, ego_cov = ego_trajectory[t]
        u = controls[t]

        p_roll[t], g_roll[t], sigma_roll[t] = rollover_risk(ego_mean, ego_cov, vehicle_params, road, u, p)
        p_slip[t], g_slip[t], sigma_slip[t] = slip_risk(ego_mean, ego_cov, vehicle_params, road, u, p)
        p_lane[t], g_lane[t], sigma_lane[t] = lane_departure_risk(ego_mean, ego_cov, road, ego_dims[1],
                                                                    lane_index, p)
        p_road[t], g_road[t], sigma_road[t] = road_departure_risk(ego_mean, ego_cov, road, ego_dims[1], p)

        n_means = [traj[t][0] for traj in neighbor_trajectories]
        n_covs = [traj[t][1] for traj in neighbor_trajectories]
        agg, per_neighbor = car_to_car_risk(ego_mean, ego_cov, road, ego_dims, n_means, n_covs, neighbor_dims, p)
        p_c2c[t] = agg
        if per_neighbor:
            worst = max(per_neighbor, key=lambda pn: pn[0])
            g_c2c[t] = worst[1]
            sigma_c2c[t] = worst[2]
        else:
            g_c2c[t] = np.inf
            sigma_c2c[t] = p.sigma_min_c2c

    return RiskProfile(
        p_slip=p_slip, p_roll=p_roll, p_lane=p_lane, p_road=p_road, p_c2c=p_c2c,
        g_slip=g_slip, g_roll=g_roll, g_lane=g_lane, g_road=g_road, g_c2c=g_c2c,
        sigma_slip=sigma_slip, sigma_roll=sigma_roll, sigma_lane=sigma_lane,
        sigma_road=sigma_road, sigma_c2c=sigma_c2c,
    )
