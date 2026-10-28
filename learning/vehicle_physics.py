"""
The single place for steering/curvature conversions, realized tire-force
quantities, physical event detectors and PARALO risk inputs
(docs/physics_audit.md). Nothing here changes the simulator's dynamics:
every function reads the state CarDynamics already defines.

Commanded vs. realized:
  delta_cmd   the front-wheel angle the controller applied.
  kappa_real  = r / v, the realized path curvature (yaw rate over speed);
              0 below LOW_SPEED_MPS (curvature is undefined at standstill, and
              every v-dependent risk is negligible there).
  delta_eq    = atan(L kappa_real), the kinematic-bicycle steering angle that
              would produce that curvature without tire slip. This -- never
              delta_cmd -- is what a kinematic-radius model (PARALO rollover /
              slippage) must be given; delta_cmd also contains the understeer
              angle K a_y (K ~ 0.034 rad s^2/m for the default SUV).

Event detectors (versioned; historical results keep their own version):
  rollover_legacy_v1    P_roll(v_x, delta_cmd) > 0.5 -- the env's original
                        ROLLOVER event. Fires at a_y ~ 2 m/s^2 on the default
                        curve at 20 m/s (false positive; audit §A2).
  rollover_physical_v2  quasi-static rigid-body tipping onset |a_y| >= SSF g,
                        a_y the REALIZED body lateral acceleration
                        (F_yf + F_yr)/m. SSF = W_c/(2 h_CG) = 0.969 -> 9.50 m/s^2.
                        Reduced-order: no suspension roll, no tripping, no
                        dynamic overshoot. Since |a_y| <= mu g in this tire
                        model, it can only fire when mu >= SSF.
  slip_lateral_saturation  some axle's requested lateral force C alpha reaches
                        mu F_z (where the simulator clips it): the axle is in
                        the sliding regime of the linear-saturated tire.
  slip_friction_circle  vehicle-level combined demand
                        sqrt((m a_x)^2 + (F_yf,req + F_yr,req)^2) >= mu (F_zf + F_zr).
                        The simulator has NO longitudinal tire force (a_x is
                        applied directly), so this is the only place where a
                        braking-plus-cornering friction limit exists; it is an
                        indicator only, the dynamics are not limited by it.
"""

import math
from dataclasses import dataclass, asdict

import numpy as np

from model.car.config import VehicleParameters, G
from model.car.model import CarDynamics
from risks.rollover import rollover, TrajectoryStep as RolloverStep

VEHICLE_PHYSICS_VERSION = "vehicle-physics-v1"
ROLLOVER_EVENT_MODELS = ("legacy_v1", "physical_v2")
LEGACY_ROLLOVER_PROB_THRESHOLD = 0.5   # == learning.env.ROLLOVER_PROB_THRESHOLD (legacy_v1)
LOW_SPEED_MPS = 1.0
PARALO_STEER_INPUTS = ("applied", "kinematic_equivalent")


# ── steering / curvature ────────────────────────────────────────────────────
def understeer_gradient(vp: VehicleParameters) -> float:
    """[rad s^2/m] K = m/L (l_r/C_f - l_f/C_r); steady-state delta = L kappa + K a_y."""
    return vp.m / vp.L * (vp.l_r / vp.C_f - vp.l_f / vp.C_r)


def steady_state_yaw_rate(v, delta, vp: VehicleParameters, K: float | None = None):
    """r = v tan(delta) / (L + K v^2): the linear single-track steady state (K = 0: kinematic)."""
    K = understeer_gradient(vp) if K is None else K
    v = np.asarray(v, dtype=float)
    return v * np.tan(delta) / (vp.L + K * v * v)


def realized_curvature(v, r, v_min: float = LOW_SPEED_MPS):
    """kappa = r / v, 0 where |v| < v_min (same sign as r: SAE +right)."""
    v, r = np.asarray(v, dtype=float), np.asarray(r, dtype=float)
    ok = np.abs(v) >= v_min
    return np.where(ok, r / np.where(ok, np.abs(v), 1.0), 0.0)


def delta_equivalent(kappa, vp: VehicleParameters):
    """Kinematic-bicycle steering angle for path curvature kappa."""
    return np.arctan(vp.L * np.asarray(kappa, dtype=float))


def paralo_steer_input(mode: str, delta_applied, v, r, vp: VehicleParameters):
    """The steering angle a kinematic-radius PARALO model is given.
    "applied": the held/applied delta (legacy; wrong for an understeering vehicle).
    "kinematic_equivalent": delta_eq of the (predicted or realized) yaw rate r --
    the MPC passes its own PREDICTED r, the evaluator the simulator's realized r."""
    assert mode in PARALO_STEER_INPUTS, mode
    if mode == "applied":
        return np.asarray(delta_applied, dtype=float)
    return delta_equivalent(realized_curvature(v, r), vp)


# ── realized tire quantities (read off CarDynamics, never re-modelled) ──────
@dataclass(frozen=True)
class TireState:
    F_yf: float          # [N] realized (clipped) lateral forces
    F_yr: float
    F_yf_req: float      # [N] requested C alpha (before clipping)
    F_yr_req: float
    F_zf: float          # [N] normal loads incl. longitudinal transfer
    F_zr: float
    a_x: float           # [m/s^2] applied longitudinal acceleration
    a_y: float           # [m/s^2] realized body lateral acceleration (F_yf + F_yr)/m = dv_y/dt + r v_x
    sideslip_rad: float  # atan2(v_y, |v_x|)
    mu: float

    def to_dict(self) -> dict:
        return asdict(self)


def tire_state(state_vec, accel: float, delta: float, kappa: float, mu: float, vp: VehicleParameters) -> TireState:
    """state_vec = CarState.as_array(); kappa in the dynamics' sign convention (-lane.kappa)."""
    d = CarDynamics(np.asarray(state_vec, dtype=float), accel, delta, kappa, mu)
    d.p = vp
    v_x, v_y = float(state_vec[3]), float(state_vec[4])
    F_yf, F_yr = float(d.F_yf), float(d.F_yr)
    return TireState(F_yf, F_yr, float(vp.C_f * d.alpha_f), float(vp.C_r * d.alpha_r), float(d.F_zf), float(d.F_zr),
                     float(accel), (F_yf + F_yr) / vp.m, math.atan2(v_y, max(abs(v_x), 1e-6)), float(mu))


def slip_lateral_saturation(ts: TireState) -> bool:
    return abs(ts.F_yf_req) >= ts.mu * ts.F_zf or abs(ts.F_yr_req) >= ts.mu * ts.F_zr


def friction_demand(ts: TireState, vp: VehicleParameters) -> float:
    """Requested combined force / available (vehicle level); >= 1 means the friction circle is exceeded."""
    return math.hypot(vp.m * ts.a_x, ts.F_yf_req + ts.F_yr_req) / max(ts.mu * (ts.F_zf + ts.F_zr), 1e-9)


def slip_friction_circle(ts: TireState, vp: VehicleParameters) -> bool:
    return friction_demand(ts, vp) >= 1.0


# ── rollover ────────────────────────────────────────────────────────────────
def rollover_accel_threshold(vp: VehicleParameters, g: float = G) -> float:
    """[m/s^2] SSF g = W_c g / (2 h_CG): quasi-static rigid-body tipping onset."""
    return vp.SSF * g


def p_roll_legacy(v_x: float, delta_cmd: float, vp: VehicleParameters) -> float:
    return rollover([RolloverStep(v=v_x, delta=delta_cmd)], vehicle_params=vp)[0]


def rollover_legacy_v1(v_x: float, delta_cmd: float, vp: VehicleParameters) -> bool:
    return p_roll_legacy(v_x, delta_cmd, vp) > LEGACY_ROLLOVER_PROB_THRESHOLD


def rollover_physical_v2(ts: TireState, vp: VehicleParameters) -> bool:
    return abs(ts.a_y) >= rollover_accel_threshold(vp)


# ── friction information available to the controller ───────────────────────
FRICTION_MODES = ("oracle", "estimated", "conservative")


@dataclass(frozen=True)
class FrictionInfoConfig:
    """mu_hat the CONTROLLER is given (the simulator always uses the scenario's true road_mu).
    oracle:       mu_hat = mu_true (perfect information; the initial-experiment baseline)
    estimated:    mu_hat = clip(mu_true + e, mu_min, mu_max), e = bias + N(0, noise_std) drawn ONCE per
                  episode from the realization seed (constant over the episode)
    conservative: mu_hat = clip(mu_est - margin, mu_min, mu_max), mu_est the 'estimated' value
    U_mu (PARALO's friction uncertainty) is a separate, fixed model parameter -- not e."""
    mode: str = "oracle"
    bias: float = 0.0
    noise_std: float = 0.0
    margin: float = 0.0
    mu_min: float = 0.1
    mu_max: float = 1.2

    def __post_init__(self):
        assert self.mode in FRICTION_MODES, self.mode
        assert self.noise_std >= 0 and self.margin >= 0 and 0 < self.mu_min < self.mu_max
        assert self.mode != "oracle" or (self.bias == 0 and self.noise_std == 0 and self.margin == 0), \
            "oracle friction mode takes no bias/noise/margin -- choose 'estimated' or 'conservative'"

    def estimate(self, mu_true: float, seed: int | None) -> float:
        if self.mode == "oracle":
            return float(mu_true)
        e = self.bias + (np.random.default_rng(seed).normal(0.0, self.noise_std) if self.noise_std > 0 else 0.0)
        mu = mu_true + e - (self.margin if self.mode == "conservative" else 0.0)
        return float(np.clip(mu, self.mu_min, self.mu_max))


__all__ = ["VEHICLE_PHYSICS_VERSION", "ROLLOVER_EVENT_MODELS", "LOW_SPEED_MPS", "PARALO_STEER_INPUTS",
           "understeer_gradient", "steady_state_yaw_rate", "realized_curvature", "delta_equivalent",
           "paralo_steer_input", "TireState", "tire_state", "slip_lateral_saturation", "friction_demand",
           "slip_friction_circle", "rollover_accel_threshold", "p_roll_legacy", "rollover_legacy_v1",
           "rollover_physical_v2", "FRICTION_MODES", "FrictionInfoConfig"]
