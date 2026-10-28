"""
PARALO-based sampling MPC (docs/constrained_design.md §3). Every
`replan_every_steps` env steps it enumerates candidate ABSOLUTE input pairs
(a, delta), holds each over the whole horizon (subject to the env's own
jerk and steering-rate limits, modelled as ramps), predicts the ego and the
surrounding traffic, and commits to the candidate minimizing

    sum_{n=1..N} l(x_n, u; w)

    l = w_speed ((v - v_ref)/v_ref)^2 + w_accel (a/4)^2 + w_steer (delta/0.5)^2
      + w_c2c P_c2c + w_roll P_roll + w_slip P_slip + w_lane P_lane
      + c_jerk (j/10)^2 + c_latacc (a_y/4)^2 + c_road P_road

w (ParaloWeights, 7 numbers) is what the constrained minimax optimizer tunes;
c_jerk, c_latacc, c_road are FIXED controller parameters (ParaloMPCConfig).
The P terms are the per-state PARALO probabilities (learning.paralo_vec --
the risks/ models unchanged, vectorized).

Prediction:
  ego   steady-state single-track ("kinematic with understeer") in the ego's own
        lane Frenet frame, exactly the env's frame and sign conventions (SAE +right,
        kappa_dyn = -lane.kappa):
            s'    = v cos(e_psi) / (1 - kappa e_y)
            e_y'  = v sin(e_psi)
            r     = v tan(delta) / (L + K_us v^2)
            e_psi'= r - kappa s'
            v'    = a
        K_us = m/L (l_r/C_f - l_f/C_r) is the vehicle's understeer gradient. With
        K_us = 0 this is the plain kinematic bicycle; the simulated SUV understeers
        so strongly (K_us ~ 0.034 rad s^2/m) that the plain kinematic model
        over-predicts yaw rate ~6x at 20 m/s (docs/constrained_design.md §2.3), so the
        corrected form is the default (`understeer_gradient=None` -> computed).
  surr  constant velocity (longitudinal v_x, lateral v_x sin e_psi + v_y cos e_psi),
        backbone frame -- the same model learning.risk_eval used.

PARALO risk inputs (learning.vehicle_physics, the single conversion point): rollover and
slippage are kinematic-radius models, so by default they are given delta_eq =
atan(L r_pred / v) of the candidate's own PREDICTED yaw rate ("kinematic_equivalent")
-- never the held delta, which also contains the understeer angle K a_y and made both
read ~1 at 2 m/s^2 real lateral acceleration (docs/physics_audit.md §A2), and never
simulator ground truth. "applied" reproduces the paralo-mpc-v1 behaviour.
Road friction: the simulator uses the scenario's road_mu; the MPC is given mu_hat from
a FrictionInfoConfig (oracle by default: mu_hat = road_mu, the perfect-information
baseline). PARALO's slip-angle limit alpha_max_deg_at_mu_1 is exposed unchanged (8.6
deg, the paper value); the simulator's linear tire saturates at ~33.8 deg * mu (front),
a documented calibration mismatch -- change it only as a separately validated version.

Lane departure is measured against the lane centre NEAREST to each predicted
state (no target-lane variable exists); road departure against the paved road.
"""

import math
from dataclasses import dataclass, asdict, fields

import numpy as np

from initialization.traffic_init import CAR_LENGTH, CAR_WIDTH
from risks.departure import DepartureParams
from risks.slippage import SlippageParams
from learning.env import ACCEL_MIN, ACCEL_MAX, DELTA_MAX, NOMINAL_SPEED, MAX_JERK, MAX_STEERING_RATE
from learning.ego_control import EgoView
from learning.paralo_vec import p_c2c_pairs, p_slip_vec, p_departure_vec, p_rollover_vec
from learning.vehicle_physics import FrictionInfoConfig, PARALO_STEER_INPUTS, paralo_steer_input, understeer_gradient

PARALO_MPC_VERSION = "paralo-mpc-v2"   # v1: PARALO given the applied delta; no friction-information interface

# ── w: the tunable weights (name, lo, hi); all searched on a log scale ─────
PW_SPEC: tuple[tuple[str, float, float], ...] = (
    ("speed",  0.05, 20.0),
    ("accel",  0.005, 5.0),
    ("steer",  0.01, 100.0),
    ("c2c",    0.5, 500.0),
    ("roll",   0.5, 500.0),
    ("slip",   0.01, 100.0),
    ("lane",   0.01, 100.0),
)
PW_NAMES = tuple(n for n, *_ in PW_SPEC)
PW_LO = np.array([lo for _, lo, _ in PW_SPEC])
PW_HI = np.array([hi for _, _, hi in PW_SPEC])


@dataclass(frozen=True)
class ParaloWeights:
    speed: float = 1.0
    accel: float = 0.1
    steer: float = 1.0
    c2c: float = 20.0
    roll: float = 20.0
    slip: float = 1.0
    lane: float = 1.0

    def __post_init__(self):
        v = self.to_vector()
        assert np.all(v >= PW_LO - 1e-12) and np.all(v <= PW_HI + 1e-12), f"w outside PW_SPEC bounds: {self}"

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in PW_NAMES], dtype=float)

    @staticmethod
    def from_vector(v) -> "ParaloWeights":
        v = np.asarray(v, dtype=float)
        assert v.shape == (len(PW_NAMES),), v.shape
        return ParaloWeights(**{n: float(x) for n, x in zip(PW_NAMES, v)})


def pw_to_unit(w) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    return (np.log(w) - np.log(PW_LO)) / (np.log(PW_HI) - np.log(PW_LO))


def pw_from_unit(u) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return np.exp(np.log(PW_LO) + u * (np.log(PW_HI) - np.log(PW_LO)))


# ── Fixed controller configuration ─────────────────────────────────────────
@dataclass(frozen=True)
class ParaloMPCConfig:
    horizon_s: float = 3.0
    dt_pred_s: float = 0.1
    replan_every_steps: int = 2                      # env steps (DT = 0.05 s) -> replan every 0.1 s
    accel_levels: tuple[float, ...] = (-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0)
    steer_offsets: tuple[float, ...] = (0.0, 0.002, -0.002, 0.005, -0.005, 0.01, -0.01,
                                        0.02, -0.02, 0.04, -0.04, 0.08, -0.08)   # around the current delta
    v_ref_mps: float = NOMINAL_SPEED
    understeer_gradient: float | None = None         # [rad s^2/m]; None -> computed from VehicleParameters
    # fixed (non-optimized) stage-cost coefficients
    c_jerk: float = 0.1
    c_latacc: float = 1.0
    c_road: float = 20.0
    # normalizers (docs/proposal_paralo_mpc.md §2.1)
    accel_ref_mps2: float = 4.0
    steer_ref_rad: float = DELTA_MAX
    jerk_ref_mps3: float = 10.0
    latacc_ref_mps2: float = 4.0
    # PARALO model inputs
    sigma_ego_m: float = 0.3                         # risk modules' default position std
    sigma_obj_m: float = 0.3
    rollover_U_H_m: float = 0.1
    slip_mu_uncertainty: float = 0.05                # PARALO U_mu (model parameter, NOT estimation error)
    slip_alpha_max_deg_at_mu_1: float = 8.6          # PARALO paper value (uncalibrated to the simulator's tire)
    paralo_steer_input: str = "kinematic_equivalent" # see module docstring; "applied" = paralo-mpc-v1
    # friction information (learning.vehicle_physics.FrictionInfoConfig); mu_max of PARALO slippage = mu_hat
    friction_mode: str = "oracle"
    friction_bias: float = 0.0
    friction_noise_std: float = 0.0
    friction_margin: float = 0.0

    def __post_init__(self):
        assert self.horizon_s > 0 and self.dt_pred_s > 0 and self.replan_every_steps >= 1
        assert all(ACCEL_MIN <= a <= ACCEL_MAX for a in self.accel_levels)
        assert 0.0 in self.steer_offsets
        assert self.paralo_steer_input in PARALO_STEER_INPUTS
        self.friction_info()   # validates the friction fields

    def friction_info(self) -> FrictionInfoConfig:
        return FrictionInfoConfig(mode=self.friction_mode, bias=self.friction_bias,
                                  noise_std=self.friction_noise_std, margin=self.friction_margin)

    @property
    def n_pred(self) -> int:
        return max(1, int(round(self.horizon_s / self.dt_pred_s)))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["accel_levels"], d["steer_offsets"] = list(self.accel_levels), list(self.steer_offsets)
        return d

    @staticmethod
    def from_dict(d: dict) -> "ParaloMPCConfig":
        d = dict(d)
        for k in ("accel_levels", "steer_offsets"):
            if k in d:
                d[k] = tuple(d[k])
        known = {f.name for f in fields(ParaloMPCConfig)}
        assert set(d) <= known, f"unknown ParaloMPCConfig fields: {set(d) - known}"
        return ParaloMPCConfig(**d)


TERMS = ("speed", "accel", "steer", "c2c", "roll", "slip", "lane", "jerk", "latacc", "road")


class ParaloMPC:
    """EgoController -- see module docstring."""

    def __init__(self, config: ParaloMPCConfig, weights: ParaloWeights):
        self.cfg, self.w = config, weights
        self.solver_failures = 0
        self.n_plans = 0
        self.diag = {}

    def reset(self, view: EgoView, scenario_info: dict) -> None:
        self._step = 0
        self._cmd = (0.0, float(view.ego.state.delta))
        self.mu_hat = self.cfg.friction_info().estimate(view.road.mu, scenario_info.get("realization_seed"))
        self.solver_failures = 0
        self.n_plans = 0
        self.diag = {"slip_active": 0, "slip_indistinct": 0, "slip_saturated_all": 0, "slip_decisive": 0}

    def act(self, view: EgoView) -> tuple[float, float]:
        if self._step % self.cfg.replan_every_steps == 0:
            self._plan(view)
        self._step += 1
        return self._cmd

    # -- prediction + cost --
    def candidate_terms(self, view: EgoView) -> tuple[np.ndarray, np.ndarray, dict]:
        """(accel per candidate, delta per candidate, {term: per-candidate sum over the horizon}).
        Terms are UNWEIGHTED; see stage_weights()."""
        cfg, road, env = self.cfg, view.road, view.env
        st, vp = view.ego.state, view.ego.vehicle_params
        K = understeer_gradient(vp) if cfg.understeer_gradient is None else cfg.understeer_gradient
        a0, d0 = float(env._prev_accel), float(st.delta)

        d_t = np.clip(d0 + np.array(cfg.steer_offsets), -DELTA_MAX, DELTA_MAX)
        acc = np.repeat(np.array(cfg.accel_levels, dtype=float), len(d_t))
        dlt = np.tile(d_t, len(cfg.accel_levels))
        C, N, dt = len(acc), cfg.n_pred, cfg.dt_pred_s
        t = dt * np.arange(1, N + 1)
        a_n = a0 + np.clip(acc[:, None] - a0, -MAX_JERK * t, MAX_JERK * t)                 # (C, N) ramped
        d_n = d0 + np.clip(dlt[:, None] - d0, -MAX_STEERING_RATE * t, MAX_STEERING_RATE * t)

        lane = road.lanes[st.lane]
        s = np.full(C, st.s)
        e_y = np.full(C, st.e_y)
        e_psi = np.full(C, st.e_psi)
        v = np.full(C, max(st.v_x, 0.0))
        S, EY, V, R = (np.empty((C, N)) for _ in range(4))
        for n in range(N):
            idx = np.clip(np.round(s / road.ds).astype(int), 0, len(lane.kappa) - 1)
            kappa = -lane.kappa[idx]
            s_dot = v * np.cos(e_psi) / (1.0 - kappa * e_y)
            r = v * np.tan(d_n[:, n]) / (vp.L + K * v * v)
            s = s + s_dot * dt
            e_y = e_y + v * np.sin(e_psi) * dt
            e_psi = e_psi + (r - kappa * s_dot) * dt
            v = np.maximum(0.0, v + a_n[:, n] * dt)
            S[:, n], EY[:, n], V[:, n], R[:, n] = s, e_y, v, r
        Y = -lane.offset + EY                                                                # backbone frame

        # surrounding traffic, constant velocity (backbone frame)
        ag = [a.car.state for a in view.agents]
        s_j = np.array([c.s for c in ag])[:, None] + np.array([c.v_x for c in ag])[:, None] * t[None, :]
        y_j = (np.array([-road.lanes[c.lane].offset + c.e_y for c in ag])[:, None]
               + np.array([c.v_x * math.sin(c.e_psi) + c.v_y * math.cos(c.e_psi) for c in ag])[:, None] * t[None, :])
        var = cfg.sigma_ego_m ** 2 + cfg.sigma_obj_m ** 2
        p_c2c = p_c2c_pairs(S[:, None, :], Y[:, None, :], s_j[None], y_j[None], CAR_LENGTH, CAR_WIDTH, var, var)
        p_c2c = p_c2c.max(axis=1) if len(ag) else np.zeros((C, N))

        d_paralo = paralo_steer_input(cfg.paralo_steer_input, d_n, V, R, vp)
        p_roll = p_rollover_vec(V, d_paralo, vp, cfg.rollover_U_H_m)
        p_slip = p_slip_vec(V, d_paralo, a_n, vp, SlippageParams(mu_max=self.mu_hat, mu_uncertainty=cfg.slip_mu_uncertainty,
                                                                 alpha_max_deg_at_mu_1=cfg.slip_alpha_max_deg_at_mu_1))
        centers = np.array([-l.offset for l in road.lanes])
        nearest = centers[np.argmin(np.abs(Y[..., None] - centers), axis=-1)]
        p_lane = p_departure_vec(Y - nearest, DepartureParams(lane_width_m=road.l_w, vehicle_width_m=CAR_WIDTH,
                                                              default_sigma_lateral_m=cfg.sigma_ego_m))
        p_road = p_departure_vec(Y - centers.mean(), DepartureParams(lane_width_m=road.l_w * len(centers),
                                                                     vehicle_width_m=CAR_WIDTH,
                                                                     default_sigma_lateral_m=cfg.sigma_ego_m))
        jerk = np.diff(np.concatenate([np.full((C, 1), a0), a_n], axis=1), axis=1) / dt
        terms = {
            "speed": (((V - cfg.v_ref_mps) / cfg.v_ref_mps) ** 2).sum(1),
            "accel": ((a_n / cfg.accel_ref_mps2) ** 2).sum(1),
            "steer": ((d_n / cfg.steer_ref_rad) ** 2).sum(1),
            "c2c": p_c2c.sum(1), "roll": p_roll.sum(1), "slip": p_slip.sum(1), "lane": p_lane.sum(1),
            "jerk": ((jerk / cfg.jerk_ref_mps3) ** 2).sum(1),
            "latacc": ((V * R / cfg.latacc_ref_mps2) ** 2).sum(1),
            "road": p_road.sum(1),
        }
        return acc, dlt, terms

    def stage_weights(self) -> dict[str, float]:
        w, c = self.w, self.cfg
        return {"speed": w.speed, "accel": w.accel, "steer": w.steer, "c2c": w.c2c, "roll": w.roll,
                "slip": w.slip, "lane": w.lane, "jerk": c.c_jerk, "latacc": c.c_latacc, "road": c.c_road}

    def candidate_costs(self, view: EgoView):
        acc, dlt, terms = self.candidate_terms(view)
        sw = self.stage_weights()
        return acc, dlt, sum(sw[k] * terms[k] for k in TERMS), terms

    def _plan(self, view: EgoView) -> None:
        acc, dlt, cost, terms = self.candidate_costs(view)
        self.n_plans += 1
        if not np.all(np.isfinite(cost)):
            self.solver_failures += 1                     # documented fallback: brake, hold steering
            self._cmd = (ACCEL_MIN, float(view.ego.state.delta))
            return
        best = int(np.argmin(cost))                       # deterministic tie-break: first minimum
        self._cmd = (float(acc[best]), float(dlt[best]))
        # slippage-transition diagnostics: does the slip term separate the candidates, and does it decide?
        slip = terms["slip"]
        if slip.max() > 1e-3:
            self.diag["slip_active"] += 1
            self.diag["slip_indistinct"] += int(slip.max() - slip.min() < 1e-6)
            self.diag["slip_saturated_all"] += int(slip.min() >= 0.99 * self.cfg.n_pred)
            self.diag["slip_decisive"] += int(int(np.argmin(cost - self.w.slip * slip)) != best)


__all__ = ["PARALO_MPC_VERSION", "ParaloMPC", "ParaloMPCConfig", "ParaloWeights", "PW_NAMES", "PW_SPEC", "pw_to_unit", "pw_from_unit",
           "understeer_gradient", "TERMS"]
