"""
PROPOSED ego MPC (a research default, not an established requirement --
see docs/minimax_design.md §2): a sampling-based MPC that, every
`replan_every_steps` env steps, enumerates a finite set of candidate plans
(constant longitudinal acceleration a_c x target lane l_c), predicts each
over a short horizon with a simplified model, scores it with the weighted
internal cost L_MPC(w), and commits to the cheapest. Every env step, the
committed target lane is tracked by the repository's own far-near steering
law + curvature feedforward (controllers.far_near) -- the same law, sign
conventions and moderate-preset gains surrounding traffic already uses.

Why sampling, not gradient-based: the risk terms (box overlap, P_roll) are
non-smooth, the lane decision is discrete, and enumeration can never fail to
return an answer -- "solver failure" reduces to a non-finite cost, handled
by a documented fallback (brake at ACCEL_MIN, hold nearest lane).

Prediction model (deliberately simpler than the simulator -- a model
mismatch is normal for MPC and is what the closed-loop evaluation measures):
  ego:  v_k = max(0, v_0 + a_c t_k), s by forward-Euler on v;
        backbone lateral offset y_k = y_t + (y_0 - y_t) exp(-t_k / T_lat);
        steering implied by path curvature kappa_SAE(s_k) + y''_k / v_k^2.
  surr: constant longitudinal speed, constant lateral velocity
        (v_x sin(e_psi) + v_y cos(e_psi)), clipped to the paved road.
Frames: SAE J670 throughout (+e_y right), backbone e_y = -lane.offset + e_y,
road kappa negated exactly as EgoTrafficEnv does before Car.step.

w (MPCWeights) is the ONLY thing the minimax optimizer tunes. MPCConfig
(horizon, candidate grid, prediction constants) is fixed per experiment.
"""

import math
from dataclasses import dataclass, field, asdict, fields

import numpy as np
from scipy.special import ndtr

from controllers.far_near import (
    far_near_lookahead_offset, far_near_steering, far_near_curvature_feedforward, FAR_NEAR_PRESETS,
)
from initialization.traffic_init import CAR_LENGTH, CAR_WIDTH
from model.car.config import VehicleParameters
from risks.rollover import RolloverParams
from learning.env import ACCEL_MIN, ACCEL_MAX, DELTA_MAX, NOMINAL_SPEED, DT
from learning.ego_control import EgoView

_EPS = 1e-9
_R_MAX = 1.0e6


# ── w: the tunable parameter vector ────────────────────────────────────────
# (name, lo, hi, scale). "log" components are searched log-uniformly by the
# optimizer (see to_unit/from_unit); lower bounds are strictly positive so no
# safety term can be switched off entirely.
W_SPEC: tuple[tuple[str, float, float, str], ...] = (
    ("speed",          0.05,  20.0,  "log"),   # speed-tracking weight
    ("accel",          0.005,  5.0,  "log"),   # control effort (accel magnitude + change)
    ("lane_change",    0.01,  10.0,  "log"),   # cost of leaving / switching the target lane
    ("collision",      0.5,  500.0,  "log"),   # predicted proximity / overlap with surr cars
    ("rollover",       0.5,  500.0,  "log"),   # predicted P_roll (risks.rollover's model, vectorized)
    ("road_edge",      0.5,  500.0,  "log"),   # predicted margin to the paved road edge
    ("time_headway_s", 0.5,    3.0,  "linear"),  # [s] safety-distance scale d_safe = d_min + T_h v
)
W_NAMES = tuple(n for n, *_ in W_SPEC)
W_LO = np.array([lo for _, lo, _, _ in W_SPEC])
W_HI = np.array([hi for _, _, hi, _ in W_SPEC])
W_LOG = np.array([sc == "log" for *_, sc in W_SPEC])


@dataclass(frozen=True)
class MPCWeights:
    speed: float = 1.0
    accel: float = 0.1
    lane_change: float = 0.5
    collision: float = 20.0
    rollover: float = 20.0
    road_edge: float = 20.0
    time_headway_s: float = 1.5

    def __post_init__(self):
        v = self.to_vector()
        assert np.all(v >= W_LO - 1e-12) and np.all(v <= W_HI + 1e-12), f"w outside W_SPEC bounds: {self}"

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in W_NAMES], dtype=float)

    @staticmethod
    def from_vector(v) -> "MPCWeights":
        v = np.asarray(v, dtype=float)
        assert v.shape == (len(W_NAMES),), v.shape
        return MPCWeights(**{n: float(x) for n, x in zip(W_NAMES, v)})


def w_to_unit(w: np.ndarray) -> np.ndarray:
    """Natural-units w -> optimizer coordinates u in [0, 1]^d."""
    w = np.asarray(w, dtype=float)
    lin = (w - W_LO) / (W_HI - W_LO)
    log = (np.log(w) - np.log(W_LO)) / (np.log(W_HI) - np.log(W_LO))
    return np.where(W_LOG, log, lin)


def w_from_unit(u: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    lin = W_LO + u * (W_HI - W_LO)
    log = np.exp(np.log(W_LO) + u * (np.log(W_HI) - np.log(W_LO)))
    return np.where(W_LOG, log, lin)


# ── Fixed controller configuration (not optimized) ─────────────────────────
@dataclass(frozen=True)
class MPCConfig:
    horizon_s: float = 3.0
    dt_pred_s: float = 0.1
    replan_every_steps: int = 2           # env steps (DT=0.05 s) between plan decisions
    accel_levels: tuple[float, ...] = (-8.0, -6.0, -4.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0)
    v_ref_mps: float = NOMINAL_SPEED      # speed the tracking term pulls toward
    lateral_time_constant_s: float = 1.2  # T_lat: decay of the tracking error about the lateral reference
    lane_change_duration_s: float = FAR_NEAR_PRESETS[2].lane_change_duration   # lateral reference ramps one
                                          # lane width over this long -- same as moderate surr traffic; a
                                          # step reference saturates far-near and rolls this SUV over
    d_min_m: float = 2.0                  # standstill safety distance in d_safe
    lateral_margin_m: float = 0.5         # decay length of lateral proximity / edge terms
    overlap_penalty: float = 10.0         # added (per unit time fraction) for a predicted box overlap
    accel_scale_mps2: float = 8.0         # normalizes the accel-effort term
    rollover_U_H_m: float = RolloverParams().U_H

    def __post_init__(self):
        assert self.horizon_s > 0 and self.dt_pred_s > 0 and self.replan_every_steps >= 1
        assert all(ACCEL_MIN <= a <= ACCEL_MAX for a in self.accel_levels), "accel_levels outside env box"

    @property
    def n_pred(self) -> int:
        return max(1, int(round(self.horizon_s / self.dt_pred_s)))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["accel_levels"] = list(self.accel_levels)
        return d

    @staticmethod
    def from_dict(d: dict) -> "MPCConfig":
        d = dict(d)
        if "accel_levels" in d:
            d["accel_levels"] = tuple(d["accel_levels"])
        known = {f.name for f in fields(MPCConfig)}
        assert set(d) <= known, f"unknown MPCConfig fields: {set(d) - known}"
        return MPCConfig(**d)


def p_rollover_vec(v: np.ndarray, delta: np.ndarray, vp: VehicleParameters, U_H: float, g: float = 9.81) -> np.ndarray:
    """Elementwise P_roll -- the exact formulas of risks.rollover (PARALO
    eqs. 5.2-5.6), vectorized for the MPC's candidate x horizon arrays.
    tests/test_mpc_controller.py checks agreement with risks.rollover."""
    v = np.maximum(v, 0.0)
    beta = np.arctan(vp.l_r * np.tan(delta) / (vp.L + _EPS))
    sb = np.sin(beta)
    R = np.where(np.abs(sb) > _EPS, np.abs(vp.l_r / np.where(np.abs(sb) > _EPS, sb, 1.0)), _R_MAX)
    v_safe = np.sqrt(np.maximum(0.0, g * R * vp.W_c / (2.0 * vp.h_CG)))
    delta_v = np.sqrt(np.maximum(0.0, g * R * vp.W_c / (4.0 * vp.h_CG ** 3))) * U_H
    sigma = np.maximum(delta_v / 3.0, 1e-6)
    return np.clip(ndtr((v - v_safe) / sigma), 0.0, 1.0)


class SamplingMPC:
    """EgoController implementation -- see module docstring."""

    def __init__(self, config: MPCConfig, weights: MPCWeights):
        self.cfg = config
        self.w = weights
        self._fn = FAR_NEAR_PRESETS[2]   # moderate-preset lateral tracking gains
        self.solver_failures = 0
        self.n_plans = 0

    # -- EgoController API --
    def reset(self, view: EgoView, scenario_info: dict) -> None:
        self._step = 0
        self._target_lane = self._nearest_lane(view)
        self._y_ref = float(self._lane_centers(view.road)[self._target_lane])
        self._accel_cmd = 0.0
        self.solver_failures = 0
        self.n_plans = 0

    def _lat_rate(self, road) -> float:
        return road.l_w / self.cfg.lane_change_duration_s   # [m/s] lateral reference speed

    def act(self, view: EgoView) -> tuple[float, float]:
        if self._step % self.cfg.replan_every_steps == 0:
            self._plan(view)
        self._step += 1
        y_t = float(self._lane_centers(view.road)[self._target_lane])
        max_move = self._lat_rate(view.road) * DT
        self._y_ref += float(np.clip(y_t - self._y_ref, -max_move, max_move))
        return float(np.clip(self._accel_cmd, ACCEL_MIN, ACCEL_MAX)), self._steer(view, self._y_ref)

    # -- internals --
    @staticmethod
    def _lane_centers(road) -> np.ndarray:
        return np.array([-lane.offset for lane in road.lanes])   # backbone frame, SAE +right

    @staticmethod
    def _backbone_e_y(road, state) -> float:
        return -road.lanes[state.lane].offset + state.e_y

    def _nearest_lane(self, view: EgoView) -> int:
        y = self._backbone_e_y(view.road, view.ego.state)
        return int(np.argmin(np.abs(self._lane_centers(view.road) - y)))

    def _steer(self, view: EgoView, y_ref: float) -> float:
        """Far-near tracking of the backbone lateral reference y_ref."""
        road, st = view.road, view.ego.state
        e_rel = self._backbone_e_y(road, st) - y_ref
        lane = self._nearest_lane(view)
        v = max(st.v_x, 0.0)
        e_near = far_near_lookahead_offset(e_rel, st.e_psi, self._fn.L_n)
        d_far = v * self._fn.T_f
        e_far = far_near_lookahead_offset(e_rel, st.e_psi, d_far)
        delta = far_near_steering(e_near, e_far, v, self._fn.k_n, self._fn.k_f)
        kappa_preview = -road.lanes[lane].kappa[road.index_at(st.s + d_far)]
        delta += far_near_curvature_feedforward(kappa_preview, view.ego.vehicle_params.L)
        return float(np.clip(delta, -DELTA_MAX, DELTA_MAX))

    def candidate_costs(self, view: EgoView) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(accel per candidate, lane per candidate, cost per candidate) --
        exposed for tests and diagnostics."""
        cfg, w, road = self.cfg, self.w, view.road
        st, vp = view.ego.state, view.ego.vehicle_params
        centers = self._lane_centers(road)
        nearest = self._nearest_lane(view)
        lanes = [l for l in (nearest - 1, nearest, nearest + 1) if 0 <= l < len(centers)]
        acc = np.repeat(np.array(cfg.accel_levels, dtype=float), len(lanes))
        lane_c = np.tile(np.array(lanes), len(cfg.accel_levels))

        N, dt = cfg.n_pred, cfg.dt_pred_s
        t = dt * np.arange(1, N + 1)                                # (N,)
        v = np.maximum(0.0, st.v_x + acc[:, None] * t[None, :])     # (C, N)
        s = st.s + np.cumsum(v, axis=1) * dt
        # lateral: the reference ramps toward the candidate lane centre at the lane-change rate (exactly
        # what act() does); the car follows it with a first-order tracking error decaying from now.
        y0 = self._backbone_e_y(road, st)
        y_t = centers[lane_c][:, None]
        rate = self._lat_rate(road)
        y_ref = self._y_ref + np.clip(y_t - self._y_ref, -rate * t[None, :], rate * t[None, :])
        decay = np.exp(-t[None, :] / cfg.lateral_time_constant_s)
        y = y_ref + (y0 - self._y_ref) * decay
        y_dd = (y0 - self._y_ref) / cfg.lateral_time_constant_s ** 2 * decay   # ramp corners neglected
        idx = np.clip(np.round(s / road.ds).astype(int), 0, len(road.kappa) - 1)
        curv = -road.kappa[idx] + y_dd / np.maximum(v, 1.0) ** 2
        delta_pred = np.arctan(vp.L * curv)

        # surrounding traffic, constant-velocity prediction
        agents = view.agents
        s_j = np.array([a.car.state.s for a in agents])[:, None] + np.array(
            [a.car.state.v_x for a in agents])[:, None] * t[None, :]                      # (J, N)
        y_j0 = np.array([-road.lanes[a.car.state.lane].offset + a.car.state.e_y for a in agents])
        vy_j = np.array([a.car.state.v_x * math.sin(a.car.state.e_psi) + a.car.state.v_y * math.cos(a.car.state.e_psi)
                         for a in agents])
        half_w = road.lanes[-1].offset + road.lanes[-1].width / 2
        y_j = np.clip(y_j0[:, None] + vy_j[:, None] * t[None, :], -half_w, half_w)

        ds = np.abs(s[:, None, :] - s_j[None]) - CAR_LENGTH                              # (C, J, N)
        dy = np.abs(y[:, None, :] - y_j[None]) - CAR_WIDTH
        d_safe = (cfg.d_min_m + w.time_headway_s * v)[:, None, :]
        prox = np.exp(-np.maximum(ds, 0.0) / d_safe) * np.exp(-np.maximum(dy, 0.0) / cfg.lateral_margin_m)
        overlap = (ds < 0.0) & (dy < 0.0)
        c_collision = prox.max(axis=1).mean(axis=1) + cfg.overlap_penalty * overlap.any(axis=1).mean(axis=1)

        c_rollover = p_rollover_vec(v, delta_pred, vp, cfg.rollover_U_H_m).mean(axis=1)
        margin = half_w - (np.abs(y) + CAR_WIDTH / 2)
        c_edge = np.exp(-np.maximum(margin, 0.0) / cfg.lateral_margin_m).mean(axis=1) + cfg.overlap_penalty * (margin < 0).mean(axis=1)
        c_speed = (((v - cfg.v_ref_mps) / cfg.v_ref_mps) ** 2).mean(axis=1)
        c_accel = (acc / cfg.accel_scale_mps2) ** 2 + ((acc - self._accel_cmd) / (ACCEL_MAX - ACCEL_MIN)) ** 2
        c_lane = (lane_c != nearest).astype(float) + (lane_c != self._target_lane).astype(float)

        cost = (w.speed * c_speed + w.accel * c_accel + w.lane_change * c_lane
                + w.collision * c_collision + w.rollover * c_rollover + w.road_edge * c_edge)
        return acc, lane_c, cost

    def _plan(self, view: EgoView) -> None:
        acc, lane_c, cost = self.candidate_costs(view)
        self.n_plans += 1
        if not np.all(np.isfinite(cost)):
            # Documented fallback: count it, brake hard, hold the nearest lane.
            self.solver_failures += 1
            self._accel_cmd, self._target_lane = ACCEL_MIN, self._nearest_lane(view)
            return
        best = int(np.argmin(cost))   # deterministic tie-break: first minimum
        self._accel_cmd, self._target_lane = float(acc[best]), int(lane_c[best])
