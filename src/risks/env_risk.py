"""
Risk probability functions — SAE J670 convention.

e_y in ego state is within-lane, positive rightward (SAE J670).
Absolute position: y_abs = lane_center_abs(ego_lane) + e_y.

P_slip, P_roll : depend only on vehicle dynamics (v_x, a, delta).
P_c2c          : car-to-car, uses absolute (s, y_abs) for all vehicles.
P_ld           : lane departure, uses within-lane |e_y|.
"""

import numpy as np
from scipy.special import ndtr

import dynamics.env_dynamics as env_dynamics
from environment.scenarios import SandwichScenario
from config.config import (
    l_r, l_f, L, m, g, C_f, C_r,
    W_c, L_veh, h_cg,
    mu, k_f, alpha_max_base,
    U_hcg, U_mu,
    sensor_accuracy,
    w_slip, w_roll, w_c2c, w_ld,
    T_horizon, dt_pred, lambda_disc,
)

Phi = ndtr

N       = int(T_horizon / dt_pred)
_t      = np.arange(1, N + 1) * dt_pred
_raw_w  = np.exp(-lambda_disc * _t)
weights = _raw_w / _raw_w.sum()

_K_US = m * l_r / (L * C_f) - m * l_f / (L * C_r)


# ── Zero-control predictor ────────────────────────────────────────────────────

def predict_zero_control(ego_state, s_ego: float, ego_lane: int,
                         vehicles: list, scenario: SandwichScenario):
    """
    N-step prediction with a=0, delta=0 for the ego.
    Surrounding vehicles travel at constant speed with fixed lateral position.

    Returns
    -------
    ego_fut : dict of shape-(N,) arrays — e_y, y_abs, e_psi, v_x, s
    sur_fut : list of dicts (one per vehicle, including invalid) — s, y_abs
    """
    state    = ego_state.copy()
    s        = float(s_ego)
    lane     = ego_lane
    k0, k1   = scenario.k0, scenario.k1
    n_lanes  = scenario.n_lanes
    lw       = scenario.lane_width

    e_y_arr   = np.empty(N)
    y_abs_arr = np.empty(N)
    e_psi_arr = np.empty(N)
    v_x_arr   = np.empty(N)
    s_arr     = np.empty(N)

    for k in range(N):
        state, s, lane = env_dynamics.step(
            state, 0.0, 0.0, s, lane, k0, k1, n_lanes, lw, dt_pred
        )
        e_y_arr[k]   = state[0]
        y_abs_arr[k] = env_dynamics.lane_center_abs(lane, n_lanes, lw) + state[0]
        e_psi_arr[k] = state[2]
        v_x_arr[k]   = state[4]
        s_arr[k]     = s

    ego_fut = {
        'e_y':   e_y_arr,
        'y_abs': y_abs_arr,
        'e_psi': e_psi_arr,
        'v_x':   v_x_arr,
        's':     s_arr,
    }

    sur_fut = []
    for veh in vehicles:
        if veh['valid']:
            sur_fut.append({
                's':     veh['s'] + veh['v_x'] * _t,
                'y_abs': np.full(N, veh['y_abs']),
            })
        else:
            sur_fut.append(None)   # placeholder — ignored in P_c2c

    return ego_fut, sur_fut


# ── P_slip ────────────────────────────────────────────────────────────────────

def P_slip(ego_fut, a: float, delta: float) -> float:
    v_x_future = ego_fut['v_x']
    abs_delta  = abs(delta)
    v_x0       = max(float(v_x_future[0]), 0.1)

    R_turn = 500.0 if abs_delta < 1e-6 else float(
        np.clip(L * (1.0 + _K_US * v_x0 ** 2) / abs_delta, 1.0, 500.0)
    )

    beta      = np.arctan(l_r * np.tan(delta) / L)
    mu_eff    = mu * np.exp(-k_f * v_x_future)
    alpha_max = alpha_max_base * mu_eff

    v_SS = np.sqrt(np.maximum(mu_eff * g * R_turn, 0.0))
    v_US = np.sqrt(np.maximum(alpha_max * R_turn * C_f * L / (m * l_r), 0.0))
    v_OS = np.sqrt(np.maximum(alpha_max * R_turn * C_r * L / (m * l_f), 0.0))
    v_safe = np.minimum(np.minimum(v_SS, v_US), v_OS)

    term1 = np.sqrt(max(R_turn * C_f * L / (4 * m * l_r * alpha_max_base), 0.0)) * U_mu * 8.6 * np.pi / 180
    term2 = np.sqrt(max(R_turn * C_r * L / (4 * m * l_f * alpha_max_base), 0.0)) * U_mu * 8.6 * np.pi / 180
    term3 = np.sqrt(max(g * R_turn / (4 * mu), 0.0)) * U_mu
    dv    = max(term1, term2, term3, 1e-3)

    P_CS = Phi((v_x_future - v_safe) / (dv / 3))
    P_BS = 1.0 - Phi((a - (-g * mu * np.cos(beta))) / (g * U_mu / 3))

    return float(np.dot(weights, 1.0 - (1.0 - P_BS) * (1.0 - P_CS)))


# ── P_roll ────────────────────────────────────────────────────────────────────

def P_roll(ego_fut, delta: float) -> float:
    v_x_future = ego_fut['v_x']
    abs_delta  = abs(delta)
    v_x0       = max(float(v_x_future[0]), 0.1)

    R_turn = 500.0 if abs_delta < 1e-6 else float(
        np.clip(L * (1.0 + _K_US * v_x0 ** 2) / abs_delta, 1.0, 500.0)
    )

    v_safe = np.sqrt(max(g * R_turn * W_c / (2 * h_cg), 0.0))
    dv     = max(np.sqrt(max(g * R_turn * W_c / (4 * h_cg ** 3), 0.0)) * U_hcg, 1e-3)

    return float(np.dot(weights, Phi((v_x_future - v_safe) / (dv / 3))))


# ── P_c2c ─────────────────────────────────────────────────────────────────────

def P_c2c(ego_fut, sur_fut) -> float:
    """
    Collision probability with any valid surrounding vehicle.
    Uses absolute arc-length s and absolute lateral position y_abs.
    """
    s_ego  = ego_fut['s']
    y_ego  = ego_fut['y_abs']
    P_none = np.ones(N)

    for veh in sur_fut:
        if veh is None:
            continue
        s_i  = veh['s']
        y_i  = veh['y_abs']

        dist  = np.sqrt((s_ego - s_i) ** 2 + (y_ego - y_i) ** 2)
        sigma = np.maximum((1.0 - sensor_accuracy) * dist / 3.0, 1e-3)

        Px = (Phi((s_ego + L_veh/2 - (s_i - L_veh/2)) / sigma)
            - Phi((s_ego - L_veh/2 - (s_i + L_veh/2)) / sigma))
        Py = (Phi((y_ego + W_c/2  - (y_i - W_c/2))  / sigma)
            - Phi((y_ego - W_c/2  - (y_i + W_c/2))  / sigma))

        P_none *= 1.0 - np.clip(Px, 0, 1) * np.clip(Py, 0, 1)

    return float(np.max(1.0 - P_none))


# ── P_ld ──────────────────────────────────────────────────────────────────────

def P_ld(ego_fut, lane_width: float) -> float:
    """
    Lane-departure risk: probability that vehicle width protrudes past lane edge.
    Uses SAE J670 within-lane e_y.
    """
    e_y   = ego_fut['e_y']
    sigma = max(lane_width * (1.0 - sensor_accuracy) / 3.0, 1e-3)
    P_arr = Phi((np.abs(e_y) + W_c / 2 - lane_width / 2) / sigma)
    return float(np.dot(weights, P_arr))


# ── Total cost ────────────────────────────────────────────────────────────────

def total_cost(ego_fut, sur_fut, a: float, delta: float,
               scenario: SandwichScenario, risk_weights=None):
    if risk_weights is None:
        risk_weights = {'slip': w_slip, 'roll': w_roll, 'c2c': w_c2c, 'ld': w_ld}

    ps = P_slip(ego_fut, a, delta)
    pr = P_roll(ego_fut, delta)
    pc = P_c2c(ego_fut, sur_fut)
    pl = P_ld(ego_fut, scenario.lane_width)

    J = (risk_weights.get('slip', w_slip) * ps
         + risk_weights.get('roll', w_roll) * pr
         + risk_weights.get('c2c',  w_c2c)  * pc
         + risk_weights.get('ld',   w_ld)   * pl)

    return J, ps, pr, pc, pl
