"""
Frenet-Serret bicycle-model dynamics — SAE J670 lateral convention.

State vector: [e_y, psi, e_psi, psi_dot, v_x, v_y]
  e_y     : within-lane lateral offset (m), SAE J670 — zero at lane centre,
             positive rightward (toward lane 0 / road right edge)
  psi     : absolute yaw in world frame (rad), 0 = initial road heading
  e_psi   : heading error = psi - theta_road(s)  (rad)
  psi_dot : yaw rate (rad/s)
  v_x     : longitudinal speed in vehicle frame (m/s)
  v_y     : lateral speed in vehicle frame (m/s), positive leftward (SAE)

Arc-length s is tracked as a separate float (returned by step).
ego_lane is tracked separately (returned by step).

Frenet reference : left road boundary.
Curvature        : κ(s) = A·sin(ω·s)  (positive = left turn).
  kappa_A     : amplitude (rad/m)
  kappa_omega : angular frequency (rad/m); wavelength λ = 2π/ω
"""

import numpy as np
from config.config import (l_r, l_f, L, m, I_zz, h_cg, g,
                    C_f, C_r, mu, Fz_nom, C_d)


# ── Road geometry ─────────────────────────────────────────────────────────────

def kappa(s: float, kappa_A: float, kappa_omega: float) -> float:
    """κ(s) = A·sin(ω·s)  (rad/m)."""
    return kappa_A * np.sin(kappa_omega * float(s))


def kappa_arr(s_arr, kappa_A: float, kappa_omega: float):
    """Vectorised curvature."""
    s_arr = np.asarray(s_arr, float)
    return kappa_A * np.sin(kappa_omega * s_arr)


def theta_road(s, kappa_A: float, kappa_omega: float):
    """Road heading θ(s) = (A/ω)·(1 − cos(ω·s)), equals 0 at s=0."""
    s = np.asarray(s, float)
    if kappa_omega == 0.0 or kappa_A == 0.0:
        return np.zeros_like(s)
    return (kappa_A / kappa_omega) * (1.0 - np.cos(kappa_omega * s))


def lane_center_abs(lane: int, n_lanes: int, lane_width: float) -> float:
    """
    Rightward distance from left road boundary to centre of lane.
    Lane 0 = rightmost (largest y_abs), lane n_lanes-1 = leftmost (smallest y_abs).
    """
    return (n_lanes - lane - 0.5) * lane_width


# ── Dynamics step ─────────────────────────────────────────────────────────────

def step(state, a: float, delta: float, s: float, ego_lane: int,
         kappa_A: float, kappa_omega: float, n_lanes: int, lane_width: float, dt: float):
    """
    Single forward-Euler step of the bicycle model.

    Parameters
    ----------
    state       : [e_y, psi, e_psi, psi_dot, v_x, v_y]
    a           : longitudinal acceleration (m/s²)
    delta       : front steering angle (rad)
    s           : arc-length along left road boundary (m)
    ego_lane    : current lane index (0=rightmost, n_lanes-1=leftmost)
    kappa_A     : curvature amplitude (rad/m)
    kappa_omega : curvature angular frequency (rad/m)
    n_lanes     : number of lanes
    lane_width  : lane width (m)
    dt          : timestep (s)

    Returns
    -------
    new_state : np.ndarray  [e_y, psi, e_psi, psi_dot, v_x, v_y]
    new_s     : float       updated arc-length
    new_lane  : int         updated lane (may differ if lane change occurred)
    """
    e_y, psi, e_psi, psi_dot, v_x, v_y = state
    v_x = max(float(v_x), 0.1)

    # ── Bicycle model (vehicle frame) ─────────────────────────────────────────
    alpha_r = np.arctan((v_y - l_r * psi_dot) / v_x)
    alpha_f = np.arctan((v_y + l_f * psi_dot) / v_x) - delta

    Fzr = max((l_f * m * g + (a - v_y * psi_dot) * m * h_cg) / L, 0.0)
    Fzf = max((l_r * m * g - (a - v_y * psi_dot) * m * h_cg) / L, 0.0)

    Fyr = -C_r * alpha_r * mu * Fzr / Fz_nom
    Fyf = -C_f * alpha_f * mu * Fzf / Fz_nom * np.cos(delta)

    v_x_dot  = a - C_d / m * v_x ** 2 + psi_dot * v_y
    v_y_dot  = (Fyf + Fyr) / m - psi_dot * v_x
    psi_ddot = (l_f * Fyf - l_r * Fyr) / I_zz

    # ── Frenet kinematics (SAE J670: e_y positive rightward) ──────────────────
    # y_abs = absolute rightward distance from left road boundary
    y_abs = lane_center_abs(ego_lane, n_lanes, lane_width) + e_y
    kap   = kappa(s, kappa_A, kappa_omega)
    denom = max(1.0 + kap * y_abs, 1e-3)

    s_dot     = (v_x * np.cos(e_psi) - v_y * np.sin(e_psi)) / denom
    e_y_dot   = -v_x * np.sin(e_psi) - v_y * np.cos(e_psi)   # SAE: + = rightward
    e_psi_dot = psi_dot - kap * s_dot

    # ── Euler integration ─────────────────────────────────────────────────────
    e_y_new     = e_y     + e_y_dot   * dt
    psi_new     = psi     + psi_dot   * dt
    e_psi_new   = e_psi   + e_psi_dot * dt
    psi_dot_new = psi_dot + psi_ddot  * dt
    v_x_new     = max(v_x + v_x_dot  * dt, 0.0)
    v_y_new     = v_y     + v_y_dot   * dt
    s_new       = s       + s_dot     * dt

    # ── Lane transitions ──────────────────────────────────────────────────────
    new_lane = ego_lane
    if e_y_new > lane_width / 2:        # crossed right boundary
        e_y_new  -= lane_width
        new_lane -= 1                    # rightward → smaller index
    elif e_y_new < -lane_width / 2:     # crossed left boundary
        e_y_new  += lane_width
        new_lane += 1                    # leftward  → larger index

    new_state = np.array([e_y_new, psi_new, e_psi_new, psi_dot_new, v_x_new, v_y_new])
    return new_state, float(s_new), new_lane
