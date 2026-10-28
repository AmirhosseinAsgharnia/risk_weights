"""
Per-state PARALO risk probabilities, vectorized over arbitrary array shapes
(candidates x horizon x objects) for the PARALO MPC (learning.paralo_mpc).

Every function is the EXACT formula of the corresponding risks/ module
evaluated on a single-state trajectory -- the modules' `prob` output for a
one-step trajectory. tests/test_paralo_vec.py checks agreement elementwise
against risks.c2c / risks.slippage / risks.departure (rollover's port,
learning.mpc.p_rollover_vec, is checked in tests/test_mpc_controller.py).
Nothing here changes a model; only the loops are replaced by numpy.

Per-state definitions (what "P at state x" means in the MPC stage cost):
    c2c        max over surrounding objects of the Gaussian rectangle-overlap
               probability (risks.c2c: prob = max_{j,k} P_jk, here k fixed)
    slippage   P_slip = 1 - (1 - P_CS)(1 - P_BS)   (PARALO eq. 5.19)
    departure  P_LD = 1 - [Phi((h - D)/sigma_D) - Phi((-h - D)/sigma_D)], against
               a STRAIGHT, EXACT centreline (coefficients (0, 0, c), zero
               covariance) -- the simulator's map is ground truth, so
               sigma_D = max(sigma_y, min_sigma_y)
"""

import numpy as np
from scipy.special import ndtr

from model.car.config import VehicleParameters
from risks.c2c import C2CParams
from risks.departure import DepartureParams
from risks.slippage import SlippageParams
from learning.mpc import p_rollover_vec

_EPS = 1e-9          # risks.slippage._EPS / risks.c2c._EPS
_PROB_FLOOR = 1e-6   # risks.slippage._PROB_FLOOR
_R_MAX = 1.0e6       # risks.slippage.compute_radius's "infinite" radius


def _axis_interval_prob(mu, var, h, var_floor):
    """risks.c2c.axis_interval_probability, elementwise."""
    var = np.asarray(var, dtype=float)
    sigma = np.sqrt(np.maximum(var, var_floor))
    gauss = np.clip(ndtr((h - mu) / sigma) - ndtr((-h - mu) / sigma), 0.0, 1.0)
    step = ((-h <= mu) & (mu <= h)).astype(float)
    return np.where(var <= var_floor, step, gauss)


def p_c2c_pairs(ego_x, ego_y, obj_x, obj_y, hx: float, hy: float, var_rx: float, var_ry: float,
                var_floor: float = C2CParams().var_floor) -> np.ndarray:
    """Overlap probability for each (ego state, object state) pair, broadcast
    over the inputs. hx/hy: half-extents of the collision rectangle (ego half
    + object half); var_rx/var_ry: relative-position variances (ego + object)."""
    p_long = _axis_interval_prob(np.asarray(obj_x) - np.asarray(ego_x), var_rx, hx, var_floor)
    p_lat = _axis_interval_prob(np.asarray(obj_y) - np.asarray(ego_y), var_ry, hy, var_floor)
    return np.clip(p_long * p_lat, 0.0, 1.0)


def p_slip_vec(v, delta, a, vp: VehicleParameters, params: SlippageParams) -> np.ndarray:
    """risks.slippage.compute_slip_probability's P_slip, elementwise."""
    v = np.maximum(np.asarray(v, dtype=float), 0.0)
    delta = np.asarray(delta, dtype=float)
    a = np.asarray(a, dtype=float)
    g, lr, lf, L, m = params.g, vp.l_r, vp.l_f, vp.L, vp.m
    mu = params.mu_max * np.exp(-params.friction_speed_decay_k * np.abs(v))
    beta = np.arctan(lr * np.tan(delta) / (L + _EPS))
    sb = np.sin(beta)
    R = np.where(np.abs(sb) < _EPS, _R_MAX, np.abs(lr / np.where(np.abs(sb) < _EPS, 1.0, sb)))
    deg_to_rad = params.alpha_max_deg_at_mu_1 * np.pi / 180.0
    alpha_max = mu * deg_to_rad
    U_alpha = params.mu_uncertainty * deg_to_rad
    v_ss = np.sqrt(np.maximum(0.0, mu * g * R))
    v_us = np.sqrt(np.maximum(0.0, alpha_max * R * vp.C_f * L / (m * lr + _EPS)))
    v_os = np.sqrt(np.maximum(0.0, alpha_max * R * vp.C_r * L / (m * lf + _EPS)))
    v_safe = np.minimum(np.minimum(v_ss, v_us), v_os)
    term_us = np.sqrt(np.maximum(0.0, R * vp.C_f * L / np.maximum(4.0 * m * lr * alpha_max, _EPS))) * U_alpha
    term_os = np.sqrt(np.maximum(0.0, R * vp.C_r * L / np.maximum(4.0 * m * lf * alpha_max, _EPS))) * U_alpha
    term_ss = np.sqrt(np.maximum(0.0, g * R / np.maximum(4.0 * mu, _EPS))) * params.mu_uncertainty
    U_cs = np.maximum(np.maximum(np.maximum(term_us, term_os), term_ss), _PROB_FLOOR)
    p_cs = np.clip(ndtr((v - v_safe) / (U_cs / 3.0)), 0.0, 1.0)
    a_min = -mu * g * np.cos(beta)
    U_bs = max(g * params.mu_uncertainty, _PROB_FLOOR)
    p_bs = np.clip(1.0 - ndtr((a - a_min) / (U_bs / 3.0)), 0.0, 1.0)
    return np.clip(1.0 - (1.0 - p_cs) * (1.0 - p_bs), 0.0, 1.0)


def p_departure_vec(D, params: DepartureParams, var_y: float | None = None) -> np.ndarray:
    """risks.departure's P_LD for lateral clearance D from a straight, exact
    centreline, elementwise."""
    h = params.lane_width_m / 2.0 - params.vehicle_width_m / 2.0
    assert h > 0.0, params
    var = params.default_sigma_lateral_m ** 2 if var_y is None else var_y
    sigma = np.sqrt(max(var, params.min_sigma_y_m ** 2))
    D = np.asarray(D, dtype=float)
    return np.clip(1.0 - (ndtr((h - D) / sigma) - ndtr((-h - D) / sigma)), 0.0, 1.0)


__all__ = ["p_c2c_pairs", "p_slip_vec", "p_departure_vec", "p_rollover_vec"]
