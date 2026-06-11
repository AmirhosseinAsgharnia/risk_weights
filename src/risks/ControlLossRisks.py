import numpy as np
import numpy.typing as npt
from scipy.stats import norm

# Column indices matching FEATURE_NAMES in dataloader.py
VX_COL = 4
VY_COL = 5


def rollover_probability(
    data: npt.ArrayLike,
    constants: dict,
    vx_col: int = VX_COL,
    vy_col: int = VY_COL,
) -> np.ndarray:
    """
    Compute per-timestep rollover probability.

    Accepts three layouts (feature axis must be last before any agent axis):
      (T, F)     — single agent          → returns (T,)
      (T, F, N)  — N agents              → returns (T, N)
      (S, T, F)  — batch of S agents     → returns (S, T)

    vx_col / vy_col default to the Waymo FEATURE_NAMES indices (4, 5).
    Override for custom trajectory arrays.
    """
    arr = np.asarray(data, dtype=np.float32)
    ndim = arr.ndim

    if ndim == 2:           # (T, F)
        vx = arr[:, vx_col]
        vy = arr[:, vy_col]
    elif ndim == 3 and arr.shape[1] > arr.shape[2]:   # (T, F, N)
        vx = arr[:, vx_col, :]
        vy = arr[:, vy_col, :]
    elif ndim == 3:         # (S, T, F)
        vx = arr[:, :, vx_col]
        vy = arr[:, :, vy_col]
    else:
        raise ValueError(f"Expected 2-D or 3-D array, got shape {arr.shape}")

    beta = np.arctan2(vy, vx)
    V = np.sqrt(vx ** 2 + vy ** 2)

    # avoid division by zero at beta ≈ 0 (straight motion)
    sin_beta = np.sin(beta)
    sin_beta = np.where(np.abs(sin_beta) < 1e-6, 1e-6, sin_beta)

    R = np.abs(constants['l_r'] / sin_beta)

    v_safe  = np.sqrt(9.81 * R * constants['W'] / (2 * constants['h_cg']))
    sigma_v = np.sqrt((9.81 * R * constants['W']) / (4 * constants['h_cg'] ** 3))

    return norm.cdf((V - v_safe) / (sigma_v / 3))


def slippage_probability(
    data: npt.ArrayLike,
    constants: dict,
    vx_col: int = VX_COL,
    vy_col: int = VY_COL,
) -> np.ndarray:
    """
    Compute per-timestep slippage (control-loss) probability.

    v_safe = min(v_safe_ss, v_safe_us, v_safe_os)
      v_safe_ss = sqrt(mu * g * R)
      v_safe_us = sqrt(alpha_max * R * C_f * L / (m * l_r))
      v_safe_os = sqrt(alpha_max * R * C_r * L / (m * l_f))

    sigma_v = Delta_v / 3  where Delta_v propagates uncertainty in alpha_max and mu:
      Delta_v = max(
        sqrt(R * C_f * L / (4 * m * l_r * alpha_max)) * U_alpha,
        sqrt(R * C_r * L / (4 * m * l_f * alpha_max)) * U_alpha,
        sqrt(g * R / (4 * mu)) * U_mu,
      )

    P_CS = Phi((V - v_safe) / (Delta_v / 3))

    Required constants keys: mu, alpha_max, C_f, C_r, l_f, l_r, mass, U_alpha, U_mu
    Array layouts: same as rollover_probability.
    """
    arr = np.asarray(data, dtype=np.float32)
    ndim = arr.ndim

    if ndim == 2:
        vx = arr[:, vx_col]
        vy = arr[:, vy_col]
    elif ndim == 3 and arr.shape[1] > arr.shape[2]:
        vx = arr[:, vx_col, :]
        vy = arr[:, vy_col, :]
    elif ndim == 3:
        vx = arr[:, :, vx_col]
        vy = arr[:, :, vy_col]
    else:
        raise ValueError(f"Expected 2-D or 3-D array, got shape {arr.shape}")

    beta = np.arctan2(vy, vx)
    V = np.sqrt(vx ** 2 + vy ** 2)

    sin_beta = np.sin(beta)
    sin_beta = np.where(np.abs(sin_beta) < 1e-6, 1e-6, sin_beta)
    R = np.abs(constants['l_r'] / sin_beta)

    g = 9.81
    mu        = constants['mu']
    alpha_max = constants['alpha_max']
    C_f       = constants['C_f']
    C_r       = constants['C_r']
    l_f       = constants['l_f']
    l_r       = constants['l_r']
    m         = constants['mass']
    L         = l_f + l_r
    U_alpha   = constants['U_alpha']
    U_mu      = constants['U_mu']

    v_safe_ss = np.sqrt(mu * g * R)
    v_safe_us = np.sqrt(alpha_max * R * C_f * L / (m * l_r))
    v_safe_os = np.sqrt(alpha_max * R * C_r * L / (m * l_f))
    v_safe    = np.minimum(v_safe_ss, np.minimum(v_safe_us, v_safe_os))

    dv_us = np.sqrt(R * C_f * L / (4 * m * l_r * alpha_max)) * U_alpha
    dv_os = np.sqrt(R * C_r * L / (4 * m * l_f * alpha_max)) * U_alpha
    dv_ss = np.sqrt(g * R / (4 * mu)) * U_mu
    delta_v = np.maximum(dv_us, np.maximum(dv_os, dv_ss))

    return norm.cdf((V - v_safe) / (delta_v / 3))


if __name__ == "__main__":
    # synthetic single-agent test (custom column layout: vx=2, vy=3)
    traj = np.zeros((20, 5), dtype=np.float32)
    traj[:, 0] = np.linspace(0, 5, 20)
    traj[:, 1] = traj[:, 0] ** 2
    traj[:, 2] = traj[:, 0] ** 3
    traj[:, 3] = np.gradient(traj[:, 1], traj[:, 0])
    traj[:, 4] = np.gradient(traj[:, 2], traj[:, 0])

    constants = {
        'mass': 1500, 'h_cg': 1.2, 'l_f': 1.2, 'l_r': 1.6,
        'W': 2, 'C_f': 19000, 'C_r': 33000, 'mu': 0.8,
        'alpha_max': 0.1, 'U_alpha': 0.01, 'U_mu': 0.05,
    }

    p_rollover  = rollover_probability(traj, constants, vx_col=2, vy_col=3)
    p_slippage  = slippage_probability(traj, constants, vx_col=2, vy_col=3)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(traj[:, 0], p_rollover,  label='rollover')
    ax.plot(traj[:, 0], p_slippage,  label='slippage')
    ax.legend()
    plt.tight_layout()
    plt.show()
