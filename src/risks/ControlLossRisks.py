import numpy as np
import numpy.typing as npt
from scipy.stats import norm

# Column indices matching FEATURE_NAMES in dataloader.py
VX_COL = 4
VY_COL = 5


def slippage_probability(
    data: npt.ArrayLike,
    constants: dict,
    vx_col: int = VX_COL,
    vy_col: int = VY_COL,
) -> np.ndarray:
    """
    Compute per-timestep slippage probability.

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

    v_safe_ss = np.sqrt(constants['mu'] * 9.81 * R)
    sigma_v = np.sqrt((9.81 * R * constants['W']) / (4 * constants['h_cg'] ** 3))

    return norm.cdf((V - v_safe_ss) / (sigma_v / 3))


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
        'W': 2, 'C_f': 19000, 'C_r': 33000, 'mu': 1,
    }

    prob = slippage_probability(traj, constants, vx_col=2, vy_col=3)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(traj[:, 0], prob)
    plt.tight_layout()
    plt.show()
