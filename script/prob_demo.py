"""
prob_demo.py — adaptive search of (kappa_A, kappa_omega) → P_slip.

For each (A, ω) pair, runs the MPC-controlled ego on that road geometry
(no surrounding vehicles) and records the 95th-percentile P_slip encountered.

The old uniform grid was expensive because it spent the same number of MPC
evaluations in boring low-risk regions as near dangerous ridges. This version
uses cached adaptive cell refinement:

1. Seed the full domain with a small coarse grid.
2. Score each cell by its current max risk plus a local-variation bonus.
3. Split only the most promising cells into four children.
4. Plot a dense interpolated heatmap from the irregular samples.

Run: python3 prob_demo.py
"""

from dataclasses import dataclass
import os
import numpy as np

import dynamics.env_dynamics as env_dynamics
import environment.env_scenario as env_scenario
import risks.env_risk as env_risk
from environment.scenarios import SandwichScenario
from controller.mpc import MPC

# ── Search domain ──────────────────────────────────────────────────────────────

A_MIN, A_MAX         = 0.0, 0.05      # curvature amplitude (rad/m)
OMEGA_MIN, OMEGA_MAX = 0.005, 0.15    # curvature frequency (rad/m)

# Adaptive-search controls. Increase REFINE_ROUNDS or CELLS_PER_ROUND when you
# want a sharper map; increase EXPLORATION_WEIGHT when dangerous regions are
# narrow and may hide between coarse samples.
def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


COARSE_N_A          = _env_int('PROB_COARSE_N_A', 5)
COARSE_N_OMEGA      = _env_int('PROB_COARSE_N_OMEGA', 5)
REFINE_ROUNDS       = _env_int('PROB_REFINE_ROUNDS', 4)
CELLS_PER_ROUND     = _env_int('PROB_CELLS_PER_ROUND', 10)
EXPLORATION_WEIGHT  = _env_float('PROB_EXPLORATION_WEIGHT', 0.5)
HEATMAP_N_A         = _env_int('PROB_HEATMAP_N_A', 80)
HEATMAP_N_OMEGA     = _env_int('PROB_HEATMAP_N_OMEGA', 80)
TOP_K_REGIONS       = _env_int('PROB_TOP_K_REGIONS', 10)
SKIP_PLOT           = bool(_env_int('PROB_SKIP_PLOT', 0))

# ── Fixed scenario (road only, no surrounding vehicles) ────────────────────────

_BASE = dict(
    n_lanes    = 1,
    ego_lane   = 0,
    lane_width = 3.7,
    ego_ey     = 0.0,
    ego_epsi   = 0.0,
    v0         = 20.0,
    vehicles   = [],
    dt         = _env_float('PROB_DT', 0.05),
    t_end      = _env_float('PROB_T_END', 8.0),
)

RISK_WEIGHTS = {'c2c': 0.35, 'slip': 0.15, 'roll': 0.15, 'ld': 0.35}
V_REF        = 20.0
MPC_INTERVAL = _env_float('PROB_MPC_INTERVAL', 0.05)  # s between MPC solves

# ── Single-run evaluator ───────────────────────────────────────────────────────

def eval_road(kappa_A: float, kappa_omega: float) -> float:
    sc  = SandwichScenario(kappa_A=kappa_A, kappa_omega=kappa_omega, **_BASE)
    mpc = MPC(risk_weights=RISK_WEIGHTS, v_ref=V_REF, w_v=0.001)

    ego_state = sc.ego_init_state()
    ego_lane  = sc.ego_lane
    # Start at first curvature peak: κ(s₀)=A when ω·s₀=π/2
    s_ego     = (np.pi / 2) / kappa_omega if kappa_omega > 0 else 0.0
    # psi must equal θ(s₀) so that e_psi = psi − θ(s₀) = 0 is consistent
    ego_state[1] = float(env_dynamics.theta_road(s_ego, kappa_A, kappa_omega))
    vehicles  = env_scenario.init_vehicles(sc, s_ego_init=s_ego)

    n_steps   = int(sc.t_end / sc.dt)
    mpc_every = max(1, int(MPC_INTERVAL / sc.dt))
    a, delta  = 0.0, 0.0
    pslip_log = []

    for k in range(n_steps):
        if k % mpc_every == 0:
            a, delta = mpc.solve(ego_state, s_ego, ego_lane, vehicles, sc)

        ego_fut, _ = mpc._predict(
            ego_state, s_ego, ego_lane, vehicles, sc, a, delta
        )
        pslip_log.append(env_risk.P_slip(ego_fut, a, delta))

        ego_state, s_ego, ego_lane = env_dynamics.step(
            ego_state, a, delta, s_ego, ego_lane,
            sc.kappa_A, sc.kappa_omega, sc.n_lanes, sc.lane_width, sc.dt,
        )
        if ego_lane < -1 or ego_lane >= sc.n_lanes + 1:
            break

    return float(np.percentile(pslip_log, 95)) if pslip_log else 0.0

# ── Adaptive search ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Cell:
    a0: float
    a1: float
    o0: float
    o1: float

    @property
    def center(self):
        return ((self.a0 + self.a1) / 2, (self.o0 + self.o1) / 2)

    @property
    def corners(self):
        return (
            (self.a0, self.o0),
            (self.a0, self.o1),
            (self.a1, self.o0),
            (self.a1, self.o1),
        )

    def children(self):
        am, om = self.center
        return (
            Cell(self.a0, am, self.o0, om),
            Cell(am, self.a1, self.o0, om),
            Cell(self.a0, am, om, self.o1),
            Cell(am, self.a1, om, self.o1),
        )


def _key(A: float, omega: float):
    """Stable key for points produced by repeated midpoint splits."""
    return (round(float(A), 12), round(float(omega), 12))


def adaptive_search():
    cache = {}

    def evaluate(A: float, omega: float) -> float:
        key = _key(A, omega)
        if key not in cache:
            cache[key] = eval_road(A, omega)
        return cache[key]

    def cell_values(cell: Cell):
        pts = (*cell.corners, cell.center)
        return [evaluate(A, omega) for A, omega in pts]

    def cell_priority(cell: Cell) -> float:
        vals = cell_values(cell)
        local_max = max(vals)
        local_variation = max(vals) - min(vals)
        return local_max + EXPLORATION_WEIGHT * local_variation

    a_edges = np.linspace(A_MIN, A_MAX, COARSE_N_A)
    o_edges = np.linspace(OMEGA_MIN, OMEGA_MAX, COARSE_N_OMEGA)
    cells = [
        Cell(a_edges[j], a_edges[j + 1], o_edges[i], o_edges[i + 1])
        for i in range(len(o_edges) - 1)
        for j in range(len(a_edges) - 1)
    ]

    print(f"Adaptive search: seed {COARSE_N_A} x {COARSE_N_OMEGA} grid, "
          f"{REFINE_ROUNDS} refinement rounds, max "
          f"{CELLS_PER_ROUND} cells/round")

    for round_idx in range(REFINE_ROUNDS):
        ranked = sorted(cells, key=cell_priority, reverse=True)
        selected = ranked[:min(CELLS_PER_ROUND, len(ranked))]
        selected_set = set(selected)
        kept = [cell for cell in cells if cell not in selected_set]
        children = [child for cell in selected for child in cell.children()]
        cells = kept + children

        best_key = max(cache, key=cache.get)
        best_val = cache[best_key]
        print(f"  round {round_idx + 1}/{REFINE_ROUNDS}: "
              f"samples={len(cache):4d}, best P_slip={best_val:.4f} "
              f"at A={best_key[0]:.6f}, omega={best_key[1]:.6f}")

    # One final scoring pass evaluates centers/corners of the latest children.
    for cell in cells:
        cell_values(cell)

    points = np.array([[A, omega] for A, omega in cache.keys()])
    values = np.array([cache[_key(A, omega)] for A, omega in points])
    order = np.argsort(values)[::-1]
    return points, values, order, cells


def plot_results(points, values, order):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    boundary_levels = [0.1, 0.3, 0.5, 0.7]

    A_vals = np.linspace(A_MIN, A_MAX, HEATMAP_N_A)
    OMEGA_vals = np.linspace(OMEGA_MIN, OMEGA_MAX, HEATMAP_N_OMEGA)
    A_mesh, OMEGA_mesh = np.meshgrid(A_vals, OMEGA_vals)
    P_grid = griddata(points, values, (A_mesh, OMEGA_mesh), method='linear')

    # Linear interpolation can leave NaNs on skinny boundary triangles. Fill
    # those with nearest-neighbour values so the heatmap covers the full domain.
    nan_mask = np.isnan(P_grid)
    if np.any(nan_mask):
        P_grid[nan_mask] = griddata(
            points, values, (A_mesh[nan_mask], OMEGA_mesh[nan_mask]),
            method='nearest',
        )

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.pcolormesh(A_mesh, OMEGA_mesh, P_grid,
                       cmap='YlOrRd', vmin=0.0,
                       vmax=max(P_grid.max(), 1e-3), shading='auto')
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('P_slip 95th percentile', fontsize=11)

    levels = [
        level for level in boundary_levels
        if P_grid.min() < level < P_grid.max()
    ]
    if levels:
        cs = ax.contour(A_mesh, OMEGA_mesh, P_grid, levels=levels,
                        colors='steelblue', linewidths=1.5)
        ax.clabel(cs, fmt='P=%.1f', fontsize=8, inline=True)

    ax.scatter(points[:, 0], points[:, 1], s=8, c='black', alpha=0.35,
               linewidths=0, label='evaluated roads')
    ax.scatter(points[order[:TOP_K_REGIONS], 0],
               points[order[:TOP_K_REGIONS], 1],
               s=30, facecolors='none', edgecolors='cyan', linewidths=1.2,
               label='top samples')
    ax.legend(loc='upper right', fontsize=8, frameon=True)

    ax.set_xlabel('Curvature amplitude  A  (rad/m)', fontsize=11)
    ax.set_ylabel('Curvature frequency  omega  (rad/m)', fontsize=11)
    ax.set_title('Adaptive P_slip search: dangerous A/omega regions',
                 fontsize=12)

    plt.tight_layout()
    plt.savefig('prob_demo.png', dpi=150)
    print("Saved: prob_demo.png")


def main():
    points, values, order, _ = adaptive_search()

    print("\nMost dangerous sampled regions:")
    for rank, idx in enumerate(order[:TOP_K_REGIONS], start=1):
        A, omega = points[idx]
        print(f"  {rank:2d}. P_slip={values[idx]:.4f}  "
              f"A={A:.6f} rad/m  omega={omega:.6f} rad/m")

    if SKIP_PLOT:
        print("Skipped plot because PROB_SKIP_PLOT=1")
    else:
        plot_results(points, values, order)


if __name__ == '__main__':
    main()
