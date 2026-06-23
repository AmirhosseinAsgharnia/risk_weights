"""
prob_demo.py — 2D sweep of (kappa_A, kappa_omega) → P_slip heatmap.

For each (A, ω) pair, runs the MPC-controlled ego on that road geometry
(no surrounding vehicles) and records the peak P_slip encountered.
Plots a heatmap with iso-risk boundary contours.

MPC is re-solved every MPC_INTERVAL seconds (zero-order hold in between)
to keep the grid evaluation fast.

Run: python prob_demo.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dynamics.env_dynamics as env_dynamics
import environment.env_scenario as env_scenario
import risks.env_risk as env_risk
from environment.scenarios import SandwichScenario
from controller.mpc import MPC

# ── Grid ───────────────────────────────────────────────────────────────────────

A_vals     = np.linspace(0.0,   0.02,  15)    # curvature amplitude (rad/m)
OMEGA_vals = np.linspace(0.005, 0.10,  15)    # curvature frequency (rad/m)

# ── Fixed scenario (road only, no surrounding vehicles) ────────────────────────

_BASE = dict(
    n_lanes    = 1,
    ego_lane   = 0,
    lane_width = 3.7,
    ego_ey     = 0.0,
    ego_epsi   = 0.0,
    v0         = 20.0,
    vehicles   = [],
    dt         = 0.05,
    t_end      = 8.0,
)

RISK_WEIGHTS = {'c2c': 0.35, 'slip': 0.15, 'roll': 0.15, 'ld': 0.35}
V_REF        = 20.0
MPC_INTERVAL = 0.05    # s between MPC solves

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

# ── Grid search ────────────────────────────────────────────────────────────────

nA, nO  = len(A_vals), len(OMEGA_vals)
P_grid  = np.zeros((nO, nA))

print(f"Grid search: {nA} × {nO} = {nA * nO} evaluations "
      f"(MPC every {MPC_INTERVAL} s)")

for i, omega in enumerate(OMEGA_vals):
    for j, A in enumerate(A_vals):
        P_grid[i, j] = eval_road(A, omega)
    print(f"  ω={omega:.4f} done  —  row max P_slip={P_grid[i].max():.4f}")

# ── Plot: heatmap + iso-risk boundaries ───────────────────────────────────────

BOUNDARY_LEVELS = [0.1, 0.3, 0.5, 0.7]   # P_slip iso-contours to highlight

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.pcolormesh(A_vals, OMEGA_vals, P_grid,
                   cmap='YlOrRd', vmin=0.0, vmax=max(P_grid.max(), 1e-3),
                   shading='auto')
cb = plt.colorbar(im, ax=ax)
cb.set_label('Peak P_slip', fontsize=11)

# Iso-risk boundary contours
cs = ax.contour(A_vals, OMEGA_vals, P_grid,
                levels=BOUNDARY_LEVELS,
                colors='steelblue', linewidths=1.5)
ax.clabel(cs, fmt='P=%.1f', fontsize=8, inline=True)

ax.set_xlabel('Curvature amplitude  A  (rad/m)', fontsize=11)
ax.set_ylabel('Curvature frequency  ω  (rad/m)', fontsize=11)
ax.set_title('Peak P_slip — iso-risk boundaries', fontsize=12)

plt.tight_layout()
plt.savefig('prob_demo.png', dpi=150)
print("Saved: prob_demo.png")
