"""
Controller demo — MPC drives the ego vehicle.

Road and environment parameters are set here.
Run:  python controller_demo.py
"""

import numpy as np
import dynamics.env_dynamics as env_dynamics
import environment.env_scenario as env_scenario
import risks.env_risk as env_risk
import visualize.env_visualize as env_visualize
from environment.scenarios import SandwichScenario
from controller.mpc import MPC

# ── Risk weights ───────────────────────────────────────────────────────────────

risk_weights = {
    'c2c':  0.5,
    'slip': 0.0,
    'roll': 0.0,
    'ld':   0.5,
}

# ── Scenario ──────────────────────────────────────────────────────────────────

scenario = SandwichScenario(
    n_lanes    = 3,
    ego_lane   = 1,           # 0=rightmost, 1=middle, 2=leftmost
    lane_width = 3.7,

    # κ(s) = A·sin(ω·s)  — try kappa_A=0 for straight road; λ = 2π/ω
    kappa_A     = 0.005,    # amplitude (rad/m)
    kappa_omega = 0.030,    # frequency (rad/m); λ ≈ 209 m

    ego_ey   = 0.0,           # within-lane SAE offset (m), 0 = lane centre
    ego_epsi = 0.0,           # initial heading error (rad)
    v0       = 20.0,          # m/s

    # Sandwich: vehicle ahead and behind in same lane, one on the side
    vehicles = [
        {'x_rel':  15.0, 'v_x': 18.0, 'lane': 1},   # ahead, same lane (slow)
        {'x_rel':  -8.0, 'v_x': 20.0, 'lane': 1},   # behind, same lane (fast)
        {'x_rel':   0.0, 'v_x': 28.0, 'lane': 2},   # side, rightmost lane
    ],

    dt    = 0.05,
    t_end = 8.0,
)

print("Env vector:", scenario.to_vector())
print(scenario)


# ── Simulation ────────────────────────────────────────────────────────────────

def run(sc: SandwichScenario, quiet: bool = False):
    mpc = MPC(risk_weights=risk_weights, v_ref=20.0, w_v=0.01)

    ego_state = sc.ego_init_state()
    ego_lane  = sc.ego_lane
    s_ego     = 0.0

    vehicles  = env_scenario.init_vehicles(sc, s_ego_init=s_ego)
    road_geom = env_visualize.RoadGeometry(sc.kappa_A, sc.kappa_omega, s_max=600.0)

    history = []
    frames  = []
    n_steps = int(sc.t_end / sc.dt)

    for step_idx in range(n_steps):
        t_now = step_idx * sc.dt

        # ── MPC: compute optimal control ──────────────────────────────────────
        a_opt, delta_opt = mpc.solve(ego_state, s_ego, ego_lane, vehicles, sc)

        # ── Risk assessment at chosen control ─────────────────────────────────
        ego_fut, sur_fut = mpc._predict(
            ego_state, s_ego, ego_lane, vehicles, sc, a_opt, delta_opt
        )
        J, ps, pr, pc, pl = env_risk.total_cost(
            ego_fut, sur_fut, a_opt, delta_opt, sc, risk_weights
        )

        if not quiet:
            e_y, psi, e_psi, *_ = ego_state
            print(
                f"t={t_now:5.2f}s  lane={ego_lane}  s={s_ego:.1f}  "
                f"e_y={e_y:+.3f}  eψ={np.degrees(e_psi):.2f}°  "
                f"v_x={ego_state[4]:.2f}  "
                f"a={a_opt:+.2f}  δ={np.degrees(delta_opt):+.2f}°  "
                f"J={J:.4f}  Pc2c={pc:.3f}  Pld={pl:.3f} Pslip={ps:.3f}  Proll={pr:.3f}"
            )

        history.append({
            't': t_now, 'ego_state': ego_state.copy(),
            'ego_lane': ego_lane, 's_ego': s_ego,
            'a': a_opt, 'delta': delta_opt,
            'J': J, 'P_slip': ps, 'P_roll': pr, 'P_c2c': pc, 'P_ld': pl,
        })

        # ── Render frame ──────────────────────────────────────────────────────
        frames.append(env_visualize._draw_frame(
            ego_state  = ego_state,
            ego_lane   = ego_lane,
            s_ego      = s_ego,
            vehicles   = vehicles,
            ego_fut    = ego_fut,
            sur_fut    = sur_fut,
            risks      = {'P_slip': ps, 'P_roll': pr, 'P_c2c': pc, 'P_ld': pl},
            road_geom  = road_geom,
            scenario   = sc,
            step_n     = step_idx,
        ))

        # ── Advance dynamics ──────────────────────────────────────────────────
        ego_state, s_ego, ego_lane = env_dynamics.step(
            ego_state, a_opt, delta_opt, s_ego, ego_lane,
            sc.kappa_A, sc.kappa_omega, sc.n_lanes, sc.lane_width, sc.dt,
        )
        vehicles = env_scenario.step_vehicles(vehicles, sc.dt)

        # ── Road departure check ──────────────────────────────────────────────
        if ego_lane < 0 or ego_lane >= sc.n_lanes:
            if not quiet:
                print(f"Road departure at t={t_now:.2f}s  (lane={ego_lane})")
            break

    return history, frames


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    history, frames = run(scenario)
    print(f"\n{len(history)} steps simulated.")
    env_visualize.make_gif(frames, path='controller_demo.gif', fps=20)
