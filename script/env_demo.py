"""
Environment demo — a=delta=0 for all vehicles.

Edit the SandwichScenario block to explore different configurations.
Run:  python env_demo.py
"""

import numpy as np
import dynamics.env_dynamics as env_dynamics
import environment.env_scenario as env_scenario
import risks.env_risk as env_risk
import visualize.env_visualize as env_visualize
from environment.scenarios import SandwichScenario

# ── Scenario ──────────────────────────────────────────────────────────────────

scenario = SandwichScenario(
    n_lanes    = 3,
    ego_lane   = 1,           # 0=rightmost, 1=middle, 2=leftmost
    lane_width = 3.7,

    # κ(s) = k0 + k1·s  (rad/m)  — try k0=0, k1=0 for straight road
    k0 = -0.001,
    k1 = 0.00001,

    ego_ey   = 0.0,           # within-lane SAE offset (m), 0 = lane centre
    ego_epsi = 0.0,           # initial heading error (rad)
    v0       = 25.0,          # m/s

    # Up to 10 vehicles; omitted slots are automatically invalid
    vehicles = [
        {'x_rel':  15.0, 'v_x': 20.0, 'lane': 1},   # ahead, same lane
        {'x_rel':  -8.0, 'v_x': 20.0, 'lane': 1},   # behind, same lane
        {'x_rel':   0.0, 'v_x': 28.0, 'lane': 0},   # side, rightmost lane
    ],

    dt    = 0.05,
    t_end = 8.0,
)

# Show the env vector for debugging
print("Env vector:", scenario.to_vector())
print(scenario)


# ── Simulation ────────────────────────────────────────────────────────────────

def run(sc: SandwichScenario, quiet: bool = False):
    ego_state = sc.ego_init_state()          # [e_y, psi, e_psi, psi_dot, v_x, v_y]
    ego_lane  = sc.ego_lane
    s_ego     = 0.0

    vehicles  = env_scenario.init_vehicles(sc, s_ego_init=s_ego)
    road_geom = env_visualize.RoadGeometry(sc.k0, sc.k1, s_max=600.0)

    history = []
    frames  = []
    n_steps = int(sc.t_end / sc.dt)

    for step_idx in range(n_steps):
        t_now = step_idx * sc.dt

        # ── Risk assessment ───────────────────────────────────────────────────
        ego_fut, sur_fut = env_risk.predict_zero_control(
            ego_state, s_ego, ego_lane, vehicles, sc
        )
        J, ps, pr, pc, pl = env_risk.total_cost(
            ego_fut, sur_fut, 0.0, 0.0, sc
        )

        if not quiet:
            e_y, psi, e_psi, *_ = ego_state
            print(
                f"t={t_now:5.2f}s  lane={ego_lane}  s={s_ego:.1f}  "
                f"e_y={e_y:+.3f}  ψ={np.degrees(psi):.2f}°  "
                f"eψ={np.degrees(e_psi):.2f}°  v_x={ego_state[4]:.2f}  "
                f"J={J:.4f}  Pc2c={pc:.3f}  Pld={pl:.3f}"
            )

        history.append({
            't': t_now, 'ego_state': ego_state.copy(),
            'ego_lane': ego_lane, 's_ego': s_ego,
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
            ego_state, 0.0, 0.0, s_ego, ego_lane,
            sc.k0, sc.k1, sc.n_lanes, sc.lane_width, sc.dt,
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
    env_visualize.make_gif(frames, path='env_demo.gif', fps=20)
