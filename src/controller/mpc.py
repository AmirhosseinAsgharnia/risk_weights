import numpy as np
from scipy.optimize import minimize

import dynamics.env_dynamics as env_dynamics
import risks.env_risk as env_risk
from config.config import (
    a_min, a_max, delta_min, delta_max,
    T_horizon, dt_pred,
    w_slip, w_roll, w_c2c, w_ld,
)

_N  = int(T_horizon / dt_pred)
_t  = np.arange(1, _N + 1) * dt_pred


class MPC:
    def __init__(self, risk_weights=None, v_ref: float = 20.0, w_v: float = 0.01):
        self._prev_u = np.array([0.0, 0.0])
        self.risk_weights = risk_weights or {
            'slip': w_slip, 'roll': w_roll, 'c2c': w_c2c, 'ld': w_ld,
        }
        self.v_ref = v_ref   # target cruise speed (m/s)
        self.w_v   = w_v     # weight on speed-tracking term

    def _predict(self, ego_state, s_ego, ego_lane, vehicles, scenario, a, delta):
        """Constant-(a, delta) N-step prediction of ego and surrounding vehicles."""
        state   = ego_state.copy()
        s       = float(s_ego)
        lane    = ego_lane
        kappa_A, kappa_omega = scenario.kappa_A, scenario.kappa_omega
        n_lanes = scenario.n_lanes
        lw      = scenario.lane_width

        e_y_arr   = np.empty(_N)
        y_abs_arr = np.empty(_N)
        e_psi_arr = np.empty(_N)
        v_x_arr   = np.empty(_N)
        s_arr     = np.empty(_N)

        for k in range(_N):
            state, s, lane = env_dynamics.step(
                state, a, delta, s, lane, kappa_A, kappa_omega, n_lanes, lw, dt_pred
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
                accel = float(veh.get('accel', 0.0))
                sur_fut.append({
                    's':     veh['s'] + veh['v_x'] * _t + 0.5 * accel * _t ** 2,
                    'y_abs': np.full(_N, veh['y_abs']),
                })
            else:
                sur_fut.append(None)

        return ego_fut, sur_fut

    def solve(self, ego_state, s_ego, ego_lane, vehicles, scenario):
        """
        Solve the 2-variable MPC problem for current ego state.

        Returns (a_opt, delta_opt).
        """
        rw = self.risk_weights

        def cost(u):
            a, delta = float(u[0]), float(u[1])
            ego_fut, sur_fut = self._predict(
                ego_state, s_ego, ego_lane, vehicles, scenario, a, delta
            )
            ps = env_risk.P_slip(ego_fut, a, delta)
            pr = env_risk.P_roll(ego_fut, delta)
            pc = env_risk.P_c2c(ego_fut, sur_fut)
            pl = env_risk.P_ld(ego_fut, scenario.lane_width)
            J_risk  = (rw['slip'] * ps + rw['roll'] * pr
                       + rw['c2c']  * pc + rw['ld']   * pl)
            J_speed = self.w_v * (ego_fut['v_x'][-1] - self.v_ref) ** 2
            return J_risk + J_speed + 1e-2 * delta ** 2

        bounds = [(a_min, a_max), (delta_min, delta_max)]

        candidates = [
            self._prev_u.copy(),
            np.array([0.0,  0.0]),
            np.array([-3.0, 0.0]),
        ]

        best_val = np.inf
        u_opt    = self._prev_u.copy()
        for u_init in candidates:
            r = minimize(cost, u_init, method='SLSQP', bounds=bounds,
                         options={'maxiter': 50, 'ftol': 1e-4})
            if r.fun < best_val:
                best_val = r.fun
                u_opt    = r.x

        self._prev_u = u_opt.copy()
        return float(u_opt[0]), float(u_opt[1])
