import numpy as np
from model.config import VehicleParameters

G = 9.81
V_MIN = 1.0   # [m/s] low-speed guard: alpha = atan2(., vx) blows up as vx -> 0


class CarDynamics:
    """
    Dynamic bicycle model in Frenet-Serret coordinates.

    State  x = [ s, e_y, e_psi, vx, vy, r ]
        s      [m]     arclength along the reference path
        e_y    [m]     lateral offset from the centerline
        e_psi  [rad]   heading error w.r.t. the centerline tangent
        vx     [m/s]   longitudinal velocity, body frame
        vy     [m/s]   lateral velocity, body frame
        r      [rad/s] yaw rate

    Input  u = [ a_x, delta ]
    Scene  kappa [1/m] path curvature at s,  mu [-] friction coefficient
    """

    IDX_S, IDX_EY, IDX_EPSI, IDX_VX, IDX_VY, IDX_R = range(6)

    def __init__(self, p: VehicleParameters | None = None):
        self.p = p or VehicleParameters()

    # ---------- tire slip angles ----------

    def alpha_f(self, x, delta):
        vx = max(x[self.IDX_VX], V_MIN)
        return np.arctan2(x[self.IDX_VY] + self.p.l_f * x[self.IDX_R], vx) - delta

    def alpha_r(self, x):
        vx = max(x[self.IDX_VX], V_MIN)
        return np.arctan2(x[self.IDX_VY] - self.p.l_r * x[self.IDX_R], vx)

    # ---------- vertical loads (longitudinal load transfer) ----------

    def normal_loads(self, ax_body):
        """Eq. 5.1d/f. ax_body is the *body-frame* longitudinal accel."""
        p = self.p
        Fzf = (p.l_r * p.m * G - ax_body * p.m * p.h_CG) / p.L
        Fzr = (p.l_f * p.m * G + ax_body * p.m * p.h_CG) / p.L
        return max(Fzf, 0.0), max(Fzr, 0.0)   # wheels cannot pull on the road

    # ---------- lateral tire forces ----------

    def lateral_forces(self, x, delta, mu, ax_body):
        p = self.p
        Fzf, Fzr = self.normal_loads(ax_body)
        a_f = self.alpha_f(x, delta)
        a_r = self.alpha_r(x)

        F_yf = -p.C_f * a_f * mu * (Fzf / p.Fz_nom) * np.cos(delta)
        F_yr = -p.C_r * a_r * mu * (Fzr / p.Fz_nom)
        return F_yf, F_yr

    # ---------- continuous-time dynamics ----------

    def state_space(self, x, u, kappa, mu):
        p = self.p
        a_x, delta = u
        _, e_y, e_psi, vx, vy, r = x

        # body-frame longitudinal accel, incl. aero drag (Eq. 5.1a)
        ax_body = a_x - (p.C_d / p.m) * vx**2

        F_yf, F_yr = self.lateral_forces(x, delta, mu, ax_body)

        # Frenet projection. Guard the 1 - kappa*e_y singularity.
        denom = 1.0 - kappa * e_y
        denom = np.sign(denom) * max(abs(denom), 1e-3)
        s_dot = (vx * np.cos(e_psi) - vy * np.sin(e_psi)) / denom

        dxdt = np.zeros(6)
        dxdt[self.IDX_S]    = s_dot
        dxdt[self.IDX_EY]   = vx * np.sin(e_psi) + vy * np.cos(e_psi)
        dxdt[self.IDX_EPSI] = r - kappa * s_dot
        dxdt[self.IDX_VX]   = ax_body + vy * r
        dxdt[self.IDX_VY]   = -vx * r + (F_yf + F_yr) / p.m
        dxdt[self.IDX_R]    = (p.l_f * F_yf - p.l_r * F_yr) / p.I_zz
        return dxdt

    # ---------- integrator ----------

    def step(self, x, u, kappa, mu, dt):
        f = lambda z: self.state_space(z, u, kappa, mu)
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)