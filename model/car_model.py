import numpy as np
from dataclasses import dataclass
from model.config import VehicleParameters

G = 9.81


class SimulationDivergedError(RuntimeError):
    """Raised when a vehicle state becomes non-finite (NaN/inf)."""


def assert_finite(state: np.ndarray, label: str = "state") -> None:
    if not np.all(np.isfinite(state)):
        raise SimulationDivergedError(f"{label} is non-finite: {state}")


@dataclass(frozen=True)
class TireDiagnostics:
    alpha_f: float
    alpha_r: float
    Fzf: float
    Fzr: float
    Fyf_unsat: float
    Fyr_unsat: float
    Fyf: float          # saturated (the value actually used by state_space)
    Fyr: float
    saturation_ratio: float  # max(|Fyf_unsat|/(mu*Fzf), |Fyr_unsat|/(mu*Fzr)); >1 means saturated


class EgoModel:
    """
    Dynamic bicycle model in Frenet-Serret coordinates, with a SATURATED LINEAR
    tire model (independent per-axle clamp |Fy| <= mu*Fz). This is not a
    combined-slip friction ellipse or a nonlinear (Pacejka) tire law -- this
    model never distributes longitudinal force per axle, so a true ellipse
    isn't meaningful here without a larger refactor. Treat it as "linear tire,
    can't exceed the friction limit," nothing more.

    State  x = [ s, e_y, e_psi, vx, vy, r ]
        s      [m]     arclength along the reference path
        e_y    [m]     lateral offset from the centerline (RIGHT-positive)
        e_psi  [rad]   heading error w.r.t. the centerline tangent
        vx     [m/s]   longitudinal velocity, body frame
        vy     [m/s]   lateral velocity, body frame
        r      [rad/s] yaw rate

    Input  u = [ a_x, delta ]
        a_x    [m/s^2] commanded propulsion acceleration BEFORE aerodynamic drag
                       (state_space subtracts (C_d/m)*vx**2 from it internally).
                       A controller that wants a specific NET acceleration
                       (e.g. IDM, which reasons about net accel) must go through
                       drag_compensated_ax() first -- never apply drag twice.
        delta  [rad]   front-wheel steering angle

    Scene  kappa [1/m] path curvature at s,  mu [-] friction coefficient
    """

    IDX_S, IDX_EY, IDX_EPSI, IDX_VX, IDX_VY, IDX_R = range(6)

    def __init__(self, p: VehicleParameters | None = None, v_min: float = 1.0):
        self.p = p or VehicleParameters()
        self.v_min = v_min  # [m/s] low-speed guard: alpha = atan2(., vx) blows up as vx -> 0

    # ---------- tire slip angles ----------

    def alpha_f(self, x, delta):
        vx = max(x[self.IDX_VX], self.v_min)
        return np.arctan2(x[self.IDX_VY] + self.p.l_f * x[self.IDX_R], vx) - delta

    def alpha_r(self, x):
        vx = max(x[self.IDX_VX], self.v_min)
        return np.arctan2(x[self.IDX_VY] - self.p.l_r * x[self.IDX_R], vx)

    # ---------- vertical loads (longitudinal load transfer) ----------

    def normal_loads(self, ax_body):
        """Eq. 5.1d/f. ax_body is the *body-frame* longitudinal accel."""
        p = self.p
        Fzf = (p.l_r * p.m * G - ax_body * p.m * p.h_CG) / p.L
        Fzr = (p.l_f * p.m * G + ax_body * p.m * p.h_CG) / p.L
        return max(Fzf, 0.0), max(Fzr, 0.0)   # wheels cannot pull on the road

    # ---------- lateral tire forces (saturated linear model) ----------

    def lateral_forces(self, x, delta, mu, ax_body):
        p = self.p
        Fzf, Fzr = self.normal_loads(ax_body)
        a_f = self.alpha_f(x, delta)
        a_r = self.alpha_r(x)

        Fyf_unsat = -p.C_f * a_f * mu * (Fzf / p.Fz_nom) * np.cos(delta)
        Fyr_unsat = -p.C_r * a_r * mu * (Fzr / p.Fz_nom)

        Fyf_max = mu * Fzf
        Fyr_max = mu * Fzr
        Fyf = float(np.clip(Fyf_unsat, -Fyf_max, Fyf_max))
        Fyr = float(np.clip(Fyr_unsat, -Fyr_max, Fyr_max))
        return Fyf, Fyr

    def tire_diagnostics(self, x, u, kappa, mu) -> TireDiagnostics:
        """Everything a logger would want: slip angles, raw + saturated forces,
        and a saturation ratio (>1 means the linear force demand exceeds grip)."""
        p = self.p
        a_x, delta = u
        vx = x[self.IDX_VX]
        ax_body = a_x - (p.C_d / p.m) * vx**2
        Fzf, Fzr = self.normal_loads(ax_body)
        a_f, a_r = self.alpha_f(x, delta), self.alpha_r(x)

        Fyf_unsat = -p.C_f * a_f * mu * (Fzf / p.Fz_nom) * np.cos(delta)
        Fyr_unsat = -p.C_r * a_r * mu * (Fzr / p.Fz_nom)
        Fyf_max, Fyr_max = mu * Fzf, mu * Fzr
        Fyf = float(np.clip(Fyf_unsat, -Fyf_max, Fyf_max))
        Fyr = float(np.clip(Fyr_unsat, -Fyr_max, Fyr_max))

        ratio_f = abs(Fyf_unsat) / Fyf_max if Fyf_max > 1e-6 else 0.0
        ratio_r = abs(Fyr_unsat) / Fyr_max if Fyr_max > 1e-6 else 0.0

        return TireDiagnostics(
            alpha_f=float(a_f), alpha_r=float(a_r), Fzf=float(Fzf), Fzr=float(Fzr),
            Fyf_unsat=float(Fyf_unsat), Fyr_unsat=float(Fyr_unsat),
            Fyf=Fyf, Fyr=Fyr, saturation_ratio=float(max(ratio_f, ratio_r)),
        )

    # ---------- longitudinal actuation convention ----------

    def drag_compensated_ax(self, a_net: float, vx: float) -> float:
        """
        Convert a desired NET body-frame longitudinal accel (IDM's output
        convention -- "how fast should I actually speed up/slow down") into the
        a_x INPUT that produces it, since state_space subtracts aero drag from
        a_x internally. Callers must route every net-accel command through this
        exactly once; applying drag compensation twice double-counts it.
        """
        return a_net + (self.p.C_d / self.p.m) * vx**2

    # ---------- continuous-time dynamics ----------

    def state_space(self, x, u, kappa, mu):
        p = self.p
        a_x, delta = u
        _, e_y, e_psi, vx, vy, r = x

        # body-frame longitudinal accel, incl. aero drag (Eq. 5.1a)
        ax_body = a_x - (p.C_d / p.m) * vx**2

        F_yf, F_yr = self.lateral_forces(x, delta, mu, ax_body)

        # Frenet projection. Guard the 1 - kappa*e_y singularity (np.sign(0) == 0,
        # so a raw sign*max approach would divide by zero exactly at denom == 0).
        denom = 1.0 - kappa * e_y
        sign = 1.0 if denom >= 0 else -1.0
        denom = sign * max(abs(denom), 1e-3)
        s_dot = (vx * np.cos(e_psi) - vy * np.sin(e_psi)) / denom

        dxdt = np.zeros(6)
        dxdt[self.IDX_S]    = s_dot
        # e_y is RIGHT-positive (road.py convention); a positive heading
        # error (nose turned toward local +y, i.e. LEFT) must therefore
        # DECREASE e_y, hence the minus sign.
        dxdt[self.IDX_EY]   = -(vx * np.sin(e_psi) + vy * np.cos(e_psi))
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
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        assert_finite(x_next, "EgoModel.step result")
        return x_next
