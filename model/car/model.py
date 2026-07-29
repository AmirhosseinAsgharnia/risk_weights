import numpy as np
from dataclasses import dataclass
from model.car.config import VehicleParameters

G = 9.81

class Dynamics:
    """The Frenet Serret Coordinate model of a vehicle:
    [s e_y e_psi v_x v_y r]
    We use SAE J670 Frame: X-Forward Y-Right and Z-Down
    The equations are based on Rajamanian's book. But since I am using SAE coordinates, there are sign changes"""

    IDX_s, IDX_e_y, IDX_e_psi, IDX_v_x, IDX_v_y, IDX_r = range(6)

    def __init__(self , states , accel , delta, kappa, mu) -> None:
        """
        Args:
            states: State vector [s, e_y, e_psi, v_x, v_y, r], indexed by IDX_*.
            accel: [m/s^2] Commanded longitudinal acceleration.
            delta: [rad] Front wheel steering angle.
            kappa: [1/m] Curvature of the reference path at s.
            mu: [-] Road-tire friction coefficient.
        """

        self.p = VehicleParameters()

        self.states = states
        self.d_states = np.zeros_like(states)
        self.accel  = accel
        self.delta  = delta
        self.kappa  = kappa
        self.mu     = mu

    def step(self):
        """Compute the state derivatives d_states for the current states, storing
        the result in-place. Does not integrate/advance self.states."""

        p = self.p

        s     = self.states [self.IDX_s]
        e_y   = self.states [self.IDX_e_y]
        e_psi = self.states [self.IDX_e_psi]
        v_x = self.states [self.IDX_v_x]
        v_y = self.states [self.IDX_v_y]
        r   = self.states [self.IDX_r]

        self.d_states [self.IDX_s]     = (v_x * np.cos(e_psi) - v_y * np.sin(e_psi)) / (1 - self.kappa * e_y)
        self.d_states [self.IDX_e_y]   = v_x * np.sin(e_psi) + v_y * np.cos(e_psi)
        self.d_states [self.IDX_e_psi] = r - self.kappa * self.d_states [self.IDX_s]
        self.d_states [self.IDX_v_x]   = self.accel + r * v_y
        self.d_states [self.IDX_v_y]   = - r * v_x + (self.F_yf + self.F_yr) / p.m
        self.d_states [self.IDX_r]     = (p.l_f * self.F_yf - p.l_r * self.F_yr) / p.I_zz


    @property
    def alpha_f(self):
        """Front Slip Angle: alpha_f = delta - theta_v_f"""
        p = self.p
        return self.delta - np.atan2(self.states[self.IDX_v_y] + p.l_f * self.states[self.IDX_r], self.states[self.IDX_v_x])

    @property
    def alpha_r(self):
        """Rear Slip Angle: alpha_r = 0 - theta_v_r"""
        p = self.p
        return - np.atan2(self.states[self.IDX_v_y] - p.l_r * self.states[self.IDX_r], self.states[self.IDX_v_x])

    @property
    def F_yf(self):
        """[N] Front axle lateral tire force: linear cornering-stiffness model,
        saturated at +/- mu * F_zf to respect the friction circle."""
        p = self.p
        F_yf_ = p.C_f * self.alpha_f
        return min(max(F_yf_ , - self.mu * self.F_zf) , self.mu * self.F_zf)

    @property
    def F_yr(self):
        """[N] Rear axle lateral tire force: linear cornering-stiffness model,
        saturated at +/- mu * F_zr to respect the friction circle."""
        p = self.p
        F_yr_ = p.C_r * self.alpha_r
        return min(max(F_yr_ , - self.mu * self.F_zr) , self.mu * self.F_zr)

    @property
    def F_zf(self):
        """[N] Front axle normal (vertical) load, including longitudinal
        load transfer from self.accel."""
        p = self.p
        return (p.m * G * p.l_r - p.m * p.h_CG * self.accel) / p.L

    @property
    def F_zr(self):
        """[N] Rear axle normal (vertical) load, including longitudinal
        load transfer from self.accel."""
        p = self.p
        return (p.m * G * p.l_f + p.m * p.h_CG * self.accel) / p.L