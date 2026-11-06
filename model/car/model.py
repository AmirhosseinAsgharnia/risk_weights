import math
from functools import cached_property
from typing import NamedTuple

import numpy as np
from model.car.config import VehicleParameters, G

_LOAD_TRANSFER_ITERATIONS = 5   # fixed-point passes on a_x -> F_z -> F_x, F_y -> a_x; worst residual 1e-3 m/s^2


class TireForces(NamedTuple):
    """One evaluation of the combined tire model [N, rad, m/s^2]. "req" is what the tire is asked for
    before friction limits; the unsuffixed value is what it delivers. Forces are in each axle's wheel
    frame; a_x/a_y are the realized body-frame accelerations from those forces (no r*v terms)."""
    alpha_f: float
    alpha_r: float
    F_xf_req: float
    F_xr_req: float
    F_yf_req: float
    F_yr_req: float
    F_xf: float
    F_xr: float
    F_yf: float
    F_yr: float
    F_yf_max: float      # lateral capacity left after F_xf
    F_yr_max: float
    F_zf: float
    F_zr: float
    a_x: float
    a_y: float


def _clip(x: float, lim: float) -> float:
    return lim if x > lim else (-lim if x < -lim else x)


def tire_forces(v_x: float, v_y: float, r: float, accel: float, delta: float, mu: float,
                p: VehicleParameters) -> TireForces:
    """Friction-circle bicycle tire model, F_x,i^2 + F_y,i^2 <= (mu F_z,i)^2 on each axle.

    accel is the longitudinal tire-force request expressed per unit mass (F_req = m accel). It is
    split by p.brake_front_share / p.drive_front_share. Where an axle's (F_x, F_y) request leaves the
    circle, F_x is scaled so the vector keeps its direction, then F_y is clipped to what is left of the
    circle, sqrt((mu F_z)^2 - F_x^2) -- the same factor, so the force still points along the slip. Giving
    F_x the whole circle first would be wrong: a full stop would leave F_y = 0 on both axles, nothing
    would damp yaw, and the brake force along the steered wheel would spin the car against its steering
    (647 of 1944 braking turns). A brake force opposes motion and fades out below p.v_stop, so braking
    stops the car and never reverses it.

    Slip angle comes from the wheel-frame velocity: alpha = -atan2(v_wheel_y, max(|v_wheel_x|, v_tire_min)).
    For v_wheel_x >= v_tire_min this is exactly delta - atan2(v_y + l_f r, v_x) (0 - atan2(v_y - l_r r, v_x)
    at the rear). Below that it is a damper on wheel side-slip, whose equilibrium is kinematic rolling.

    Load transfer is quasi-static on the REALIZED a_x (which itself depends on F_z, hence the fixed-point
    passes); F_zf + F_zr = m g and both stay >= 0. The returned forces satisfy the circle exactly for the
    returned F_z."""
    cd, sd = math.cos(delta), math.sin(delta)
    v_fy = v_y + p.l_f * r
    alpha_f = -math.atan2(-v_x * sd + v_fy * cd, max(abs(v_x * cd + v_fy * sd), p.v_tire_min))
    alpha_r = -math.atan2(v_y - p.l_r * r, max(abs(v_x), p.v_tire_min))
    F_yf_req, F_yr_req = p.C_f * alpha_f, p.C_r * alpha_r

    F_x_req = p.m * accel
    if accel < 0.0:
        # The fade window widens with any request above mu g, so the realized slope near rest stays at
        # mu g / v_stop: a steeper one is stiff for RK4 and sticks in a limit cycle at a small speed.
        F_x_req *= _clip(v_x / (p.v_stop * max(1.0, -accel / max(mu * G, 1e-9))), 1.0)
        share = p.brake_front_share
    else:
        share = p.drive_front_share

    weight = p.m * G
    a_x = _clip(F_x_req / p.m, mu * G)
    for _ in range(_LOAD_TRANSFER_ITERATIONS):
        F_zf = min(max((weight * p.l_r - p.m * p.h_CG * a_x) / p.L, 0.0), weight)
        F_zr = weight - F_zf
        front = F_zf / weight if share is None else share
        F_xf_req, F_xr_req = front * F_x_req, (1.0 - front) * F_x_req
        lim_f, lim_r = mu * F_zf, mu * F_zr
        F_xf = F_xf_req * min(1.0, lim_f / max(math.hypot(F_xf_req, F_yf_req), 1e-9))
        F_xr = F_xr_req * min(1.0, lim_r / max(math.hypot(F_xr_req, F_yr_req), 1e-9))
        F_yf_max = math.sqrt(max(lim_f * lim_f - F_xf * F_xf, 0.0))
        F_yr_max = math.sqrt(max(lim_r * lim_r - F_xr * F_xr, 0.0))
        F_yf, F_yr = _clip(F_yf_req, F_yf_max), _clip(F_yr_req, F_yr_max)
        a_x = (F_xf * cd - F_yf * sd + F_xr) / p.m
    a_y = (F_xf * sd + F_yf * cd + F_yr) / p.m
    return TireForces(alpha_f, alpha_r, F_xf_req, F_xr_req, F_yf_req, F_yr_req, F_xf, F_xr, F_yf, F_yr,
                      F_yf_max, F_yr_max, F_zf, F_zr, a_x, a_y)


class CarDynamics:
    """The Frenet Serret Coordinate model of a vehicle:
    [s e_y e_psi v_x v_y r]
    We use SAE J670 Frame: X-Forward Y-Right and Z-Down
    The equations are based on Rajamanian's book. But since I am using SAE coordinates, there are sign changes

    Body dynamics, with the front wheel-frame forces rotated by delta (tire_forces gives every force):
        m (dv_x - r v_y) = F_xf cos(delta) - F_yf sin(delta) + F_xr
        m (dv_y + r v_x) = F_xf sin(delta) + F_yf cos(delta) + F_yr
        I_zz dr = l_f (F_xf sin(delta) + F_yf cos(delta)) - l_r F_yr
    accel is a request: the state only ever sees the saturated forces."""

    IDX_s, IDX_e_y, IDX_e_psi, IDX_v_x, IDX_v_y, IDX_r = range(6)

    def __init__(self , states , accel , delta, kappa, mu) -> None:
        """
        Args:
            states: State vector [s, e_y, e_psi, v_x, v_y, r], indexed by IDX_*.
            accel: [m/s^2] Requested longitudinal acceleration (F_req / m), before friction limits.
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

    @cached_property
    def tires(self) -> TireForces:
        """Tire forces at self.states, computed once per instance: Car.step builds one per RK4 stage."""
        return tire_forces(float(self.states[self.IDX_v_x]), float(self.states[self.IDX_v_y]),
                           float(self.states[self.IDX_r]), self.accel, self.delta, self.mu, self.p)

    def step(self):
        """Compute the state derivatives d_states for the current states, storing
        the result in-place. Does not integrate/advance self.states."""

        p = self.p
        t = self.tires

        s     = self.states [self.IDX_s]
        e_y   = self.states [self.IDX_e_y]
        e_psi = self.states [self.IDX_e_psi]
        v_x = self.states [self.IDX_v_x]
        v_y = self.states [self.IDX_v_y]
        r   = self.states [self.IDX_r]

        F_body_yf = t.F_xf * math.sin(self.delta) + t.F_yf * math.cos(self.delta)

        self.d_states [self.IDX_s]     = (v_x * np.cos(e_psi) - v_y * np.sin(e_psi)) / (1 - self.kappa * e_y)
        self.d_states [self.IDX_e_y]   = v_x * np.sin(e_psi) + v_y * np.cos(e_psi)
        self.d_states [self.IDX_e_psi] = r - self.kappa * self.d_states [self.IDX_s]
        self.d_states [self.IDX_v_x]   = t.a_x + r * v_y
        self.d_states [self.IDX_v_y]   = t.a_y - r * v_x
        self.d_states [self.IDX_r]     = (p.l_f * F_body_yf - p.l_r * t.F_yr) / p.I_zz

    @property
    def alpha_f(self):
        """Front slip angle (see tire_forces): delta - theta_v_f above the low-speed floor."""
        return self.tires.alpha_f

    @property
    def alpha_r(self):
        """Rear slip angle (see tire_forces): 0 - theta_v_r above the low-speed floor."""
        return self.tires.alpha_r

    @property
    def F_xf(self):
        """[N] Front axle realized longitudinal force, wheel frame."""
        return self.tires.F_xf

    @property
    def F_xr(self):
        """[N] Rear axle realized longitudinal force."""
        return self.tires.F_xr

    @property
    def F_yf(self):
        """[N] Front axle realized lateral force, wheel frame."""
        return self.tires.F_yf

    @property
    def F_yr(self):
        """[N] Rear axle realized lateral force."""
        return self.tires.F_yr

    @property
    def F_zf(self):
        """[N] Front axle normal load, including load transfer from the realized a_x."""
        return self.tires.F_zf

    @property
    def F_zr(self):
        """[N] Rear axle normal load, including load transfer from the realized a_x."""
        return self.tires.F_zr
