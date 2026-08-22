import itertools
import numpy as np
from dataclasses import dataclass, replace
from model.car.config import VehicleParameters
from model.car.model import CarDynamics
from solvers.rk4 import rk4

_id_counter = itertools.count()


@dataclass
class CarState:
    """A car's dynamics state (CarDynamics convention, [s, e_y, e_psi, v_x,
    v_y, r]) plus the bits Car tracks alongside it. x/y (global Cartesian)
    and lane are left None -- assigning them is a road/scenario concern,
    not this class's."""

    s:     float = 0.0   # [m] arclength along the road centreline
    e_y:   float = 0.0   # [m] lateral offset from the centreline, SAE +right
    e_psi: float = 0.0   # [rad] heading error relative to the centreline
    v_x:   float = 0.0   # [m/s] longitudinal speed, body frame
    v_y:   float = 0.0   # [m/s] lateral speed, body frame
    r:     float = 0.0   # [rad/s] yaw rate
    delta: float = 0.0   # [rad] last commanded front steering angle
    x:     float | None = None   # [m] global Cartesian position
    y:     float | None = None   # [m] global Cartesian position
    lane:  int   | None = None   # current lane index

    def as_array(self) -> np.ndarray:
        """[s, e_y, e_psi, v_x, v_y, r] -- the CarDynamics state vector."""
        return np.array([self.s, self.e_y, self.e_psi, self.v_x, self.v_y, self.r])


class Car:
    """A single vehicle: owns its dynamics model and state, and integrates
    itself forward one timestep at a time given commanded (accel, delta)
    and the road curvature/friction at its current position.

    `controller` is accepted and stored but not used yet -- until it's
    wired in, step() takes (accel, delta) directly from the caller instead
    of computing them itself.
    """

    def __init__(self,
                 car_id: int | None = None,
                 model=CarDynamics,
                 controller=None,
                 state: CarState | None = None,
                 vehicle_params: VehicleParameters | None = None,
                 behaviour: int = 2) -> None:

        assert behaviour in (1, 2, 3), f"behaviour must be 1 (conservative), 2 (moderate) or 3 (aggressive), got {behaviour}"

        self.car_id = car_id if car_id is not None else next(_id_counter)
        self.model = model
        self.controller = controller
        self.state = state if state is not None else CarState()
        self.vehicle_params = vehicle_params or VehicleParameters()
        # Driving style: 1 = conservative, 2 = moderate, 3 = aggressive.
        # Selects which IDM/far-near parameter preset this car drives with
        # -- see controllers.idm.IDM_PRESETS and
        # controllers.far_near.FAR_NEAR_PRESETS.
        self.behaviour = behaviour

    def step(self, accel: float, delta: float, kappa: float, mu: float, dt: float) -> CarState:
        """Advance the dynamics state by dt under the commanded (accel,
        delta), given the road curvature/friction at the car's current
        position -- both are the caller's responsibility to look up (this
        class holds no reference to a road).

        RK4-integrates self.model (CarDynamics by default), rebuilding it
        at every stage: CarDynamics computes its derivative from the
        states/inputs it was constructed with rather than exposing a
        standalone f(t, x), so each rk4 stage needs its own instance at
        that stage's trial state.
        """
        def f(t, x):
            dynamics = self.model(x, accel, delta, kappa, mu)
            dynamics.p = self.vehicle_params
            dynamics.step()
            return dynamics.d_states

        s, e_y, e_psi, v_x, v_y, r = rk4(f, 0.0, self.state.as_array(), dt)
        self.state = replace(self.state, s=s, e_y=e_y, e_psi=e_psi,
                              v_x=v_x, v_y=v_y, r=r, delta=delta)
        return self.state
