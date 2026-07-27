import numpy as np
from dataclasses import dataclass
from model.road import RoadScenario

S_EGO0 = 30.0     # [m] ego always starts here
CAR_LENGTH = 4.0  # [m]
CAR_WIDTH = 2.0   # [m]


@dataclass
class Car:
    lane: int
    state: np.ndarray
    length: float = CAR_LENGTH
    width: float = CAR_WIDTH
    v0: float | None = None  # [m/s] desired (IDM) speed; None for non-IDM-controlled cars


def frenet_to_global(road: RoadScenario, s, e_y, e_psi=0.0):
    """Global (x, y, psi) of a point at arclength s, offset e_y from the
    reference centreline (same sign convention as road.d_c)."""
    idx = int(round(s / road.ds))
    idx = min(max(idx, 0), len(road.s) - 1)

    psi_ref = road.heading[idx]
    n_x = np.sin(psi_ref)
    n_y = -np.cos(psi_ref)

    # Both ends of the road are straight by construction (kappa = 0), so
    # linearly extrapolating along the boundary heading is exact for s
    # outside [0, s_max) -- not just an approximation to stop position
    # (and hence heading) from freezing at the last tabulated sample.
    ds_extra = s - road.s[idx]

    x = road.x[idx] + ds_extra * np.cos(psi_ref) + e_y * n_x
    y = road.y[idx] + ds_extra * np.sin(psi_ref) + e_y * n_y
    return x, y, psi_ref + e_psi


def init_ego(road: RoadScenario, lane=1, speed=25.0) -> Car:
    """
    Ego initial condition, in Frenet coordinates (state for EgoModel).
    state = [s, e_y, e_psi, vx, vy, r]
    """
    if not (0 <= lane < road.lane_num):
        raise ValueError(f"lane must be in [0, {road.lane_num - 1}], got {lane}")

    e_y = road.d_c[lane]
    state = np.array([S_EGO0, e_y, 0.0, speed, 0.0, 0.0])
    return Car(lane=lane, state=state)


def init_surr(road: RoadScenario, ego_speed, cars) -> list[Car]:
    """
    Surrounding-car initial conditions, in Frenet coordinates
    (state for EgoModel), placed relative to the ego car.
    state = [s, e_y, e_psi, vx, vy, r]

    cars: iterable of (N, dS, dV)
        N   lane index of the surrounding car
        dS  [m]   longitudinal offset from the ego car: s_car = S_EGO0 + dS
        dV  [m/s] offset from the ego car's speed defining this car's own
            *desired* (IDM v0) speed: v0_car = ego_speed + dV. A dV-based
            initial speed alone would just be a transient -- IDM relaxes v
            back to v0 within a few seconds regardless of where it started,
            so dV only produces a lasting behavioural difference (a durably
            slower/faster car) if it sets v0, not just the initial v.
    """
    surr = []
    for N, dS, dV in cars:
        if not (0 <= N < road.lane_num):
            raise ValueError(f"lane must be in [0, {road.lane_num - 1}], got {N}")

        v0 = ego_speed + dV
        state = np.array([S_EGO0 + dS, road.d_c[N], 0.0, v0, 0.0, 0.0])
        surr.append(Car(lane=N, state=state, v0=v0))
    return surr
