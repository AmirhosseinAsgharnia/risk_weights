"""
Traffic generation: place N_c surrounding cars around a (not-yet-existing)
ego position, each with a random lane/desired-speed/behaviour, and give
each one its IDM *steady-state* speed for the gap it was placed in -- so
the scenario starts already car-following, not transiently accelerating/
braking to reach equilibrium.
"""

import numpy as np
from dataclasses import dataclass, field

from model.car.car import Car, CarState
from model.car.config import VehicleParameters
from low_level_controller.idm import idm_accel, IDM_PRESETS

CAR_LENGTH = 4.0   # [m] used for bumper-to-bumper gap, matching the body
                    # rectangle tests/dynamics_test.py draws cars with.
CAR_WIDTH  = 2.0    # [m] shared with tests/traffic_test.py's plotted body
                    # width and model.collision's overlap test.

SPEED_RANGES: dict[int, tuple[float, float]] = {   # [m/s] desired-speed range per behaviour
    1: (15.0, 20.0),   # conservative
    2: (17.5, 22.5),   # moderate
    3: (20.0, 25.0),   # aggressive
}


@dataclass
class TrafficAgent:
    """A traffic car plus the scenario-level bookkeeping Car/CarState don't
    own themselves (v0 is a per-car IDM preference, not a behaviour-fixed
    constant -- see IDM_PRESETS; target_lane/lane_change_t0/prev_delta are
    MOBIL/far-near runtime state, not general vehicle-dynamics state;
    crashed/crash_t/crash_v are set by model.collision once this car has
    been in a plastic collision -- see that module for what CRASHED means
    and why v0 keeps its pre-crash value rather than being cleared)."""
    car: Car
    v0: float                              # [m/s] this car's own IDM desired speed
    target_lane: int                       # lane it's heading for; == car.state.lane when not changing
    lane_change_t0: float | None = None    # sim time the active lane change started; None if not changing
    prev_delta: float = 0.0                # for clip_steering_rate
    last_lane_change_t: float | None = None   # sim time the last lane change committed; None if never
    crashed: bool = False                  # once True, permanent: skip IDM/MOBIL, passive obstacle
    crash_t: float | None = None           # [s] sim time this car crashed; None if not crashed
    crash_v: float | None = None           # [m/s] momentum-conserved speed at the moment of its crash


def steady_state_speed(v0: float, gap: float, v_leader: float, idm_params,
                        tol: float = 1e-4, max_iter: int = 60) -> float:
    """
    The speed v in [0, v0] at which idm_accel(v, gap, v-v_leader, v0, ...)
    == 0 -- i.e. the speed this car settles to, unaccelerating, given a
    leader at a fixed gap moving at v_leader (gap=inf, any v_leader ==>
    free-road, returns v0). idm_accel(v, ...) is monotonically decreasing
    in v (both the free-road and interaction terms shrink as v grows), so
    bisection applies directly. Degenerate case: if even v=0 already wants
    to decelerate (gap tighter than the standstill jam distance s0), there
    is no equilibrium above a standstill -- return 0 rather than search.
    """
    def f(v):
        return idm_accel(v, gap, v - v_leader, v0, idm_params.a_max, idm_params.b,
                          idm_params.s0, idm_params.T, idm_params.delta)

    lo, hi = 0.0, v0
    if f(lo) <= 0.0:
        return lo
    if f(hi) >= 0.0:
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def generate_traffic(
        n_cars: int,
        ego_s: float = 50.0,
        d_range: tuple[float, float] = (-40.0, 40.0),
        lanes: tuple[int, ...] = (0, 1, 2),
        speed_ranges: dict[int, tuple[float, float]] = SPEED_RANGES,
        behaviours: tuple[int, ...] = (1, 2, 3),
        min_gap: float = 6.0,
        max_place_attempts: int = 200,
        rng: np.random.Generator | None = None,
) -> list[TrafficAgent]:
    """
    Place n_cars around ego_s (no ego car itself -- ego_s is just the
    reference point d is measured from). Each car gets an independent
    random lane, longitudinal offset d ~ U(d_range) (so s = ego_s + d,
    resampled up to max_place_attempts times if it would land within
    min_gap of an already-placed car in the same lane -- unconstrained
    i.i.d. placement can otherwise spawn two cars overlapping, which sends
    IDM's (s_star/gap)^2 term toward infinity and blows up the first
    integration step), behaviour ~ choice(behaviours) (selecting its
    IDM_PRESETS/FAR_NEAR_PRESETS entry), and desired speed v0 ~
    U(speed_ranges[behaviour]) -- conservative/moderate/aggressive draw
    from their own (narrower, and offset) range rather than sharing one.

    Initial speeds are each car's IDM *steady-state* speed for the gap it
    landed in: within each lane, cars are resolved front-to-back (largest
    s first) so every car's leader has an already-known steady-state speed
    by the time it's this car's turn; the front-most car in each lane (no
    leader) gets its own v0 outright.
    """
    rng = rng or np.random.default_rng()

    cars: list[Car] = []
    v0s: list[float] = []
    s_taken_by_lane: dict[int, list[float]] = {lane: [] for lane in lanes}
    for _ in range(n_cars):
        lane = int(rng.choice(lanes))
        for _ in range(max_place_attempts):
            d = float(rng.uniform(*d_range))
            s_candidate = ego_s + d
            if all(abs(s_candidate - s) >= min_gap for s in s_taken_by_lane[lane]):
                break
        s_taken_by_lane[lane].append(s_candidate)
        behaviour = int(rng.choice(behaviours))
        v0 = float(rng.uniform(*speed_ranges[behaviour]))

        state = CarState(s=s_candidate, e_y=0.0, e_psi=0.0, v_x=v0, lane=lane)
        car = Car(state=state, vehicle_params=VehicleParameters(), behaviour=behaviour)
        cars.append(car)
        v0s.append(v0)

    for lane in lanes:
        idx_in_lane = [i for i, c in enumerate(cars) if c.state.lane == lane]
        idx_in_lane.sort(key=lambda i: cars[i].state.s, reverse=True)  # front-most first

        leader_s, leader_v = None, None
        for i in idx_in_lane:
            car, v0 = cars[i], v0s[i]
            idm_params = IDM_PRESETS[car.behaviour]
            if leader_s is None:
                gap, v_leader = float("inf"), 0.0
            else:
                gap = leader_s - car.state.s - CAR_LENGTH
                v_leader = leader_v
            v_ss = steady_state_speed(v0, gap, v_leader, idm_params)
            car.state.v_x = v_ss
            leader_s, leader_v = car.state.s, v_ss

    return [
        TrafficAgent(car=car, v0=v0, target_lane=car.state.lane)
        for car, v0 in zip(cars, v0s)
    ]
