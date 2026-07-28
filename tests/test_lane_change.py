import numpy as np
from model.road import RoadScenario
from model.car_model import EgoModel
from model.car_init import Car
from model.IDM_MOBIL import LaneChangeParams, steering_cmd
from model.lane_change import (
    _q, _q_dot, occupied_lanes, lane_change_reference,
    start_lane_change, try_complete_lane_change, LaneChangeTolerances,
)


def test_quintic_boundary_conditions():
    assert _q(0.0) == 0.0
    assert _q(1.0) == 1.0
    assert _q_dot(0.0) == 0.0
    assert _q_dot(1.0) == 0.0
    # monotonic and within [0, 1] in between
    taus = np.linspace(0, 1, 21)
    values = [_q(t) for t in taus]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert all(b >= a for a, b in zip(values, values[1:]))


def _car(road, lane, s=100.0):
    return Car(current_lane=lane, target_lane=lane, state=np.array([s, road.d_c[lane], 0.0, 20.0, 0.0, 0.0]))


def test_reference_holds_current_lane_when_not_changing():
    road = RoadScenario(lane_num=3)
    car = _car(road, lane=1)
    e_y_ref, e_y_ref_rate = lane_change_reference(car, road, t=5.0)
    assert e_y_ref == road.d_c[1]
    assert e_y_ref_rate == 0.0


def test_start_lane_change_captures_e_y_start_once():
    road = RoadScenario(lane_num=3)
    car = _car(road, lane=1)
    car.state[1] = 0.37  # pretend it's already drifted a bit before the decision
    start_lane_change(car, target_lane=0, t=10.0)

    assert car.target_lane == 0
    assert car.lane_change_active is True
    assert car.lane_change_start_time == 10.0
    assert car.e_y_start == 0.37

    # moving the car afterward must NOT change e_y_start
    car.state[1] = 1.5
    assert car.e_y_start == 0.37


def test_reference_blends_from_start_to_target_and_back_to_zero_rate():
    road = RoadScenario(lane_num=3)
    car = _car(road, lane=1)
    start_lane_change(car, target_lane=0, t=0.0)  # e_y_start = 0, target = road.d_c[0] = +l_w

    e_y_0, rate_0 = lane_change_reference(car, road, t=0.0)
    e_y_mid, rate_mid = lane_change_reference(car, road, t=car.lane_change_duration / 2)
    e_y_end, rate_end = lane_change_reference(car, road, t=car.lane_change_duration)
    e_y_after, rate_after = lane_change_reference(car, road, t=car.lane_change_duration + 5.0)

    assert np.isclose(e_y_0, 0.0) and rate_0 == 0.0
    assert 0.0 < e_y_mid < road.d_c[0]
    assert rate_mid > 0  # moving toward the target
    assert np.isclose(e_y_end, road.d_c[0])
    assert np.isclose(rate_end, 0.0, atol=1e-9)
    # tau clips at 1 -- holding past the nominal duration doesn't overshoot the reference
    assert np.isclose(e_y_after, road.d_c[0])
    assert rate_after == 0.0


def test_completion_requires_both_duration_elapsed_and_convergence():
    road = RoadScenario(lane_num=3)
    car = _car(road, lane=1)
    start_lane_change(car, target_lane=0, t=0.0)
    tol = LaneChangeTolerances()

    # duration not elapsed yet, even if (implausibly) already converged
    car.state[1] = road.d_c[0]
    car.state[2] = 0.0
    car.state[4] = 0.0
    assert try_complete_lane_change(car, road, t=0.1, tol=tol) is False
    assert car.lane_change_active is True

    # duration elapsed but not converged (still far from target)
    car.state[1] = road.d_c[1]
    assert try_complete_lane_change(car, road, t=car.lane_change_duration + 0.1, tol=tol) is False
    assert car.lane_change_active is True

    # both satisfied -> completes, cooldown reference set on completion
    car.state[1] = road.d_c[0]
    ok = try_complete_lane_change(car, road, t=car.lane_change_duration + 0.2, tol=tol)
    assert ok is True
    assert car.lane_change_active is False
    assert car.current_lane == 0
    assert car.last_lane_change_time == car.lane_change_duration + 0.2


def test_occupied_lanes_straddles_two_lanes_mid_change():
    road = RoadScenario(lane_num=3)  # lanes at d_c = [+l_w, 0, -l_w], l_w = 4.0
    centered = occupied_lanes(road, e_y=0.0, width=2.0)
    assert centered == (1,)

    straddling = occupied_lanes(road, e_y=road.l_w / 2.0, width=2.0)  # exactly on the 0/1 boundary
    assert set(straddling) >= {0, 1}


def test_steering_direction_is_correct_for_a_rightward_target():
    """
    e_y is right-positive. Moving from e_y=0 toward a target e_y_target=+4
    (to the right) requires e_psi > 0 at some point during the approach
    (nose turned toward local +y/left initially decreases e_y -- wait, per
    EgoModel: de_y/dt = -(vx sin(e_psi) + vy cos(e_psi)), so e_y INCREASES
    (moves right) when e_psi is NEGATIVE. We check the sign a short time
    into the transition (not at tau=0, where the quintic's rate is exactly
    zero by construction -- a deliberate smooth start, not a bug), and
    confirm the car actually ends up closer to the target, not further away.
    """
    road = RoadScenario(kappa_max=0.0, lane_num=3)
    model = EgoModel()
    lc_p = LaneChangeParams()

    car = _car(road, lane=1, s=100.0)  # e_y starts at road.d_c[1] = 0.0
    start_lane_change(car, target_lane=0, t=0.0)  # target e_y = road.d_c[0] = +4.0
    e_y_target = road.d_c[0]

    dt = 0.1
    t = 0.0
    e_y_initial = car.state[1]
    saw_negative_e_psi = False
    for _ in range(15):  # 1.5s into a 3s manoeuvre
        e_y_ref, e_y_ref_rate = lane_change_reference(car, road, t)
        delta, car.lane_error_integral = steering_cmd(
            car.state, 0.0, e_y_ref, e_y_ref_rate, model.p.L, lc_p,
            integral=car.lane_error_integral, prev_delta=car.last_delta, dt=dt,
        )
        car.last_delta = delta
        u = (0.0, delta)
        car.state = model.step(car.state, u, kappa=0.0, mu=1.0, dt=dt)
        if car.state[2] < 0:
            saw_negative_e_psi = True
        t += dt

    assert saw_negative_e_psi  # correct sign to increase e_y toward +4
    assert car.state[1] > e_y_initial  # genuinely moved toward the target, not away from it


def test_lane_change_settles_without_sustained_oscillation():
    road = RoadScenario(kappa_max=0.0, lane_num=3)
    model = EgoModel()
    lc_p = LaneChangeParams()
    tol = LaneChangeTolerances()

    car = _car(road, lane=0, s=100.0)
    start_lane_change(car, target_lane=1, t=0.0)
    e_y_target = road.d_c[1]

    dt = 0.1
    t = 0.0
    overshoots = []
    for _ in range(150):  # 15s -- well past the 3s nominal duration
        e_y_ref, e_y_ref_rate = lane_change_reference(car, road, t)
        delta, car.lane_error_integral = steering_cmd(
            car.state, 0.0, e_y_ref, e_y_ref_rate, model.p.L, lc_p,
            integral=car.lane_error_integral, prev_delta=car.last_delta, dt=dt,
        )
        car.last_delta = delta
        car.state = model.step(car.state, (0.0, delta), kappa=0.0, mu=1.0, dt=dt)
        t += dt
        try_complete_lane_change(car, road, t, tol)
        if t > car.lane_change_duration:
            overshoots.append(abs(car.state[1] - e_y_target))

    # settles close to the target and stays there -- no sustained oscillation
    assert max(overshoots[-30:]) < 0.5
    assert not car.lane_change_active  # actually completed within the run
