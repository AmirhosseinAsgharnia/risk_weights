import copy
import numpy as np
import pytest
from model.simulation import (
    build_demo_scenario, Simulation, validate_scenario, ScenarioValidationError,
)
from model.car_init import Car


BENIGN_THETA_ROAD = [0.005, 50, 1.0, 225.0]  # gentle curve, full friction -- a "nominal" scenario
BENIGN_SURR_CARS = [(0, -30.0, 0.0), (1, 40.0, -2.0), (2, -20.0, 1.0)]


def _benign_scenario():
    return build_demo_scenario(BENIGN_THETA_ROAD, ego_lane=1, ego_speed=20.0, surr_cars=BENIGN_SURR_CARS)


def test_nominal_scenario_runs_with_no_nan_or_inf_states():
    scenario = _benign_scenario()
    sim = Simulation(scenario)
    result = sim.run(n_steps=200)

    assert result.numerical_failure is False
    for record in result.history:
        for log in record.actors:
            assert np.all(np.isfinite(log.state)), f"non-finite state for actor {log.actor_id} at t={log.time}"


def test_nominal_scenario_has_no_collision_or_departure():
    scenario = _benign_scenario()
    result = Simulation(scenario).run(n_steps=200)
    assert result.completed
    assert not result.collision
    assert not result.road_departure
    assert not result.loss_of_control


def test_nominal_lane_change_completes_without_excessive_oscillation():
    scenario = _benign_scenario()
    result = Simulation(scenario).run(n_steps=200)
    assert result.lane_changes >= 1

    # For any actor that has fully completed its manoeuvre well before the
    # tail (not lane_change_active at any point in the window), e_y should
    # sit tight on the lane centreline -- no sustained post-completion
    # ringing. Actors still mid-manoeuvre this late are excluded here (that's
    # exactly what test_lane_change.py's isolated, controlled-timing test
    # already covers precisely).
    road = scenario.road
    per_actor_late_error = {}
    still_changing = set()
    for record in result.history[-10:]:
        for log in record.actors:
            if log.lane_change_active:
                still_changing.add(log.actor_id)
            target_e_y = road.d_c[log.target_lane]
            err = abs(log.state[1] - target_e_y)
            per_actor_late_error.setdefault(log.actor_id, []).append(err)

    checked_any = False
    for actor_id, errors in per_actor_late_error.items():
        if actor_id in still_changing:
            continue
        checked_any = True
        assert max(errors) < 0.3, f"actor {actor_id} still oscillating late in the run: {errors}"
    assert checked_any, "every actor was still mid-lane-change in the tail window -- test can't verify settling"


def test_rendering_style_read_only_access_does_not_mutate_history():
    scenario = _benign_scenario()
    result = Simulation(scenario).run(n_steps=50)

    before = copy.deepcopy(result.history[-1].actors[0].state)

    # do exactly what tests/scene_test.py's renderer does: read poses/state
    # into plain tuples/dicts, nothing more
    frames = [{log.actor_id: log.pose for log in record.actors} for record in result.history]
    _ = [log.state.copy() for record in result.history for log in record.actors]
    assert len(frames) == len(result.history)

    after = result.history[-1].actors[0].state
    assert np.array_equal(before, after)


def test_actor_ids_are_stable_and_unique():
    scenario = _benign_scenario()
    all_cars = [scenario.ego] + scenario.traffic
    ids = [c.id for c in all_cars]
    assert len(ids) == len(set(ids))  # unique
    # ids don't change across steps
    sim = Simulation(scenario)
    sim.step()
    assert [c.id for c in [scenario.ego] + scenario.traffic] == ids


def test_validate_scenario_rejects_out_of_range_lane():
    scenario = _benign_scenario()
    scenario.ego.current_lane = 99
    scenario.ego.target_lane = 99
    with pytest.raises(ScenarioValidationError):
        validate_scenario(scenario)


def test_validate_scenario_rejects_initial_overlap():
    scenario = _benign_scenario()
    # force the first traffic car to exactly overlap the ego
    scenario.traffic[0].current_lane = scenario.ego.current_lane
    scenario.traffic[0].target_lane = scenario.ego.current_lane
    scenario.traffic[0].state[0] = scenario.ego.state[0]
    scenario.traffic[0].state[1] = scenario.ego.state[1]
    with pytest.raises(ScenarioValidationError):
        validate_scenario(scenario)


def test_validate_scenario_rejects_bad_dt():
    scenario = _benign_scenario()
    scenario.dt = 0.0
    with pytest.raises(ScenarioValidationError):
        validate_scenario(scenario)


def test_validate_scenario_rejects_non_finite_initial_state():
    scenario = _benign_scenario()
    scenario.traffic[0].state[0] = np.nan
    with pytest.raises(ScenarioValidationError):
        validate_scenario(scenario)


def test_harsh_scenario_produces_a_reported_rear_end_collision():
    """
    Documents a known, expected outcome: the ego is an intentionally
    non-reactive cruise/lane-hold placeholder (see model.simulation.
    CruiseLaneHoldController), so a slow car merging directly in front of it
    is not avoided -- and the collision detector must catch it, not silently
    let bodies interpenetrate.
    """
    theta_road = [0.01, 50, 0.5, 225.0]
    surr_cars = [
        (0, -40.0, 1.0), (0, -5.0, 5.0), (0, 35.0, -5.0), (0, 80.0, 4.0),
        (1, -25.0, 1.0), (1, 50.0, -2.0),
        (2, -45.0, 3.0), (2, 10.0, -4.0), (2, 55.0, 0.0),
    ]
    scenario = build_demo_scenario(theta_road, ego_lane=1, ego_speed=15.0, surr_cars=surr_cars)
    result = Simulation(scenario).run(n_steps=200)
    assert result.collision is True
    assert result.completed is False  # stop_on_collision defaults to True
