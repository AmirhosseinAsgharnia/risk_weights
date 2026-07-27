import numpy as np
from model.road import RoadScenario
from model.car_init import Car
from model.IDM_MOBIL import IDMParams, MOBILParams
from model.traffic import build_snapshot
from model.collision import (
    rectangle_corners, sat_overlap, broad_phase_overlap, narrow_phase_collision,
    check_collisions, check_road_departure, check_lane_index_valid, check_singularity_risk,
)


def _car(lane, s, e_y=None, road=None):
    e_y = road.d_c[lane] if e_y is None else e_y
    return Car(current_lane=lane, target_lane=lane, state=np.array([s, e_y, 0.0, 20.0, 0.0, 0.0]))


def _snapshot(road, cars):
    return build_snapshot(cars, road, IDMParams(), MOBILParams())


def test_sat_overlap_identical_position_overlaps():
    corners = rectangle_corners(0.0, 0.0, 0.0, length=4.0, width=2.0)
    assert sat_overlap(corners, corners) is True


def test_sat_overlap_far_apart_does_not_overlap():
    a = rectangle_corners(0.0, 0.0, 0.0, length=4.0, width=2.0)
    b = rectangle_corners(100.0, 100.0, 0.0, length=4.0, width=2.0)
    assert sat_overlap(a, b) is False


def test_sat_overlap_rotated_rectangle_side_by_side():
    # two 4x2 rectangles centred 1.5m apart laterally overlap; 3m apart do not
    a = rectangle_corners(0.0, 0.0, 0.0, length=4.0, width=2.0)
    b_close = rectangle_corners(0.0, 1.5, 0.0, length=4.0, width=2.0)
    b_far = rectangle_corners(0.0, 3.5, 0.0, length=4.0, width=2.0)
    assert sat_overlap(a, b_close) is True
    assert sat_overlap(a, b_far) is False


def test_rear_end_collision_detected_for_same_lane_overlap():
    road = RoadScenario(lane_num=2)
    a = _car(0, s=100.0, road=road)
    b = _car(0, s=101.0, road=road)  # 1m apart, well within combined length -> overlap
    snapshot = _snapshot(road, [a, b])
    events = check_collisions(snapshot, road, t=0.0)
    assert len(events) == 1
    assert set(events[0].actor_ids) == {a.id, b.id}
    assert events[0].collision_type == "rear_end"


def test_no_collision_when_far_apart():
    road = RoadScenario(lane_num=2)
    a = _car(0, s=100.0, road=road)
    b = _car(0, s=200.0, road=road)
    snapshot = _snapshot(road, [a, b])
    assert check_collisions(snapshot, road, t=0.0) == []


def test_broad_phase_rejects_pairs_that_cannot_possibly_overlap():
    road = RoadScenario(lane_num=3)
    a = _car(0, s=100.0, road=road)
    b = _car(2, s=100.0, road=road)  # same s, but two full lanes apart laterally
    snapshot = _snapshot(road, [a, b])
    assert broad_phase_overlap(snapshot[0], snapshot[1]) is False
    assert narrow_phase_collision(snapshot[0], snapshot[1], road) is False


def test_road_departure_uses_body_extent_not_just_centre():
    road = RoadScenario(lane_num=3)  # outer edges at +-(l_w + l_w/2) = +-6.0
    # centre is off the road, but check_road_departure is defined as the BODY
    # being entirely outside the boundary -- a centre-only check would flag
    # this car sooner (or a merely-straddling car later); this confirms it
    # uses the full [e_y-width/2, e_y+width/2] extent as specified.
    car = _car(1, s=100.0, e_y=7.5, road=road)  # body spans [6.5, 8.5], fully past +6.0
    snapshot = _snapshot(road, [car])
    assert check_road_departure(snapshot[0], road) is True

    straddling = _car(1, s=100.0, e_y=5.5, road=road)  # body spans [4.5, 6.5]: only partly past
    snapshot2 = _snapshot(road, [straddling])
    assert check_road_departure(snapshot2[0], road) is False

    car_in_bounds = _car(1, s=100.0, e_y=0.0, road=road)
    snapshot2 = _snapshot(road, [car_in_bounds])
    assert check_road_departure(snapshot2[0], road) is False


def test_lane_index_valid():
    assert check_lane_index_valid(0, 3) is True
    assert check_lane_index_valid(2, 3) is True
    assert check_lane_index_valid(3, 3) is False
    assert check_lane_index_valid(-1, 3) is False


def test_singularity_risk_near_1_over_kappa():
    # 1 - kappa*e_y close to 0
    assert check_singularity_risk(kappa=0.02, e_y=49.5, threshold=0.1) is True
    assert check_singularity_risk(kappa=0.02, e_y=0.0, threshold=0.1) is False
