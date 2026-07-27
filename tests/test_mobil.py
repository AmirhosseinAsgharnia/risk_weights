import numpy as np
from model.road import RoadScenario
from model.car_init import Car
from model.IDM_MOBIL import IDMParams, MOBILParams
from model.traffic import (
    build_snapshot, evaluate_lane_candidates, select_best_candidate,
    propose_lane_changes, resolve_conflicts,
)


def _car(road, current_lane, s, v, idm_v0=25.0):
    return Car(
        current_lane=current_lane, target_lane=current_lane,
        state=np.array([s, road.d_c[current_lane], 0.0, v, 0.0, 0.0]),
        idm_params=IDMParams(v0=idm_v0),
    )


def _snapshot(road, cars):
    return build_snapshot(cars, road, IDMParams(), MOBILParams(a_thr=0.1))


def test_best_incentive_selection_has_no_left_right_bias():
    """lane0 (current-1) is only mildly better; lane2 (current+1) is far
    better (empty, no leader at all) -- the winner must be lane2, proving
    this isn't a first-valid-candidate (left-first) selection."""
    road = RoadScenario(kappa_max=0.0, lane_num=3)
    mover = _car(road, current_lane=1, s=100.0, v=20.0)
    slow_leader_lane1 = _car(road, current_lane=1, s=110.0, v=8.0, idm_v0=8.0)  # blocks lane1 badly
    moderate_leader_lane0 = _car(road, current_lane=0, s=140.0, v=20.0)  # lane0: some room, not free
    cars = [mover, slow_leader_lane1, moderate_leader_lane0]
    snapshot = _snapshot(road, cars)

    candidates = evaluate_lane_candidates(snapshot, mover.id, road)
    by_lane = {c.lane: c for c in candidates}
    assert by_lane[2].incentive > by_lane[0].incentive > 0

    best = select_best_candidate(candidates)
    assert best.lane == 2


def test_staying_wins_when_no_candidate_clears_threshold():
    road = RoadScenario(kappa_max=0.0, lane_num=3)
    mover = _car(road, current_lane=1, s=100.0, v=20.0)
    # both neighbours are already-free-flowing at the same speed: no real incentive either way
    left = _car(road, current_lane=0, s=100.0, v=20.0)
    right = _car(road, current_lane=2, s=100.0, v=20.0)
    snapshot = _snapshot(road, [mover, left, right])

    candidates = evaluate_lane_candidates(snapshot, mover.id, road)
    best = select_best_candidate(candidates)
    assert best.lane == 1  # stays


def test_mobil_safety_vetoes_an_unsafe_merge_even_with_high_incentive():
    road = RoadScenario(kappa_max=0.0, lane_num=2)
    mover = _car(road, current_lane=0, s=100.0, v=25.0)
    blocking_leader = _car(road, current_lane=0, s=101.0, v=5.0, idm_v0=5.0)  # forces huge incentive to flee
    # a follower right behind the merge point in lane1, moving fast -> merging in front of it is unsafe
    tailgater = _car(road, current_lane=1, s=99.5, v=30.0, idm_v0=30.0)
    snapshot = _snapshot(road, [mover, blocking_leader, tailgater])

    candidates = evaluate_lane_candidates(snapshot, mover.id, road)
    by_lane = {c.lane: c for c in candidates}
    assert by_lane[1].safe is False
    assert by_lane[1].valid is False


def test_proposals_are_independent_of_snapshot_order():
    road = RoadScenario(kappa_max=0.0, lane_num=3)
    mover = _car(road, current_lane=1, s=100.0, v=20.0)
    slow_leader = _car(road, current_lane=1, s=110.0, v=8.0, idm_v0=8.0)
    other = _car(road, current_lane=0, s=300.0, v=20.0)  # far away, irrelevant

    cars = [mover, slow_leader, other]
    ids = {c.id for c in cars}
    snap_a = build_snapshot(cars, road, IDMParams(), MOBILParams(a_thr=0.1))
    snap_b = build_snapshot(list(reversed(cars)), road, IDMParams(), MOBILParams(a_thr=0.1))

    props_a, _ = propose_lane_changes(snap_a, road, t=100.0, cooldown=3.0, eligible_ids=ids)
    props_b, _ = propose_lane_changes(snap_b, road, t=100.0, cooldown=3.0, eligible_ids=ids)

    result_a = {(p.actor_id, p.to_lane) for p in props_a}
    result_b = {(p.actor_id, p.to_lane) for p in props_b}
    assert result_a == result_b
    assert result_a  # sanity: a proposal actually happened


def test_conflict_resolution_is_deterministic_regardless_of_input_order():
    road = RoadScenario(kappa_max=0.0, lane_num=2)
    a = _car(road, current_lane=0, s=100.0, v=20.0)
    b = _car(road, current_lane=0, s=101.0, v=20.0)  # right next to `a`, same target lane -> conflict
    snapshot = _snapshot(road, [a, b])

    from model.traffic import LaneProposal
    proposal_a = LaneProposal(actor_id=a.id, from_lane=0, to_lane=1, incentive=1.0, safety_margin=5.0, candidates=())
    proposal_b = LaneProposal(actor_id=b.id, from_lane=0, to_lane=1, incentive=0.5, safety_margin=5.0, candidates=())

    accepted_1, rejected_1 = resolve_conflicts([proposal_a, proposal_b], snapshot, min_gap=4.0)
    accepted_2, rejected_2 = resolve_conflicts([proposal_b, proposal_a], snapshot, min_gap=4.0)

    assert [p.actor_id for p in accepted_1] == [p.actor_id for p in accepted_2] == [a.id]
    assert [p.actor_id for p, _ in rejected_1] == [p.actor_id for p, _ in rejected_2] == [b.id]
