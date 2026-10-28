import dataclasses

import numpy as np
import pytest

from learning.closed_loop import FAMILIES, ScenarioSpec, Stratum, build_controller
from learning.paralo_mpc import ParaloWeights
from learning.ego_control import ConstantController
from learning.env import EgoTrafficEnv
from learning.feasibility_cutin import CutinConfig
from learning.vehicle_physics import slip_lateral_saturation
from learning.physical_eval import (
    DEFAULT_CONSTRAINTS, DIAG_PEAKS, EVENTS, PEAKS, ConstraintSpec, constraints_from_config,
    run_physical_episode, summarize_scenario,
)

SPEC = ScenarioSpec.make(Stratum("cutin", 1), CutinConfig().theta())


def _ep(seed, shortfall, success=False, **events):
    return {"realization_seed": seed, "progress_shortfall": shortfall, "success": success, "timeout": False,
            "solver_failures": 0, "events": {e: events.get(e, False) for e in EVENTS},
            "elapsed_s": 20.0, "max_episode_seconds": 40.0, **{p: float(seed % 7) for p in PEAKS + DIAG_PEAKS}}


def test_collision_is_recorded_separately_and_shortfall_uses_actual_progress():
    rec = run_physical_episode(ConstantController(4.0, 0.0), SPEC, 1000)   # floors it into the traffic ahead
    assert rec["events"]["collision"] and not rec["success"] and rec["terminated_by_safety_failure"]
    assert rec["progress_shortfall"] == pytest.approx(1.0 - rec["progress_m"] / rec["goal_distance_m"])
    assert 0.0 < rec["progress_shortfall"] <= 1.0


def test_shortfall_is_bounded_and_zero_only_at_the_goal():
    s = summarize_scenario([_ep(1, 0.0, success=True), _ep(2, 0.4)])
    assert s["progress_shortfall"] == pytest.approx(0.2) and s["completion_rate"] == 0.5
    rec = run_physical_episode(ConstantController(-8.0, 0.0), SPEC, 1000)   # stands still
    assert not rec["success"] and 0.9 < rec["progress_shortfall"] <= 1.0
    ok = run_physical_episode(build_controller({"name": "paralo_mpc"}, ParaloWeights().to_vector()), SPEC, 1000)
    assert ok["success"] and ok["progress_shortfall"] == 0.0 and not any(ok["events"][e] for e in
                                                                          ("collision", "rollover", "road_departure"))


def test_event_rates_wilson_intervals_and_peaks():
    s = summarize_scenario([_ep(1, 0.0, collision=True), _ep(2, 0.0), _ep(3, 0.0, slippage=True), _ep(4, 0.0)])
    assert s["collision_count"] == 1 and s["collision_rate"] == 0.25 and s["rollover_rate"] == 0.0
    assert s["rollover_wilson95"][0] == 0.0 and s["rollover_wilson95"][1] > 0.0   # 0/4 is not "probability 0"
    assert s["peak_abs_jerk"] == 4.0


def test_constraints_are_independent_and_monitored_ones_never_violate():
    cs = {c.name: c for c in constraints_from_config(DEFAULT_CONSTRAINTS)}
    s = summarize_scenario([_ep(1, 0.0, collision=True), _ep(2, 0.0, lane_departure=True)])
    assert cs["collision"].violated(cs["collision"].value(s))
    assert not cs["rollover"].violated(cs["rollover"].value(s))
    lane = cs["lane_departure"]
    assert not lane.enforced and lane.value(s) == 0.5 and not lane.violated(1.0) and lane.violation(1.0) is None
    wil = ConstraintSpec("c", "collision_rate", 0.0, "1/episode", statistic="wilson_upper")
    assert wil.violated(wil.value(summarize_scenario([_ep(1, 0.0), _ep(2, 0.0)])))   # bound > 0 with 0 events
    with pytest.raises(AssertionError):
        ConstraintSpec("x", "peak_abs_jerk", 10.0, "m/s^2")          # wrong units
    with pytest.raises(AssertionError):
        constraints_from_config([DEFAULT_CONSTRAINTS[0], DEFAULT_CONSTRAINTS[0]])


def test_slippage_event_is_tire_saturation_in_the_simulator_model():
    env = EgoTrafficEnv(arena=FAMILIES["cutin"][1](SPEC.config(1000)), mode="fixed")
    env.reset()
    assert not slip_lateral_saturation(env.ego_tire_state())
    env.ego_car.state = dataclasses.replace(env.ego_car.state, v_x=20.0, v_y=8.0)   # rear slip angle ~0.38 rad
    assert slip_lateral_saturation(env.ego_tire_state())


def test_completion_time_ranks_every_success_before_every_failure():
    s_ok = summarize_scenario([_ep(1, 0.0, success=True)])
    s_fail = summarize_scenario([_ep(1, 0.0, success=False)])
    assert s_ok["completion_time"] == pytest.approx(0.5) and s_fail["completion_time"] >= 1.0
