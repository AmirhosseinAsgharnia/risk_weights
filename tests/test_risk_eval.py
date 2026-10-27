import dataclasses

import numpy as np
import pytest

from learning.closed_loop import EvalCostConfig, FAMILIES, ScenarioSpec, Stratum, build_controller, run_episode
from learning.env import EgoTrafficEnv
from learning.feasibility_cutin import CutinConfig
from learning.mpc import MPCWeights
from learning.risk_eval import CHANNELS, predicted_risks

SPEC = ScenarioSpec.make(Stratum("cutin", 1), CutinConfig().theta())
NOW = EvalCostConfig(risk_horizon_s=0.0)   # instantaneous: only the current state


def _env():
    env = EgoTrafficEnv(arena=FAMILIES["cutin"][1](SPEC.config(0)), mode="fixed")
    env.reset()
    return env


def _set(env, **kw):
    for k, v in kw.items():
        setattr(env.ego_car.state, k, v)


def test_all_channels_present_and_probabilities_in_unit_interval():
    r = predicted_risks(_env(), EvalCostConfig())
    assert set(r) == set(CHANNELS) and all(0.0 <= v <= 1.0 for v in r.values())


def test_c2c_with_vanishing_sigma_reproduces_the_simulator_collision_test():
    exact = dataclasses.replace(NOW, risk_sigma_ego_m=1e-7, risk_sigma_obj_m=1e-7)
    env = _env()
    assert not env._ego_collided() and predicted_risks(env, exact)["c2c"] == 0.0
    front = env.agents[0].car.state                    # front_1, ego's lane
    _set(env, s=front.s - 3.0)                         # |ds| = 3 < CAR_LENGTH
    assert env._ego_collided() and predicted_risks(env, exact)["c2c"] == 1.0


def test_c2c_is_predictive():
    env = _env()
    front = env.agents[0].car.state
    _set(env, s=front.s - 4.0 - 8.0, v_x=front.v_x + 10.0)   # 8 m gap, closing at 10 m/s
    assert predicted_risks(env, NOW)["c2c"] < 0.01
    assert predicted_risks(env, EvalCostConfig(risk_horizon_s=2.0))["c2c"] > 0.99


def test_lane_and_road_departure_geometry():
    env = _env()                                       # ego on its lane centre = road centre (middle lane)
    r = predicted_risks(env, NOW)
    assert r["lane_departure"] < 1e-3 and r["road_departure"] < 1e-12
    _set(env, e_y=1.0)                                 # body edge on the lane line: h = (4 - 2)/2 = 1
    assert predicted_risks(env, NOW)["lane_departure"] == pytest.approx(0.5, abs=1e-6)
    _set(env, e_y=5.0)                                 # body edge on the road edge: h = (12 - 2)/2 = 5
    assert predicted_risks(env, NOW)["road_departure"] == pytest.approx(0.5, abs=1e-6)
    _set(env, e_y=-5.0)                                # SAE sign symmetric
    assert predicted_risks(env, NOW)["road_departure"] == pytest.approx(0.5, abs=1e-6)


def test_rollover_matches_env_and_slippage_uses_scenario_friction():
    env = _env()
    _set(env, delta=0.04)
    assert predicted_risks(env, NOW)["rollover"] == pytest.approx(env._ego_p_rollover())
    _set(env, delta=0.012, v_x=20.0)
    assert predicted_risks(env, NOW)["slippage"] < 0.01      # road mu = 1.0
    env.road.mu = 0.4
    assert predicted_risks(env, NOW)["slippage"] > 0.99


def test_score_measure_is_normalized_paralo_risk():
    env = _env()
    _set(env, e_y=1.0)
    s = predicted_risks(env, dataclasses.replace(NOW, risk_measure="score"))
    p = predicted_risks(env, NOW)
    m, v, v_ref = env.ego_car.vehicle_params.m, env.ego_car.state.v_x, NOW.v_ref_mps
    assert s["lane_departure"] == pytest.approx(0.5 * m * v ** 2 * p["lane_departure"] / (0.5 * m * v_ref ** 2))


def test_betas_weight_the_risk_term_and_are_recorded():
    w = MPCWeights().to_vector()
    zero = EvalCostConfig(**{f"beta_{ch}": 0.0 for ch in CHANNELS})
    r0 = run_episode(build_controller({"name": "mpc"}, w), SPEC, 0, zero)
    r1 = run_episode(build_controller({"name": "mpc"}, w), SPEC, 0, EvalCostConfig())
    assert r0["cost_terms"]["risk"] == 0.0 and r1["cost_terms"]["risk"] > 0.0
    assert r0["risk_integrated"] == r1["risk_integrated"]      # same trajectory: betas don't affect control
    assert r1["cost_terms"]["risk"] == pytest.approx(sum(r1["risk_integrated"].values()), rel=1e-9)
    with pytest.raises(AssertionError):
        EvalCostConfig(beta_c2c=-1.0)
