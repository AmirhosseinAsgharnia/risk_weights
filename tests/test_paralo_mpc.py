import dataclasses

import numpy as np
import pytest

from model.car.car import Car
from model.car.config import VehicleParameters
from learning.closed_loop import FAMILIES, ScenarioSpec, Stratum, build_controller
from learning.ego_control import EgoView, to_env_action
from learning.env import ACCEL_MAX, ACCEL_MIN, DELTA_MAX, DT, MAX_STEERING_RATE, EgoTrafficEnv
from learning.feasibility_cutin import CutinConfig
from learning.paralo_mpc import (
    PW_NAMES, TERMS, ParaloMPC, ParaloMPCConfig, ParaloWeights, pw_from_unit, pw_to_unit, understeer_gradient,
)
from learning.physical_eval import deterministic_part, run_physical_episode

SPEC = ScenarioSpec.make(Stratum("cutin", 1), CutinConfig().theta())


def _env(seed=1000):
    env = EgoTrafficEnv(arena=FAMILIES["cutin"][1](SPEC.config(seed)), mode="fixed")
    obs, _ = env.reset()
    return env, obs


def _mpc(**w):
    return ParaloMPC(ParaloMPCConfig(), ParaloWeights(**w))


def test_weights_unit_roundtrip_and_bounds():
    w = ParaloWeights().to_vector()
    np.testing.assert_allclose(pw_from_unit(pw_to_unit(w)), w)
    assert np.all((pw_to_unit(w) > 0) & (pw_to_unit(w) < 1))
    with pytest.raises(AssertionError):
        ParaloWeights(c2c=0.0)
    assert len(PW_NAMES) == 7


def test_prediction_yaw_model_matches_the_simulator_steady_state():
    vp = VehicleParameters()
    K = understeer_gradient(vp)
    for v, d in ((10.0, 0.03), (20.0, 0.06)):
        car = Car()
        car.state = dataclasses.replace(car.state, v_x=v, delta=d)
        for _ in range(200):
            car.step(0.0, d, 0.0, 1.0, DT)
        assert car.state.r == pytest.approx(v * np.tan(d) / (vp.L + K * v * v), rel=0.02)


def test_candidates_are_absolute_held_inputs_within_bounds_and_horizon():
    env, obs = _env()
    m = _mpc()
    m.reset(EgoView(env, obs), {})
    acc, dlt, terms = m.candidate_terms(EgoView(env, obs))
    cfg = m.cfg
    assert len(acc) == len(cfg.accel_levels) * len(cfg.steer_offsets) and cfg.n_pred == 30
    assert np.all((acc >= ACCEL_MIN) & (acc <= ACCEL_MAX)) and np.all(np.abs(dlt) <= DELTA_MAX)
    np.testing.assert_allclose(np.unique(dlt), np.unique(np.clip(env.ego_car.state.delta + np.array(cfg.steer_offsets),
                                                                 -DELTA_MAX, DELTA_MAX)))
    assert set(terms) == set(TERMS) and all(t.shape == acc.shape for t in terms.values())
    assert all(np.all((terms[k] >= 0) & (terms[k] <= cfg.n_pred + 1e-9)) for k in ("c2c", "roll", "slip", "lane", "road"))


def test_cost_uses_the_weight_vector_and_all_five_risks():
    env, obs = _env()
    view = EgoView(env, obs)
    m1, m2 = _mpc(), _mpc(slip=2.0, c2c=40.0)
    for m in (m1, m2):
        m.reset(view, {})
    _, _, c1, terms = m1.candidate_costs(view)
    _, _, c2, _ = m2.candidate_costs(view)
    np.testing.assert_allclose(c2 - c1, 1.0 * terms["slip"] + 20.0 * terms["c2c"], atol=1e-9)
    # c2c: ego placed 3 m behind the front car -> overlapping now
    front = env.agents[0].car.state
    env.ego_car.state = dataclasses.replace(env.ego_car.state, s=front.s - 3.0)
    assert m1.candidate_terms(view)[2]["c2c"].max() >= 1.0
    # slippage reacts to the friction the MPC is given (mu_hat, fixed at reset); lane/road departure to e_y
    env.ego_car.state = dataclasses.replace(env.ego_car.state, s=front.s - 60.0, delta=0.05)
    slip_dry = m1.candidate_terms(view)[2]["slip"].sum()
    env.road.mu = 0.4
    assert m1.candidate_terms(view)[2]["slip"].sum() == slip_dry         # no mid-episode ground-truth peeking
    m1.reset(view, {})
    assert m1.mu_hat == 0.4 and m1.candidate_terms(view)[2]["slip"].sum() > slip_dry
    env.ego_car.state = dataclasses.replace(env.ego_car.state, e_y=1.9, delta=0.0)
    t = m1.candidate_terms(view)[2]
    assert t["lane"].max() > 1.0 and t["roll"].min() >= 0.0 and t["road"].min() >= 0.0


def test_closed_loop_respects_actuator_limits_and_keeps_weights():
    env, obs = _env()
    m = _mpc()
    w_before = m.w
    m.reset(EgoView(env, obs), {})
    prev_delta = env.ego_car.state.delta
    for _ in range(80):
        a, d = m.act(EgoView(env, obs))
        assert ACCEL_MIN <= a <= ACCEL_MAX and abs(d) <= DELTA_MAX
        obs, *_ = env.step(to_env_action(a, d))
        assert abs(env.ego_car.state.delta - prev_delta) <= MAX_STEERING_RATE * DT + 1e-12
        prev_delta = env.ego_car.state.delta
    assert m.w is w_before and m.n_plans == 40   # replans every 2 env steps; weights never reset


def test_episode_is_deterministic():
    w = ParaloWeights().to_vector()
    r1 = run_physical_episode(build_controller({"name": "paralo_mpc"}, w), SPEC, 1000)
    r2 = run_physical_episode(build_controller({"name": "paralo_mpc"}, w), SPEC, 1000)
    assert deterministic_part(r1) == deterministic_part(r2)
    assert r1["n_plans"] > 0 and set(r1["mpc_diagnostics"]) == {"slip_active", "slip_indistinct",
                                                                "slip_saturated_all", "slip_decisive"}
