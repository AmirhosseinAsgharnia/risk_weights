import dataclasses

import numpy as np
import pytest

import learning.env as env_mod
from learning.closed_loop import EvalCostConfig, ScenarioSpec, Stratum, FAMILIES, run_episode, build_controller
from learning.ego_control import ConstantController, EgoView, to_env_action
from learning.env import ACCEL_MAX, ACCEL_MIN, DELTA_MAX, DT, EgoTrafficEnv, MAX_STEERING_RATE
from learning.feasibility_cutin import CutinConfig
from learning.mpc import (
    MPCConfig, MPCWeights, SamplingMPC, W_HI, W_LO, p_rollover_vec, w_from_unit, w_to_unit,
)
from model.car.config import VehicleParameters
from risks.rollover import TrajectoryStep, rollover

SPEC = ScenarioSpec.make(Stratum("cutin", 1), CutinConfig().theta())
COST = EvalCostConfig()


def test_vectorized_rollover_matches_risks_module():
    vp = VehicleParameters()
    rng = np.random.default_rng(0)
    v = rng.uniform(0, 35, 200)
    d = rng.uniform(-0.5, 0.5, 200)
    d[:5] = 0.0
    ref = np.array([rollover([TrajectoryStep(v=a, delta=b)], vehicle_params=vp)[0] for a, b in zip(v, d)])
    assert np.allclose(p_rollover_vec(v, d, vp, U_H=0.1), ref, atol=1e-9)


def test_to_env_action_inverts_env_scaling_and_clips():
    env = EgoTrafficEnv()
    for a, d in [(-8.0, -0.5), (4.0, 0.5), (0.0, 0.0), (1.3, -0.21)]:
        assert env._unscale_action(to_env_action(a, d)) == pytest.approx((a, d), abs=1e-5)
    assert env._unscale_action(to_env_action(50.0, -3.0)) == pytest.approx((ACCEL_MAX, -DELTA_MAX), abs=1e-6)


def test_w_unit_mapping_roundtrip_and_bounds():
    w = MPCWeights().to_vector()
    assert np.allclose(w_from_unit(w_to_unit(w)), w)
    assert np.allclose(w_from_unit(np.zeros(len(w))), W_LO) and np.allclose(w_from_unit(np.ones(len(w))), W_HI)
    assert np.allclose(w_from_unit(np.full(len(w), 7.0)), W_HI)   # clipped into the box
    with pytest.raises(AssertionError):
        MPCWeights(collision=0.0)   # safety terms cannot be switched off


def _drive(controller, n_steps=200, seed=0):
    env = EgoTrafficEnv(arena=FAMILIES["cutin"][1](SPEC.config(seed)), mode="fixed")
    obs, _ = env.reset()
    controller.reset(EgoView(env, obs), {})
    cmds, applied = [], []
    for _ in range(n_steps):
        a, d = controller.act(EgoView(env, obs))
        cmds.append((a, d))
        obs, _, term, trunc, _ = env.step(to_env_action(a, d))
        applied.append((env._prev_accel, env.ego_car.state.delta))
        if term or trunc:
            break
    return np.array(cmds), np.array(applied)


def test_mpc_commands_and_applied_actuation_respect_bounds():
    cmds, applied = _drive(SamplingMPC(MPCConfig(), MPCWeights()))
    assert np.all(cmds[:, 0] >= ACCEL_MIN) and np.all(cmds[:, 0] <= ACCEL_MAX)
    assert np.all(np.abs(cmds[:, 1]) <= DELTA_MAX + 1e-12)
    assert np.all(np.abs(np.diff(applied[:, 1])) <= MAX_STEERING_RATE * DT + 1e-9)


def test_closed_loop_reuses_env_dynamics_exactly():
    """A controller goes through EgoTrafficEnv.step unchanged: same states as stepping the env directly."""
    _, applied_ctrl = _drive(ConstantController(0.5, 0.01), n_steps=60)
    env = EgoTrafficEnv(arena=FAMILIES["cutin"][1](SPEC.config(0)), mode="fixed")
    env.reset()
    direct = []
    for _ in range(60):
        env.step(to_env_action(0.5, 0.01))
        direct.append((env._prev_accel, env.ego_car.state.delta))
    assert np.array_equal(applied_ctrl, np.array(direct)[:len(applied_ctrl)])


def test_episode_is_deterministic_and_w_changes_behaviour():
    w0 = MPCWeights().to_vector()
    r1 = run_episode(build_controller({"name": "mpc"}, w0), SPEC, 7, COST)
    r2 = run_episode(build_controller({"name": "mpc"}, w0), SPEC, 7, COST)
    assert r1 == r2
    risky = MPCWeights(collision=0.5, rollover=0.5, road_edge=0.5, speed=20.0, time_headway_s=0.5).to_vector()
    r3 = run_episode(build_controller({"name": "mpc"}, risky), SPEC, 7, COST)
    assert (r3["cost"], r3["outcome"], r3["steps"]) != (r1["cost"], r1["outcome"], r1["steps"])


def test_outcome_and_cost_independent_of_ppo_reward(monkeypatch):
    w0 = MPCWeights().to_vector()
    base = run_episode(build_controller({"name": "mpc"}, w0), SPEC, 3, COST)
    for name in ("PROGRESS_REWARD_SCALE", "COLLISION_PENALTY", "FINISH_BONUS", "SURVIVAL_REWARD", "LANE_TRACKING_SCALE"):
        monkeypatch.setattr(env_mod, name, getattr(env_mod, name) * 13.0 + 1.0)
    assert run_episode(build_controller({"name": "mpc"}, w0), SPEC, 3, COST) == base
    assert base["success"] == (base["outcome"] == "finished")


def test_mpc_fallback_on_nonfinite_cost(monkeypatch):
    mpc = SamplingMPC(MPCConfig(), MPCWeights())
    real = mpc.candidate_costs

    def broken(view):
        acc, lanes, cost = real(view)
        return acc, lanes, cost * np.nan

    monkeypatch.setattr(mpc, "candidate_costs", broken)
    cmds, _ = _drive(mpc, n_steps=4)
    assert mpc.solver_failures >= 1 and np.all(cmds[:, 0] == ACCEL_MIN)


def test_eval_cost_config_guards_and_terminal_penalties():
    with pytest.raises(AssertionError):
        EvalCostConfig(penalty_collision=30.0)   # below the 4 x 10 running-cost bound
    c = EvalCostConfig()
    assert c.terminal_penalty({"failure_reason": "collision", "outcome": "safety_failure"}, 400) == c.penalty_collision
    assert c.terminal_penalty({"failure_reason": None, "outcome": "finished", "progress_m": 400}, 400) == 0.0
    t = c.terminal_penalty({"failure_reason": None, "outcome": "timeout", "progress_m": 100.0}, 400)
    assert t == pytest.approx(c.penalty_timeout_base + c.penalty_timeout_shortfall * 0.75)
    assert dataclasses.asdict(c) == dataclasses.asdict(EvalCostConfig(**dataclasses.asdict(c)))
