"""
Tests for EgoTrafficEnv's episode-outcome plumbing (learning.env): success/
failure/timeout classification, Gymnasium terminated/truncated semantics,
actuator rate limiting, observation shape, goal configurability, and
learning.eval_batch's feasibility labeling.

These deliberately do NOT re-verify the underlying physics/risk models
(model.collision, risks.rollover -- those are exercised elsewhere/by
construction) -- they test whether EgoTrafficEnv classifies a given
physical state correctly, so most failure-mode tests inject state directly
(same white-box style tests/test_scenario.py already uses, reaching into
env.agents/env.ego_car) rather than fighting the full physics stack to
reach a rare state indirectly. No full PPO training is required anywhere
here -- see tests/test_scenario.py's own header for how to run this file.
"""

import numpy as np
import pytest

from learning.env import (
    EgoTrafficEnv, Outcome, FailureReason, N_SURR, DT,
    ACCEL_MIN, ACCEL_MAX, DELTA_MAX, MAX_STEERING_RATE, MAX_JERK,
)
from learning.scenario import ScenarioConfig, CriticalActorConfig
from learning.eval_batch import label_feasibility


def _cruise_action() -> np.ndarray:
    """(accel=0, delta=0) -- see EgoTrafficEnv._unscale_action's linear map."""
    return np.array([2.0 * (0.0 - ACCEL_MIN) / (ACCEL_MAX - ACCEL_MIN) - 1.0, 0.0], dtype=np.float32)


def _run_until_done(env: EgoTrafficEnv, action: np.ndarray, obs, max_steps: int = 2000):
    terminated = truncated = False
    info = {}
    ep_return = 0.0
    steps = 0
    while not (terminated or truncated) and steps < max_steps:
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        steps += 1
    assert terminated or truncated, "episode did not end within max_steps -- test setup problem"
    return obs, ep_return, terminated, truncated, info


# ── Collision / rollover / off-road detection (state injection -- see module docstring) ────────────

def test_collision_detection():
    env = EgoTrafficEnv(scenario_config=ScenarioConfig(seed=0), mode="fixed")
    env.reset()

    front = env.agents[0].car.state   # index 0 is always the "front" critical actor -- see generate_scenario
    env.ego_car.state.s = front.s
    env.ego_car.state.lane = front.lane
    env.ego_car.state.e_y = front.e_y

    _, reward, terminated, truncated, info = env.step(_cruise_action())

    assert terminated is True and truncated is False
    assert info["outcome"] == Outcome.SAFETY_FAILURE.value
    assert info["failure_reason"] == FailureReason.COLLISION.value
    assert info["success"] is False
    assert info["collided"] is True


def test_rollover_detection():
    env = EgoTrafficEnv(scenario_config=ScenarioConfig(seed=0), mode="fixed")
    env.reset()

    # Pre-arm the rate limiter (state.delta is the "previous applied delta") so full lock applies on
    # the very first step, and push speed well above the rollover threshold speed at full lock.
    env.ego_car.state.delta = DELTA_MAX
    env.ego_car.state.v_x = 40.0

    _, reward, terminated, truncated, info = env.step(np.array([0.0, 1.0], dtype=np.float32))

    assert terminated is True and truncated is False
    assert info["outcome"] == Outcome.SAFETY_FAILURE.value
    assert info["failure_reason"] == FailureReason.ROLLOVER.value
    assert info["success"] is False
    assert info["max_p_rollover"] > 0.5


def test_off_road_detection():
    env = EgoTrafficEnv(scenario_config=ScenarioConfig(seed=0), mode="fixed")
    env.reset()

    env.ego_car.state.e_y = env._road_half_width * 5   # far past the paved shoulder, straight steering/cruise

    _, reward, terminated, truncated, info = env.step(_cruise_action())

    assert terminated is True and truncated is False
    assert info["outcome"] == Outcome.SAFETY_FAILURE.value
    assert info["failure_reason"] == FailureReason.OFF_ROAD.value
    assert info["success"] is False
    assert info["max_road_departure_m"] > env._road_half_width


# ── Success / goal configuration ────────────────────────────────────────────

def test_goal_distance_is_configurable_and_success_detection():
    cfg = ScenarioConfig(seed=0, goal_distance_m=5.0, max_episode_seconds=5.0)
    env = EgoTrafficEnv(scenario_config=cfg, mode="fixed")
    obs, _ = env.reset()

    obs, ep_return, terminated, truncated, info = _run_until_done(env, _cruise_action(), obs)

    assert terminated is True and truncated is False
    assert info["outcome"] == Outcome.FINISHED.value
    assert info["success"] is True
    assert info["failure_reason"] is None


# ── Timeout handling: correct terminated/truncated, timeout != success ──────

def test_timeout_is_not_success():
    cfg = ScenarioConfig(seed=0, goal_distance_m=10_000.0, max_episode_seconds=1.0,
                          rear=CriticalActorConfig(role="rear", gap=40.0))
    env = EgoTrafficEnv(scenario_config=cfg, mode="fixed")
    obs, _ = env.reset()

    obs, ep_return, terminated, truncated, info = _run_until_done(env, _cruise_action(), obs)

    assert truncated is True and terminated is False
    assert info["outcome"] == Outcome.TIMEOUT.value
    assert info["success"] is False
    assert info["failure_reason"] is None


# ── Actuator rate limiting ───────────────────────────────────────────────────

def test_steering_and_accel_are_rate_limited():
    env = EgoTrafficEnv(scenario_config=ScenarioConfig(seed=0), mode="fixed")
    env.reset()
    assert env.ego_car.state.delta == 0.0
    assert env._prev_accel == 0.0

    env.step(np.array([1.0, 1.0], dtype=np.float32))   # full accel + full right steer, from a standing start

    assert env.ego_car.state.delta == pytest.approx(MAX_STEERING_RATE * DT)
    assert env.ego_car.state.delta < DELTA_MAX   # never reached the raw commanded lock in one step
    assert env._prev_accel == pytest.approx(min(ACCEL_MAX, MAX_JERK * DT))
    assert env._prev_accel < ACCEL_MAX   # jerk limiting binds before the raw commanded accel does


# ── Observation shape/bounds/dtype ──────────────────────────────────────────

def test_observation_shape_bounds_dtype():
    env = EgoTrafficEnv(scenario_config=ScenarioConfig(seed=0), mode="fixed")
    obs, _ = env.reset()
    assert obs.shape == (7 + 5 * N_SURR,) == (82,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)

    obs2, *_ = env.step(env.action_space.sample())
    assert obs2.shape == obs.shape
    assert obs2.dtype == np.float32
    assert env.observation_space.contains(obs2)


# ── Stable actor-to-slot ordering ────────────────────────────────────────────

def test_actor_slot_ordering_is_deterministic():
    cfg = ScenarioConfig(seed=123)

    env_a = EgoTrafficEnv(scenario_config=cfg, mode="fixed", worker_rank=0)
    env_a.reset()
    ids_a = [agent.car.car_id for agent in env_a.agents]
    roles_a = [r["role"] for r in env_a.get_realized_scenario()["agents"]]

    # A different worker_rank must be ignored in "fixed" mode.
    env_b = EgoTrafficEnv(scenario_config=cfg, mode="fixed", worker_rank=7)
    env_b.reset()
    ids_b = [agent.car.car_id for agent in env_b.agents]
    assert ids_a == ids_b

    # Stable role -> car_id convention (see learning.scenario.generate_scenario).
    assert roles_a[0] == "front" and roles_a[1] == "rear" and roles_a[2] == "blocker"
    assert ids_a[:3] == [0, 1, 2]

    # A second reset on the SAME instance must reproduce the same order too.
    env_a.reset()
    assert [agent.car.car_id for agent in env_a.agents] == ids_a


# ── Evaluation label generation ─────────────────────────────────────────────

def test_feasibility_labels():
    solved = [{"n_episodes": 10, "success_rate": 1.0, "repeatability": "OK"}]
    unsolved = [{"n_episodes": 10, "success_rate": 0.0, "repeatability": "OK"}]
    incomplete = [{"n_episodes": 0, "success_rate": 0.0, "repeatability": "OK"}]
    disagreeing = [{"n_episodes": 5, "success_rate": 0.4, "repeatability": "INCONCLUSIVE"}]

    assert label_feasibility(solved, success_threshold=1.0)[0] == "FEASIBLE"
    assert label_feasibility(unsolved, success_threshold=1.0)[0] == "NOT_SOLVED"
    assert label_feasibility(incomplete, success_threshold=1.0)[0] == "INCONCLUSIVE"
    assert label_feasibility(disagreeing, success_threshold=1.0)[0] == "INCONCLUSIVE"
    assert label_feasibility([], success_threshold=1.0)[0] == "INCONCLUSIVE"

    # One good seed among several is enough for FEASIBLE.
    mixed = [unsolved[0], solved[0]]
    assert label_feasibility(mixed, success_threshold=1.0)[0] == "FEASIBLE"


# ── Zero-action ("stop and survive") policy must not be feasible ───────────

def test_zero_action_policy_is_not_successful_or_dominant():
    # Comfortably unreachable while braking to a stop -- see ScenarioConfig.goal_distance_m's comment
    # for why the default (150m) is already sized against this kind of concern; this test uses an
    # explicit, larger value so the assertion doesn't depend on exact braking-distance arithmetic.
    cfg = ScenarioConfig(seed=0, goal_distance_m=300.0,
                          rear=CriticalActorConfig(role="rear", gap=40.0))

    env = EgoTrafficEnv(scenario_config=cfg, mode="fixed")
    obs, _ = env.reset()
    _, zero_return, terminated, truncated, info = _run_until_done(env, np.zeros(2, dtype=np.float32), obs)
    assert info["success"] is False
    assert info["outcome"] != Outcome.FINISHED.value

    # A reasonable forward-driving (cruise) policy should out-earn passively braking to a stop --
    # a stopped/near-stopped ego must not be the higher-return option merely for surviving.
    env2 = EgoTrafficEnv(scenario_config=cfg, mode="fixed")
    obs2, _ = env2.reset()
    _, cruise_return, *_ = _run_until_done(env2, _cruise_action(), obs2)

    assert cruise_return > zero_return
