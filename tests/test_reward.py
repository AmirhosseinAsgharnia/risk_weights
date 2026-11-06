"""
The PPO training reward (learning.env.RewardConfig): r = r_progress + r_terminal - r_time.

White-box in the style of tests/test_env.py: states are injected to reach exact conditions. `_open_road`
removes traffic so s only moves as the test intends. Tests A-K follow the reward specification.
"""

import dataclasses
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import learning.env as env_mod
from learning.env import (
    ACCEL_MAX, ACCEL_MIN, DEFAULT_PPO_GAMMA, DELTA_MAX, EGO_S0, EgoTrafficEnv, FailureReason, Outcome, RewardConfig,
)
from learning.feasibility_cutin import CutinConfig, CutinRuntime
from learning.feasibility_sandwich import SandwichConfig, SandwichRuntime
from learning.feasibility_train_one import RunParams, train_and_evaluate
from learning.scenario import ScenarioConfig

T = RewardConfig().time_penalty_per_step


def _action(accel: float = 0.0, delta: float = 0.0) -> np.ndarray:
    return np.array([2.0 * (accel - ACCEL_MIN) / (ACCEL_MAX - ACCEL_MIN) - 1.0, delta / DELTA_MAX], dtype=np.float32)


BRAKE = _action(ACCEL_MIN)


def _scenario_env(goal: float = 100.0, seconds: float = 20.0, **kw) -> EgoTrafficEnv:
    env = EgoTrafficEnv(scenario_config=ScenarioConfig(seed=0, goal_distance_m=goal, max_episode_seconds=seconds),
                        mode="fixed", **kw)
    env.reset()
    return env


def _open_road(goal: float = 100.0, seconds: float = 20.0, v_x: float = 0.0, **kw) -> EgoTrafficEnv:
    env = _scenario_env(goal, seconds, **kw)
    env.agents.clear()
    env.ego_car.state.v_x = v_x
    return env


def _hold(env: EgoTrafficEnv, v_x: float, n: int) -> list[dict]:
    """n steps at speed v_x, re-injected each step; v_x < 0 moves ego backward along s."""
    infos = []
    for _ in range(n):
        env.ego_car.state.v_x = v_x
        infos.append(env.step(_action())[-1])
    return infos


def _on_front_car(env: EgoTrafficEnv) -> None:
    front = env.agents[0].car.state
    st = env.ego_car.state
    st.s, st.lane, st.e_y = front.s, front.lane, front.e_y


def _random_episode(env: EgoTrafficEnv, seed: int) -> list[tuple[float, dict]]:
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    out, done = [], False
    while not done:
        _, r, term, trunc, info = env.step(rng.uniform(-1.0, 1.0, 2).astype(np.float32))
        out.append((r, info))
        done = term or trunc
    return out


# ── A: forward progress ─────────────────────────────────────────────────────
def test_a_forward_progress_is_normalized_by_goal_distance():
    env = _open_road(goal=200.0, v_x=10.0)
    s0 = env.ego_car.state.s
    _, r, _, _, info = env.step(_action())
    ds = env.ego_car.state.s - s0
    assert ds > 0.4
    assert info["reward_progress"] == pytest.approx(ds / 200.0, abs=1e-12)
    assert r == pytest.approx(ds / 200.0 - T, abs=1e-12)


@pytest.mark.parametrize("family, runtime, cfg", [("cutin", CutinRuntime, CutinConfig()),
                                                  ("sandwich", SandwichRuntime, SandwichConfig())])
def test_a_goal_distance_is_distance_travelled_not_absolute_station(family, runtime, cfg):
    env = EgoTrafficEnv(arena=runtime(cfg), mode="fixed")
    env.reset()
    assert env._goal_s == EGO_S0 + cfg.goal_distance_m and env._goal_distance == cfg.goal_distance_m
    total = sum(env.step(_action())[-1]["reward_progress"] for _ in range(20))
    assert total == pytest.approx((env.ego_car.state.s - EGO_S0) / cfg.goal_distance_m, abs=1e-12)


def test_a_legacy_path_goal_is_the_road_end():
    env = EgoTrafficEnv()
    env.reset(seed=0)
    assert env._goal_distance == env.road.s_max - EGO_S0


# ── B: no progress ──────────────────────────────────────────────────────────
def test_b_stationary_vehicle_gets_only_the_time_penalty():
    env = _open_road()
    for _ in range(5):
        _, r, term, trunc, info = env.step(BRAKE)
        assert (info["reward_progress"], info["reward_terminal"], info["reward_time"]) == (0.0, 0.0, T)
        assert r == -T and not term and not trunc


# ── C: reverse ──────────────────────────────────────────────────────────────
def test_c_moving_backward_pays_no_progress():
    env = _open_road()
    s0 = env.ego_car.state.s
    infos = _hold(env, -5.0, 10)
    assert env.ego_car.state.s < s0 - 2.0
    assert all(i["reward_progress"] == 0.0 for i in infos)


def test_c_driving_the_wrong_way_pays_no_progress():
    env = _open_road(v_x=10.0)
    env.ego_car.state.e_psi = math.pi   # moving forward in the body frame, backward along the road
    s0 = env.ego_car.state.s
    infos = [env.step(_action())[-1] for _ in range(10)]
    assert env.ego_car.state.s < s0 - 2.0
    assert all(i["reward_progress"] == 0.0 for i in infos)


# ── D: oscillation ──────────────────────────────────────────────────────────
def test_d_retracing_ground_pays_nothing():
    env = _open_road(goal=100.0)
    s0 = env.ego_car.state.s
    first = _hold(env, 10.0, 20)
    peak = env.ego_car.state.s
    assert sum(i["reward_progress"] for i in first) == pytest.approx((peak - s0) / 100.0, abs=1e-12)

    back = _hold(env, -10.0, 10)
    assert all(i["reward_progress"] == 0.0 for i in back)
    regain = []
    while env.ego_car.state.s < peak - 0.6:   # 0.5 m per step: stops short of the peak
        regain += _hold(env, 10.0, 1)
    assert all(i["reward_progress"] == 0.0 for i in regain)

    beyond = _hold(env, 10.0, 10)
    assert env.ego_car.state.s > peak + 3.0
    assert sum(i["reward_progress"] for i in beyond) == pytest.approx((env.ego_car.state.s - peak) / 100.0, abs=1e-12)
    assert beyond[-1]["episode_progress"] == pytest.approx((env.ego_car.state.s - s0) / 100.0, abs=1e-12)


# ── E: success ──────────────────────────────────────────────────────────────
def test_e_goal_completion_pays_the_success_reward_exactly_once():
    env = _open_road(goal=5.0, v_x=10.0)
    infos, done = [], False
    while not done:
        _, r, term, trunc, info = env.step(_action())
        infos.append(info)
        done = term or trunc
    assert [i["reward_terminal"] for i in infos[:-1]] == [0.0] * (len(infos) - 1)
    assert infos[-1]["reward_terminal"] == 3.0
    assert term and not trunc and info["success"] and info["outcome"] == Outcome.FINISHED.value
    assert info["episode_progress"] == 1.0
    assert sum(i["reward_progress"] for i in infos) == pytest.approx(1.0, abs=1e-12)


# ── F: collision ────────────────────────────────────────────────────────────
def test_f_collision_costs_the_collision_penalty():
    env = _scenario_env()
    _on_front_car(env)
    _, r, term, _, info = env.step(_action())
    assert term and info["failure_reason"] == FailureReason.COLLISION.value
    assert info["reward_terminal"] == -5.0 and r == pytest.approx(info["reward_progress"] - 5.0 - T)
    assert info["success"] is False


def test_f_collision_on_the_goal_step_is_a_failure():
    env = _scenario_env(goal=5.0)
    _on_front_car(env)
    assert env.ego_car.state.s >= env._goal_s
    _, _, term, _, info = env.step(_action())
    assert term and info["finished"] is True and info["collided"] is True
    assert info["success"] is False and info["outcome"] == Outcome.SAFETY_FAILURE.value
    assert info["reward_terminal"] == -5.0


# ── G: road departure, rollover, priority ───────────────────────────────────
def _force_off_road(env: EgoTrafficEnv) -> None:
    env.ego_car.state.e_y = env._road_half_width * 5


def _force_legacy_rollover(env: EgoTrafficEnv) -> None:
    env.ego_car.state.delta = DELTA_MAX
    env.ego_car.state.v_x = 40.0


ROLL_ACTION = np.array([0.0, 1.0], dtype=np.float32)


@pytest.mark.parametrize("goal", [100.0, 1.0])
def test_g_road_departure_outranks_the_goal(goal):
    env = _open_road(goal=goal, v_x=30.0)
    _force_off_road(env)
    _, _, term, _, info = env.step(_action())
    assert term and info["failure_reason"] == FailureReason.OFF_ROAD.value
    assert info["reward_terminal"] == -5.0 and info["success"] is False


@pytest.mark.parametrize("goal", [100.0, 1.0])
def test_g_rollover_outranks_the_goal(goal):
    env = _open_road(goal=goal)
    _force_legacy_rollover(env)
    _, _, term, _, info = env.step(ROLL_ACTION)
    assert term and info["failure_reason"] == FailureReason.ROLLOVER.value
    assert info["reward_terminal"] == -5.0 and info["success"] is False


def test_g_physical_rollover_event_gets_the_rollover_penalty():
    env = _open_road(rollover_event="physical_v2", v_x=20.0)
    env.ego_car.state.v_y = -20.0   # a full lateral slide: realized |a_y| reaches SSF g
    _, _, term, _, info = env.step(_action())
    assert term and info["failure_reason"] == FailureReason.ROLLOVER.value and info["reward_terminal"] == -5.0


def test_g_simultaneous_failures_pay_once_by_priority():
    distinct = RewardConfig(collision_penalty=5.0, rollover_penalty=6.0, road_departure_penalty=7.0)
    env = _open_road(reward_config=distinct)
    _force_legacy_rollover(env)
    _force_off_road(env)
    _, _, _, _, info = env.step(ROLL_ACTION)
    assert info["rolled_over"] and info["off_road"]
    assert info["failure_reason"] == FailureReason.ROLLOVER.value and info["reward_terminal"] == -6.0

    env = _scenario_env(reward_config=distinct)
    _on_front_car(env)
    _force_legacy_rollover(env)
    _, _, _, _, info = env.step(ROLL_ACTION)
    assert info["collided"] and info["rolled_over"]
    assert info["failure_reason"] == FailureReason.COLLISION.value and info["reward_terminal"] == -5.0


# ── H: timeout ──────────────────────────────────────────────────────────────
def test_h_timeout_costs_one_and_is_never_success():
    env = _open_road(seconds=0.25)
    infos, done = [], False
    while not done:
        _, r, term, trunc, info = env.step(BRAKE)
        infos.append(info)
        done = term or trunc
    assert len(infos) == 5 and trunc and not term
    assert [i["reward_terminal"] for i in infos] == [0.0, 0.0, 0.0, 0.0, -1.0]
    assert r == -1.0 - T
    assert info["outcome"] == Outcome.TIMEOUT.value and info["timeout"] and not info["success"]


def test_h_reaching_the_goal_on_the_last_step_is_success_not_timeout():
    env = _open_road(goal=0.4, seconds=0.05, v_x=10.0)
    _, _, term, trunc, info = env.step(_action())
    assert term and not trunc and info["success"] and info["reward_terminal"] == 3.0 and not info["timeout"]


# ── I: decomposition and episode bookkeeping ────────────────────────────────
ALL_ENVS = [lambda: _scenario_env(goal=60.0, seconds=6.0),
            lambda: EgoTrafficEnv(arena=CutinRuntime(CutinConfig()), mode="distribution", rollover_event="physical_v2"),
            lambda: EgoTrafficEnv(arena=SandwichRuntime(SandwichConfig()), mode="distribution", rollover_event="physical_v2")]


@pytest.mark.parametrize("make", ALL_ENVS)
def test_i_reward_is_exactly_its_three_components(make):
    env = make()
    for seed in range(3):
        ep = _random_episode(env, seed)
        ret = 0.0
        for n, (r, i) in enumerate(ep, start=1):
            ret += r
            assert r == i["reward_total"] == i["reward_progress"] + i["reward_terminal"] - i["reward_time"]
            assert i["reward_time"] == T and math.isfinite(r)
            assert i["episode_length"] == n and i["episode_return"] == pytest.approx(ret, abs=1e-12)
        assert sum(i["reward_terminal"] != 0.0 for _, i in ep) == 1 and ep[-1][1]["reward_terminal"] != 0.0


def test_i_time_penalty_applies_on_the_final_step():
    env = _open_road(goal=1.0, v_x=30.0)
    _, r, term, _, info = env.step(_action())
    assert term and r == pytest.approx(1.0 + 3.0 - T, abs=1e-12)


# ── J: boundedness ──────────────────────────────────────────────────────────
def test_j_overshooting_the_goal_is_capped():
    env = _open_road(goal=1.0, v_x=60.0)   # 3 m in the one step that crosses a 1 m goal
    _, _, term, _, info = env.step(_action())
    assert term and env.ego_car.state.s > env._goal_s + 1.0
    assert info["reward_progress"] == pytest.approx(1.0, abs=1e-12) and info["episode_progress"] == 1.0


def test_j_progress_stays_bounded_while_oscillating_near_the_goal():
    env = _open_road(goal=4.0, seconds=30.0)
    total = 0.0
    for _ in range(40):
        total += sum(i["reward_progress"] for i in _hold(env, 10.0, 6))
        total += sum(i["reward_progress"] for i in _hold(env, -10.0, 6))
        assert total <= 1.0
    assert env.step(_action())[-1]["episode_progress"] <= 1.0


@pytest.mark.parametrize("make", ALL_ENVS)
def test_j_random_episodes_stay_bounded(make):
    env = make()
    for seed in range(3):
        ep = _random_episode(env, seed)
        assert 0.0 <= sum(i["reward_progress"] for _, i in ep) <= 1.0 + 1e-12


# ── K: compatibility, independence, removals ────────────────────────────────
@pytest.mark.parametrize("make", ALL_ENVS)
def test_k_gymnasium_interface(make):
    check_env(make(), skip_render_check=True)


def test_k_spaces_are_unchanged():
    arena = EgoTrafficEnv(arena=CutinRuntime(CutinConfig()))
    scenario = EgoTrafficEnv(scenario_config=ScenarioConfig(seed=0))
    assert (arena.observation_space.shape, scenario.observation_space.shape) == ((37,), (82,))
    assert arena.action_space.shape == scenario.action_space.shape == (2,)


def test_k_ppo_initializes_and_collects_rollouts():
    vec = DummyVecEnv([lambda: EgoTrafficEnv(arena=CutinRuntime(CutinConfig()), mode="fixed",
                                             rollover_event="physical_v2")])
    model = PPO("MlpPolicy", vec, n_steps=32, batch_size=32, n_epochs=1, device="cpu", gamma=DEFAULT_PPO_GAMMA, seed=0)
    model.learn(total_timesteps=32)
    assert model.gamma == 0.999 and model.num_timesteps == 32


def test_k_training_entry_point_uses_and_stamps_gamma_and_reward(tmp_path):
    assert RunParams(out="x").gamma == 0.999
    params = RunParams(out=str(tmp_path / "p"), timesteps=64, n_envs=1, no_early_stop=True, episodes=1,
                       gamma=0.995, ppo_seed=0)
    meta = train_and_evaluate("cutin", CutinConfig(), params, verbose=0)
    assert PPO.load(tmp_path / "p", device="cpu").gamma == 0.995
    assert meta["ppo_gamma"] == 0.995 and meta["reward_version"] == env_mod.REWARD_VERSION
    assert meta["reward_config"] == dataclasses.asdict(RewardConfig())


def test_k_outcomes_and_trajectories_do_not_depend_on_the_reward():
    other = RewardConfig(progress_reward_scale=9.0, success_reward=-7.0, collision_penalty=-2.0, rollover_penalty=0.0,
                         road_departure_penalty=0.0, timeout_penalty=11.0, time_penalty_per_step=0.3)
    a = _random_episode(_scenario_env(goal=60.0, seconds=6.0), 0)
    b = _random_episode(_scenario_env(goal=60.0, seconds=6.0, reward_config=other), 0)
    keys = ("outcome", "success", "failure_reason", "timeout", "s", "progress_m", "episode_progress")
    assert [{k: i[k] for k in keys} for _, i in a] == [{k: i[k] for k in keys} for _, i in b]


def test_k_tire_saturation_alone_neither_ends_the_episode_nor_costs_reward():
    env = _open_road(goal=300.0, seconds=2.0, v_x=20.0)
    env.road.mu = 0.3
    saturated = 0
    for _ in range(30):
        _, _, term, _, info = env.step(_action(ACCEL_MIN))
        assert not term and info["reward_terminal"] == 0.0
        saturated = info["tire_saturated_steps"] + (info["max_friction_demand"] >= 1.0)
    assert saturated > 0


@pytest.mark.parametrize("name", ["PROGRESS_REWARD_BOOST", "PROGRESS_REWARD_SCALE", "SURVIVAL_REWARD", "COLLISION_PENALTY",
                                  "ROLLOVER_PENALTY", "OFFROAD_PENALTY", "LANE_TRACKING_SCALE", "FINISH_BONUS"])
def test_k_old_reward_terms_are_gone(name):
    assert not hasattr(env_mod, name)


def test_k_lane_deviation_and_controls_do_not_affect_the_reward():
    a, b = _open_road(v_x=10.0), _open_road(v_x=10.0)
    b.ego_car.state.e_y, b.ego_car.state.e_psi, b.ego_car.state.delta = 1.0, 0.0, 0.1
    b._prev_accel = 3.0
    ia, ib = a.step(_action())[-1], b.step(_action(3.0, 0.2))[-1]
    assert ia["reward_terminal"] == ib["reward_terminal"] == 0.0
    assert ib["reward_total"] == pytest.approx(ib["reward_progress"] - T, abs=1e-15)
    assert ia["reward_total"] == pytest.approx(ia["reward_progress"] - T, abs=1e-15)
