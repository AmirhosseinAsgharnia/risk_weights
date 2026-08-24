"""
Tests for the compact scenario parameterization (learning.scenario) and
its wiring into learning.env.EgoTrafficEnv. See
/home/amir/.claude/plans/replicated-sleeping-firefly.md for the design
this implements.

Run with: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_scenario.py -v
(this venv has an unrelated pre-existing issue where ROS's launch_testing/
launch_ros pytest plugins auto-load and fail on a missing pyyaml -- not
something to fix here).
"""

import json

import numpy as np
import pytest

from model.road.road import Road
from initialization.traffic_init import CAR_LENGTH
from learning.scenario import (
    ScenarioConfig, CriticalActorConfig, ScenarioConfigError,
    generate_scenario, derive_seed, N_SURR, N_BACKGROUND, N_CRITICAL,
)
from learning.env import EgoTrafficEnv, LANE_NUM, ROAD_KWARGS


def _road() -> Road:
    return Road(**ROAD_KWARGS)


def _bodies(realized: dict) -> list[tuple[str, float, int]]:
    """(name, s, lane) for ego + every agent in a realized-scenario dict."""
    out = [("ego", realized["ego"]["s"], realized["ego"]["lane"])]
    out += [(f"{a['role']}#{a['id']}", a["s"], a["lane"]) for a in realized["agents"]]
    return out


# ── 1/2: exactly 15 agents, exactly 3 critical ──────────────────────────────

def test_exactly_15_agents():
    road = _road()
    rng = np.random.default_rng(0)
    agents, ego_v0, realized = generate_scenario(ScenarioConfig(), road, ego_s=50.0, ego_lane=1, rng=rng)
    assert len(agents) == 15 == N_SURR
    assert len(realized["agents"]) == 15


def test_exactly_three_critical_roles():
    road = _road()
    rng = np.random.default_rng(0)
    _, _, realized = generate_scenario(ScenarioConfig(), road, ego_s=50.0, ego_lane=1, rng=rng)
    roles = [a["role"] for a in realized["agents"]]
    assert roles.count("front") == 1
    assert roles.count("rear") == 1
    assert roles.count("blocker") == 1
    assert roles.count("background") == N_BACKGROUND == 12
    assert sum(r != "background" for r in roles) == N_CRITICAL == 3


# ── 3: fixed mode reproducibility ───────────────────────────────────────────

def test_fixed_mode_reproducible():
    cfg = ScenarioConfig(seed=7)
    env = EgoTrafficEnv(scenario_config=cfg, mode="fixed", worker_rank=0)
    obs_a, _ = env.reset()
    realized_a = env.get_realized_scenario()
    obs_b, _ = env.reset()
    realized_b = env.get_realized_scenario()
    assert realized_a == realized_b
    assert np.array_equal(obs_a, obs_b)

    # A second, freshly-constructed env (different worker_rank even) must
    # still land on the identical realization in fixed mode.
    env2 = EgoTrafficEnv(scenario_config=cfg, mode="fixed", worker_rank=5)
    obs_c, _ = env2.reset()
    realized_c = env2.get_realized_scenario()
    assert realized_a == realized_c


# ── 4: distribution mode varies ─────────────────────────────────────────────

def test_distribution_mode_varies():
    cfg = ScenarioConfig(seed=7)
    env = EgoTrafficEnv(scenario_config=cfg, mode="distribution", worker_rank=0)
    obs_a, _ = env.reset()
    realized_a = env.get_realized_scenario()
    obs_b, _ = env.reset()
    realized_b = env.get_realized_scenario()
    assert realized_a != realized_b
    # Critical actors' *positions* (deterministic given cfg) stay
    # identical across resets...
    for role in ("front", "rear", "blocker"):
        a = next(x for x in realized_a["agents"] if x["role"] == role)
        b = next(x for x in realized_b["agents"] if x["role"] == role)
        assert a["s"] == b["s"] and a["lane"] == b["lane"]
    # ...but NOT their v_x: ego's own speed is now drawn stochastically
    # each episode (Normal(background_mean_speed, background_speed_std),
    # same as background traffic -- see learning.scenario's module
    # docstring), and critical actor speed is ego_v0 + relative_speed, so
    # it legitimately varies episode to episode too.
    assert realized_a["ego"]["v_x"] != realized_b["ego"]["v_x"]
    # ...the background realization varies as well, of course.
    bg_a = [x["s"] for x in realized_a["agents"] if x["role"] == "background"]
    bg_b = [x["s"] for x in realized_b["agents"] if x["role"] == "background"]
    assert bg_a != bg_b


# ── 5: distribution mode reproducible across separate runs/instances ───────

def test_distribution_mode_reproducible_across_runs():
    cfg = ScenarioConfig(seed=42)

    def run_sequence(n_resets: int) -> list[dict]:
        env = EgoTrafficEnv(scenario_config=cfg, mode="distribution", worker_rank=2)
        out = []
        for _ in range(n_resets):
            env.reset()
            out.append(env.get_realized_scenario())
        return out

    seq1 = run_sequence(4)
    seq2 = run_sequence(4)   # fresh instance, same "experiment" replayed
    assert seq1 == seq2


# ── 6: no initial overlap ───────────────────────────────────────────────────

def test_no_initial_overlap():
    road = _road()
    for seed in range(15):
        rng = np.random.default_rng(seed)
        _, _, realized = generate_scenario(ScenarioConfig(seed=seed), road, ego_s=50.0, ego_lane=1, rng=rng)
        bodies = _bodies(realized)
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                n1, s1, lane1 = bodies[i]
                n2, s2, lane2 = bodies[j]
                if lane1 == lane2:
                    assert abs(s1 - s2) >= CAR_LENGTH, (
                        f"seed={seed}: {n1} and {n2} overlap in lane {lane1} "
                        f"(|ds|={abs(s1 - s2):.3f} < {CAR_LENGTH})")


# ── 7: critical actor placement (gap/lane/speed/behavior + CAR_LENGTH conv.) ─

def test_critical_actor_placement():
    # background_speed_std=0.0 makes ego_v0's draw deterministic (always
    # exactly background_mean_speed) so this test can assert exact values --
    # ego no longer has its own fixed speed input (it's drawn the same way
    # background traffic speed is, see learning.scenario's module docstring).
    cfg = ScenarioConfig(
        background_mean_speed=18.0, background_speed_std=0.0,
        front=CriticalActorConfig(role="front", behavior="aggressive", relative_speed=-3.0, gap=25.0),
        rear=CriticalActorConfig(role="rear", behavior="conservative", relative_speed=2.0, gap=15.0),
        blocker=CriticalActorConfig(role="blocker", behavior="moderate", relative_speed=1.5,
                                     lane_offset=-1, relative_s=5.0),
    )
    road = _road()
    rng = np.random.default_rng(0)
    ego_s, ego_lane = 50.0, 1
    _, ego_v0, realized = generate_scenario(cfg, road, ego_s=ego_s, ego_lane=ego_lane, rng=rng)

    assert ego_v0 == 18.0

    front = next(a for a in realized["agents"] if a["role"] == "front")
    assert front["s"] == pytest.approx(ego_s + 25.0 + CAR_LENGTH)
    assert front["lane"] == ego_lane
    assert front["v_x"] == pytest.approx(18.0 - 3.0)
    assert front["behaviour"] == "aggressive"

    rear = next(a for a in realized["agents"] if a["role"] == "rear")
    assert rear["s"] == pytest.approx(ego_s - 15.0 - CAR_LENGTH)
    assert rear["lane"] == ego_lane
    assert rear["v_x"] == pytest.approx(18.0 + 2.0)
    assert rear["behaviour"] == "conservative"

    blocker = next(a for a in realized["agents"] if a["role"] == "blocker")
    assert blocker["s"] == pytest.approx(ego_s + 5.0)
    assert blocker["lane"] == ego_lane - 1
    assert blocker["v_x"] == pytest.approx(18.0 + 1.5)
    assert blocker["behaviour"] == "moderate"


# ── 8: background speed distribution ────────────────────────────────────────

def test_background_speed_distribution():
    cfg = ScenarioConfig(background_mean_speed=22.0, background_speed_std=1.0, seed=0)
    road = _road()
    speeds = []
    for seed in range(40):   # many scenarios -> ~480 background cars, enough for a loose statistical check
        rng = np.random.default_rng(seed)
        _, _, realized = generate_scenario(cfg, road, ego_s=50.0, ego_lane=1, rng=rng)
        speeds += [a["v0"] for a in realized["agents"] if a["role"] == "background"]

    speeds = np.array(speeds)
    assert speeds.min() >= 0.0   # clip bound
    assert speeds.max() <= 40.0  # clip bound
    # loose check (not tight, to avoid flakiness): sample mean within 1
    # sample-std-error-ish band of the configured mean.
    assert abs(speeds.mean() - 22.0) < 0.5


# ── 9: invalid blocker lane raises clearly ──────────────────────────────────

def test_invalid_blocker_lane_raises():
    road = _road()
    rng = np.random.default_rng(0)
    cfg = ScenarioConfig(blocker=CriticalActorConfig(role="blocker", lane_offset=-1, relative_s=0.0))
    with pytest.raises(ScenarioConfigError, match="outside"):
        generate_scenario(cfg, road, ego_s=50.0, ego_lane=0, rng=rng)   # ego already in lane 0, offset -1 -> lane -1

    cfg2 = ScenarioConfig(blocker=CriticalActorConfig(role="blocker", lane_offset=2, relative_s=0.0))
    with pytest.raises(ScenarioConfigError, match="-1 or \\+1"):
        generate_scenario(cfg2, road, ego_s=50.0, ego_lane=1, rng=rng)


# ── 10: realized scenario round-trips through JSON and replays exactly ─────

def test_realized_scenario_replay():
    cfg = ScenarioConfig(seed=13)
    road = _road()
    seed = derive_seed(cfg.seed, worker_rank=0, episode_index=0)
    rng = np.random.default_rng(seed)
    _, _, realized = generate_scenario(cfg, road, ego_s=50.0, ego_lane=1, rng=rng,
                                        worker_rank=0, episode_index=0, scenario_seed=seed)

    round_tripped = json.loads(json.dumps(realized))
    assert round_tripped == realized

    # Replay: re-derive the same seed from the round-tripped seed_info and
    # regenerate -- must reproduce the identical realization.
    si = round_tripped["seed_info"]
    replay_cfg = ScenarioConfig.from_dict(round_tripped["scenario_config"])
    replay_seed = derive_seed(si["base_seed"], si["worker_rank"], si["episode_index"])
    assert replay_seed == si["scenario_seed"]
    replay_rng = np.random.default_rng(replay_seed)
    _, _, replay_realized = generate_scenario(replay_cfg, road, ego_s=50.0, ego_lane=1, rng=replay_rng,
                                               worker_rank=si["worker_rank"], episode_index=si["episode_index"],
                                               scenario_seed=replay_seed)
    assert replay_realized == realized


# ── 11: parallel workers don't produce identical episodes ─────────────────

def test_parallel_workers_differ():
    cfg = ScenarioConfig(seed=99)
    env_a = EgoTrafficEnv(scenario_config=cfg, mode="distribution", worker_rank=0)
    env_b = EgoTrafficEnv(scenario_config=cfg, mode="distribution", worker_rank=1)
    env_a.reset()
    env_b.reset()
    assert env_a.get_realized_scenario() != env_b.get_realized_scenario()


# ── 12: observation shape/dtype regression check ────────────────────────────

def test_observation_shape_dtype():
    env = EgoTrafficEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (5 + 3 * N_SURR,) == (50,)
    assert obs.dtype == np.float32

    obs2, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs2.shape == obs.shape
    assert obs2.dtype == np.float32
