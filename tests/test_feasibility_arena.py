"""
Tests for the feasibility-surrogate scenario families (learning.
feasibility_common / feasibility_cutin / feasibility_sandwich) and their
integration into EgoTrafficEnv (learning.env's `arena` parameter).

Run with: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_feasibility_arena.py -v
"""

import numpy as np
import pytest

from model.road.road import Road
from initialization.traffic_init import CAR_LENGTH

from learning.env import EgoTrafficEnv, ARENA_N_SURR, MAX_STEERING_RATE
from learning.feasibility_common import (
    ScenarioFamilyConfigError, road_segment_bounds, road_segment_at, scenario_id,
)
from learning.feasibility_cutin import (
    CutinConfig, generate_cutin_arena, CutinRuntime, SLOT_NAMES as CUTIN_SLOTS,
    S_CUTIN_MIN_M, S_CUTIN_MAX_M,
)
from learning.feasibility_sandwich import (
    SandwichConfig, generate_sandwich_arena, SandwichRuntime, SLOT_NAMES as SW_SLOTS,
)


def _road() -> Road:
    return Road(s_max=500, kappa_max=0.005, L_clothoid=60, mu=1.0, lane_num=3)


def _cruise_action():
    from learning.env import ACCEL_MIN, ACCEL_MAX
    return np.array([2.0 * (0.0 - ACCEL_MIN) / (ACCEL_MAX - ACCEL_MIN) - 1.0, 0.0], dtype=np.float32)


# ── Generation, ordering, geometry ──────────────────────────────────────────

@pytest.mark.parametrize("gen,cfg_cls,slots", [
    (generate_cutin_arena, CutinConfig, CUTIN_SLOTS),
    (generate_sandwich_arena, SandwichConfig, SW_SLOTS),
])
def test_six_agents_fixed_semantic_order(gen, cfg_cls, slots):
    assert slots == ("front_1", "front_2", "rear_1", "rear_2", "blocker_1", "blocker_2")
    road = _road()
    rng = np.random.default_rng(0)
    agents, ego_v0, realized = gen(cfg_cls(seed=0), road, ego_s=50.0, ego_lane=1, rng=rng)
    assert len(agents) == 6 == ARENA_N_SURR
    assert [a.car.car_id for a in agents] == [0, 1, 2, 3, 4, 5]
    assert [r["role"] for r in realized["agents"]] == list(slots)


@pytest.mark.parametrize("gen,cfg_cls", [
    (generate_cutin_arena, CutinConfig),
    (generate_sandwich_arena, SandwichConfig),
])
def test_no_initial_overlap(gen, cfg_cls):
    road = _road()
    for seed in range(10):
        rng = np.random.default_rng(seed)
        agents, _, _ = gen(cfg_cls(seed=seed), road, ego_s=50.0, ego_lane=1, rng=rng)
        cars = [a.car.state for a in agents]
        for i in range(len(cars)):
            for j in range(i + 1, len(cars)):
                if cars[i].lane == cars[j].lane:
                    assert abs(cars[i].s - cars[j].s) >= CAR_LENGTH, f"seed={seed} overlap {i},{j}"


def test_cutin_blocker_gap_geometry():
    road = _road()
    rng = np.random.default_rng(0)
    cfg = CutinConfig(seed=0, mode="exact", cutter_gap_m=20.0, escape_gap_length_m=15.0)
    agents, ego_v0, _ = generate_cutin_arena(cfg, road, ego_s=50.0, ego_lane=1, rng=rng)
    blocker_1, blocker_2 = agents[4].car.state, agents[5].car.state
    assert blocker_1.lane == blocker_2.lane == 2   # blocker_side default +1
    bumper_gap = (blocker_2.s - blocker_1.s) - CAR_LENGTH
    assert bumper_gap == pytest.approx(cfg.escape_gap_length_m)


def test_sandwich_blocker_gap_geometry():
    road = _road()
    rng = np.random.default_rng(0)
    cfg = SandwichConfig(seed=0, mode="exact", escape_gap_center_m=5.0, escape_gap_length_m=18.0)
    agents, ego_v0, _ = generate_sandwich_arena(cfg, road, ego_s=50.0, ego_lane=1, rng=rng)
    blocker_1, blocker_2 = agents[4].car.state, agents[5].car.state
    bumper_gap = (blocker_2.s - blocker_1.s) - CAR_LENGTH
    assert bumper_gap == pytest.approx(cfg.escape_gap_length_m)
    center = (blocker_1.s + blocker_2.s) / 2.0
    assert center == pytest.approx(50.0 + cfg.escape_gap_center_m)


@pytest.mark.parametrize("cfg_cls,gen,side", [
    (CutinConfig, generate_cutin_arena, -1),
    (CutinConfig, generate_cutin_arena, 2),
    (SandwichConfig, generate_sandwich_arena, -1),
])
def test_blocker_side_validation(cfg_cls, gen, side):
    road = _road()
    rng = np.random.default_rng(0)
    if side == -1:
        # ego in lane 0 (leftmost) + blocker_side -1 -> lane -1, invalid
        with pytest.raises(ScenarioFamilyConfigError):
            gen(cfg_cls(seed=0, blocker_side=side), road, ego_s=50.0, ego_lane=0, rng=rng)
    else:
        with pytest.raises(ScenarioFamilyConfigError):
            gen(cfg_cls(seed=0, blocker_side=side), road, ego_s=50.0, ego_lane=0, rng=rng)   # type: ignore


def test_event_station_out_of_domain_rejected():
    road = _road()
    rng = np.random.default_rng(0)
    with pytest.raises(ScenarioFamilyConfigError):
        generate_cutin_arena(CutinConfig(seed=0, s_cutin_m=S_CUTIN_MAX_M + 50.0), road, 50.0, 1, rng)
    with pytest.raises(ScenarioFamilyConfigError):
        generate_cutin_arena(CutinConfig(seed=0, s_cutin_m=S_CUTIN_MIN_M - 50.0), road, 50.0, 1, rng)


# ── Reproducibility: deterministic from seed, theta separate from realization noise ─────────

def test_deterministic_reproduction_from_seed():
    road = _road()
    cfg = CutinConfig(seed=7, mode="robust")
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    agents_a, v0_a, real_a = generate_cutin_arena(cfg, road, 50.0, 1, rng_a)
    agents_b, v0_b, real_b = generate_cutin_arena(cfg, road, 50.0, 1, rng_b)
    assert v0_a == v0_b
    assert real_a == real_b
    for a, b in zip(agents_a, agents_b):
        assert a.car.state.s == b.car.state.s
        assert a.car.state.v_x == b.car.state.v_x


def test_exact_mode_has_zero_perturbation():
    road = _road()
    cfg_exact = CutinConfig(seed=7, mode="exact")
    a1, v1, _ = generate_cutin_arena(cfg_exact, road, 50.0, 1, np.random.default_rng(1))
    a2, v2, _ = generate_cutin_arena(cfg_exact, road, 50.0, 1, np.random.default_rng(2))
    # Different rng streams must NOT matter in exact mode.
    assert v1 == v2
    for x, y in zip(a1, a2):
        assert x.car.state.s == y.car.state.s
        assert x.car.state.v_x == y.car.state.v_x


def test_robust_mode_varies_across_seeds():
    road = _road()
    cfg = CutinConfig(seed=7, mode="robust")
    a1, v1, _ = generate_cutin_arena(cfg, road, 50.0, 1, np.random.default_rng(1))
    a2, v2, _ = generate_cutin_arena(cfg, road, 50.0, 1, np.random.default_rng(2))
    assert v1 != v2 or any(x.car.state.s != y.car.state.s for x, y in zip(a1, a2))


def test_theta_separate_from_realization_noise():
    """Same nominal theta, different realization seeds -> same theta/scenario_id,
    different realized (perturbed) positions in robust mode."""
    cfg = CutinConfig(seed=7, mode="robust")
    road = _road()
    a1, v1, r1 = generate_cutin_arena(cfg, road, 50.0, 1, np.random.default_rng(1))
    a2, v2, r2 = generate_cutin_arena(cfg, road, 50.0, 1, np.random.default_rng(2))
    assert cfg.scenario_id() == scenario_id("cutin", cfg.blocker_side, cfg.mode, cfg.theta())
    assert r1["s_cutin_realized_m"] != r2["s_cutin_realized_m"]
    assert r1["s_cutin_nominal_m"] == r2["s_cutin_nominal_m"] == cfg.s_cutin_m


# ── Road-segment stratification ─────────────────────────────────────────────

def test_road_segments_cover_whole_road_in_order():
    road = _road()
    bounds = road_segment_bounds(road)
    names = [b[0] for b in bounds]
    assert names == ["pre_curve_straight", "clothoid_entry", "constant_curve",
                      "clothoid_exit", "post_curve_straight"]
    assert bounds[0][1] == 0.0
    assert bounds[-1][2] == road.s_max
    for (_, _, hi_a), (_, lo_b, _) in zip(bounds, bounds[1:]):
        assert hi_a == lo_b


def test_road_segment_at_selects_each_segment():
    road = _road()
    assert road_segment_at(road, 10.0) == "pre_curve_straight"
    assert road_segment_at(road, 180.0) == "clothoid_entry"
    assert road_segment_at(road, 250.0) == "constant_curve"
    assert road_segment_at(road, 320.0) == "clothoid_exit"
    assert road_segment_at(road, 450.0) == "post_curve_straight"


# ── Stable scenario IDs ──────────────────────────────────────────────────────

def test_scenario_ids_distinguish_meaningfully_different_theta():
    a = scenario_id("cutin", 1, "robust", {"x": 1.0, "y": 2.0})
    b = scenario_id("cutin", 1, "robust", {"x": 1.0, "y": 2.0 + 1e-6})
    c = scenario_id("cutin", -1, "robust", {"x": 1.0, "y": 2.0})   # side differs
    d = scenario_id("sandwich", 1, "robust", {"x": 1.0, "y": 2.0})   # family differs
    assert len({a, b, c, d}) == 4


def test_scenario_id_stable_and_deterministic():
    theta = {"x": 1.0, "y": 2.0}
    assert scenario_id("cutin", 1, "robust", theta) == scenario_id("cutin", 1, "robust", dict(theta))


# ── EgoTrafficEnv integration: spatial triggers, obs shape, legacy env untouched ────────────

def test_arena_and_scenario_config_are_mutually_exclusive():
    from learning.scenario import ScenarioConfig
    with pytest.raises(ValueError):
        EgoTrafficEnv(scenario_config=ScenarioConfig(), arena=CutinRuntime(CutinConfig()))


def test_arena_observation_shape():
    env = EgoTrafficEnv(arena=CutinRuntime(CutinConfig(seed=0)), mode="fixed")
    obs, _ = env.reset()
    assert obs.shape == (37,) == (7 + 5 * ARENA_N_SURR,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)


def test_legacy_env_observation_shape_unchanged():
    """The 15-vehicle legacy/scenario paths must be completely unaffected by the arena addition."""
    env = EgoTrafficEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (82,)


def test_no_event_before_configured_station():
    cfg = CutinConfig(seed=0, mode="exact", s_cutin_m=300.0)
    env = EgoTrafficEnv(arena=CutinRuntime(cfg), mode="fixed")
    obs, _ = env.reset()
    for _ in range(20):   # ~1s, ego starts at s=50 -- nowhere near s=300 yet
        obs, reward, term, trunc, info = env.step(_cruise_action())
        assert info["arena"]["event_triggered"] is False
        if term or trunc:
            break


def test_event_triggers_once_and_is_latched():
    cfg = CutinConfig(seed=0, mode="exact", s_cutin_m=110.0)
    env = EgoTrafficEnv(arena=CutinRuntime(cfg), mode="fixed")
    obs, _ = env.reset()
    trigger_times = []
    for _ in range(150):
        obs, reward, term, trunc, info = env.step(_cruise_action())
        if info["arena"]["event_triggered"]:
            trigger_times.append(info["arena"]["event_trigger_time_s"])
        if term or trunc:
            break
    assert len(trigger_times) > 0
    assert len(set(trigger_times)) == 1   # the recorded trigger time never changes once latched


def test_higher_speed_changes_trigger_time_not_station():
    """Same nominal event station -- a faster cruise action must reach it sooner in time, but the
    recorded trigger station itself (event_trigger_ego_s) must be the same (station is a spatial,
    not time, condition)."""
    def run(action):
        cfg = CutinConfig(seed=0, mode="exact", s_cutin_m=110.0)
        env = EgoTrafficEnv(arena=CutinRuntime(cfg), mode="fixed")
        obs, _ = env.reset()
        for _ in range(300):
            obs, reward, term, trunc, info = env.step(action)
            if info["arena"]["event_triggered"]:
                return info["arena"]["event_trigger_time_s"], info["arena"]["event_trigger_ego_s"]
            if term or trunc:
                break
        raise AssertionError("never triggered")

    from learning.env import ACCEL_MIN, ACCEL_MAX
    slow = np.array([2.0 * (-2.0 - ACCEL_MIN) / (ACCEL_MAX - ACCEL_MIN) - 1.0, 0.0], dtype=np.float32)
    fast = np.array([1.0, 0.0], dtype=np.float32)   # max accel
    t_slow, s_slow = run(slow)
    t_fast, s_fast = run(fast)
    assert t_fast < t_slow
    assert s_slow == pytest.approx(s_fast, abs=0.5)   # same station (within one integration step's slack)


# ── Cut-in specific ───────────────────────────────────────────────────────

def test_cutin_prescribed_change_bypasses_mobil_and_completes():
    """With discretionary MOBIL disabled (threshold=inf), the ONLY way the cutter ends up in ego's
    lane is the prescribed event -- this proves that mechanism actually works end to end. A large
    cutter_gap_m keeps the naive non-avoiding cruise test policy from colliding with the cutter
    before the (multi-second) lane change has time to complete."""
    cfg = CutinConfig(seed=0, mode="exact", s_cutin_m=110.0, blocker_side=1, cutter_gap_m=45.0)
    env = EgoTrafficEnv(arena=CutinRuntime(cfg), mode="fixed")
    obs, _ = env.reset()
    ego_lane = env.ego_car.state.lane
    cutter_before = next(a for a in env.agents if a.car.car_id == 4).car.state.lane
    assert cutter_before != ego_lane

    completed = False
    for _ in range(300):
        obs, reward, term, trunc, info = env.step(_cruise_action())
        if info["arena"]["cutin_completed"]:
            completed = True
            break
        if term or trunc:
            break
    assert completed
    cutter_after = next(a for a in env.agents if a.car.car_id == 4).car.state.lane
    assert cutter_after == ego_lane


def test_cutin_no_lateral_teleport():
    """e_y must change continuously (bounded step-to-step delta), never jump discontinuously."""
    cfg = CutinConfig(seed=0, mode="exact", s_cutin_m=110.0)
    env = EgoTrafficEnv(arena=CutinRuntime(cfg), mode="fixed")
    obs, _ = env.reset()
    prev_e_y = next(a for a in env.agents if a.car.car_id == 4).car.state.e_y
    max_step_e_y_change = 0.0
    for _ in range(300):
        obs, reward, term, trunc, info = env.step(_cruise_action())
        cutter = next(a for a in env.agents if a.car.car_id == 4)
        max_step_e_y_change = max(max_step_e_y_change, abs(cutter.car.state.e_y - prev_e_y))
        prev_e_y = cutter.car.state.e_y
        if term or trunc:
            break
    # A lane's full offset is ~4m; a real per-step change should be a small fraction of that at DT=0.05s.
    assert max_step_e_y_change < 1.0


# ── Sandwich specific ────────────────────────────────────────────────────

def test_sandwich_front_decelerates_monotonically_to_zero_and_holds():
    cfg = SandwichConfig(seed=0, mode="exact", s_stop_m=110.0, front_relative_speed_mps=0.0)
    env = EgoTrafficEnv(arena=SandwichRuntime(cfg), mode="fixed")
    obs, _ = env.reset()
    speeds = []
    for _ in range(300):
        obs, reward, term, trunc, info = env.step(_cruise_action())
        stopper = next(a for a in env.agents if a.car.car_id == 0)
        speeds.append(stopper.car.state.v_x)
        if info["arena"]["front_stopped"] and len(speeds) > 5:
            break
        if term or trunc:
            break
    assert all(v >= -1e-6 for v in speeds), "front_1's speed must never go negative"
    post_trigger = speeds[speeds.index(max(speeds)):]   # from peak speed (trigger) onward, should be non-increasing until 0
    # after the trigger the speed should be non-increasing (monotonic decel) until it bottoms at 0
    decreasing_part = []
    for v in post_trigger:
        decreasing_part.append(v)
        if v <= 1e-6:
            break
    assert all(a >= b - 1e-9 for a, b in zip(decreasing_part, decreasing_part[1:]))
    # Not exactly 0.0: the clamped-accel formula (see SandwichRuntime.pre_surr_step) is exact for
    # the isolated dv_x/dt=accel ODE, but RK4 still integrates the full 6-state CarDynamics model,
    # whose tiny (numerical-noise-level) r*v_y coupling leaves a negligible residual -- physically
    # stopped (<1mm/s) well within one step, not a real drift.
    assert speeds[-1] == pytest.approx(0.0, abs=1e-2)


def test_sandwich_idm_never_resumes_after_stop():
    cfg = SandwichConfig(seed=0, mode="exact", s_stop_m=110.0)
    env = EgoTrafficEnv(arena=SandwichRuntime(cfg), mode="fixed")
    obs, _ = env.reset()
    stopped_seen = False
    for _ in range(400):
        obs, reward, term, trunc, info = env.step(_cruise_action())
        if info["arena"]["front_stopped"]:
            stopped_seen = True
            stopper = next(a for a in env.agents if a.car.car_id == 0)
            assert stopper.car.state.v_x == pytest.approx(0.0, abs=1e-2)   # see the sibling test's own comment
        if term or trunc:
            break
    assert stopped_seen


def test_sandwich_deterministic_reproduction():
    cfg = SandwichConfig(seed=42, mode="robust")
    def run():
        env = EgoTrafficEnv(arena=SandwichRuntime(cfg), mode="fixed")
        obs, _ = env.reset()
        trace = []
        for _ in range(50):
            obs, reward, term, trunc, info = env.step(_cruise_action())
            trace.append((float(obs[0]), info["arena"]["event_triggered"]))
            if term or trunc:
                break
        return trace
    assert run() == run()
