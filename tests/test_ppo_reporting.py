"""
PPO evaluation, analysis, figures and reports (docs/description.md §23). Uses tests/ppo_fixtures.py, whose
numbers are invented and always marked as such, plus one really trained tiny policy for the evaluator's protocol.
"""

import csv
import json
import math
import subprocess
import sys

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from learning.env import EgoTrafficEnv, RewardConfig
from learning.eval_batch import run_episodes, wilson_ci
from learning.evaluate_ppo import EXACT_REPEATS, evaluate_policy, load_policy_meta
from learning.feasibility_cutin import CutinConfig, CutinRuntime
from learning.feasibility_sandwich import SandwichConfig
from learning.feasibility_train_one import RunParams, train_and_evaluate
from learning.minimax_feasibility import ADMITTED, REJECTED, UNCERTAIN, demonstrated
from learning.plot_ppo_results import (
    collision_location, event_index, footprint, make_all, rank_rows, representatives, select_slice, _is_full_grid,
)
from learning import plot_ppo_results as plots
from learning import animate_ppo_results as anim
from learning import ppo_analysis as A
from learning.ppo_metrics import training_env_kwargs
from learning.ppo_experiment import (
    ADMISSION_P0, EARLY_STOP_RANK, EVALUATION_RANKS, ExperimentConfig, FAMILIES, SELECTION_RANK, TEST_RANK, TRAINING_RANK,
    VALIDATION_RANK, arena_config, create_experiment, read_json, sample_scenarios, write_json,
)
from learning.ppo_metrics import TrainingMetricsCallback, ValidationCallback
from learning.report_ppo_results import build_report, markdown_to_html
from learning.scenario import derive_seed
from learning.trajectory_log import (
    FRICTION_TOLERANCE, Trajectory, TrajectoryRecorder, friction_utilization, load_trajectory, physical_validation, save_trajectory,
)
from model.road.road import Road
from initialization.traffic_init import CAR_LENGTH, CAR_WIDTH
from tests.ppo_fixtures import ScriptedPolicy, build_fixture, synthetic_record


@pytest.fixture(scope="module")
def fixture_exp(tmp_path_factory):
    exp = build_fixture(tmp_path_factory.mktemp("res"))
    make_all(exp)
    build_report(exp, html_export=True)
    return exp


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One really trained (barely) policy, evaluated in both modes, through the real entry points."""
    root = tmp_path_factory.mktemp("real")
    cfg = CutinConfig(mode="robust", seed=3, max_episode_seconds=4.0)
    exp = create_experiment(ExperimentConfig(name="tiny", strata=[["cutin", 1]], n_thetas=1), root, "smoke", scenarios=[])
    out = exp / "training/p"
    out.mkdir()
    train_and_evaluate("cutin", cfg, RunParams(out=str(out / "model"), timesteps=64, n_envs=1, no_early_stop=True, episodes=0, ppo_seed=0,
                                               metrics_path=str(out / "metrics.jsonl"), eval_freq_timesteps=1024, validation_episodes=2), verbose=0)
    ckpt = out / "model.zip"
    robust = read_json(evaluate_policy(ckpt, exp, "robust", test_episodes=4, selection_episodes=2, trajectories_per_outcome=1))
    exact = read_json(evaluate_policy(ckpt, exp, "exact", test_episodes=4, selection_episodes=2))
    return {"exp": exp, "ckpt": ckpt, "robust": robust, "exact": exact, "metrics": out / "metrics.jsonl"}


def _rollout_traj(family="cutin", policy=None, cfg=None) -> Trajectory:
    cfg = cfg or CutinConfig(mode="robust", road_kappa_max=0.01, max_episode_seconds=20.0)
    env = EgoTrafficEnv(arena=FAMILIES[family][1](cfg), mode="distribution", worker_rank=TEST_RANK, rollover_event="physical_v2")
    policy = policy or ScriptedPolicy()
    policy.env = env
    rec = TrajectoryRecorder()
    results = run_episodes(policy, env, 1, observer=rec)
    packed = rec.arrays(env, results[0])
    return Trajectory(packed["arrays"], packed["meta"])


def _forces(fxf, fyf, fzf, fxr=0.0, fyr=0.0, fzr=5000.0, mu=1.0) -> Trajectory:
    arrays = {f"ego__{k}": np.atleast_1d(np.asarray(v, dtype=float)) for k, v in
              dict(F_xf=fxf, F_yf=fyf, F_zf=fzf, F_xr=fxr, F_yr=fyr, F_zr=fzr, t=0.0).items()}
    return Trajectory(arrays, {"mu": mu})


# ── serialization and identification ────────────────────────────────────────
def test_trajectory_round_trip(tmp_path):
    traj = _rollout_traj()
    save_trajectory(tmp_path / "a/t.npz", traj.arrays, traj.meta)
    back = load_trajectory(tmp_path / "a/t.npz")
    assert back.meta == json.loads(json.dumps(traj.meta))
    assert set(back.arrays) == set(traj.arrays)
    assert all(np.array_equal(back.arrays[k], traj.arrays[k], equal_nan=True) for k in traj.arrays)


def test_evaluation_record_round_trip(tmp_path):
    rec = synthetic_record("cutin", 1, CutinConfig().theta(), 7, 10, "abc123456789")
    write_json(tmp_path / "r.json", rec)
    back = read_json(tmp_path / "r.json")
    assert A.scenario_row(back)["k"] == 7 and back["summary"]["n_success"] == 7


@pytest.mark.parametrize("family, side, n_params", [("cutin", 1, 8), ("cutin", -1, 8), ("sandwich", 1, 10), ("sandwich", -1, 10)])
@pytest.mark.parametrize("design", ["lhs", "sobol"])
def test_sampled_rows_rebuild_the_same_scenario(family, side, n_params, design):
    rows = sample_scenarios(ExperimentConfig(name="x", strata=[[family, side]], n_thetas=3, design=design))
    assert len(rows) == 3 and len({r["scenario_id"] for r in rows}) == 3
    for r in rows:
        assert len(r["theta"]) == n_params
        cfg = arena_config(family, r)
        assert cfg.scenario_id() == r["scenario_id"] and cfg.seed == r["scenario_seed"] and cfg.blocker_side == side


def test_sampling_matches_the_existing_sampler():
    from learning.feasibility_sampling import sample_cutin
    expected = sample_cutin(3, 1, 5, method="lhs")[0]
    rows = sample_scenarios(ExperimentConfig(name="x", strata=[["cutin", 1]], n_thetas=3, sampling_seed=5))
    assert [r["theta"] for r in rows] == [c.theta() for c in expected]


def test_grid_design_is_a_validated_product():
    c = ExperimentConfig(name="x", strata=[["cutin", 1]], design="grid", grid_levels={"road_mu": [0.4, 0.8], "cutter_gap_m": [20.0, 40.0, 60.0]}, grid_fixed={"front_gap_m": 50.0})
    rows = sample_scenarios(c)
    assert len(rows) == 6 and all(r["theta"]["front_gap_m"] == 50.0 for r in rows)
    assert select_slice(rows, ("road_mu", "cutter_gap_m"), None)[2] is True


def test_evaluation_mode_changes_realization_mode_not_theta():
    row = sample_scenarios(ExperimentConfig(name="x", strata=[["cutin", 1]], n_thetas=1))[0]
    robust, exact = arena_config("cutin", row, "robust"), arena_config("cutin", row, "exact")
    assert robust.theta() == exact.theta() and robust.mode == "robust" and exact.mode == "exact"
    assert robust.scenario_id() != exact.scenario_id()


def test_results_are_not_silently_overwritten(tmp_path, trained):
    c = ExperimentConfig(name="n", strata=[["cutin", 1]], n_thetas=1)
    create_experiment(c, tmp_path)
    with pytest.raises(SystemExit):
        create_experiment(c, tmp_path)
    with pytest.raises(SystemExit):
        evaluate_policy(trained["ckpt"], trained["exp"], "robust", test_episodes=4, selection_episodes=2, trajectories_per_outcome=1)


def test_manifest_records_what_a_reader_needs(fixture_exp):
    m = read_json(fixture_exp / "manifest.json")
    p = m["provenance"]
    assert p["git_commit"] and p["software"]["stable-baselines3"] and p["dt_s"] == 0.05 and p["vehicle_parameters"]["m"] > 0
    assert m["reward_config"]["success_reward"] == 3.0 and m["admission_p0"] == 0.8 and m["definitions"]["physical_feasibility"]
    assert m["config"]["ppo_seeds"] == [0, 1] and m["seed_streams"]["test"] == TEST_RANK


# ── seeds and independence ──────────────────────────────────────────────────
def test_seed_streams_are_disjoint():
    ranks = (TRAINING_RANK, EARLY_STOP_RANK, *EVALUATION_RANKS)
    assert len(set(ranks)) == len(ranks)
    seeds = [derive_seed(7, r, e) for r in ranks for e in range(200)]
    assert len(set(seeds)) == len(seeds)


def _training_seeds(cfg, worker, resets):
    env = EgoTrafficEnv(arena=CutinRuntime(cfg), **training_env_kwargs(cfg.mode, worker))
    seeds = []
    for _ in range(resets):
        env.reset()
        seeds.append(env.get_realized_scenario()["seed_info"]["scenario_seed"])
    return seeds


def test_robust_training_draws_a_new_realization_every_reset_and_worker():
    cfg = CutinConfig(mode="robust", seed=7)
    first, second = _training_seeds(cfg, 0, 5), _training_seeds(cfg, 1, 5)
    assert len(set(first + second)) == 10


def test_exact_training_keeps_one_realization():
    cfg = CutinConfig(mode="exact", seed=7)
    assert len(set(_training_seeds(cfg, 0, 3) + _training_seeds(cfg, 1, 3))) == 1


def test_training_never_reuses_an_evaluation_realization():
    cfg = CutinConfig(mode="robust", seed=7)
    train = {derive_seed(cfg.seed, TRAINING_RANK + w, e) for w in range(64) for e in range(500)}
    held_out = {derive_seed(cfg.seed, r, e) for r in (0, EARLY_STOP_RANK, *EVALUATION_RANKS) for e in range(500)}
    assert not train & held_out


@pytest.mark.parametrize("mode,env_mode", [("robust", "distribution"), ("exact", "fixed")])
def test_training_entry_point_uses_the_mode_it_records(tmp_path, monkeypatch, mode, env_mode):
    """The workers train_and_evaluate really builds, not the helper in isolation."""
    built = []

    class Recording(DummyVecEnv):
        def __init__(self, fns):
            super().__init__(fns)
            built.append(self)

    monkeypatch.setattr("learning.feasibility_train_one.SubprocVecEnv", Recording)
    cfg = CutinConfig(mode=mode, seed=3, max_episode_seconds=2.0)
    meta = train_and_evaluate("cutin", cfg, RunParams(out=str(tmp_path / "model"), timesteps=64, n_envs=2, no_early_stop=True, episodes=0, ppo_seed=0), verbose=0)
    assert meta["training_realization"]["env_mode"] == env_mode
    assert [e.unwrapped.mode for e in built[0].envs] == [env_mode, env_mode]
    infos = [e.unwrapped.get_realized_scenario()["seed_info"] for e in built[0].envs]
    if mode == "robust":
        assert [i["worker_rank"] for i in infos] == [TRAINING_RANK, TRAINING_RANK + 1]
        assert all(i["episode_index"] > 0 for i in infos)   # workers really moved on to new realizations
    else:
        assert all(i["episode_index"] == 0 for i in infos)


def test_manifest_states_what_training_saw(tmp_path):
    robust = create_experiment(ExperimentConfig(name="r", strata=[["cutin", 1]], n_thetas=1), tmp_path, "smoke", scenarios=[])
    exact = create_experiment(ExperimentConfig(name="e", strata=[["cutin", 1]], n_thetas=1, training_mode="exact"), tmp_path, "smoke", scenarios=[])
    assert read_json(robust / "manifest.json")["training_realization"]["env_mode"] == "distribution"
    assert read_json(exact / "manifest.json")["training_realization"]["env_mode"] == "fixed"


def test_evaluation_uses_only_the_test_stream(trained):
    r = trained["robust"]
    train_seed = derive_seed(r["scenario"]["scenario_seed"], TRAINING_RANK, 0)
    infos = [x["seed_info"] for x in r["rollouts"]]
    assert {i["worker_rank"] for i in infos} == {TEST_RANK} and [i["episode_index"] for i in infos] == [0, 1, 2, 3]
    assert len({i["scenario_seed"] for i in infos}) == 4 and train_seed not in {i["scenario_seed"] for i in infos}
    assert r["protocol"]["selection_rank"] == SELECTION_RANK and r["selection"]["n"] == 2


def test_exact_mode_is_one_outcome_not_repeated_trials(trained):
    e = trained["exact"]
    assert e["independent"] is False and e["n_independent"] == 1 and e["n_distinct_rollouts"] == 1 and e["repeatable"] is True
    assert e["summary"]["n_episodes"] == EXACT_REPEATS and "wilson_95ci" not in e["summary"]
    row = A.scenario_row(e)
    assert (row["n"], row["n_rollouts"], row["wilson_lower"], row["status"]) == (1, 1, None, A.NOT_APPLICABLE)


def test_robust_mode_is_independent(trained):
    r = trained["robust"]
    assert r["independent"] is True and r["n_independent"] == 4 and r["n_distinct_rollouts"] == 4
    assert A.scenario_row(r)["status"] != A.NOT_APPLICABLE


def test_policy_identity_comes_from_metadata(trained):
    p = trained["robust"]["policy"]
    assert p["reward_version"] and p["vehicle_physics_version"] == "vehicle-physics-v2" and p["trained_mode"] == "robust"
    assert p["ppo_hyperparameters"]["gamma"] == 0.999 and len(p["checkpoint_sha256"]) == 64 and p["obs_dim"] == 37


def test_missing_metadata_is_refused(tmp_path):
    (tmp_path / "model.zip").write_bytes(b"x")
    with pytest.raises(SystemExit):
        load_policy_meta(tmp_path / "model.zip")


# ── counting, Wilson, admission, selection ──────────────────────────────────
@pytest.mark.parametrize("k, n", [(0, 10), (10, 10), (7, 20), (87, 100)])
def test_success_counts_and_wilson_use_the_existing_interval(k, n):
    r = A.scenario_row(synthetic_record("cutin", 1, CutinConfig().theta(), k, n, "abc123456789"))
    assert (r["k"], r["n"]) == (k, n) and r["p_hat"] == k / n and (r["wilson_lower"], r["wilson_upper"]) == wilson_ci(k, n)
    assert r["n_collision"] + r["n_timeout"] == n - k


def test_wilson_known_value():
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)


@pytest.mark.parametrize("k, n, expected", [(16, 16, ADMITTED), (15, 15, UNCERTAIN), (100, 100, ADMITTED), (80, 100, UNCERTAIN),
                                            (72, 100, REJECTED), (0, 5, REJECTED), (4, 5, UNCERTAIN)])
def test_admission_status(k, n, expected):
    assert A.admission_status(k, n) == expected
    assert (A.admission_status(k, n) == ADMITTED) == demonstrated(k, n, ADMISSION_P0)


def test_admission_is_undefined_without_independent_data():
    assert A.admission_status(1, 1, independent=False) == A.NOT_APPLICABLE and A.admission_status(0, 0) == A.NOT_APPLICABLE


def test_seed_selection_ignores_test_results():
    theta = CutinConfig().theta()
    lucky = synthetic_record("cutin", 1, theta, 100, 100, "aaaaaaaa1111", selection=(1, 10))
    chosen = synthetic_record("cutin", 1, theta, 40, 100, "aaaaaaaa2222", selection=(9, 10))
    chosen["policy"]["ppo_seed"] = 1
    rows = A.scenario_table([lucky, chosen])
    assert len(rows) == 1 and rows[0]["policy_id"] == "aaaaaaaa2222" and rows[0]["k"] == 40 and rows[0]["n_seeds_considered"] == 2


def test_rate_tables_keep_scenarios_and_rollouts_separate():
    recs = [synthetic_record("cutin", 1, {**CutinConfig().theta(), "front_gap_m": g}, k, 100, f"{g:08.0f}0000") for g, k in ((30, 100), (40, 0))]
    a = A.table_a(A.scenario_table(recs), [{"family": "cutin", "blocker_side": 1}] * 5)[0]
    assert a["scenarios_sampled"] == 5 and a["scenarios_evaluated"] == 2 and a["admitted"] == 1 and a["not_admitted"] == 1 and a["rollouts"] == 200
    assert a["rollout_collision_rate"] == pytest.approx(50 / 200) and a["mean_success_time_s"] == 20.0


def test_exact_rows_take_no_admission_counts():
    rec = synthetic_record("cutin", 1, CutinConfig().theta(), 1, 1, "abc123456789", eval_mode="exact")
    a = A.table_a(A.scenario_table([rec]), [])[0]
    assert (a["admitted"], a["uncertain"], a["not_admitted"]) == (None, None, None)


def test_rollout_outcomes_do_not_depend_on_the_reward():
    cfg = CutinConfig(mode="robust", road_kappa_max=0.0, max_episode_seconds=5.0)
    runs = []
    for reward in (None, RewardConfig(success_reward=-9.0, collision_penalty=-5.0, timeout_penalty=50.0, time_penalty_per_step=1.0)):
        env = EgoTrafficEnv(arena=CutinRuntime(cfg), mode="distribution", worker_rank=TEST_RANK, reward_config=reward)
        policy = ScriptedPolicy()
        policy.env = env
        runs.append(run_episodes(policy, env, 2))
    keys = ("outcome", "success", "failure_reason", "progress_m", "elapsed_s", "episode_progress")
    assert [{k: r[k] for k in keys} for r in runs[0]] == [{k: r[k] for k in keys} for r in runs[1]]
    assert [r["return"] for r in runs[0]] != [r["return"] for r in runs[1]]


# ── tire utilization and physical validation ────────────────────────────────
def test_eta_is_combined_force_over_limit():
    u = friction_utilization(_forces([3000.0], [4000.0], [10000.0], mu=0.5))
    assert u["front"][0] == pytest.approx(1.0) and u["rear"][0] == 0.0
    assert friction_utilization(_forces([3000.0], [0.0], [10000.0], mu=1.0))["front"][0] == pytest.approx(0.3)


def test_zero_normal_force_is_handled():
    u = friction_utilization(_forces([0.0, 100.0], [0.0, 0.0], [0.0, 10000.0]))
    assert math.isnan(u["front"][0]) and u["front"][1] == pytest.approx(0.01)
    assert physical_validation(_forces([0.0], [0.0], [0.0]))["friction_violations"] == 0


def test_force_on_an_unloaded_axle_is_a_violation():
    assert physical_validation(_forces([50.0], [0.0], [0.0]))["status"] == "violation"


def test_violation_is_counted_and_sized():
    v = physical_validation(_forces([6000.0, 1000.0], [0.0, 0.0], [5000.0, 5000.0]))
    assert v["status"] == "violation" and v["friction_violations"] == 1 and v["max_violation"] == pytest.approx(0.2)
    assert v["peak_eta_front"] == pytest.approx(1.2)


def test_limit_within_tolerance_is_valid():
    v = physical_validation(_forces([5000.0 * (1 + FRICTION_TOLERANCE / 2)], [0.0], [5000.0]))
    assert v["status"] == "valid" and v["friction_violations"] == 0


def test_saturation_is_not_a_validity_failure():
    """Hard braking at low friction saturates the tires, and is physically valid."""
    traj = _rollout_traj(policy=ScriptedPolicy(accel=-8.0), cfg=CutinConfig(mode="robust", road_mu=0.4, road_kappa_max=0.0, max_episode_seconds=4.0))
    v = physical_validation(traj)
    assert v["status"] == "valid" and v["peak_eta_front"] == pytest.approx(1.0, abs=1e-6) and v["peak_long_decel"] == pytest.approx(0.4 * 9.81, abs=0.05)
    assert v["stability_loss"] is False


def test_physical_diagnostics_are_simulator_values():
    traj = _rollout_traj()
    v = physical_validation(traj)
    assert v["min_clearance_m"] == pytest.approx(np.nanmin(traj.ego("clearance_m")))
    assert v["duration_s"] == pytest.approx(traj.ego("t")[-1]) and v["peak_abs_steering_rad"] == pytest.approx(np.nanmax(np.abs(traj.ego("delta_applied"))))


def test_missing_diagnostics_are_unavailable_not_invented():
    traj = _rollout_traj()
    slim = Trajectory({k: v for k, v in traj.arrays.items() if not k.startswith("ego__F_")}, traj.meta)
    v = physical_validation(slim)
    assert friction_utilization(slim) is None and v["status"] == "unavailable" and v["peak_eta_front"] is None and v["friction_violations"] is None
    rec = synthetic_record("cutin", 1, CutinConfig().theta(), 5, 10, "abc123456789")
    for r in rec["rollouts"]:
        r.pop("physical")
    row = A.scenario_row(rec)
    assert row["physical_status"] == "unavailable" and row["peak_tire_utilization"] is None


def test_first_row_has_no_applied_control():
    traj = _rollout_traj()
    assert traj.ego("t")[0] == 0.0 and np.isnan(traj.ego("accel_applied")[0]) and np.isnan(traj.ego("F_xf")[0])
    assert np.isfinite(traj.ego("F_xf")[1:]).all() and traj.surr("x").shape == (len(traj), 6)


# ── progress, completion, Frenet to world ───────────────────────────────────
def test_progress_and_completion_time_from_a_real_success():
    traj_policy = ScriptedPolicy(0.0, 3.0, 225.0)
    cfg = SandwichConfig(blocker_side=1, mode="robust", road_mu=0.9)
    env = EgoTrafficEnv(arena=FAMILIES["sandwich"][1](cfg), mode="distribution", worker_rank=TEST_RANK, rollover_event="physical_v2")
    traj_policy.env = env
    r = run_episodes(traj_policy, env, 1)[0]
    assert r["success"] and r["episode_progress"] == 1.0 and r["elapsed_s"] == pytest.approx(r["steps"] * 0.05)


def test_recorded_pose_is_the_road_transform_on_a_curve():
    traj = _rollout_traj()
    road = plots.build_road(traj.meta)
    assert road.kappa_max > 0
    for i in range(0, len(traj), 20):
        x, y, h = road.frenet_to_global(traj.ego("s")[i], traj.ego("e_y_backbone")[i], traj.ego("e_psi")[i])
        assert (x, y, h) == pytest.approx((traj.ego("x")[i], traj.ego("y")[i], traj.ego("heading")[i]), abs=1e-9)


def test_lateral_positions_are_recorded_in_the_backbone_frame():
    """The arena's ego lane has offset 0, so only a side lane can tell the two frames apart."""
    env = EgoTrafficEnv(arena=CutinRuntime(CutinConfig(mode="robust", road_kappa_max=0.01)), mode="fixed")
    env.reset()
    env.ego_car.state.lane = 0
    env._update_ego_pose()
    rec = TrajectoryRecorder()
    rec.start(env)
    assert rec.rows["e_y"][0] == 0.0 and rec.rows["e_y_backbone"][0] == -env.road.lanes[0].offset != 0.0


def test_recorded_traffic_poses_are_the_road_transform():
    traj = _rollout_traj()
    road = plots.build_road(traj.meta)
    assert np.ptp(traj.surr("e_y_backbone")) > 2.0   # traffic in several lanes
    for i in (0, len(traj) // 2, len(traj) - 1):
        for j in range(traj.surr("x").shape[1]):
            pose = road.frenet_to_global(traj.surr("s")[i, j], traj.surr("e_y_backbone")[i, j], traj.surr("e_psi")[i, j])
            assert pose[:2] == pytest.approx((traj.surr("x")[i, j], traj.surr("y")[i, j]), abs=0.05)


def test_world_coordinates_are_not_frenet_on_a_curve():
    road = Road(s_max=500, kappa_max=0.02, L_clothoid=60, mu=1.0, lane_num=3)
    s = road.L_enter + road.L_clothoid + road.L_curve / 2
    x0, y0, psi = road.frenet_to_global(s, 0.0)
    x1, y1, _ = road.frenet_to_global(s, 2.0)
    assert abs(x0 - s) > 5.0 and abs(y0) > 5.0
    assert math.hypot(x1 - x0, y1 - y0) == pytest.approx(2.0, abs=1e-9)
    assert (x1 - x0) * math.cos(psi) + (y1 - y0) * math.sin(psi) == pytest.approx(0.0, abs=1e-9)
    assert (x1 - x0) * (-math.sin(psi)) + (y1 - y0) * math.cos(psi) == pytest.approx(-2.0)   # +e_y is to the right of travel


# ── footprints, collisions, events ──────────────────────────────────────────
def test_footprint_has_vehicle_dimensions_and_heading():
    h = 0.7
    c = footprint(10.0, -3.0, h)
    assert c.mean(axis=0) == pytest.approx([10.0, -3.0])
    assert np.linalg.norm(c[0] - c[3]) == pytest.approx(CAR_LENGTH) and np.linalg.norm(c[0] - c[1]) == pytest.approx(CAR_WIDTH)
    long_axis = c[0] - c[3]
    assert math.atan2(long_axis[1], long_axis[0]) == pytest.approx(h)


def test_collision_location_comes_from_the_simulator_flag():
    traj = _rollout_traj(cfg=CutinConfig(mode="robust", road_mu=0.4, cutter_gap_m=10.0, s_cutin_m=120.0))
    assert traj.meta["failure_reason"] == "collision"
    hit = collision_location(traj)
    assert (hit["x"], hit["y"]) == (traj.ego("x")[-1], traj.ego("y")[-1])
    i = int(np.nonzero(traj.surr("car_id") == hit["car_id"])[0][0])
    assert math.hypot(traj.surr("x")[-1, i] - hit["x"], traj.surr("y")[-1, i] - hit["y"]) < CAR_LENGTH * 1.5
    assert event_index(traj) is not None and traj.ego("s")[event_index(traj)] >= traj.meta["arena"]["event_trigger_ego_s"] - 2.0


def test_no_collision_marker_without_a_collision():
    assert collision_location(_rollout_traj(cfg=CutinConfig(mode="robust", road_kappa_max=0.0, max_episode_seconds=1.0))) is None


# ── maps ────────────────────────────────────────────────────────────────────
def _map_rows(thetas):
    return [{"theta": t} for t in thetas]


def test_slice_is_only_claimed_when_every_other_parameter_is_held():
    base = {"a": 1.0, "b": 2.0, "c": 3.0}
    rows, text, is_slice = select_slice(_map_rows([base, {**base, "a": 2.0}]), ("a", "b"), None)
    assert is_slice and text == "Slice: c=3"
    rows, text, is_slice = select_slice(_map_rows([base, {**base, "c": 4.0}]), ("a", "b"), None)
    assert not is_slice and "Projection" in text and "c" in text


def test_fix_filters_to_the_requested_slice():
    base = {"a": 1.0, "b": 2.0, "c": 3.0}
    rows, _, is_slice = select_slice(_map_rows([base, {**base, "c": 4.0}]), ("a", "b"), {"c": 4.0})
    assert len(rows) == 1 and is_slice
    assert select_slice(_map_rows([base]), ("a", "b"), {"c": 9.0})[0] == []


def test_grid_detection_needs_a_complete_product():
    assert _is_full_grid([1, 1, 2, 2], [3, 4, 3, 4]) and not _is_full_grid([1, 1, 2], [3, 4, 3]) and not _is_full_grid([1, 2, 1], [1, 2, 3])


# ── comparisons ─────────────────────────────────────────────────────────────
def _pair(pid: str, exact: int, robust_k: int, theta=None):
    theta = theta or CutinConfig().theta()
    return [synthetic_record("cutin", 1, theta, exact, 1, pid, eval_mode="exact"), synthetic_record("cutin", 1, theta, robust_k, 100, pid)]


def test_robustness_pairs_the_same_policy():
    recs = _pair("aaaaaaaa0000", 1, 90) + _pair("bbbbbbbb0000", 0, 30)
    t = A.robustness_table(A.scenario_table(recs))[0]
    assert t["policies"] == 2 and t["exact_success"] == 0.5 and t["robust_success"] == pytest.approx(0.6)
    assert t["difference"] == pytest.approx(0.1) and t["trained_mode"] == "robust" and t["robust_ci"][0] < 0.6 < t["robust_ci"][1]


def test_robustness_needs_both_modes():
    assert A.robustness_table(A.scenario_table([synthetic_record("cutin", 1, CutinConfig().theta(), 5, 10, "aaaaaaaa0000")])) == []


def test_single_policy_has_no_interval():
    assert A.robustness_table(A.scenario_table(_pair("aaaaaaaa0000", 1, 90)))[0]["robust_ci"] is None


def _rewards(reward_b="other-v1", physics_b="vehicle-physics-v2", shared=True):
    theta = CutinConfig().theta()
    a = synthetic_record("cutin", 1, theta, 80, 100, "aaaaaaaa0000")
    b = synthetic_record("cutin", 1, theta, 60, 100, "aaaaaaaa0000" if shared else "bbbbbbbb0000", reward=reward_b, physics=physics_b)
    return [a, b]


def test_reward_comparison_is_equivalent_when_only_the_reward_differs():
    recs = _rewards()
    table, issues = A.table_c(recs, A.scenario_table(recs))
    assert issues == [] and len(table) == 2 and {t["comparison"] for t in table} == {"equivalent"}
    assert sorted(t["heldout_success_rate"] for t in table) == [0.6, 0.8]


def test_different_dynamics_make_the_comparison_non_equivalent():
    recs = _rewards(physics_b="vehicle-physics-v1")
    table, issues = A.table_c(recs, A.scenario_table(recs))
    assert any("vehicle dynamics" in i for i in issues) and {t["comparison"] for t in table} == {"non-equivalent"}


def test_unrecorded_reward_is_never_guessed():
    recs = _rewards(reward_b=None)
    recs[1]["policy"]["checkpoint"] = "learning/ppo_goal-progress-v1.zip"   # a tempting file name
    assert A.reward_key(recs[1]["policy"]) == "unrecorded"
    assert any("reward configuration not recorded" in i for i in A.table_c(recs, A.scenario_table(recs))[1])


def test_different_scenario_sets_are_flagged():
    recs = _rewards(shared=False)
    assert any("scenario sets differ" in i for i in A.table_c(recs, A.scenario_table(recs))[1])


def test_one_reward_gives_no_table():
    rec = synthetic_record("cutin", 1, CutinConfig().theta(), 80, 100, "aaaaaaaa0000")
    assert A.table_c([rec], A.scenario_table([rec]))[0] == []


# ── training metrics ────────────────────────────────────────────────────────
def test_metrics_file_separates_training_from_validation(trained):
    rows = [json.loads(line) for line in open(trained["metrics"])]
    train, val = [r for r in rows if r["kind"] == "train"], [r for r in rows if r["kind"] == "validation"]
    assert train and val and val[0]["deterministic"] is True
    assert {"approx_kl", "explained_variance", "entropy_loss", "learning_rate", "value_loss"} <= set(train[-1])
    assert train[-1]["approx_kl"] is not None and train[-1]["n_episodes"] >= 0


def _small_ppo(callback=None):
    def make():
        return EgoTrafficEnv(arena=CutinRuntime(CutinConfig(mode="exact", max_episode_seconds=2.0)), mode="fixed", rollover_event="physical_v2")
    model = PPO("MlpPolicy", DummyVecEnv([make]), n_steps=64, batch_size=32, n_epochs=1, device="cpu", seed=0)
    model.learn(128, callback=callback(make) if callback else None)
    return model


def test_logging_does_not_change_training(tmp_path):
    from stable_baselines3.common.callbacks import CallbackList
    base = _small_ppo()
    logged = _small_ppo(lambda make: CallbackList([TrainingMetricsCallback(str(tmp_path / "m.jsonl")), ValidationCallback(make, str(tmp_path / "m.jsonl"), 64, 1)]))
    for a, b in zip(base.policy.parameters(), logged.policy.parameters()):
        assert (a == b).all()
    assert any(json.loads(line)["kind"] == "validation" for line in open(tmp_path / "m.jsonl"))


def test_validation_episodes_repeat_exactly():
    envs = []

    def make():
        envs.append(EgoTrafficEnv(arena=CutinRuntime(CutinConfig(mode="robust", max_episode_seconds=0.2)), mode="distribution", worker_rank=VALIDATION_RANK))
        return envs[-1]

    seeds = []
    for _ in range(2):
        env = make()
        env.reset()
        seeds.append(env.get_realized_scenario()["seed_info"]["scenario_seed"])
    assert seeds[0] == seeds[1]


def test_curves_do_not_extrapolate_past_a_run():
    runs = [{"train": [{"timesteps": t, "x": 1.0} for t in (0, 10)]}, {"train": [{"timesteps": t, "x": 3.0} for t in (0, 20)]}]
    c = A.aligned_curves(runs, "train", "x", n_grid=21)
    assert c["n_runs"] == 2 and c["count"][0] == 2 and c["count"][-1] == 1
    assert c["mean"][0] == 2.0 and c["mean"][-1] == 3.0 and c["std"][-1] == 0.0


def test_unrecorded_metric_stays_missing():
    assert A.aligned_curves([{"train": [{"timesteps": 1, "x": None}]}], "train", "x") is None


# ── figures, animation, reports ─────────────────────────────────────────────
def test_figures_are_made_from_saved_records(fixture_exp):
    index = read_json(fixture_exp / "figures/index.json")
    names = {f["name"] for f in index["figures"]}
    for expected in ("learning_validation", "learning_training", "learning_diagnostics", "wilson_intervals_p_hat", "robustness", "trajectory_cutin_success",
                     "trajectory_cutin_failure", "trajectory_sandwich_success", "trajectory_sandwich_failure", "physics_low_friction", "map_empirical_cutin_pos1_road_mu__cutter_gap_m"):
        assert expected in names
    for name in names:
        assert (fixture_exp / f"figures/{name}.pdf").stat().st_size > 0 and (fixture_exp / f"figures/{name}.png").stat().st_size > 0
    assert index["data_class"] == "synthetic_test_fixture"


def test_missing_trajectory_category_is_skipped_not_filled(fixture_exp, tmp_path):
    records = [r for r in A.load_records(fixture_exp) if r["scenario"]["family"] == "cutin"]
    found, missing = representatives(records)
    assert "sandwich_success" in missing and "sandwich_success" not in found


def test_exact_and_single_rollouts_stay_out_of_maps_and_wilson(fixture_exp):
    rows = A.scenario_table(A.load_records(fixture_exp))
    assert any(not r["independent"] for r in rows)
    assert all(r["status"] == A.NOT_APPLICABLE for r in rows if not r["independent"])


def test_ranking_options():
    rows = [{"p_hat": 0.5, "wilson_lower": 0.1, "stratum": "b", "mean_peak_tire_utilization": 0.9}, {"p_hat": 0.6, "wilson_lower": 0.05, "stratum": "a", "mean_peak_tire_utilization": None}]
    assert rank_rows(rows, "p_hat")[0]["p_hat"] == 0.5 and rank_rows(rows, "lower")[0]["p_hat"] == 0.6
    assert rank_rows(rows, "family")[0]["stratum"] == "a" and rank_rows(rows, "difficulty")[0]["p_hat"] == 0.5


def test_real_results_plot_with_zero_successes(trained):
    assert make_all(trained["exp"]).made   # k = 0 made errorbar fail before its bounds were clipped


def test_animation_from_saved_rollout(fixture_exp, monkeypatch):
    records = A.load_records(fixture_exp)
    path, rec, r = anim.find_rollout(records, "r0cccccc", None, "robust", 0)
    traj = load_trajectory(fixture_exp / path)
    traj.arrays = {k: v[:8] if k.startswith(("ego__", "surr__x", "surr__y", "surr__heading", "surr__crashed")) else v for k, v in traj.arrays.items()}
    monkeypatch.setattr(anim.FFMpegWriter, "isAvailable", classmethod(lambda cls: False))
    out = anim.animate(traj, fixture_exp / "animations/t", "test", "notice")
    assert out.suffix == ".gif" and out.stat().st_size > 0


def test_animation_rejects_unknown_rollouts(fixture_exp):
    with pytest.raises(SystemExit):
        anim.find_rollout(A.load_records(fixture_exp), "nope", None, "robust", 0)


def test_evaluation_does_not_depend_on_animation_or_plotting():
    code = "import sys, learning.evaluate_ppo; print('learning.animate_ppo_results' in sys.modules, 'learning.plot_ppo_results' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "False False"


def test_report_marks_demonstration_data_and_links_figures(fixture_exp):
    text = (fixture_exp / "reports/report.md").read_text()
    assert text.startswith("> **DEMONSTRATION (synthetic_test_fixture).**")
    for i, heading in enumerate(("Experiment identification", "PPO training summary", "Evaluation methodology", "Success and failure", "Feasibility admission",
                                 "Robustness", "Physical-validity", "Representative maneuvers", "Reward comparison", "Limitations"), start=1):
        assert f"## {i}. {heading}" in text
    links = [l.split("](")[1].rstrip(")") for l in text.splitlines() if l.startswith("![")]
    assert links and all((fixture_exp / "reports" / l).exists() for l in links)
    assert "physically impossible" in text and "Surrogate prediction" in text and "Statistical admission" in text


def test_real_experiment_report_has_no_demonstration_banner_only_when_marked(tmp_path):
    exp = create_experiment(ExperimentConfig(name="e", strata=[["cutin", 1]], n_thetas=1), tmp_path)
    text = build_report(exp).read_text()
    assert "DEMONSTRATION" not in text and "No figures have been generated yet" in text


def test_tables_are_exported(fixture_exp):
    t = fixture_exp / "reports/tables"
    with open(t / "table_a_feasibility.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows and {"stratum", "admitted", "rollouts", "mean_success_time_s"} <= set(rows[0])
    assert (t / "table_a_feasibility.tex").read_text().startswith(r"\begin{tabular}")
    assert r"\_" in (t / "table_b_physical_validity.tex").read_text()
    assert (t / "table_c_reward_comparison.csv").exists()


def test_html_export_is_self_consistent(fixture_exp):
    page = (fixture_exp / "reports/report.html").read_text()
    assert "<table>" in page and "<img src=" in page and "<blockquote>" in page and "<script" not in page
    assert "&lt;b&gt;" in markdown_to_html("a <b> c", "t")
