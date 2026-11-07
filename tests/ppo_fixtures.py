"""
Small deterministic experiment for the reporting tests and demonstrations. Marked data_class =
"synthetic_test_fixture", so every figure and report built from it says so. Scenario counts, k/N values and
training curves are invented; trajectories are real simulator rollouts of scripted (non-PPO) controllers.

    python -m tests.ppo_fixtures results/ppo_feasibility     # builds demo_fixture there, with figures and report
"""

import math
import sys
from pathlib import Path

import numpy as np

from learning.env import ACCEL_MAX, ACCEL_MIN, DELTA_MAX, EgoTrafficEnv
from learning.eval_batch import run_episodes, summarize_seed
from learning.evaluate_ppo import EVALUATION_SCHEMA, outcome_class
from learning.feasibility_cutin import CutinConfig
from learning.feasibility_sandwich import SandwichConfig
from learning.ppo_experiment import (
    ExperimentConfig, FAMILIES, TEST_RANK, create_experiment, provenance, run_path, write_json, write_jsonl,
)
from learning.trajectory_log import TrajectoryRecorder


class ScriptedPolicy:
    """Stands in for a PPO model in run_episodes: a constant speed request and a proportional lane follower that
    moves to target_e_y (road backbone frame, SAE +right) once ego passes change_at_s. Not a trained policy."""

    def __init__(self, accel: float = 0.0, target_e_y: float = 0.0, change_at_s: float = math.inf):
        self.accel, self.target_e_y, self.change_at_s = accel, target_e_y, change_at_s
        self.env = None

    def predict(self, obs, deterministic=True):
        st = self.env.ego_car.state
        target = self.target_e_y if st.s >= self.change_at_s else 0.0
        delta = float(np.clip(-0.06 * (self.env._ego_backbone_e_y() - target) - 1.0 * st.e_psi, -DELTA_MAX, DELTA_MAX))
        return np.array([2.0 * (self.accel - ACCEL_MIN) / (ACCEL_MAX - ACCEL_MIN) - 1.0, delta / DELTA_MAX], dtype=np.float32), None


def _policy(pid: str, trained: str = "robust", reward: str | None = "goal-progress-v1", physics: str | None = "vehicle-physics-v2") -> dict:
    return {"policy_id": pid, "checkpoint": f"fixture/{pid}.zip", "checkpoint_sha256": pid, "evaluated_checkpoint": "best", "ppo_seed": 0,
            "trained_mode": trained, "training_scenario_id": pid[:8], "training_scenario_seed": 0, "timesteps_actual": 1000,
            "reward_version": reward, "reward_config": {"success_reward": 3.0} if reward else None, "ppo_gamma": 0.999, "ppo_hyperparameters": {},
            "vehicle_physics_version": physics, "rollover_event_model": "physical_v2", "obs_dim": 37, "action_dim": 2, "dt_s": 0.05}


def synthetic_record(family: str, side: int, theta: dict, k: int, n: int, pid: str, *, eval_mode: str = "robust", selection: tuple = (5, 10),
                     **policy_kw) -> dict:
    """A record with invented outcomes, for testing the analysis layer: never run through a simulator."""
    independent = eval_mode == "robust"
    outcomes = [("finished", True, None)] * k + [("safety_failure", False, "collision")] * ((n - k + 1) // 2) + [("timeout", False, None)] * ((n - k) // 2)
    rollouts = [{"episode": i, "outcome": o, "success": s, "failure_reason": fr, "timeout": o == "timeout", "steps": 400, "return": 3.8 if s else -2.0,
                 "progress_m": 400.0 if s else 200.0, "episode_progress": 1.0 if s else 0.5, "elapsed_s": 20.0 if s else 12.0,
                 "physical": {"status": "valid", "peak_eta_front": 0.5 + 0.4 * i / max(n, 1), "peak_eta_rear": 0.4, "friction_violations": 0, "max_violation": 0.0,
                              "peak_abs_lat_accel": 4.0, "peak_long_decel": 6.0, "min_clearance_m": 1.5, "min_road_edge_clearance_m": 0.5, "stability_loss": False}}
                for i, (o, s, fr) in enumerate(outcomes)]
    policy = _policy(pid, **policy_kw)
    return {"schema": EVALUATION_SCHEMA, "policy": policy, "provenance": {**provenance(), "vehicle_physics_version": "vehicle-physics-v2"},
            "protocol": {"eval_mode": eval_mode, "deterministic": True, "test_rank": TEST_RANK, "test_episodes": n, "rollover_event": "physical_v2"},
            "scenario": {"family": family, "blocker_side": side, "scenario_id": f"{pid[:8]}-{eval_mode}", "training_scenario_id": pid[:8], "theta": theta},
            "independent": independent, "n_independent": n if independent else 1, "summary": summarize_seed(rollouts, "distribution" if independent else "fixed", 1.0),
            "selection": {"k": selection[0], "n": selection[1]}, "rollouts": rollouts}


def _write_record(exp: Path, rec: dict) -> None:
    sc = rec["scenario"]
    rel = run_path(sc["family"], sc["blocker_side"], rec["policy"]["training_scenario_id"], rec["policy"]["ppo_seed"])
    write_json(exp / "evaluations" / rel / f"{rec['protocol']['eval_mode']}_{rec['policy']['policy_id']}.json", rec)


def _real_rollout_record(exp: Path, family: str, cfg, policy: ScriptedPolicy, pid: str) -> None:
    """A record from a real simulator rollout of a scripted controller, with its trajectory saved."""
    env = EgoTrafficEnv(arena=FAMILIES[family][1](cfg), mode="distribution", worker_rank=TEST_RANK, rollover_event="physical_v2")
    policy.env = env
    rel = run_path(family, cfg.blocker_side, pid[:8], 0)

    def keep(result):
        result["trajectory"] = str(Path("trajectories") / rel / "robust" / f"ep_{result['episode']:05d}.npz")
        return exp / result["trajectory"]

    rollouts = run_episodes(policy, env, 1, observer=TrajectoryRecorder(keep))
    rec = synthetic_record(family, cfg.blocker_side, cfg.theta(), 0, 0, pid)
    # A single scripted rollout: shown in trajectory plots, never counted as a statistical estimate.
    rec.update(rollouts=rollouts, summary=summarize_seed(rollouts, "fixed", 1.0), n_independent=1, independent=False)
    _write_record(exp, rec)
    return outcome_class(rollouts[0])


def _metrics(exp: Path, family: str, side: int, sid: str, seed: int, shift: float) -> None:
    """Invented learning curves that end early for one seed, as early stopping does."""
    rel = run_path(family, side, sid, seed)
    end = 10 - seed
    train = [{"kind": "train", "timesteps": 2048 * i * 4, "n_episodes": 8, "ep_return_mean": -4 + 0.7 * i + shift, "ep_length_mean": 200 + 20 * i, "success_rate": 0.08 * i,
              "collision_rate": 0.8 - 0.07 * i, "off_road_rate": 0.1, "rollover_rate": 0.0, "timeout_rate": 0.02 * i, "approx_kl": 0.02 / i, "explained_variance": 0.08 * i,
              "entropy_loss": -2.8 + 0.1 * i, "value_loss": 3.0 / i, "policy_gradient_loss": -0.01, "learning_rate": 3e-4, "loss": 1.0} for i in range(1, end + 1)]
    val = [{"kind": "validation", "timesteps": 16384 * i, "n_episodes": 20, "deterministic": True, "success_rate": min(1.0, 0.1 * i + shift / 20), "collision_rate": max(0.0, 0.7 - 0.08 * i),
            "off_road_rate": 0.05, "rollover_rate": 0.0, "timeout_rate": 0.05, "ep_return_mean": -3 + 0.6 * i, "ep_length_mean": 220 + 15 * i} for i in range(1, end // 2 + 1)]
    write_jsonl(exp / "training" / rel / "metrics.jsonl", train + val)
    write_json(exp / "training" / rel / "model.meta.json", {"scenario_family": family, "blocker_side": side, "scenario_id": sid, "ppo_seed": seed,
                                                            "reward_version": "goal-progress-v1", "reward_config": {"success_reward": 3.0}})


def build_fixture(root: Path, name: str = "demo_fixture") -> Path:
    scenarios = []
    for family, cfg_cls in (("cutin", CutinConfig), ("sandwich", SandwichConfig)):
        cfg = cfg_cls(blocker_side=1, mode="robust")
        scenarios += [{"family": family, "blocker_side": 1, "mode": "robust", "scenario_id": f"{family}{i}", "scenario_seed": i, "theta": cfg.theta(),
                       "goal_distance_m": 400.0, "max_episode_seconds": 35.0, "min_progress_m": 100.0} for i in range(30)]
    exp = create_experiment(ExperimentConfig(name=name, n_thetas=30, ppo_seeds=[0, 1], eval_modes=["robust", "exact"]), root, "synthetic_test_fixture", scenarios=scenarios)
    rng = np.random.default_rng(0)
    for i, mu in enumerate(np.linspace(0.4, 1.0, 5)):            # a complete 5 x 4 grid: a true slice
        for j, gap in enumerate((15.0, 30.0, 45.0, 60.0)):
            k = int(np.clip(round(100 * (0.25 + 0.75 * (i / 4) * (j + 1) / 4)), 0, 100))
            theta = {**CutinConfig().theta(), "road_mu": float(mu), "cutter_gap_m": gap}
            pid = f"c{i}{j}aaaaaaaaaa"
            _write_record(exp, synthetic_record("cutin", 1, theta, k, 100, pid))
            _write_record(exp, synthetic_record("cutin", 1, theta, 1 if k > 50 else 0, 1, pid, eval_mode="exact"))
    for i in range(12):                                           # irregular: every parameter varies
        bounds = FAMILIES["sandwich"][2]
        theta = {n: float(rng.uniform(lo, hi)) for n, (lo, hi) in bounds.items()}
        pid = f"s{i:02d}bbbbbbbbbb"
        _write_record(exp, synthetic_record("sandwich", 1, theta, int(rng.integers(0, 101)), 100, pid))
    for seed, shift in ((0, 0.0), (1, 1.0)):
        _metrics(exp, "cutin", 1, "metrics0", seed, shift)
        _metrics(exp, "sandwich", 1, "metrics1", seed, shift * 0.5)
    _real_rollout_record(exp, "cutin", CutinConfig(blocker_side=1, mode="robust", road_mu=0.9), ScriptedPolicy(), "r0ccccccccccc")
    _real_rollout_record(exp, "cutin", CutinConfig(blocker_side=1, mode="robust", road_mu=0.4, cutter_gap_m=10.0, s_cutin_m=120.0), ScriptedPolicy(), "r1ddddddddddd")
    _real_rollout_record(exp, "sandwich", SandwichConfig(blocker_side=1, mode="robust", road_mu=0.9, escape_gap_length_m=40.0, escape_gap_center_m=0.0, front_gap_m=20.0), ScriptedPolicy(0.0, 1.5, 120.0), "r2eeeeeeeeeee")
    _real_rollout_record(exp, "sandwich", SandwichConfig(blocker_side=-1, mode="robust", road_mu=0.5), ScriptedPolicy(), "r3fffffffffff")
    return exp


if __name__ == "__main__":
    from learning.plot_ppo_results import make_all
    from learning.report_ppo_results import build_report
    exp = build_fixture(Path(sys.argv[1]))
    figures = make_all(exp)
    print(build_report(exp, html_export=True), len(figures.made), "figures;", "skipped:", [s["name"] for s in figures.skipped])
