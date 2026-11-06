"""
Independent evaluation of saved PPO policies (docs/description.md §23). Outcomes come from
learning.eval_batch.run_episodes and EgoTrafficEnv's info, so there is one rollout loop and one definition of
success. The record is a saved JSON file; figures, tables and reports are built from these without rerunning.

Protocol for one (policy, evaluation mode):
  selection stream   SELECTION_RANK. Used only to choose among PPO seeds of a scenario.
  test stream        TEST_RANK. The only data behind reported rates and admission. Disjoint from training,
                     early stopping, validation and selection (learning.ppo_experiment).

exact mode has no perturbation, so a deterministic policy repeats the same rollout. It runs EXACT_REPEATS times
only to confirm that, and n_independent = 1: it is one deterministic outcome, carries no confidence interval
and takes no statistical admission. Robust rollouts, and stochastic-policy rollouts, are independent draws.

Usage:
    python -m learning.evaluate_ppo --experiment results/ppo_feasibility/<name>
    python -m learning.evaluate_ppo --experiment results/ppo_feasibility/<name> --checkpoint path/to/model.zip
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from learning.env import DT, EgoTrafficEnv
from learning.eval_batch import run_episodes, summarize_seed
from learning.ppo_experiment import (
    ExperimentConfig, FAMILIES, SELECTION_RANK, TEST_RANK, arena_config, create_experiment, file_sha256, json_hash,
    load_config, provenance, read_json, run_path, write_json,
)
from learning.trajectory_log import TrajectoryRecorder

EVALUATION_SCHEMA = "ppo-evaluation-v1"
EXACT_REPEATS = 3


def outcome_class(result: dict) -> str:
    return "success" if result["success"] else (result["failure_reason"] or "timeout")


def _clean(obj):
    """Plain JSON types: info values can be numpy scalars."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj.item() if isinstance(obj, np.generic) else obj


def load_policy_meta(checkpoint) -> dict:
    meta_path = Path(str(checkpoint).removesuffix(".zip") + ".meta.json")
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found: a policy's scenario, reward and physics are read from its metadata, "
                         f"never guessed from a file name.")
    meta = read_json(meta_path)
    if meta.get("scenario_family") not in FAMILIES:
        raise SystemExit(f"{meta_path} is not a cut-in or sandwich policy.")
    return meta


def policy_identity(checkpoint, meta: dict) -> dict:
    """Unrecorded fields stay None, and downstream comparisons treat that as non-equivalent."""
    sha = file_sha256(checkpoint)
    return {"policy_id": sha[:12], "checkpoint": str(checkpoint), "checkpoint_sha256": sha,
            "evaluated_checkpoint": meta.get("evaluated_checkpoint"), "ppo_seed": meta.get("ppo_seed"),
            "trained_mode": meta["mode"], "training_scenario_id": meta["scenario_id"],
            "training_scenario_seed": meta["scenario_seed"], "timesteps_actual": meta.get("timesteps_actual"),
            "reward_version": meta.get("reward_version"), "reward_config": meta.get("reward_config"),
            "ppo_gamma": meta.get("ppo_gamma"), "ppo_hyperparameters": meta.get("ppo_hyperparameters"),
            "vehicle_physics_version": meta.get("vehicle_physics_version"),
            "rollover_event_model": meta.get("rollover_event_model"), "obs_dim": meta.get("obs_dim"),
            "action_dim": meta.get("action_dim"), "dt_s": meta.get("dt_s")}


def _env(family: str, cfg, rank: int, rollover_event: str) -> EgoTrafficEnv:
    return EgoTrafficEnv(arena=FAMILIES[family][1](cfg), mode="distribution", worker_rank=rank, rollover_event=rollover_event)


def evaluate_policy(checkpoint, exp: Path, eval_mode: str, *, test_episodes: int = 100, selection_episodes: int = 20,
                    deterministic: bool = True, eval_seed: int = 0, rollover_event: str | None = None,
                    trajectories_per_outcome: int = 3, force: bool = False) -> Path:
    exp, meta = Path(exp), load_policy_meta(checkpoint)
    family, policy = meta["scenario_family"], policy_identity(checkpoint, meta)
    rollover_event = rollover_event or meta.get("rollover_event_model") or "physical_v2"
    cfg = arena_config(family, meta, mode=eval_mode)
    independent = eval_mode == "robust" or not deterministic
    n_test = test_episodes if independent else EXACT_REPEATS

    protocol = {"eval_mode": eval_mode, "deterministic": deterministic, "test_rank": TEST_RANK, "test_episodes": n_test,
                "selection_rank": SELECTION_RANK, "selection_episodes": selection_episodes, "eval_seed": None if deterministic else eval_seed,
                "rollover_event": rollover_event, "dt_s": DT}
    rel = run_path(family, meta["blocker_side"], meta["scenario_id"], meta.get("ppo_seed"))
    out = exp / "evaluations" / rel / f"{eval_mode}_{json_hash({**protocol, 'policy': policy['policy_id']})}.json"
    if out.exists() and not force:
        raise SystemExit(f"{out} already exists -- pass --force to replace it.")

    saved = Counter()

    def keep(result):
        key = outcome_class(result)
        saved[key] += 1
        if saved[key] > trajectories_per_outcome:
            return None
        result["trajectory"] = str(Path("trajectories") / rel / eval_mode / f"ep_{result['episode']:05d}.npz")
        return exp / result["trajectory"]

    model = PPO.load(checkpoint, device="cpu")
    if not deterministic:
        model.set_random_seed(eval_seed)
    selection = run_episodes(model, _env(family, cfg, SELECTION_RANK, rollover_event), selection_episodes,
                             deterministic=deterministic, observer=TrajectoryRecorder())
    rollouts = run_episodes(model, _env(family, cfg, TEST_RANK, rollover_event), n_test, deterministic=deterministic,
                            observer=TrajectoryRecorder(keep))
    n_distinct = len({r["fingerprint"] for r in rollouts})
    record = {
        "schema": EVALUATION_SCHEMA, "policy": policy, "protocol": protocol, "provenance": provenance(),
        "scenario": {"family": family, "blocker_side": meta["blocker_side"], "scenario_id": cfg.scenario_id(),
                     "training_scenario_id": meta["scenario_id"], "theta": cfg.theta(), "scenario_seed": cfg.seed,
                     "goal_distance_m": cfg.goal_distance_m, "max_episode_seconds": cfg.max_episode_seconds},
        "independent": independent, "n_independent": n_test if independent else 1, "n_distinct_rollouts": n_distinct,
        "independence": "independent realizations" if eval_mode == "robust" else
                        ("independent draws of a stochastic policy" if independent else
                         "one deterministic realization repeated to confirm it"),
        "repeatable": None if independent else n_distinct == 1,
        "summary": summarize_seed(rollouts, "distribution" if independent else "fixed", 1.0),
        "selection": {"k": sum(bool(r["success"]) for r in selection), "n": len(selection),
                      "stability_loss": sum(bool(r["physical"]["stability_loss"]) for r in selection)},
        "rollouts": rollouts}
    write_json(out, _clean(record))
    return out


def experiment_checkpoints(exp: Path) -> list[Path]:
    return sorted(Path(exp).glob("training/*/*/*/seed_*/model.zip"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="experiment directory; created if it does not exist")
    parser.add_argument("--checkpoint", default=None, help="one model.zip; omit to evaluate all of the experiment's policies")
    parser.add_argument("--eval-modes", nargs="+", choices=["exact", "robust"], default=None)
    parser.add_argument("--test-episodes", type=int, default=None)
    parser.add_argument("--selection-episodes", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true", help="sample actions; deterministic inference is the primary protocol")
    parser.add_argument("--eval-seed", type=int, default=0, help="seeds action sampling with --stochastic")
    parser.add_argument("--rollover-event", choices=["legacy_v1", "physical_v2"], default=None)
    parser.add_argument("--trajectories-per-outcome", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    exp = Path(args.experiment)
    if not exp.exists():
        if args.checkpoint is None:
            raise SystemExit(f"{exp} does not exist; pass --checkpoint to start one from a saved policy.")
        create_experiment(ExperimentConfig(name=exp.name, strata=[], n_thetas=0), exp.parent, scenarios=[])
    c = load_config(exp)
    for ckpt in ([Path(args.checkpoint)] if args.checkpoint else experiment_checkpoints(exp)):
        for mode in (args.eval_modes or c.eval_modes):
            out = evaluate_policy(ckpt, exp, mode, test_episodes=args.test_episodes or c.test_episodes,
                                  selection_episodes=args.selection_episodes or c.selection_episodes,
                                  deterministic=c.deterministic and not args.stochastic, eval_seed=args.eval_seed,
                                  rollover_event=args.rollover_event, force=args.force,
                                  trajectories_per_outcome=c.trajectories_per_outcome if args.trajectories_per_outcome is None else args.trajectories_per_outcome)
            s = read_json(out)["summary"]
            print(f"{out}: {s['n_success']}/{s['n_episodes']} success")


if __name__ == "__main__":
    main()
