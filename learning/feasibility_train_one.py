"""
Train (and, by default, evaluate) a PPO policy against ONE manually-specified
feasibility-arena scenario -- theta is supplied directly on the command line,
not sampled. See learning.feasibility_cutin/learning.feasibility_sandwich for
the two scenario families this trains against, and learning.eval_batch for
the evaluation machinery this reuses unchanged.

"exact" mode reproduces the identical realization every reset (answers "can
PPO solve this precise scenario?"); "robust" mode applies each family's
small seeded nuisance perturbations across resets (answers "can PPO solve a
local distribution around this scenario?") -- see ArenaCommonConfig.mode.

Every theta field from both families is exposed as a flag (only the ones
belonging to --scenario-family are actually used -- see THETA_BOUNDS_CUTIN/
THETA_BOUNDS_SANDWICH for which); anything not passed keeps that family's
default. The resolved config is always printed before training starts.

--timesteps is an upper bound, not a target: by default this stops training
early once a held-out evaluation env (a DIFFERENT realization-seed stream
than training -- never the same episodes PPO is training on) shows no
improvement in mean return for --patience consecutive evaluations (each
--eval-freq-timesteps apart, only after --min-evals have happened at all) --
a deliberately cautious, patience-based rule so one lucky/unlucky evaluation
can't stop or extend training on its own (see StopTrainingOnNoModelImprovement
below). Pass --no-early-stop to always run the full --timesteps instead. The
BEST checkpoint seen (by that held-out mean return) is saved to
<out>_best/best_model.zip alongside the FINAL checkpoint at <out>.zip -- the
meta.json records which one <out>.zip actually is and why training ended.

Usage:
    python -m learning.feasibility_train_one --scenario-family cutin \\
        --timesteps 2000000 --n-envs 4 --blocker-side 1 --mode robust \\
        --scenario-seed 0 --ppo-seed 0 \\
        --s-cutin-m 200 --cutter-gap-m 25 \\
        --episodes 100 --out learning/ppo_cutin_theta0

    python -m learning.feasibility_train_one --scenario-family sandwich \\
        --timesteps 2000000 --blocker-side -1 --mode exact \\
        --s-stop-m 220 --front-gap-m 25 \\
        --episodes 100 --out learning/ppo_sandwich_thetaA
"""

import argparse
import dataclasses
import json
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from learning.env import EgoTrafficEnv
from learning.feasibility_cutin import CutinConfig, CutinRuntime
from learning.feasibility_sandwich import SandwichConfig, SandwichRuntime
from learning.eval_batch import run_episodes, summarize_seed

_CONFIG_CLASSES = {"cutin": CutinConfig, "sandwich": SandwichConfig}
_RUNTIME_CLASSES = {"cutin": CutinRuntime, "sandwich": SandwichRuntime}

# Union of every theta field across both families -- see each module's THETA_BOUNDS_* for which
# family actually consumes which flag; passing one that doesn't apply to --scenario-family is
# simply ignored (never a silent no-op on the WRONG family's field, though -- _build_config only
# ever looks up field names the chosen dataclass actually declares).
_THETA_FLAGS = {
    "road_mu": float, "road_kappa_max": float,
    "s_cutin_m": float, "cutter_gap_m": float, "cutter_relative_speed_mps": float,
    "s_stop_m": float, "rear_gap_m": float, "rear_relative_speed_mps": float,
    "escape_gap_center_m": float, "blocker_relative_speed_mps": float,
    "front_gap_m": float, "front_relative_speed_mps": float, "escape_gap_length_m": float,
}


def _meta_path(out: str) -> str:
    return f"{out}.meta.json"


def _build_config(args: argparse.Namespace):
    cfg_cls = _CONFIG_CLASSES[args.scenario_family]
    field_names = {f.name for f in dataclasses.fields(cfg_cls)}
    overrides = {name: getattr(args, name) for name in _THETA_FLAGS
                 if name in field_names and getattr(args, name) is not None}
    overrides.update(blocker_side=args.blocker_side, mode=args.mode, seed=args.scenario_seed,
                      goal_distance_m=args.goal_distance_m, max_episode_seconds=args.max_episode_seconds,
                      min_progress_m=args.min_progress_m)
    return cfg_cls(**overrides)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario-family", choices=["cutin", "sandwich"], required=True)
    parser.add_argument("--mode", choices=["exact", "robust"], default="robust")
    parser.add_argument("--blocker-side", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--scenario-seed", type=int, default=0,
                         help="this scenario's own base seed (realization-noise stream) -- kept "
                              "separate from --ppo-seed and --eval-seed.")
    parser.add_argument("--goal-distance-m", type=float, default=400.0)
    parser.add_argument("--max-episode-seconds", type=float, default=35.0)
    parser.add_argument("--min-progress-m", type=float, default=100.0)
    for name, typ in _THETA_FLAGS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=typ, default=None,
                             help=f"theta field '{name}' -- omit to keep this family's default.")

    parser.add_argument("--timesteps", type=int, default=2_000_000,
                         help="upper bound on training -- early stopping (on by default, see "
                              "--no-early-stop) will typically stop well before this.")
    parser.add_argument("--n-envs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--ppo-seed", type=int, default=None, help="PPO's own algorithm-level seed.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--force", action="store_true",
                         help="overwrite --out even if its existing .meta.json describes a different theta.")

    parser.add_argument("--no-early-stop", action="store_true",
                         help="disable early stopping -- always train the full --timesteps.")
    parser.add_argument("--eval-freq-timesteps", type=int, default=100_000,
                         help="how often (in total env-timesteps, i.e. already accounting for "
                              "--n-envs) to run a held-out evaluation for the early-stopping check.")
    parser.add_argument("--patience-episodes", type=int, default=20,
                         help="episodes per held-out evaluation -- separate from --episodes' own "
                              "FINAL evaluation after training ends.")
    parser.add_argument("--patience", type=int, default=5,
                         help="stop once this many consecutive held-out evaluations show no "
                              "improvement in mean return.")
    parser.add_argument("--min-evals", type=int, default=5,
                         help="never stop early before this many held-out evaluations have happened, "
                              "regardless of --patience -- guards against stopping on early noise "
                              "before the policy has had a real chance to improve.")

    parser.add_argument("--episodes", type=int, default=100,
                         help="held-out realization seeds to evaluate after training (0 skips evaluation).")
    parser.add_argument("--eval-seed", type=int, default=None,
                         help="seeds --stochastic evaluation's action sampling -- ignored for the "
                              "deterministic (default) evaluation mode, which has no sampling randomness.")
    parser.add_argument("--stochastic", action="store_true",
                         help="sample actions during evaluation instead of using the policy's "
                              "deterministic mean (default: deterministic).")
    args = parser.parse_args()

    if args.device not in ("auto", "cpu") and not torch.cuda.is_available():
        raise SystemExit(f"--device {args.device!r} requested but torch.cuda.is_available() is False.")

    cfg = _build_config(args)
    theta = cfg.theta()
    resolved = {"scenario_family": args.scenario_family, "scenario_id": cfg.scenario_id(),
                "blocker_side": cfg.blocker_side, "mode": cfg.mode, "scenario_seed": cfg.seed,
                "theta": theta, "goal_distance_m": cfg.goal_distance_m,
                "max_episode_seconds": cfg.max_episode_seconds, "min_progress_m": cfg.min_progress_m}
    print("Resolved scenario:")
    print(json.dumps(resolved, indent=2))

    meta_path = _meta_path(args.out)
    if os.path.exists(meta_path) and not args.force:
        with open(meta_path) as f:
            existing = json.load(f)
        if existing.get("scenario_id") != cfg.scenario_id():
            raise SystemExit(
                f"{meta_path} already exists and describes a different scenario "
                f"(scenario_id={existing.get('scenario_id')!r} vs {cfg.scenario_id()!r}) -- refusing "
                f"to overwrite {args.out}. Pass --force to overwrite anyway, or choose a different --out.")

    runtime_cls = _RUNTIME_CLASSES[args.scenario_family]

    def make_env(rank: int):
        def _init():
            return EgoTrafficEnv(arena=runtime_cls(cfg), mode="fixed", worker_rank=rank)
        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    model = PPO("MlpPolicy", vec_env, verbose=1, device=args.device, seed=args.ppo_seed)

    callback = None
    stop_reason = "reached --timesteps"
    if not args.no_early_stop:
        # worker_rank=1_000_000: far outside the training envs' own 0..n_envs-1 range, so
        # derive_seed's SeedSequence-based stream for this held-out eval env never overlaps the
        # scenario realizations PPO is actually training on -- a real held-out set, not a relabeled
        # training episode. mode="distribution" so each periodic eval sees fresh realizations too,
        # not one repeated fixed episode.
        early_stop_env = EgoTrafficEnv(arena=runtime_cls(cfg), mode="distribution", worker_rank=1_000_000)
        stop_on_plateau = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=args.patience, min_evals=args.min_evals, verbose=1)
        callback = EvalCallback(
            early_stop_env, callback_after_eval=stop_on_plateau,
            best_model_save_path=f"{args.out}_best", eval_freq=max(1, args.eval_freq_timesteps // args.n_envs),
            n_eval_episodes=args.patience_episodes, deterministic=True, verbose=1)

    model.learn(total_timesteps=args.timesteps, callback=callback)
    if callback is not None and model.num_timesteps < args.timesteps:
        stop_reason = (f"early stopping: no improvement over {args.patience} held-out evaluations "
                        f"(each {args.patience_episodes} episodes) after {args.min_evals} minimum evals")
    model.save(args.out)
    vec_env.close()

    meta = {**resolved, "ppo_seed": args.ppo_seed, "timesteps_budget": args.timesteps,
            "timesteps_actual": int(model.num_timesteps), "stop_reason": stop_reason,
            "n_envs": args.n_envs,
            "best_checkpoint": None if args.no_early_stop else f"{args.out}_best/best_model.zip"}

    if args.episodes > 0:
        eval_env = EgoTrafficEnv(arena=runtime_cls(cfg), mode="distribution", worker_rank=0)
        if args.eval_seed is not None:
            model.set_random_seed(args.eval_seed)
        results = run_episodes(model, eval_env, args.episodes, deterministic=not args.stochastic)
        summary = summarize_seed(results, mode="distribution", success_threshold=1.0)
        print(f"\n{summary['n_success']}/{summary['n_episodes']} success ({summary['success_rate']:.1%}), "
              f"95% CI {summary['wilson_95ci']} | collisions={summary['n_collisions']} "
              f"rollovers={summary['n_rollovers']} off_road={summary['n_offroad']} "
              f"timeouts={summary['n_timeouts']}")
        meta["evaluation"] = {"episodes": args.episodes, "deterministic": not args.stochastic,
                               "eval_seed": args.eval_seed, "summary": summary}

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
