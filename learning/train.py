"""
Train the ego vehicle with PPO (Stable-Baselines3) against EgoTrafficEnv
(see env.py for the scenario, observation, action, and reward definitions,
and learning.scenario for the compact ScenarioConfig this can optionally
train against instead of fully-random traffic).

Usage:
    python -m learning.train [--timesteps N] [--n-envs N] [--out PATH] [--device DEVICE]
                              [--scenario-config PATH --scenario-mode {fixed,distribution}]
                              [--base-seed N] [--ppo-seed N] [--force]
"""

import argparse
import dataclasses
import json
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from learning.env import EgoTrafficEnv, N_SURR, LANE_NUM, DT, EPISODE_SECONDS, MAX_STEPS
from learning.scenario import ScenarioConfig

_DEFAULT_N_ENVS = max(1, (os.cpu_count() or 4) - 1)   # leave one core for the main process


def _meta_path(out: str) -> str:
    return f"{out}.meta.json"


def _build_meta(args, scenario_config: ScenarioConfig | None) -> dict:
    return {
        "scenario_config": scenario_config.to_dict() if scenario_config is not None else None,
        "scenario_mode": args.scenario_mode,
        "ppo_seed": args.ppo_seed,
        "base_seed": args.base_seed,
        "args": vars(args),
        "env_constants": {
            "N_SURR": N_SURR, "LANE_NUM": LANE_NUM, "DT": DT,
            "EPISODE_SECONDS": EPISODE_SECONDS, "MAX_STEPS": MAX_STEPS,
            "obs_dim": 5 + 3 * N_SURR, "action_dim": 2,
        },
    }


def _check_overwrite(out: str, meta: dict, force: bool) -> None:
    """Refuse to clobber an existing model file's metadata with an
    incompatible experiment's, unless --force -- the model file itself
    would silently become inconsistent with any stale .meta.json left
    behind otherwise (or vice versa if only the model got overwritten)."""
    path = _meta_path(out)
    if not os.path.exists(path):
        return
    with open(path) as f:
        existing = json.load(f)
    same_scenario = existing.get("scenario_config") == meta["scenario_config"]
    same_mode = existing.get("scenario_mode") == meta["scenario_mode"]
    if same_scenario and same_mode:
        return   # re-running/continuing the same experiment -- fine
    if not force:
        raise SystemExit(
            f"{path} already exists and describes a different experiment "
            f"(scenario_mode={existing.get('scenario_mode')!r} vs {meta['scenario_mode']!r}, "
            f"scenario_config {'differs' if not same_scenario else 'matches'}) -- refusing to overwrite "
            f"{out} and its metadata. Pass --force to overwrite anyway, or choose a different --out.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type = int, default = 200_000)
    parser.add_argument("--n-envs", type = int, default = _DEFAULT_N_ENVS,
                         help = f"parallel envs, each in its own subprocess (see SubprocVecEnv below) "
                                f"-- defaults to cpu_count - 1 ({_DEFAULT_N_ENVS} on this machine).")
    parser.add_argument("--out", type = str, default = "learning/ppo_ego")
    parser.add_argument("--force", action = "store_true",
                         help = "overwrite --out even if its existing .meta.json describes a different "
                                "experiment (different scenario_config/scenario_mode).")
    parser.add_argument("--n-epochs", type = int, default = 30,
                         help = "PPO gradient-descent passes over each collected rollout (SB3 default is 10).")
    parser.add_argument("--ego-speed-std", type = float, default = 2.5,
                         help = "[m/s] ego is the road speed, not an independent quantity: its initial "
                                "v_x ~ Normal(mean(realized surr v_x), this). Only used without "
                                "--scenario-config (with one, ScenarioConfig.background_mean_speed/"
                                "background_speed_std governs ego's speed instead).")
    parser.add_argument("--device", type = str, default = "auto",
                         help = "'auto' (default, picks cuda if available), 'cuda', 'cuda:0', or 'cpu'. "
                                "Note: the environment itself always runs on CPU (it's plain Python/NumPy "
                                "physics, not batched on GPU), and this policy network is a small MLP -- "
                                "so a GPU mainly helps once you scale up network size/batch size, not "
                                "necessarily out of the box.")
    parser.add_argument("--scenario-config", type = str, default = None,
                         help = "path to a JSON-serialized ScenarioConfig (see learning.scenario). If "
                                "omitted, trains against the original fully-random traffic instead.")
    parser.add_argument("--scenario-mode", type = str, choices = ["fixed", "distribution"], default = "distribution",
                         help = "only meaningful with --scenario-config: 'fixed' reconstructs the exact "
                                "same scenario realization every episode (is this one scenario solvable "
                                "at all?); 'distribution' varies the background traffic across episodes "
                                "(what's the success rate over this scenario family?). See learning.env.")
    parser.add_argument("--base-seed", type = int, default = None,
                         help = "overrides scenario_config.seed if given (ignored without --scenario-config).")
    parser.add_argument("--ppo-seed", type = int, default = None,
                         help = "seeds PPO's own algorithm-level randomness (network init, action sampling) "
                                "-- separate from, and doesn't affect, scenario/background randomness.")
    args = parser.parse_args()

    if args.device not in ("auto", "cpu") and not torch.cuda.is_available():
        raise SystemExit(f"--device {args.device!r} requested but torch.cuda.is_available() is False "
                          f"on this machine -- install a CUDA-enabled torch build, or pass --device cpu.")

    scenario_config = None
    if args.scenario_config is not None:
        with open(args.scenario_config) as f:
            scenario_config = ScenarioConfig.from_dict(json.load(f))
        if args.base_seed is not None:
            scenario_config = dataclasses.replace(scenario_config, seed = args.base_seed)
    elif args.base_seed is not None:
        raise SystemExit("--base-seed only makes sense together with --scenario-config.")

    meta = _build_meta(args, scenario_config)
    _check_overwrite(args.out, meta, args.force)

    # SubprocVecEnv: one real OS process per env, so n_envs actually spreads
    # across cores. Built directly (rather than via make_vec_env, whose
    # default DummyVecEnv runs every env sequentially in this one process
    # -- n_envs would only ever use one core no matter how high you set it;
    # benchmarked ~3x higher fps with Subproc at n_envs=11 on a 12-core
    # machine, 326 vs 108 fps) so each subprocess gets a distinct
    # worker_rank -- required for distribution-mode scenario seeding (see
    # learning.env/learning.scenario.derive_seed) to not give parallel
    # workers identical background realizations.
    def _make_env(rank: int):
        def _init():
            return EgoTrafficEnv(ego_speed_std = args.ego_speed_std, scenario_config = scenario_config,
                                  mode = args.scenario_mode, worker_rank = rank)
        return _init

    vec_env = SubprocVecEnv([_make_env(i) for i in range(args.n_envs)])
    model = PPO("MlpPolicy", vec_env, verbose = 1, device = args.device,
                n_epochs = args.n_epochs, seed = args.ppo_seed)
    model.learn(total_timesteps = args.timesteps)
    model.save(args.out)

    with open(_meta_path(args.out), "w") as f:
        json.dump(meta, f, indent = 2)


if __name__ == "__main__":
    main()
