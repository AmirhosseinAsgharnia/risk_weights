"""
Train the ego vehicle with PPO (Stable-Baselines3) against EgoTrafficEnv
(see env.py for the scenario, observation, action, and reward definitions).

Usage:
    python -m learning.train [--timesteps N] [--n-envs N] [--out PATH] [--device DEVICE]
"""

import argparse
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from learning.env import EgoTrafficEnv

_DEFAULT_N_ENVS = max(1, (os.cpu_count() or 4) - 1)   # leave one core for the main process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type = int, default = 200_000)
    parser.add_argument("--n-envs", type = int, default = _DEFAULT_N_ENVS,
                         help = f"parallel envs, each in its own subprocess (see SubprocVecEnv below) "
                                f"-- defaults to cpu_count - 1 ({_DEFAULT_N_ENVS} on this machine).")
    parser.add_argument("--out", type = str, default = "learning/ppo_ego")
    parser.add_argument("--n-epochs", type = int, default = 30,
                         help = "PPO gradient-descent passes over each collected rollout (SB3 default is 10).")
    parser.add_argument("--ego-speed-min", type = float, default = 15.0,
                         help = "[m/s] lower bound ego's initial v_x is drawn from each episode -- "
                                "drawn the same way a surr car's v0 is (see generate_traffic).")
    parser.add_argument("--ego-speed-max", type = float, default = 25.0,
                         help = "[m/s] upper bound ego's initial v_x is drawn from each episode. "
                                "Defaults match SPEED_RANGES' overall span; pass 0 0 to start at rest.")
    parser.add_argument("--device", type = str, default = "auto",
                         help = "'auto' (default, picks cuda if available), 'cuda', 'cuda:0', or 'cpu'. "
                                "Note: the environment itself always runs on CPU (it's plain Python/NumPy "
                                "physics, not batched on GPU), and this policy network is a small MLP -- "
                                "so a GPU mainly helps once you scale up network size/batch size, not "
                                "necessarily out of the box.")
    args = parser.parse_args()

    if args.device not in ("auto", "cpu") and not torch.cuda.is_available():
        raise SystemExit(f"--device {args.device!r} requested but torch.cuda.is_available() is False "
                          f"on this machine -- install a CUDA-enabled torch build, or pass --device cpu.")

    ego_speed_range = (args.ego_speed_min, args.ego_speed_max)
    # SubprocVecEnv: one real OS process per env, so n_envs actually spreads
    # across cores. make_vec_env's default (DummyVecEnv) runs every env
    # sequentially in this single process/thread instead -- n_envs would
    # only ever use one core no matter how high you set it. Benchmarked
    # ~3x higher fps with Subproc at n_envs=11 on a 12-core machine (326
    # vs 108 fps) since this env's per-step work, while cheap, is still
    # enough pure-Python/NumPy CPU work for the parallelism to pay for its
    # own IPC overhead.
    vec_env = make_vec_env(EgoTrafficEnv, n_envs = args.n_envs, vec_env_cls = SubprocVecEnv,
                            env_kwargs = dict(ego_speed_range = ego_speed_range))
    model = PPO("MlpPolicy", vec_env, verbose = 1, device = args.device, n_epochs = args.n_epochs)
    model.learn(total_timesteps = args.timesteps)
    model.save(args.out)


if __name__ == "__main__":
    main()
