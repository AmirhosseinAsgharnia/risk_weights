"""
Train the ego vehicle with PPO (Stable-Baselines3) against EgoTrafficEnv
(see env.py for the scenario, observation, action, and reward definitions).

Usage:
    python -m learning.train [--timesteps N] [--n-envs N] [--out PATH] [--device DEVICE]
"""

import argparse

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from learning.env import EgoTrafficEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type = int, default = 200_000)
    parser.add_argument("--n-envs", type = int, default = 8)
    parser.add_argument("--out", type = str, default = "learning/ppo_ego")
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

    vec_env = make_vec_env(EgoTrafficEnv, n_envs = args.n_envs)
    model = PPO("MlpPolicy", vec_env, verbose = 1, device = args.device)
    model.learn(total_timesteps = args.timesteps)
    model.save(args.out)


if __name__ == "__main__":
    main()
