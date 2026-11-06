"""
Lightweight scalar logging for PPO training (docs/description.md §23): two SB3 callbacks that append to one
metrics.jsonl. They read what training already produces and run on the training loop's own thread.

  kind="train"       TrainingMetricsCallback: per PPO iteration, the optimizer values SB3 logged plus episode
                     statistics over the episodes that ended in that rollout. Training rollouts use a
                     stochastic policy on the training realization, so these rates are NOT evaluation results.
  kind="validation"  ValidationCallback: a deterministic evaluation on the fixed VALIDATION_RANK episodes.
                     It never updates the policy and never picks a checkpoint.

A value SB3 did not produce, or an iteration where no episode ended, is written as null, never estimated.
"""

import json
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from learning.eval_batch import run_episodes, summarize_seed

VALIDATION_RANK = 1_100_000   # learning curves' realization stream (see learning.ppo_experiment's streams)
TRAINING_RANK = 2_000_000     # first training worker; worker i uses TRAINING_RANK + i


def training_env_kwargs(cfg_mode: str, worker: int) -> dict:
    """EgoTrafficEnv arguments for training worker `worker`. Exact mode has one realization. Robust mode draws a
    new one every reset, from a rank range no evaluation stream uses, so rank 0 evaluations stay held out."""
    assert cfg_mode in ("exact", "robust"), cfg_mode
    return {"mode": "fixed" if cfg_mode == "exact" else "distribution", "worker_rank": TRAINING_RANK + worker}


def training_realization(cfg_mode: str) -> dict:
    """What a policy trained in cfg_mode saw, for the metadata and the report."""
    description = ("the single unperturbed realization of the scenario, identical for every worker and every reset"
                   if cfg_mode == "exact" else "a new perturbed realization every reset, different for every worker")
    return {"env_mode": training_env_kwargs(cfg_mode, 0)["mode"], "first_worker_rank": TRAINING_RANK, "description": description}
_OPTIMIZER_KEYS = ("loss", "policy_gradient_loss", "value_loss", "entropy_loss", "approx_kl", "clip_fraction",
                   "clip_range", "explained_variance", "learning_rate", "std")


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=float) + "\n")   # SB3 logs numpy scalars


class TrainingMetricsCallback(BaseCallback):
    """Written at the start of each rollout, after the previous iteration's train() recorded its values,
    and once more when training ends: SB3 dumps its logger before train(), so the values are there then."""

    def __init__(self, path: str):
        super().__init__()
        self.path = Path(path)
        self._episodes: list[dict] = []
        self._t0 = time.perf_counter()

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self._episodes.append(info)
        return True

    def _write(self) -> None:
        values = self.logger.name_to_value
        optimizer = {k: values.get(f"train/{k}") for k in _OPTIMIZER_KEYS}
        if not self._episodes and all(v is None for v in optimizer.values()):
            return
        eps, n = self._episodes, len(self._episodes)
        rate = lambda f: sum(bool(f(i)) for i in eps) / n if n else None   # noqa: E731
        _append(self.path, {
            "kind": "train", "timesteps": self.num_timesteps, "wall_time_s": time.perf_counter() - self._t0,
            "n_episodes": n, "ep_return_mean": sum(i["episode_return"] for i in eps) / n if n else None,
            "ep_length_mean": sum(i["episode_length"] for i in eps) / n if n else None,
            "success_rate": rate(lambda i: i["success"]), "collision_rate": rate(lambda i: i["failure_reason"] == "collision"),
            "off_road_rate": rate(lambda i: i["failure_reason"] == "off_road"),
            "rollover_rate": rate(lambda i: i["failure_reason"] == "rollover"), "timeout_rate": rate(lambda i: i["timeout"]),
            **optimizer})
        self._episodes = []

    def _on_rollout_start(self) -> None:
        self._write()

    def _on_training_end(self) -> None:
        self._write()


class ValidationCallback(BaseCallback):
    """make_env builds a fresh distribution-mode env on VALIDATION_RANK for every evaluation, so its episode
    counter restarts and each point on the curve sees exactly the same realizations."""

    def __init__(self, make_env, path: str, every_timesteps: int, episodes: int):
        super().__init__()
        self.make_env, self.path, self.every, self.episodes = make_env, Path(path), every_timesteps, episodes
        self._next = every_timesteps

    def _evaluate(self) -> None:
        results = run_episodes(self.model, self.make_env(), self.episodes, deterministic=True)
        s = summarize_seed(results, "distribution", 1.0)
        n = s["n_episodes"]
        _append(self.path, {
            "kind": "validation", "timesteps": self.num_timesteps, "n_episodes": n, "deterministic": True,
            "success_rate": s["success_rate"], "collision_rate": s["n_collisions"] / n, "off_road_rate": s["n_offroad"] / n,
            "rollover_rate": s["n_rollovers"] / n, "timeout_rate": s["n_timeouts"] / n, "ep_return_mean": s["mean_return"],
            "ep_length_mean": sum(r["steps"] for r in results) / n})

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            self._evaluate()
        return True
