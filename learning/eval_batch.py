"""
Non-visual batch evaluation of a trained ego policy: run N deterministic-
policy episodes, report collisions/rollovers/safe completions, a success
rate (with a 95% Wilson CI in distribution mode), mean/min progress, and
save each failed episode's exact realized scenario for replay.

Distinct from learning.eval, which stays the visual single-episode
matplotlib animation. This is the batch/statistics counterpart, meant for
"is this scenario solvable" (fixed mode) or "what's the success rate over
this scenario family" (distribution mode) questions -- see
learning.scenario/learning.env for what those modes mean.

Usage:
    python -m learning.eval_batch --model PATH --episodes N
                                   [--fail-dir DIR] [--out results.json]
"""

import argparse
import json
import math
import os
import warnings

from stable_baselines3 import PPO

from learning.env import EgoTrafficEnv, EGO_S0
from learning.scenario import ScenarioConfig

_Z_95 = 1.959963984540054   # two-sided 95% normal quantile


def wilson_ci(successes: int, n: int, z: float = _Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial success proportion --
    closed-form, no scipy dependency needed. Better-behaved than the
    naive normal-approximation interval at small n or p_hat near 0/1."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = successes / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _load_meta(model_path: str) -> dict | None:
    meta_path = f"{model_path}.meta.json"
    if not os.path.exists(meta_path):
        warnings.warn(
            f"{meta_path} not found -- evaluating against a fresh default (fully-random-traffic) "
            f"EgoTrafficEnv instead of whatever scenario this model was actually trained on. Pass a "
            f"model trained via learning.train (which always writes this file) for a meaningful "
            f"fixed/distribution-mode evaluation.")
        return None
    with open(meta_path) as f:
        return json.load(f)


def run_batch(model: PPO, env: EgoTrafficEnv, episodes: int, fail_dir: str | None) -> list[dict]:
    results = []
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = truncated = False
        step_idx = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic = True)
            obs, reward, terminated, truncated, info = env.step(action)
            step_idx += 1

        progress = env.ego_car.state.s - EGO_S0
        failed = bool(info["collided"] or info["rolled_over"] or info["off_road"])
        # "survived" = not a failure -- either actually finished the road (a
        # stronger, distinct outcome, tracked separately below) or merely
        # ran out the clock unharmed without reaching the end.
        results.append({
            "episode": ep,
            "collided": bool(info["collided"]),
            "rolled_over": bool(info["rolled_over"]),
            "off_road": bool(info["off_road"]),
            "finished": bool(info["finished"]),
            "survived": bool(truncated or info["finished"]),
            "steps": step_idx,
            "progress": float(progress),
        })

        if failed and fail_dir is not None:
            os.makedirs(fail_dir, exist_ok = True)
            try:
                realized = env.get_realized_scenario()
            except RuntimeError:
                realized = None   # default-random env has no realized-scenario record (see get_realized_scenario)
            with open(os.path.join(fail_dir, f"ep_{ep}.json"), "w") as f:
                json.dump({"result": results[-1], "step_failed_at": step_idx, "realized_scenario": realized},
                          f, indent = 2)

    return results


def summarize(results: list[dict], mode: str) -> dict:
    n = len(results)
    n_collisions = sum(r["collided"] for r in results)
    n_rollovers = sum(r["rolled_over"] for r in results)
    n_offroad = sum(r["off_road"] for r in results)
    n_finished = sum(r["finished"] for r in results)
    n_safe = sum(r["survived"] for r in results)
    progress = [r["progress"] for r in results]

    summary = {
        "n_episodes": n,
        "n_collisions": n_collisions,
        "n_rollovers": n_rollovers,
        "n_offroad": n_offroad,
        "n_finished": n_finished,
        "n_safe_completions": n_safe,
        "success_rate": n_safe / n if n else 0.0,
        "mean_progress": sum(progress) / n if n else 0.0,
        "min_progress": min(progress) if progress else 0.0,
    }

    if mode == "distribution":
        lo, hi = wilson_ci(n_safe, n)
        summary["wilson_95ci"] = [lo, hi]
    else:
        # Fixed mode: deterministic scenario + deterministic policy means
        # every episode should agree. Classify accordingly rather than
        # ever calling a scenario itself "unsolvable" -- PPO failing only
        # tells you about this policy, not the scenario's true difficulty.
        outcomes = {r["survived"] for r in results}
        if len(outcomes) > 1:
            summary["classification"] = "INCONCLUSIVE"
            summary["classification_reason"] = (
                f"{n} repeated fixed-mode episodes disagreed ({n_safe} survived, {n - n_safe} did not) "
                f"-- expected identical results under a deterministic scenario + deterministic policy; "
                f"this points at unexpected nondeterminism somewhere, not scenario difficulty.")
        elif n == 0:
            summary["classification"] = "INCONCLUSIVE"
            summary["classification_reason"] = "no episodes were run."
        else:
            summary["classification"] = "SOLVED_BY_POLICY" if outcomes == {True} else "NOT_SOLVED_BY_POLICY"

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default = "learning/ppo_ego")
    parser.add_argument("--episodes", type = int, default = 30)
    parser.add_argument("--fail-dir", type = str, default = None,
                         help = "directory to save each failed episode's realized scenario (for exact "
                                "replay) as ep_<i>.json. Omit to not save anything.")
    parser.add_argument("--out", type = str, default = None,
                         help = "write the full results + summary as JSON here. Always printed to stdout "
                                "regardless.")
    args = parser.parse_args()

    model = PPO.load(args.model)
    meta = _load_meta(args.model)

    if meta is not None and meta.get("scenario_config") is not None:
        scenario_config = ScenarioConfig.from_dict(meta["scenario_config"])
        mode = meta["scenario_mode"]
        env = EgoTrafficEnv(scenario_config = scenario_config, mode = mode, worker_rank = 0)
    else:
        scenario_config = None
        mode = "distribution"   # fully-random traffic has no "fixed" concept; treat statistically
        env = EgoTrafficEnv()

    results = run_batch(model, env, args.episodes, args.fail_dir)
    summary = summarize(results, mode)

    print(json.dumps(summary, indent = 2))

    if args.out is not None:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "episodes": results}, f, indent = 2)


if __name__ == "__main__":
    main()
