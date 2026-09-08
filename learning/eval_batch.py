"""
Non-visual batch evaluation of one or more trained ego policies (one per
independent PPO seed, all trained on the SAME fixed scenario): run N
episodes per model, report success/failure/timeout counts, a success rate
(with a 95% Wilson CI in distribution mode), return statistics, and a
FEASIBLE/NOT_SOLVED/INCONCLUSIVE label across the tested seeds -- see
label_feasibility's own docstring for exactly what each label does and
does not claim.

`success` here is read directly from EgoTrafficEnv's own info dict (see
learning.env's module docstring: success = scenario_goal_reached and not
(collision or rollover or off_road), computed from physical terminal
conditions, never from accumulated reward) -- this module never re-derives
it, and never treats an unharmed timeout as success.

Distinct from learning.eval, which stays the visual single-episode
matplotlib animation. This is the batch/statistics counterpart, meant for
"is this scenario solvable" (fixed mode) or "what's the success rate over
this scenario family" (distribution mode) questions -- see
learning.scenario/learning.env for what those modes mean.

Usage:
    python -m learning.eval_batch --model PATH [PATH ...] --episodes N
                                   [--stochastic] [--eval-seed N]
                                   [--success-threshold F]
                                   [--fail-dir DIR] [--out results.json]
"""

import argparse
import json
import math
import os
import warnings

from stable_baselines3 import PPO

from learning.env import EgoTrafficEnv, Outcome, FailureReason
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


def run_episodes(model: PPO, env: EgoTrafficEnv, episodes: int,
                  deterministic: bool = True, fail_dir: str | None = None) -> list[dict]:
    """Run `episodes` full rollouts of `model` against `env`, one row per
    episode built directly from EgoTrafficEnv's own terminal `info` (see
    module docstring -- outcome/success/failure_reason are never
    re-derived here)."""
    results = []
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = truncated = False
        step_idx = 0
        ep_return = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            step_idx += 1

        results.append({
            "episode": ep,
            "outcome": info["outcome"],
            "success": info["success"],
            "failure_reason": info["failure_reason"],
            "timeout": info["timeout"],
            "steps": step_idx,
            "return": ep_return,
            "progress_m": info["progress_m"],
            "distance_travelled_m": info["distance_travelled_m"],
            "elapsed_s": info["elapsed_s"],
            "min_clearance_m": info["min_clearance_m"],
            "max_p_rollover": info["max_p_rollover"],
            "max_road_departure_m": info["max_road_departure_m"],
        })

        if not info["success"] and fail_dir is not None:
            os.makedirs(fail_dir, exist_ok=True)
            try:
                realized = env.get_realized_scenario()
            except RuntimeError:
                realized = None   # default-random env has no realized-scenario record (see get_realized_scenario)
            with open(os.path.join(fail_dir, f"ep_{ep}.json"), "w") as f:
                json.dump({"result": results[-1], "step_failed_at": step_idx, "realized_scenario": realized},
                          f, indent=2)

    return results


def summarize_seed(results: list[dict], mode: str, success_threshold: float) -> dict:
    """Per-seed statistics -- see module docstring for terminology.
    `meets_threshold` combines a fixed-mode repeatability check (episodes
    MUST agree under a deterministic scenario + deterministic policy) with
    the success-rate threshold; distribution mode skips the repeatability
    requirement (background traffic is expected to vary episode to
    episode) and only applies the threshold."""
    n = len(results)
    n_success = sum(r["success"] for r in results)
    n_collisions = sum(r["failure_reason"] == FailureReason.COLLISION.value for r in results)
    n_rollovers = sum(r["failure_reason"] == FailureReason.ROLLOVER.value for r in results)
    n_offroad = sum(r["failure_reason"] == FailureReason.OFF_ROAD.value for r in results)
    n_timeouts = sum(r["outcome"] == Outcome.TIMEOUT.value for r in results)
    returns = [r["return"] for r in results]
    success_times = [r["elapsed_s"] for r in results if r["success"]]

    mean_return = sum(returns) / n if n else 0.0
    std_return = math.sqrt(sum((x - mean_return) ** 2 for x in returns) / n) if n else 0.0

    summary = {
        "n_episodes": n,
        "n_success": n_success,
        "n_safety_failures": n_collisions + n_rollovers + n_offroad,
        "n_collisions": n_collisions,
        "n_rollovers": n_rollovers,
        "n_offroad": n_offroad,
        "n_timeouts": n_timeouts,
        "success_rate": n_success / n if n else 0.0,
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_success_time_s": sum(success_times) / len(success_times) if success_times else None,
    }

    if mode == "distribution":
        lo, hi = wilson_ci(n_success, n)
        summary["wilson_95ci"] = [lo, hi]
        summary["repeatability"] = "OK"   # not a meaningful check in distribution mode -- background varies by design
    else:
        # Fixed mode: deterministic scenario + deterministic policy means every episode should agree --
        # see EgoTrafficEnv "fixed" mode. Disagreement signals unexpected nondeterminism, not scenario
        # difficulty, so it's reported distinctly rather than folded into an ordinary success rate.
        outcomes = {r["outcome"] for r in results}
        if n == 0:
            summary["repeatability"] = "INCONCLUSIVE"
            summary["repeatability_reason"] = "no episodes were run."
        elif len(outcomes) > 1:
            summary["repeatability"] = "INCONCLUSIVE"
            summary["repeatability_reason"] = (
                f"{n} repeated fixed-mode episodes disagreed (outcomes seen: {sorted(outcomes)}) -- "
                f"expected identical results under a deterministic scenario + deterministic policy; "
                f"this points at unexpected nondeterminism somewhere, not scenario difficulty.")
        else:
            summary["repeatability"] = "OK"

    summary["meets_threshold"] = (
        n > 0 and summary["repeatability"] == "OK" and summary["success_rate"] >= success_threshold)
    return summary


def label_feasibility(seed_summaries: list[dict], success_threshold: float) -> tuple[str, str]:
    """FEASIBLE / NOT_SOLVED / INCONCLUSIVE for one fixed scenario across
    however many independent PPO seeds were tested (see module docstring).

    FEASIBLE:    at least one tested seed's success rate reached
                 success_threshold under the configured evaluation
                 protocol -- PPO demonstrated a working policy exists.
    NOT_SOLVED:  every seed completed its full training/evaluation budget
                 but none reached the threshold -- "not solved by this PPO
                 procedure", NEVER "physically impossible". PPO failure is
                 not proof of infeasibility.
    INCONCLUSIVE: no seeds were evaluated, or at least one seed's
                 evaluation was incomplete/non-repeatable (0 episodes, or
                 fixed-mode episodes disagreeing -- see summarize_seed) --
                 there isn't enough evidence yet to say either way.
    """
    if not seed_summaries:
        return "INCONCLUSIVE", "no seeds were evaluated."
    if any(s["n_episodes"] == 0 or s.get("repeatability") == "INCONCLUSIVE" for s in seed_summaries):
        return "INCONCLUSIVE", "at least one seed's evaluation was incomplete or non-repeatable."
    if any(s["success_rate"] >= success_threshold for s in seed_summaries):
        return "FEASIBLE", (f"at least one of {len(seed_summaries)} tested seed(s) reached the "
                             f"{success_threshold:.0%} success-rate threshold.")
    return "NOT_SOLVED", (f"none of {len(seed_summaries)} tested seed(s) reached the "
                           f"{success_threshold:.0%} success-rate threshold under this training/evaluation "
                           f"procedure -- this is NOT proof the scenario is physically infeasible, only "
                           f"that PPO hasn't solved it (yet) under the tested budget/seeds.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+", default=["learning/ppo_ego"],
                         help="one or more trained-model paths -- one per independent PPO seed being "
                              "compared for the same fixed scenario (see learning.train --ppo-seed).")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--stochastic", action="store_true",
                         help="sample actions from each policy instead of its deterministic mean "
                              "(default: deterministic).")
    parser.add_argument("--eval-seed", type=int, default=None,
                         help="seeds action-sampling randomness for --stochastic evaluation -- kept "
                              "separate from training's ppo_seed and the scenario's own seed. Ignored "
                              "for deterministic evaluation (no sampling randomness to seed).")
    parser.add_argument("--success-threshold", type=float, default=1.0,
                         help="minimum per-seed success rate to count that seed as having solved the "
                              "scenario (default 1.0 -- fixed mode is deterministic, so a solved seed "
                              "should succeed every episode; lower this for distribution mode).")
    parser.add_argument("--fail-dir", type=str, default=None,
                         help="directory to save each failed episode's realized scenario (for exact "
                              "replay) as <model-basename>/ep_<i>.json. Omit to not save anything.")
    parser.add_argument("--out", type=str, default=None,
                         help="write the full per-seed results + feasibility verdict as JSON here. A "
                              "concise summary is always printed to stdout regardless.")
    args = parser.parse_args()

    seed_reports = []
    for model_path in args.model:
        model = PPO.load(model_path, device="cpu")   # see feasibility_train_one's own fix -- PPO.load()
                                                       # defaults device to "auto", which would silently
                                                       # grab CUDA if available regardless of this
                                                       # project's CPU-only-training rationale.
        if args.eval_seed is not None:
            model.set_random_seed(args.eval_seed)   # only affects --stochastic action sampling
        meta = _load_meta(model_path)

        if meta is not None and meta.get("scenario_config") is not None:
            scenario_config = ScenarioConfig.from_dict(meta["scenario_config"])
            mode = meta["scenario_mode"]
            env = EgoTrafficEnv(scenario_config=scenario_config, mode=mode, worker_rank=0)
        else:
            scenario_config = None
            mode = "distribution"   # fully-random traffic has no "fixed" concept; treat statistically
            env = EgoTrafficEnv()

        fail_dir = os.path.join(args.fail_dir, os.path.basename(model_path)) if args.fail_dir else None
        results = run_episodes(model, env, args.episodes, deterministic=not args.stochastic, fail_dir=fail_dir)
        summary = summarize_seed(results, mode, args.success_threshold)

        seed_reports.append({
            "model": model_path,
            "ppo_seed": meta.get("ppo_seed") if meta else None,
            "scenario_seed": (meta.get("scenario_config") or {}).get("seed") if meta else None,
            "training_timesteps": (meta.get("args") or {}).get("timesteps") if meta else None,
            "mode": mode,
            "deterministic": not args.stochastic,
            "eval_seed": args.eval_seed,
            "n_episodes": args.episodes,
            "summary": summary,
            "episodes": results,
        })

        print(f"{model_path}: {summary['n_success']}/{summary['n_episodes']} success "
              f"({summary['success_rate']:.1%}) | collisions={summary['n_collisions']} "
              f"rollovers={summary['n_rollovers']} off_road={summary['n_offroad']} "
              f"timeouts={summary['n_timeouts']} | return {summary['mean_return']:.2f} "
              f"+/- {summary['std_return']:.2f}"
              + (f" | repeatability={summary['repeatability']}" if summary.get("repeatability") != "OK" else ""))

    label, reason = label_feasibility([r["summary"] for r in seed_reports], args.success_threshold)
    print(f"\n{label}: {reason}")

    output = {
        "feasibility_label": label,
        "feasibility_reason": reason,
        "success_threshold": args.success_threshold,
        "seeds": seed_reports,
    }
    if args.out is not None:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
    else:
        print(json.dumps(
            {"feasibility_label": label, "feasibility_reason": reason,
             "seeds": [{"model": r["model"], "summary": r["summary"]} for r in seed_reports]},
            indent=2))


if __name__ == "__main__":
    main()
