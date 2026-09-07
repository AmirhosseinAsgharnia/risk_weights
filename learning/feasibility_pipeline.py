"""
Runner B: build a feasibility database for one scenario family + blocker
side, and fit the surrogate on it. Three modes:

    pilot     benchmark a few concurrency levels on THIS machine and
              recommend one -- run this first, on the real machine, before
              committing to a real `run`. Never touches the real dataset.
    run       sample n_thetas via a seeded space-filling design (learning.
              feasibility_sampling), train+evaluate each one (learning.
              feasibility_train_one.train_and_evaluate, unchanged), and
              accumulate results into the durable dataset (learning.
              feasibility_dataset). Safely resumable: re-running the same
              command skips any scenario_id that already has a result.
    fit-only  fit the surrogate (learning.feasibility_surrogate) on
              whatever's already in the dataset -- no training.

GPU note: every training job here runs with device=cpu (see
learning.feasibility_train_one/its own SB3 warning) -- a small MlpPolicy
over plain NumPy physics is CPU-bound end to end, so the lever for using
many cores is CONCURRENT scenarios (this file's own ProcessPoolExecutor),
not a bigger --n-envs on one job. `pilot` mode measures this directly rather
than assuming a concurrency level is efficient.

Usage:
    python -m learning.feasibility_pipeline --mode pilot --scenario-family cutin \\
        --blocker-side 1 --pilot-concurrencies 1,2,4,8

    python -m learning.feasibility_pipeline --mode run --scenario-family cutin \\
        --blocker-side 1 --n-thetas 24 --sampling-seed 0 --concurrency 6 \\
        --timesteps 2000000 --n-envs 4 --episodes 100

    python -m learning.feasibility_pipeline --mode fit-only --scenario-family cutin \\
        --blocker-side 1 --surrogate-out artifacts/feasibility/surrogates/cutin_side1.joblib
"""

import argparse
import dataclasses
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from learning.feasibility_cutin import CutinConfig, THETA_BOUNDS_CUTIN
from learning.feasibility_sandwich import SandwichConfig, THETA_BOUNDS_SANDWICH
from learning.feasibility_sampling import sample_cutin, sample_sandwich
from learning.feasibility_train_one import RunParams, train_and_evaluate
from learning.feasibility_dataset import (
    ARTIFACTS_ROOT, append_scenario_record, append_rollout_records, read_jsonl,
    scenarios_path, rollouts_path, scenario_result_exists, write_scenario_result, read_scenario_result,
)
from learning.feasibility_surrogate import fit_surrogate, PRELIMINARY_SCENARIO_THRESHOLD

_SAMPLERS = {"cutin": sample_cutin, "sandwich": sample_sandwich}
_THETA_BOUNDS = {"cutin": THETA_BOUNDS_CUTIN, "sandwich": THETA_BOUNDS_SANDWICH}
_BENCHMARK_CONFIGS = {"cutin": CutinConfig, "sandwich": SandwichConfig}


def _side_label(blocker_side: int) -> str:
    return "pos1" if blocker_side == 1 else "neg1"


def _policy_out_path(root: Path, family: str, blocker_side: int, scenario_id: str) -> Path:
    return root / "policies" / family / f"side_{_side_label(blocker_side)}" / scenario_id / "model"


def _scenario_record_from_result(family: str, blocker_side: int, cfg, meta: dict) -> dict:
    summary = (meta.get("evaluation") or {}).get("summary", {})
    return {
        "scenario_id": cfg.scenario_id(), "family": family, "blocker_side": blocker_side,
        "mode": cfg.mode, "theta": cfg.theta(),
        "k": summary.get("n_success"), "n": summary.get("n_episodes"),
        "success_rate": summary.get("success_rate"), "wilson_95ci": summary.get("wilson_95ci"),
        "n_collisions": summary.get("n_collisions"), "n_rollovers": summary.get("n_rollovers"),
        "n_offroad": summary.get("n_offroad"), "n_timeouts": summary.get("n_timeouts"),
        "mean_return": summary.get("mean_return"), "std_return": summary.get("std_return"),
        "mean_success_time_s": summary.get("mean_success_time_s"),
        "checkpoint": meta.get("evaluated_checkpoint"), "stop_reason": meta.get("stop_reason"),
        "timesteps_actual": meta.get("timesteps_actual"), "ppo_seed": meta.get("ppo_seed"),
    }


def _rollout_records_from_result(cfg, meta: dict) -> list[dict]:
    details = meta.get("_rollout_details") or []
    theta = cfg.theta()
    return [{"scenario_id": cfg.scenario_id(), "episode": r["episode"], "y": int(r["success"]),
              "outcome": r["outcome"], "failure_reason": r["failure_reason"], "theta": theta}
             for r in details]


def _worker_train_and_evaluate(family: str, cfg, params: RunParams, root: Path) -> dict:
    """Runs inside its own OS process (ProcessPoolExecutor). Trains+
    evaluates via the existing, unmodified train_and_evaluate, then
    durably writes this scenario's complete result to its own isolated
    file BEFORE returning -- see learning.feasibility_dataset's module
    docstring for why only this worker-owned file, never the shared
    scenarios.jsonl/rollouts.jsonl, is touched here. The coordinator (the
    process running run_campaign, never a worker) is what appends to the
    shared dataset, after collecting this return value."""
    meta = train_and_evaluate(family, cfg, params, verbose=0)
    result = {"scenario_record": _scenario_record_from_result(family, cfg.blocker_side, cfg, meta),
              "rollout_records": _rollout_records_from_result(cfg, meta)}
    write_scenario_result(family, cfg.blocker_side, cfg.scenario_id(), result, root=root)
    return result


def run_pilot(
        family: str, blocker_side: int, concurrencies: list[int], *,
        envs_per_job: int = 2, timesteps: int = 20_000, device: str = "cpu",
        torch_threads: int = RunParams.torch_threads, root: Path = ARTIFACTS_ROOT,
) -> list[dict]:
    """Times one throwaway benchmark scenario trained at each candidate
    concurrency level (C jobs at once, envs_per_job each), reports
    scenarios/hour at each, and recommends the best -- never touches the
    real dataset (writes under root/_pilot_bench/, safe to delete after).
    See module docstring: the GPU is deliberately not part of this
    benchmark by default -- every job runs on CPU (RunParams.device's own
    default, "cpu" -- see learning.feasibility_train_one) unless device= is
    explicitly overridden to something else. Concurrent jobs contending for
    one shared GPU instead of exercising CPU-core-parallelism would measure
    a completely different thing than what this pipeline is built around,
    so this is a deliberate default, not an oversight. Likewise
    torch_threads defaults low (see RunParams.torch_threads) so concurrent
    jobs don't fight each other over the same cores via PyTorch's own
    (otherwise all-cores-per-process) CPU threading."""
    bench_dir = root / "_pilot_bench"
    cfg = _BENCHMARK_CONFIGS[family](blocker_side=blocker_side, seed=0, mode="robust")
    reports = []

    for concurrency in concurrencies:
        params = RunParams(out="placeholder", timesteps=timesteps, n_envs=envs_per_job,
                            device=device, torch_threads=torch_threads, no_early_stop=True, episodes=0)
        jobs = []
        for i in range(concurrency):
            out = str(bench_dir / f"c{concurrency}_job{i}" / "model")
            jobs.append((family, cfg, dataclasses.replace(params, out=out, force=True), root))

        start = time.monotonic()
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_worker_train_and_evaluate, *job) for job in jobs]
            for f in as_completed(futures):
                f.result()   # surface any worker exception now, at benchmark time
        elapsed = time.monotonic() - start

        scenarios_per_hour = concurrency / elapsed * 3600.0
        reports.append({"concurrency": concurrency, "envs_per_job": envs_per_job,
                         "elapsed_s": elapsed, "scenarios_per_hour": scenarios_per_hour})
        print(f"concurrency={concurrency:3d} x n_envs={envs_per_job}: {elapsed:.1f}s for {concurrency} "
              f"scenarios -> {scenarios_per_hour:.1f} scenarios/hour")

    best = max(reports, key=lambda r: r["scenarios_per_hour"])
    print(f"\nRecommended: --concurrency {best['concurrency']} --n-envs {best['envs_per_job']} "
          f"(~{best['scenarios_per_hour']:.1f} scenarios/hour measured just now on this machine)")
    return reports


def run_campaign(
        family: str, blocker_side: int, n_thetas: int, sampling_seed: int, concurrency: int,
        run_params_template: RunParams, *, doe_method: str = "lhs",
        max_wall_time_hours: float | None = None, root: Path = ARTIFACTS_ROOT,
) -> None:
    """Sample n_thetas, train+evaluate each (skipping any scenario_id that
    already has a durable result -- resumable), and accumulate everything
    into (family, blocker_side)'s dataset. Reconciles any already-completed-
    but-not-yet-appended results first (the case where a previous run's
    coordinator died after a worker finished but before the append), so a
    killed run can always be safely resumed by re-running this same command."""
    sampler = _SAMPLERS[family]
    configs, rejected = sampler(n=n_thetas, blocker_side=blocker_side, seed=sampling_seed, method=doe_method)
    if rejected:
        print(f"{len(rejected)} candidate(s) rejected during sampling:")
        for r in rejected[:10]:
            print(f"  {r.reason}")
        if len(rejected) > 10:
            print(f"  ... and {len(rejected) - 10} more")

    existing_ids = {r["scenario_id"] for r in read_jsonl(scenarios_path(family, blocker_side, root))}

    # Reconciliation pass: a scenario_id can have a durable isolated result file without yet being
    # in the shared dataset if a previous run's coordinator was killed between a worker finishing
    # and this process appending it -- fold those in now, with zero extra compute.
    pending = []
    for cfg in configs:
        sid = cfg.scenario_id()
        if sid in existing_ids:
            continue
        if scenario_result_exists(family, blocker_side, sid, root=root):
            result = read_scenario_result(family, blocker_side, sid, root=root)
            append_scenario_record(family, blocker_side, result["scenario_record"], root=root)
            append_rollout_records(family, blocker_side, result["rollout_records"], root=root)
            existing_ids.add(sid)
            print(f"reconciled already-completed scenario {sid} (no retraining needed)")
            continue
        pending.append(cfg)

    print(f"{len(configs)} sampled, {len(configs) - len(pending)} already done, "
          f"{len(pending)} to train this run.")

    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for cfg in pending:
            if max_wall_time_hours is not None and (time.monotonic() - start) > max_wall_time_hours * 3600:
                print(f"wall-time budget ({max_wall_time_hours}h) reached -- not launching further jobs "
                      f"this run (already-launched jobs are allowed to finish; re-run this same command "
                      f"later to continue with the rest).")
                break
            out = _policy_out_path(root, family, blocker_side, cfg.scenario_id())
            params = dataclasses.replace(run_params_template, out=str(out), force=True,
                                          return_rollout_details=True)
            futures[executor.submit(_worker_train_and_evaluate, family, cfg, params, root)] = cfg

        for future in as_completed(futures):
            cfg = futures[future]
            try:
                result = future.result()
            except Exception as exc:   # noqa: BLE001 -- one scenario's failure must not abort the batch
                print(f"scenario {cfg.scenario_id()} FAILED: {exc!r}")
                continue
            append_scenario_record(family, blocker_side, result["scenario_record"], root=root)
            append_rollout_records(family, blocker_side, result["rollout_records"], root=root)
            rec = result["scenario_record"]
            print(f"done: {cfg.scenario_id()} -- {rec['k']}/{rec['n']} success ({rec['success_rate']:.1%})")


def run_fit_only(family: str, blocker_side: int, out_path: str, *,
                  n_splits: int = 5, n_bootstrap: int = 200, bootstrap_seed: int = 0,
                  root: Path = ARTIFACTS_ROOT) -> None:
    records = read_jsonl(rollouts_path(family, blocker_side, root))
    if not records:
        raise SystemExit(f"no rollout records found for family={family} blocker_side={blocker_side} "
                          f"under {root} -- run `--mode run` first.")
    bounds = _THETA_BOUNDS[family]
    artifact = fit_surrogate(family, blocker_side, records, bounds,
                              n_splits=n_splits, n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed)
    artifact.save(out_path)

    print(f"Best model: {artifact.model_name}")
    print(f"  Brier={artifact.report.brier:.4f}  log_loss={artifact.report.log_loss:.4f}  "
          f"ROC-AUC={artifact.report.roc_auc}")
    print(f"  fit on {artifact.n_scenarios} scenarios / {artifact.n_rollouts} rollouts")
    if artifact.preliminary:
        print(f"  ** PRELIMINARY ** (< {PRELIMINARY_SCENARIO_THRESHOLD} scenarios threshold -- "
              f"see learning.feasibility_surrogate.PRELIMINARY_SCENARIO_THRESHOLD)")
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["pilot", "run", "fit-only"], required=True)
    parser.add_argument("--scenario-family", choices=["cutin", "sandwich"], required=True)
    parser.add_argument("--blocker-side", type=int, choices=[-1, 1], required=True)
    parser.add_argument("--root", type=str, default=str(ARTIFACTS_ROOT))
    parser.add_argument("--device", type=str, default="cpu",
                         help="used by both --mode pilot and --mode run -- defaults to \"cpu\" "
                              "deliberately (NOT RunParams' own \"auto\" default, which would pick "
                              "CUDA on a GPU machine): every job here is small/CPU-bound, and "
                              "concurrent jobs contending for one shared GPU is not what this "
                              "pipeline's concurrency model assumes. Override only to verify that for "
                              "yourself.")
    parser.add_argument("--torch-threads", type=int, default=RunParams.torch_threads,
                         help="PyTorch's own CPU thread count PER JOB -- used by both --mode pilot and "
                              "--mode run. Defaults low (1, see RunParams.torch_threads) because "
                              "PyTorch otherwise claims every core for its own tensor math regardless "
                              "of how tiny this network is: harmless for one job, but with several "
                              "running concurrently (the whole point of this pipeline) every process "
                              "then fights every other one over the same cores via redundant "
                              "threading -- almost certainly why a concurrency level can look SLOWER "
                              "than running jobs one at a time if this is left at PyTorch's own "
                              "all-cores default.")

    # pilot
    parser.add_argument("--pilot-concurrencies", type=str, default="1,2,4",
                         help="comma-separated concurrency levels to benchmark.")
    parser.add_argument("--pilot-envs-per-job", type=int, default=2)
    parser.add_argument("--pilot-timesteps", type=int, default=20_000)

    # run
    parser.add_argument("--n-thetas", type=int, default=24)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--doe-method", choices=["lhs", "sobol"], default="lhs")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-wall-time-hours", type=float, default=None)
    parser.add_argument("--timesteps", type=int, default=RunParams.timesteps)
    parser.add_argument("--n-envs", type=int, default=RunParams.n_envs)
    parser.add_argument("--episodes", type=int, default=RunParams.episodes)
    parser.add_argument("--eval-freq-timesteps", type=int, default=RunParams.eval_freq_timesteps)
    parser.add_argument("--patience-episodes", type=int, default=RunParams.patience_episodes)
    parser.add_argument("--patience", type=int, default=RunParams.patience)
    parser.add_argument("--min-evals", type=int, default=RunParams.min_evals)
    parser.add_argument("--no-early-stop", action="store_true")

    # fit-only
    parser.add_argument("--surrogate-out", type=str, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=0)

    args = parser.parse_args()
    root = Path(args.root)

    if args.mode == "pilot":
        concurrencies = [int(c) for c in args.pilot_concurrencies.split(",")]
        run_pilot(args.scenario_family, args.blocker_side, concurrencies,
                  envs_per_job=args.pilot_envs_per_job, timesteps=args.pilot_timesteps,
                  device=args.device, torch_threads=args.torch_threads, root=root)

    elif args.mode == "run":
        params_template = RunParams(
            out="placeholder", timesteps=args.timesteps, n_envs=args.n_envs, episodes=args.episodes,
            device=args.device, torch_threads=args.torch_threads,
            eval_freq_timesteps=args.eval_freq_timesteps,
            patience_episodes=args.patience_episodes,
            patience=args.patience, min_evals=args.min_evals, no_early_stop=args.no_early_stop,
        )
        run_campaign(args.scenario_family, args.blocker_side, args.n_thetas, args.sampling_seed,
                     args.concurrency, params_template, doe_method=args.doe_method,
                     max_wall_time_hours=args.max_wall_time_hours, root=root)

    elif args.mode == "fit-only":
        if not args.surrogate_out:
            raise SystemExit("--surrogate-out is required for --mode fit-only")
        run_fit_only(args.scenario_family, args.blocker_side, args.surrogate_out,
                     n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed, root=root)


if __name__ == "__main__":
    main()
