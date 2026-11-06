"""
Experiment organization for PPO feasibility-oracle studies (docs/description.md §23): one directory per
experiment, holding everything needed to reproduce and interpret it. Reporting only: this file trains
through learning.feasibility_train_one unchanged and nothing here alters the PPO learning problem.

    results/ppo_feasibility/<name>/
        manifest.json     identity, provenance, definitions, data class
        configs/          experiment.json, scenarios.jsonl (every sampled scenario, trained or not)
        training/<family>/side_<pos1|neg1>/<scenario_id>/seed_<n>/   model.zip, model.meta.json, metrics.jsonl
        evaluations/<same path>/<eval_mode>_<protocol>.json          one independent evaluation record
        trajectories/<same path>/<eval_mode>/ep_<i>.npz              selected rollouts
        figures/  animations/  reports/                              derived, safe to regenerate

Seed streams. A realization is derive_seed(cfg.seed, worker_rank, episode_index), so worker ranks are the
streams. A robust policy trains on a new realization every reset, on worker ranks TRAINING_RANK + i. An
exact policy has one realization to train on. Evaluation streams are disjoint from training: experiments
here use the ranks below, and skip learning.feasibility_train_one's built-in evaluation (episodes=0).

Usage:
    python -m learning.ppo_experiment init --config path/to/experiment.json
    python -m learning.ppo_experiment train --experiment results/ppo_feasibility/<name> [--concurrency N]
"""

import argparse
import dataclasses
import hashlib
import itertools
import json
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path

from learning.env import DEFAULT_PPO_GAMMA, DEFAULT_REWARD_CONFIG, DT, REWARD_VERSION
from learning.feasibility_cutin import CutinConfig, CutinRuntime, THETA_BOUNDS_CUTIN, validate_cutin
from learning.feasibility_dataset import _atomic_write_bytes, _side_dir_name, read_jsonl
from learning.feasibility_sampling import sample_cutin, sample_sandwich
from learning.feasibility_sandwich import SandwichConfig, SandwichRuntime, THETA_BOUNDS_SANDWICH, validate_sandwich
from learning.minimax_store import git_commit
from learning.physical_eval import PHYSICAL_EVALUATOR_VERSION
from learning.ppo_metrics import TRAINING_RANK, VALIDATION_RANK, training_realization
from learning.vehicle_physics import VEHICLE_PHYSICS_VERSION
from model.car.config import VehicleParameters

EXPERIMENTS_ROOT = Path("results/ppo_feasibility")
SUBDIRS = ("configs", "training", "evaluations", "trajectories", "figures", "animations", "reports")
SCHEMA_VERSION = "ppo-experiment-v1"

# TRAINING_RANK (learning.ppo_metrics): the first training worker; worker i uses TRAINING_RANK + i
EARLY_STOP_RANK = 1_000_000    # learning.feasibility_train_one: best checkpoint and early stopping
# VALIDATION_RANK (learning.ppo_metrics): learning curves; never updates the policy or picks a checkpoint
SELECTION_RANK = 1_200_000     # picks one PPO seed per scenario, so admission data stay unselected
TEST_RANK = 1_300_000          # the only stream behind reported rates and admission decisions
EVALUATION_RANKS = (VALIDATION_RANK, SELECTION_RANK, TEST_RANK)

ADMISSION_P0 = 0.8   # == learning.minimax_feasibility.ScreeningConfig.tau_demonstrated
DATA_CLASSES = ("experiment", "smoke", "synthetic_test_fixture")   # anything but "experiment" is watermarked

FAMILIES = {"cutin": (CutinConfig, CutinRuntime, THETA_BOUNDS_CUTIN, validate_cutin, sample_cutin),
            "sandwich": (SandwichConfig, SandwichRuntime, THETA_BOUNDS_SANDWICH, validate_sandwich, sample_sandwich)}
_LANE_NUM, _EGO_LANE = 3, 1   # the standard road, as learning.feasibility_sampling

DEFINITIONS = {
    "observed_success": "rollout reached the scenario goal with no collision, rollover or road departure "
                        "(EgoTrafficEnv info['success']); a timeout is never a success",
    "estimated_success_probability": "p_hat = k/N over independent held-out rollouts of one selected policy",
    "statistical_admission": f"admitted: two-sided 95% Wilson lower bound >= p0 = {ADMISSION_P0}; "
                             f"not admitted: Wilson upper bound < p0; uncertain: otherwise",
    "surrogate_prediction": "learning.feasibility_surrogate output; a model estimate, never a rollout",
    "physical_feasibility": "not established here: admission demonstrates that PPO solved the scenario, "
                            "and failure to admit does not show the scenario is physically impossible",
}


def json_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def write_json(path: Path, obj) -> None:
    _atomic_write_bytes(Path(path), json.dumps(obj, indent=2, sort_keys=True, default=str).encode("utf-8"))


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    _atomic_write_bytes(Path(path), "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode("utf-8"))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def software_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "torch", "gymnasium", "stable-baselines3", "matplotlib", "scikit-learn"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def provenance() -> dict:
    return {"git_commit": git_commit(), "software": software_versions(), "dt_s": DT,
            "vehicle_parameters": dataclasses.asdict(VehicleParameters()),
            "vehicle_physics_version": VEHICLE_PHYSICS_VERSION,
            "physical_evaluator_version": PHYSICAL_EVALUATOR_VERSION}


@dataclass(frozen=True)
class ExperimentConfig:
    """strata: [family, blocker_side] pairs. design: "lhs" or "sobol" (learning.feasibility_sampling, n_thetas
    per stratum) or "grid": the product of grid_levels {field: values}, every other field from grid_fixed or
    the family default. That is the only design whose maps are true two-dimensional slices."""
    name: str
    strata: list = field(default_factory=lambda: [["cutin", 1], ["cutin", -1], ["sandwich", 1], ["sandwich", -1]])
    design: str = "lhs"
    n_thetas: int = 10
    sampling_seed: int = 0
    grid_levels: dict = field(default_factory=dict)
    grid_fixed: dict = field(default_factory=dict)
    training_mode: str = "robust"
    ppo_seeds: list = field(default_factory=lambda: [0, 1, 2])
    timesteps: int = 2_000_000
    n_envs: int = 4
    gamma: float = DEFAULT_PPO_GAMMA
    eval_freq_timesteps: int = 100_000
    validation_episodes: int = 20
    no_early_stop: bool = False
    rollover_event: str = "physical_v2"
    eval_modes: list = field(default_factory=lambda: ["robust", "exact"])
    selection_episodes: int = 20
    test_episodes: int = 100
    deterministic: bool = True
    trajectories_per_outcome: int = 3
    max_ego_speed_mps: float | None = None

    def __post_init__(self):
        assert self.design in ("lhs", "sobol", "grid"), self.design
        assert self.training_mode in ("exact", "robust") and set(self.eval_modes) <= {"exact", "robust"}
        assert all(f in FAMILIES and s in (-1, 1) for f, s in self.strata), self.strata


def arena_config(family: str, row: dict, mode: str | None = None):
    """The family config from a scenarios.jsonl row or a checkpoint's meta.json (same field names)."""
    return FAMILIES[family][0](blocker_side=row["blocker_side"], mode=mode or row["mode"], seed=row["scenario_seed"],
                               goal_distance_m=row["goal_distance_m"], max_episode_seconds=row["max_episode_seconds"],
                               min_progress_m=row["min_progress_m"], max_ego_speed_mps=row.get("max_ego_speed_mps"), **row["theta"])


def sample_scenarios(c: ExperimentConfig) -> list[dict]:
    """Every sampled scenario as a row. Sampling and validation are the families' own."""
    rows = []
    for family, side in c.strata:
        cfg_cls, _, _, validate, sampler = FAMILIES[family]
        if c.design == "grid":
            names = list(c.grid_levels)
            cfgs = []
            for i, values in enumerate(itertools.product(*c.grid_levels.values())):
                fixed = {k: v for k, v in c.grid_fixed.items() if k in FAMILIES[family][2]}
                cfg = cfg_cls(blocker_side=side, seed=c.sampling_seed + i, mode="robust", **fixed, **dict(zip(names, values)))
                validate(cfg, lane_num=_LANE_NUM, ego_lane=_EGO_LANE)
                cfgs.append(cfg)
        else:
            cfgs, _ = sampler(c.n_thetas, side, c.sampling_seed, method=c.design)
        for cfg in cfgs:
            cfg = dataclasses.replace(cfg, mode=c.training_mode)
            rows.append({"family": family, "blocker_side": side, "mode": cfg.mode, "scenario_id": cfg.scenario_id(),
                         "scenario_seed": cfg.seed, "theta": cfg.theta(), "goal_distance_m": cfg.goal_distance_m,
                         "max_episode_seconds": cfg.max_episode_seconds, "min_progress_m": cfg.min_progress_m})
    return rows


def create_experiment(config: ExperimentConfig, root: Path = EXPERIMENTS_ROOT, data_class: str = "experiment",
                      scenarios: list[dict] | None = None) -> Path:
    """Refuses an existing name: results are never silently replaced."""
    assert data_class in DATA_CLASSES, data_class
    exp = Path(root) / config.name
    if exp.exists():
        raise SystemExit(f"{exp} already exists -- choose another experiment name; results are never overwritten.")
    for sub in SUBDIRS:
        (exp / sub).mkdir(parents=True)
    write_json(exp / "configs/experiment.json", dataclasses.asdict(config))
    write_jsonl(exp / "configs/scenarios.jsonl", sample_scenarios(config) if scenarios is None else scenarios)
    write_json(exp / "manifest.json", {
        "schema": SCHEMA_VERSION, "name": config.name, "data_class": data_class, "provenance": provenance(),
        "config": dataclasses.asdict(config), "definitions": DEFINITIONS, "admission_p0": ADMISSION_P0,
        "seed_streams": {"training": TRAINING_RANK, "early_stopping": EARLY_STOP_RANK, "validation": VALIDATION_RANK,
                         "selection": SELECTION_RANK, "test": TEST_RANK},
        "training_realization": training_realization(config.training_mode), "reward_version": REWARD_VERSION,
        "reward_config": dataclasses.asdict(DEFAULT_REWARD_CONFIG)})
    return exp


def load_config(exp: Path) -> ExperimentConfig:
    return ExperimentConfig(**read_json(Path(exp) / "configs/experiment.json"))


def load_scenarios(exp: Path) -> list[dict]:
    return read_jsonl(Path(exp) / "configs/scenarios.jsonl")


def run_path(family: str, side: int, scenario_id: str, ppo_seed: int) -> Path:
    return Path(family) / _side_dir_name(side) / scenario_id / f"seed_{ppo_seed}"


def _train_job(exp: str, row: dict, ppo_seed: int) -> str:
    import torch
    from learning.feasibility_train_one import RunParams, train_and_evaluate
    c = load_config(exp)
    out = Path(exp) / "training" / run_path(row["family"], row["blocker_side"], row["scenario_id"], ppo_seed)
    out.mkdir(parents=True, exist_ok=True)
    params = RunParams(out=str(out / "model"), timesteps=c.timesteps, n_envs=c.n_envs, ppo_seed=ppo_seed, gamma=c.gamma,
                       eval_freq_timesteps=c.eval_freq_timesteps, no_early_stop=c.no_early_stop,
                       rollover_event=c.rollover_event, episodes=0, metrics_path=str(out / "metrics.jsonl"),
                       validation_episodes=c.validation_episodes)
    torch.set_num_threads(params.torch_threads)
    cfg = dataclasses.replace(arena_config(row["family"], row), max_ego_speed_mps=c.max_ego_speed_mps)
    train_and_evaluate(row["family"], cfg, params, verbose=0)
    return str(out)


def train_all(exp: Path, concurrency: int = 1) -> list[str]:
    """Trains every (scenario, PPO seed) with no finished model yet, so an interrupted run resumes."""
    exp, c = Path(exp), load_config(exp)
    jobs = [(str(exp), row, seed) for row in load_scenarios(exp) for seed in c.ppo_seeds
            if not (exp / "training" / run_path(row["family"], row["blocker_side"], row["scenario_id"], seed) /
                    "model.meta.json").exists()]
    done = []
    with ProcessPoolExecutor(max_workers=concurrency, mp_context=multiprocessing.get_context("spawn")) as pool:
        for fut in as_completed([pool.submit(_train_job, *job) for job in jobs]):
            done.append(fut.result())
            print(f"[{len(done)}/{len(jobs)}] {done[-1]}", flush=True)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="create the experiment directory and sample its scenarios; trains nothing")
    p.add_argument("--config", required=True, help="JSON with ExperimentConfig fields")
    p.add_argument("--root", default=str(EXPERIMENTS_ROOT))
    p.add_argument("--data-class", choices=DATA_CLASSES, default="experiment")
    p.add_argument("--scenarios", default=None, help="JSONL of scenario rows to reuse exactly, instead of sampling")
    p = sub.add_parser("train", help="train every (scenario, seed) not finished yet")
    p.add_argument("--experiment", required=True)
    p.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.command == "init":
        scenarios = None if args.scenarios is None else read_jsonl(Path(args.scenarios))
        assert scenarios != [], args.scenarios
        exp = create_experiment(ExperimentConfig(**read_json(args.config)), Path(args.root), args.data_class, scenarios=scenarios)
        print(f"{exp}: {len(load_scenarios(exp))} scenarios sampled; nothing trained.")
    else:
        train_all(Path(args.experiment), args.concurrency)


if __name__ == "__main__":
    main()
