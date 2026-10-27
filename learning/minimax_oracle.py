"""
Active feasibility-oracle refinement: send an UNCERTAIN, high-cost adversary
candidate back to the PPO feasibility oracle (learning.feasibility_train_one.
train_and_evaluate, reused unchanged) to try to DEMONSTRATE that it is
solvable.

Protocol per scenario:
  1. Train one PPO policy per seed in `ppo_seeds` (each with train_and_evaluate's
     own early stopping + held-out evaluation, `selection_episodes` episodes).
  2. Select the policy with the best selection-eval success rate.
  3. Re-evaluate ONLY the selected policy on an independent realization stream
     (worker_rank = validation_worker_rank, never used for training or
     selection) for `validation_episodes` episodes. The max over seeds in step
     2 is optimistically biased; the step-3 estimate is not.
  4. Status from the step-3 (k, n) via minimax_feasibility.demonstrated
     (Wilson 95% lower bound >= tau_demonstrated). Otherwise the scenario stays
     UNVERIFIED -- never "infeasible".

Results go to the experiment's own oracle/ directory (one atomic JSON per
scenario, checked before any retraining, so a resumed experiment never
retrains) -- the original feasibility datasets are never touched, and the
global surrogate is never refit automatically.
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

from learning.closed_loop import ScenarioSpec, FAMILIES
from learning.minimax_feasibility import FeasibilityScreen


@dataclass(frozen=True)
class OracleRefinementConfig:
    enabled: bool = False
    max_refinements: int = 2              # per experiment (expensive: each is len(ppo_seeds) PPO trainings)
    max_per_iteration: int = 1
    ppo_seeds: tuple[int, ...] = (0, 1)
    scenario_seed: int = 0                # base realization seed for training/selection envs
    timesteps: int = 2_000_000
    n_envs: int = 4
    selection_episodes: int = 50
    validation_episodes: int = 100
    validation_worker_rank: int = 2_000_000

    @staticmethod
    def from_dict(d: dict) -> "OracleRefinementConfig":
        d = dict(d)
        if "ppo_seeds" in d:
            d["ppo_seeds"] = tuple(d["ppo_seeds"])
        return OracleRefinementConfig(**d)


def _default_train_fn(family, cfg, params, verbose=0):
    from learning.feasibility_train_one import train_and_evaluate
    return train_and_evaluate(family, cfg, params, verbose=verbose)


def _default_validate_fn(model_path: str, spec: ScenarioSpec, cfg_seed: int, episodes: int, worker_rank: int) -> list[dict]:
    from stable_baselines3 import PPO
    from learning.env import EgoTrafficEnv
    from learning.eval_batch import run_episodes
    model = PPO.load(model_path, device="cpu")
    env = EgoTrafficEnv(arena=FAMILIES[spec.family][1](spec.config(seed=cfg_seed)), mode="distribution",
                        worker_rank=worker_rank)
    return run_episodes(model, env, episodes, deterministic=True)


class OracleRefiner:
    def __init__(self, config: OracleRefinementConfig, out_dir: Path, *, train_fn=None, validate_fn=None):
        self.config = config
        self.out_dir = Path(out_dir)
        self.train_fn = train_fn or _default_train_fn
        self.validate_fn = validate_fn or _default_validate_fn
        self.n_used = 0

    def result_path(self, spec: ScenarioSpec) -> Path:
        return self.out_dir / spec.stratum.label / f"{spec.scenario_id}.json"

    def budget_left(self) -> int:
        return max(0, self.config.max_refinements - self.n_used)

    def refine(self, spec: ScenarioSpec, screen: FeasibilityScreen, *, iteration: int, mpc_cost: float | None) -> dict:
        """Run (or reload) the refinement for `spec`, register the evidence on
        `screen`, and return the provenance record."""
        from learning.feasibility_dataset import _atomic_write_bytes
        from learning.feasibility_train_one import RunParams

        path = self.result_path(spec)
        if path.exists():
            rec = json.loads(path.read_text())
        else:
            self.n_used += 1
            cfg = spec.config(seed=self.config.scenario_seed)
            seeds = []
            for s in self.config.ppo_seeds:
                out = path.parent / spec.scenario_id / f"ppo_seed{s}" / "model"
                out.parent.mkdir(parents=True, exist_ok=True)
                params = RunParams(out=str(out), timesteps=self.config.timesteps, n_envs=self.config.n_envs,
                                   ppo_seed=s, episodes=self.config.selection_episodes, force=True)
                meta = self.train_fn(spec.family, cfg, params, verbose=0)
                summ = meta["evaluation"]["summary"]
                seeds.append({"ppo_seed": s, "model": str(out), "selection_success_rate": summ["success_rate"],
                              "selection_k": summ["n_success"], "selection_n": summ["n_episodes"],
                              "stop_reason": meta.get("stop_reason"), "timesteps_actual": meta.get("timesteps_actual"),
                              "evaluated_checkpoint": meta.get("evaluated_checkpoint")})
            best = max(seeds, key=lambda r: (r["selection_success_rate"], -r["ppo_seed"]))
            results = self.validate_fn(best["model"], spec, self.config.scenario_seed,
                                       self.config.validation_episodes, self.config.validation_worker_rank)
            k, n = sum(bool(r["success"]) for r in results), len(results)
            rec = {"scenario": spec.to_dict(), "iteration": iteration, "mpc_cost_at_referral": mpc_cost,
                   "config": asdict(self.config), "per_seed": seeds, "selected_ppo_seed": best["ppo_seed"],
                   "validation": {"k": k, "n": n, "worker_rank": self.config.validation_worker_rank,
                                  "outcomes": [r["outcome"] for r in results]},
                   "note": "selection success rates are max-over-seeds (optimistic); use validation only"}
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(path, json.dumps(rec, indent=2).encode())
        status = screen.record_oracle_evidence(spec, rec["validation"]["k"], rec["validation"]["n"],
                                               source=str(path))
        return {**rec, "status": status}
