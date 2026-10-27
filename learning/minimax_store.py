"""
Persistence for minimax experiments:

    <root>/experiments/<experiment_id>/
        config.json            the full experiment config (+ its scientific hash)
        manifest.json          provenance: git commit, surrogate identities, versions
        iterations.jsonl       one row per exchange iteration
        controller_candidates.jsonl   one row per (w, scenario-set) oracle call
        adversarial_candidates.jsonl  one row per screened adversary candidate
        oracle_refinements.jsonl      one row per PPO refinement
        best_weights.json, summary.json
        checkpoints/state.json resumable ExchangeState (+ oracle evidence, counters)
        cache/<hh>/<key>.json  one closed-loop EPISODE per file (atomic), keyed on every
                               cost-affecting input -- see episode_cache_key
        oracle/                PPO refinement results + policies (learning.minimax_oracle)
        plots/

Single writer: only the coordinator process writes here; worker processes
return episode records and never touch disk. JSON/JSONL helpers are the
feasibility dataset's own (atomic replace, fsync'd single-line appends,
truncated-last-line tolerant reads).
"""

import hashlib
import json
import subprocess
from pathlib import Path

from learning.feasibility_dataset import _atomic_write_bytes, _append_line, read_jsonl

# Config keys that may change on resume without changing the science (budgets, parallelism, paths).
NON_SCIENTIFIC_KEYS = ("n_workers", "artifacts_root", "experiment_id")
NON_SCIENTIFIC_EXCHANGE_KEYS = ("max_iterations", "max_wall_s", "max_oracle_evals", "stagnation_iterations")


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def scientific_config(config: dict) -> dict:
    sci = {k: v for k, v in config.items() if k not in NON_SCIENTIFIC_KEYS}
    if "exchange" in sci:
        sci["exchange"] = {k: v for k, v in sci["exchange"].items() if k not in NON_SCIENTIFIC_EXCHANGE_KEYS}
    return sci


def episode_cache_key(controller_spec: dict, w, scenario_key: str, realization_seed: int, cost_config: dict,
                      evaluator_version: str, schema_version: int) -> str:
    """Every input that affects one episode's cost. w is hashed at full float
    precision (repr round-trips exactly)."""
    return canonical_hash({"controller": controller_spec, "w": [repr(float(x)) for x in w],
                           "scenario": scenario_key, "seed": int(realization_seed), "cost": cost_config,
                           "evaluator": evaluator_version, "schema": schema_version})


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        return out.stdout.strip() + ("+dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


class ExperimentStore:
    LOGS = ("iterations", "controller_candidates", "adversarial_candidates", "oracle_refinements")

    def __init__(self, root: Path | str, experiment_id: str):
        self.dir = Path(root) / "experiments" / experiment_id
        self.cache_hits = 0
        self.cache_writes = 0

    # -- files --
    def path(self, name: str) -> Path:
        return self.dir / name

    def exists(self) -> bool:
        return self.path("config.json").exists()

    def write_json(self, name: str, obj) -> None:
        _atomic_write_bytes(self.path(name), json.dumps(obj, indent=2, sort_keys=True, default=str).encode())

    def read_json(self, name: str):
        p = self.path(name)
        return json.loads(p.read_text()) if p.exists() else None

    def append(self, log: str, record: dict) -> None:
        assert log in self.LOGS, log
        _append_line(self.path(f"{log}.jsonl"), record)

    def read_log(self, log: str) -> list[dict]:
        return read_jsonl(self.path(f"{log}.jsonl"))

    def truncate_logs_to(self, completed_iterations: int) -> None:
        """On resume: drop log rows from any iteration that never reached a
        checkpoint (iteration >= completed_iterations), so replaying it does
        not duplicate rows. Rows from the pre-loop seeding call (iteration -1)
        are kept only once iteration 0 completed."""
        for log in self.LOGS:
            rows = self.read_log(log)
            keep = [r for r in rows if completed_iterations > 0 and r.get("iteration", -1) < completed_iterations]
            if len(keep) != len(rows):
                data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in keep).encode()
                _atomic_write_bytes(self.path(f"{log}.jsonl"), data)

    # -- config --
    def init_config(self, config: dict) -> str:
        h = canonical_hash(scientific_config(config))
        self.write_json("config.json", {"config": config, "scientific_hash": h})
        return h

    def check_config(self, config: dict) -> str:
        stored = self.read_json("config.json")
        assert stored is not None, f"{self.dir} has no config.json"
        h = canonical_hash(scientific_config(config))
        if h != stored["scientific_hash"]:
            raise ValueError(f"experiment {self.dir.name}: the scientific configuration differs from the stored "
                             f"one (hash {h[:12]} vs {stored['scientific_hash'][:12]}). Start a new experiment "
                             f"instead of resuming with a changed setup.")
        return h

    # -- episode cache --
    def _cache_path(self, key: str) -> Path:
        return self.dir / "cache" / key[:2] / f"{key}.json"

    def cache_get(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if p.exists():
            self.cache_hits += 1
            return json.loads(p.read_text())
        return None

    def cache_put(self, key: str, record: dict) -> None:
        _atomic_write_bytes(self._cache_path(key), json.dumps(record, sort_keys=True).encode())
        self.cache_writes += 1
