"""
Durable, resumable storage for the feasibility pipeline's accumulated
results -- one directory tree per (scenario_family, blocker_side), two
JSON-Lines files (per-rollout binary outcomes, and per-scenario aggregate
summaries), plus isolated per-scenario result files workers write before
the coordinator ever touches the shared dataset.

Single-writer design (see learning.feasibility_pipeline): worker processes
(each training+evaluating one theta) call write_scenario_result() only --
that's an atomic, per-scenario-id file, so two workers can never race on the
same path. Only the coordinator (the main pipeline process) ever calls
append_rollout_records()/append_scenario_record(), and only after collecting
a worker's finished result -- so the shared scenarios.jsonl/rollouts.jsonl
files never see concurrent writers, which is what actually prevents
corruption (atomic replace alone does not, if two processes could still
both decide to append at once).
"""

import json
import os
import tempfile
from pathlib import Path

from learning.feasibility_common import BlockerSide, ScenarioFamily

ARTIFACTS_ROOT = Path("artifacts/feasibility")


def _side_dir_name(blocker_side: BlockerSide) -> str:
    return f"side_{'pos1' if blocker_side == 1 else 'neg1'}"


def dataset_dir(family: ScenarioFamily, blocker_side: BlockerSide, root: Path = ARTIFACTS_ROOT) -> Path:
    return root / "datasets" / family / _side_dir_name(blocker_side)


def evaluations_dir(family: ScenarioFamily, blocker_side: BlockerSide, root: Path = ARTIFACTS_ROOT) -> Path:
    return root / "evaluations" / family / _side_dir_name(blocker_side)


def scenarios_path(family: ScenarioFamily, blocker_side: BlockerSide, root: Path = ARTIFACTS_ROOT) -> Path:
    return dataset_dir(family, blocker_side, root) / "scenarios.jsonl"


def rollouts_path(family: ScenarioFamily, blocker_side: BlockerSide, root: Path = ARTIFACTS_ROOT) -> Path:
    return dataset_dir(family, blocker_side, root) / "rollouts.jsonl"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` via a same-directory temp file + os.replace --
    a reader can never observe a partially-written file, and a crash mid-
    write leaves the original (or nothing, if this is the first write) untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _append_line(path: Path, record: dict) -> None:
    """A single complete JSON line, written with one write() call and
    fsync'd -- appends never rewrite the whole file (see module docstring:
    this is safe because only one process, the coordinator, ever appends to
    a given path), so a crash mid-write can at worst leave one truncated
    trailing line, which read_jsonl skips over rather than failing on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    """All complete records in `path` -- [] if it doesn't exist yet. A
    malformed final line (e.g. from a crash mid-append) is skipped, not
    raised, so a resumed run can still read everything written so far."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError(f"{path}: malformed JSONL at line {i + 1} (not the final line -- "
                                  f"this isn't the truncated-last-write case this function tolerates)")
            # else: silently-tolerated truncated final line, per this function's own contract.
    return records


def append_scenario_record(family: ScenarioFamily, blocker_side: BlockerSide, record: dict,
                            root: Path = ARTIFACTS_ROOT) -> None:
    """Append one per-theta aggregate record (k_i, n_i, p_hat, wilson CI,
    failure counts, checkpoint paths, fingerprint, ...) to that (family,
    side)'s scenarios.jsonl. Coordinator-only -- see module docstring."""
    _append_line(scenarios_path(family, blocker_side, root), record)


def append_rollout_records(family: ScenarioFamily, blocker_side: BlockerSide, records: list[dict],
                            root: Path = ARTIFACTS_ROOT) -> None:
    """Append one binary-outcome record per rollout episode (scenario_id,
    seed, Y, outcome_reason, diagnostics) -- this is what learning.
    feasibility_surrogate actually trains on. Coordinator-only."""
    path = rollouts_path(family, blocker_side, root)
    for record in records:
        _append_line(path, record)


def scenario_result_path(family: ScenarioFamily, blocker_side: BlockerSide, scenario_id: str,
                          root: Path = ARTIFACTS_ROOT) -> Path:
    return evaluations_dir(family, blocker_side, root) / f"{scenario_id}.result.json"


def scenario_result_exists(family: ScenarioFamily, blocker_side: BlockerSide, scenario_id: str,
                            root: Path = ARTIFACTS_ROOT) -> bool:
    """True if this scenario_id already has a completed, durably-written
    result -- what the pipeline's resume logic checks before re-training a
    theta a batch already finished."""
    return scenario_result_path(family, blocker_side, scenario_id, root).exists()


def write_scenario_result(family: ScenarioFamily, blocker_side: BlockerSide, scenario_id: str,
                           record: dict, root: Path = ARTIFACTS_ROOT) -> None:
    """Durably write one scenario's complete result -- called by a WORKER,
    exactly once per scenario_id, before it returns. Atomic (tempfile +
    replace): two workers can't corrupt this even if (they never should,
    but) both somehow targeted the same scenario_id, since each write is
    independently atomic and the last one simply wins cleanly."""
    path = scenario_result_path(family, blocker_side, scenario_id, root)
    _atomic_write_bytes(path, json.dumps(record, sort_keys=True, indent=2).encode("utf-8"))


def read_scenario_result(family: ScenarioFamily, blocker_side: BlockerSide, scenario_id: str,
                          root: Path = ARTIFACTS_ROOT) -> dict | None:
    path = scenario_result_path(family, blocker_side, scenario_id, root)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
