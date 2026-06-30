"""
Run deterministic/adaptive scenario search for MPC risk-weight identification.

Required:
    PYTHONPATH=src:. python3 script/weight_sensitivity_demo.py
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from search.weight_sensitivity import (
    adaptive_theta_search,
    default_weight_set,
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _json_default(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _summary_record(result: dict) -> dict:
    return {
        "theta": asdict(result["theta"]),
        "score": result["score"],
        "weights_list": [asdict(w) for w in result["weights_list"]],
        "descriptors": result["descriptors"],
    }


def main():
    initial_samples = _env_int("WS_INITIAL_SAMPLES", 32)
    refinement_rounds = _env_int("WS_REFINEMENT_ROUNDS", 3)
    children_per_parent = _env_int("WS_CHILDREN_PER_PARENT", 4)
    top_k = _env_int("WS_TOP_K", 8)
    t_end = _env_float("WS_T_END", 8.0)
    dt = _env_float("WS_DT", 0.05)
    mpc_interval = _env_int("WS_MPC_INTERVAL", 1)
    seed = _env_int("WS_SEED", 0)
    output_dir = Path(os.environ.get("WS_OUTPUT_DIR", "outputs/weight_sensitivity"))
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = default_weight_set()
    print(
        "Weight-sensitivity search: "
        f"initial={initial_samples}, rounds={refinement_rounds}, "
        f"children/parent={children_per_parent}, top_k={top_k}, "
        f"weights={len(weights)}, t_end={t_end}, dt={dt}, seed={seed}"
    )

    results = adaptive_theta_search(
        weights_list=weights,
        initial_samples=initial_samples,
        refinement_rounds=refinement_rounds,
        children_per_parent=children_per_parent,
        top_k=top_k,
        t_end=t_end,
        dt=dt,
        mpc_interval=mpc_interval,
        seed=seed,
        verbose=True,
    )

    print("\nTop informative scenarios:")
    for rank, result in enumerate(results[:top_k], start=1):
        theta = result["theta"]
        score = result["score"]
        print(
            f"  {rank:2d}. score={score['usefulness_score']:+.3f} "
            f"activation={score['risk_activation_score']:.2f} "
            f"sensitivity={score['weight_sensitivity_score']:.2f} "
            f"pareto={score['pareto_diversity_score']:.2f} "
            f"A={theta.kappa_A:.4f} omega={theta.kappa_omega:.4f} "
            f"mu={theta.mu:.2f} v0={theta.ego_v0:.1f} "
            f"front=({theta.front_distance:.1f}m, {theta.front_speed:.1f}m/s, "
            f"decel={theta.front_decel:.1f}) "
            f"rear=({theta.rear_distance:.1f}m, {theta.rear_speed:.1f}m/s)"
        )

    summary = [_summary_record(result) for result in results]
    summary_path = output_dir / "weight_sensitivity_summary.json"
    full_path = output_dir / "weight_sensitivity_results.pkl"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    with full_path.open("wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved full results: {full_path}")


if __name__ == "__main__":
    main()
