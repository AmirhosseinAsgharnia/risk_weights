"""Scenario search for identifying risk-weight-sensitive MPC behavior.

This module does not replace the MPC. It samples candidate highway scenarios,
runs the existing controller under multiple risk-weight vectors, and scores
scenarios by how much those weights change the resulting behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

import dynamics.env_dynamics as env_dynamics
import environment.env_scenario as env_scenario
import risks.env_risk as env_risk
from config.config import L_veh, W_c
from controller.mpc import MPC
from environment.scenarios import SandwichScenario


@dataclass(frozen=True)
class ScenarioTheta:
    kappa_A: float
    kappa_omega: float
    mu: float
    ego_v0: float
    front_distance: float
    front_speed: float
    front_decel: float
    rear_distance: float
    rear_speed: float
    lane_width: float


@dataclass(frozen=True)
class RiskWeights:
    slip: float
    roll: float
    c2c: float
    ld: float


THETA_BOUNDS = {
    # These are scenario-search ranges, not validated physical operating limits.
    "kappa_A": (0.005, 0.06),
    "kappa_omega": (0.005, 0.10),
    "mu": (0.35, 1.0),
    "ego_v0": (15.0, 35.0),
    "front_distance": (10.0, 60.0),
    "front_speed": (5.0, 35.0),
    "front_decel": (0.0, 8.0),
    "rear_distance": (8.0, 50.0),
    "rear_speed": (15.0, 40.0),
    "lane_width": (3.2, 4.2),
}


def _weights_dict(weights: RiskWeights) -> dict[str, float]:
    return asdict(weights)


def make_sandwich_scenario(theta: ScenarioTheta):
    """Convert ``ScenarioTheta`` into the repo's existing ``SandwichScenario``."""
    return SandwichScenario(
        n_lanes=3,
        ego_lane=1,
        lane_width=theta.lane_width,
        kappa_A=theta.kappa_A,
        kappa_omega=theta.kappa_omega,
        ego_ey=0.0,
        ego_epsi=0.0,
        v0=theta.ego_v0,
        vehicles=[
            {
                "x_rel": theta.front_distance,
                "v_x": theta.front_speed,
                "lane": 1,
                "valid": 1,
                "accel": -abs(theta.front_decel),
            },
            {
                "x_rel": -theta.rear_distance,
                "v_x": theta.rear_speed,
                "lane": 1,
                "valid": 1,
                "accel": 0.0,
            },
        ],
    )


def generate_weight_grid(
    values=(0.0, 0.25, 0.5, 1.0, 2.0),
    normalize=True,
) -> list[RiskWeights]:
    """Generate candidate risk-weight vectors."""
    out: list[RiskWeights] = []
    seen = set()
    for slip in values:
        for roll in values:
            for c2c in values:
                for ld in values:
                    raw = np.array([slip, roll, c2c, ld], dtype=float)
                    total = float(raw.sum())
                    if total <= 0.0:
                        continue
                    vec = raw / total if normalize else raw
                    key = tuple(np.round(vec, 12))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(RiskWeights(*map(float, vec)))
    return out


def default_weight_set() -> list[RiskWeights]:
    """Return a compact set of interpretable candidate weights."""
    return [
        RiskWeights(1.0, 0.0, 0.0, 0.0),
        RiskWeights(0.0, 1.0, 0.0, 0.0),
        RiskWeights(0.0, 0.0, 1.0, 0.0),
        RiskWeights(0.0, 0.0, 0.0, 1.0),
        RiskWeights(0.25, 0.25, 0.25, 0.25),
        RiskWeights(0.5, 0.0, 0.5, 0.0),
        RiskWeights(0.0, 0.0, 0.5, 0.5),
        RiskWeights(0.5, 0.5, 0.0, 0.0),
    ]


def _set_mu(mu_value: float):
    old = (env_dynamics.mu, env_risk.mu)
    env_dynamics.mu = float(mu_value)
    env_risk.mu = float(mu_value)
    return old


def _restore_mu(old):
    env_dynamics.mu, env_risk.mu = old


def _vehicle_snapshot(vehicles):
    return [
        {k: float(v[k]) if k in {"s", "y_abs", "v_x", "accel"} else v[k]
         for k in ("s", "y_abs", "v_x", "accel", "lane", "valid") if k in v}
        for v in vehicles
    ]


def run_mpc_rollout(
    theta: ScenarioTheta,
    weights: RiskWeights,
    *,
    t_end: float,
    dt: float,
    mpc_interval: int = 1,
    verbose: bool = False,
) -> dict:
    """Simulate one scenario/weight pair with the existing MPC controller."""
    sc = make_sandwich_scenario(theta)
    sc.t_end = float(t_end)
    sc.dt = float(dt)

    old_mu = _set_mu(theta.mu)
    history = []
    vehicle_history = []
    status = {"ok": True, "reason": "completed"}
    try:
        mpc = MPC(risk_weights=_weights_dict(weights), v_ref=theta.ego_v0, w_v=0.001)
        ego_state = sc.ego_init_state()
        ego_lane = sc.ego_lane
        s_ego = (np.pi / 2.0) / sc.kappa_omega if sc.kappa_omega > 0 else 0.0
        ego_state[1] = float(env_dynamics.theta_road(s_ego, sc.kappa_A, sc.kappa_omega))
        vehicles = env_scenario.init_vehicles(sc, s_ego_init=s_ego)

        n_steps = int(sc.t_end / sc.dt)
        every = max(1, int(mpc_interval))
        a_opt, delta_opt = 0.0, 0.0

        for step_idx in range(n_steps):
            t_now = step_idx * sc.dt
            if step_idx % every == 0:
                a_opt, delta_opt = mpc.solve(ego_state, s_ego, ego_lane, vehicles, sc)

            ego_fut, sur_fut = mpc._predict(
                ego_state, s_ego, ego_lane, vehicles, sc, a_opt, delta_opt
            )
            J, ps, pr, pc, pl = env_risk.total_cost(
                ego_fut, sur_fut, a_opt, delta_opt, sc, _weights_dict(weights)
            )
            if not np.all(np.isfinite(ego_state)) or not np.isfinite(J):
                status = {"ok": False, "reason": "non-finite state or cost"}
                break

            history.append({
                "t": t_now,
                "ego_state": ego_state.copy(),
                "ego_lane": int(ego_lane),
                "s_ego": float(s_ego),
                "a": float(a_opt),
                "delta": float(delta_opt),
                "J": float(J),
                "P_slip": float(ps),
                "P_roll": float(pr),
                "P_c2c": float(pc),
                "P_ld": float(pl),
            })
            vehicle_history.append(_vehicle_snapshot(vehicles))

            ego_state, s_ego, ego_lane = env_dynamics.step(
                ego_state, a_opt, delta_opt, s_ego, ego_lane,
                sc.kappa_A, sc.kappa_omega, sc.n_lanes, sc.lane_width, sc.dt,
            )
            vehicles = env_scenario.step_vehicles(vehicles, sc.dt)

            if ego_lane < -1 or ego_lane >= sc.n_lanes + 1:
                status = {"ok": False, "reason": "road departure"}
                break

            if verbose:
                print(
                    f"t={t_now:.2f} a={a_opt:+.2f} delta={delta_opt:+.3f} "
                    f"J={J:.3f} Pc2c={pc:.3f} Pld={pl:.3f}"
                )
    except Exception as exc:  # Keep a bad candidate from killing the search.
        status = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        _restore_mu(old_mu)

    return {
        **status,
        "theta": theta,
        "weights": weights,
        "time": np.array([h["t"] for h in history], dtype=float),
        "state_history": np.array([h["ego_state"] for h in history], dtype=float),
        "lane_history": np.array([h["ego_lane"] for h in history], dtype=int),
        "s_history": np.array([h["s_ego"] for h in history], dtype=float),
        "control_history": np.array([[h["a"], h["delta"]] for h in history], dtype=float),
        "risk_history": {
            key: np.array([h[key] for h in history], dtype=float)
            for key in ("J", "P_slip", "P_roll", "P_c2c", "P_ld")
        },
        "surrounding_history": vehicle_history,
    }


def _nanmin(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.min()) if vals.size else np.nan


def _same_lane_gaps(result: dict):
    front_gaps, rear_gaps, ttc_front, ttc_rear = [], [], [], []
    states = result["state_history"]
    lanes = result["lane_history"]
    s_hist = result["s_history"]
    for i, vehicles in enumerate(result.get("surrounding_history", [])):
        ego_s = float(s_hist[i])
        ego_v = float(states[i, 4])
        ego_lane = int(lanes[i])
        for veh in vehicles:
            if not veh.get("valid") or int(veh["lane"]) != ego_lane:
                continue
            gap = float(veh["s"]) - ego_s
            rel_front = ego_v - float(veh["v_x"])
            rel_rear = float(veh["v_x"]) - ego_v
            if gap >= 0:
                front_gaps.append(gap)
                if rel_front > 1e-6:
                    ttc_front.append(gap / rel_front)
            else:
                rear_gap = -gap
                rear_gaps.append(rear_gap)
                if rel_rear > 1e-6:
                    ttc_rear.append(rear_gap / rel_rear)
    return front_gaps, rear_gaps, ttc_front, ttc_rear


def extract_behavior_descriptor(result: dict) -> dict:
    """Convert one rollout result into scalar behavior/risk descriptors."""
    risks = result.get("risk_history", {})
    controls = result.get("control_history", np.empty((0, 2)))
    states = result.get("state_history", np.empty((0, 6)))
    lanes = result.get("lane_history", np.array([], dtype=int))
    front_gaps, rear_gaps, ttc_front, ttc_rear = _same_lane_gaps(result)

    max_collision = 0.0
    for i, vehicles in enumerate(result.get("surrounding_history", [])):
        ego_s = result["s_history"][i]
        ego_y = env_dynamics.lane_center_abs(
            int(lanes[i]), 3, result["theta"].lane_width
        ) + states[i, 0]
        for veh in vehicles:
            if not veh.get("valid"):
                continue
            overlap_s = abs(ego_s - float(veh["s"])) <= L_veh
            overlap_y = abs(ego_y - float(veh["y_abs"])) <= W_c
            max_collision = max(max_collision, float(overlap_s and overlap_y))

    if controls.size:
        max_abs_accel = float(np.max(np.abs(controls[:, 0])))
        max_abs_delta = float(np.max(np.abs(controls[:, 1])))
        mean_abs_accel = float(np.mean(np.abs(controls[:, 0])))
        mean_abs_delta = float(np.mean(np.abs(controls[:, 1])))
    else:
        max_abs_accel = max_abs_delta = mean_abs_accel = mean_abs_delta = np.nan

    descriptor = {
        "ok": bool(result.get("ok", False)),
        "reason": result.get("reason", ""),
        "max_P_slip": float(np.max(risks["P_slip"])) if len(risks.get("P_slip", [])) else np.nan,
        "max_P_roll": float(np.max(risks["P_roll"])) if len(risks.get("P_roll", [])) else np.nan,
        "max_P_c2c": float(np.max(risks["P_c2c"])) if len(risks.get("P_c2c", [])) else np.nan,
        "max_P_ld": float(np.max(risks["P_ld"])) if len(risks.get("P_ld", [])) else np.nan,
        "mean_total_cost": float(np.mean(risks["J"])) if len(risks.get("J", [])) else np.nan,
        "max_total_cost": float(np.max(risks["J"])) if len(risks.get("J", [])) else np.nan,
        "min_front_gap": _nanmin(front_gaps),
        "min_rear_gap": _nanmin(rear_gaps),
        "min_ttc_front": _nanmin(ttc_front),
        "min_ttc_rear": _nanmin(ttc_rear),
        "max_abs_accel": max_abs_accel,
        "max_abs_delta": max_abs_delta,
        "mean_abs_accel": mean_abs_accel,
        "mean_abs_delta": mean_abs_delta,
        "final_lane": int(lanes[-1]) if len(lanes) else np.nan,
        "max_abs_lateral_offset": float(np.max(np.abs(states[:, 0]))) if len(states) else np.nan,
        "lane_departure_flag": bool(result.get("reason") == "road departure"),
        "collision_flag": bool(max_collision),
        "infeasible_flag": not bool(result.get("ok", False)),
    }
    return descriptor


def _matrix(descriptors: list[dict], keys: Iterable[str]) -> np.ndarray:
    return np.array([[float(d.get(k, np.nan)) for k in keys] for d in descriptors], dtype=float)


def _normalized_spread(mat: np.ndarray) -> float:
    spreads = []
    for j in range(mat.shape[1]):
        col = mat[:, j]
        col = col[np.isfinite(col)]
        if col.size < 2:
            continue
        denom = max(float(np.percentile(col, 95) - np.percentile(col, 5)), 1e-6)
        spreads.append(float(np.std(col) / denom))
    return float(np.clip(np.mean(spreads), 0.0, 1.0)) if spreads else 0.0


def _non_dominated(mat: np.ndarray) -> np.ndarray:
    finite = np.all(np.isfinite(mat), axis=1)
    idx = np.where(finite)[0]
    if len(idx) == 0:
        return np.array([], dtype=int)
    vals = mat[idx]
    keep = np.ones(len(vals), dtype=bool)
    for i, row in enumerate(vals):
        dominated = np.all(vals <= row, axis=1) & np.any(vals < row, axis=1)
        if np.any(dominated):
            keep[i] = False
    return idx[keep]


def score_scenario_usefulness(
    descriptors: list[dict],
    *,
    min_risk_activation: float = 0.05,
) -> dict:
    """Score how useful a scenario is for risk-weight identification."""
    if not descriptors:
        return {
            "usefulness_score": -3.0,
            "risk_activation_score": 0.0,
            "weight_sensitivity_score": 0.0,
            "pareto_diversity_score": 0.0,
            "infeasibility_penalty": 1.0,
            "triviality_penalty": 1.0,
            "notes": ["no descriptors"],
        }

    risk_keys = ["max_P_slip", "max_P_roll", "max_P_c2c", "max_P_ld"]
    risk_mat = _matrix(descriptors, risk_keys)
    risk_max = np.nanmax(risk_mat, axis=0) if np.any(np.isfinite(risk_mat)) else np.zeros(4)
    risk_activation_score = float(np.mean(risk_max > min_risk_activation))

    sensitivity_keys = [
        "max_P_slip", "max_P_roll", "max_P_c2c", "max_P_ld",
        "max_abs_accel", "max_abs_delta", "max_abs_lateral_offset",
        "min_front_gap", "min_rear_gap",
    ]
    weight_sensitivity_score = _normalized_spread(_matrix(descriptors, sensitivity_keys))

    pareto_keys = [
        "max_P_slip", "max_P_roll", "max_P_c2c",
        "max_P_ld", "max_abs_accel", "max_abs_delta",
    ]
    pareto_mat = _matrix(descriptors, pareto_keys)
    nd_idx = _non_dominated(pareto_mat)
    if len(nd_idx) <= 1:
        pareto_diversity_score = 0.0
    else:
        pareto_diversity_score = 0.5 * min(1.0, (len(nd_idx) - 1) / 3.0)
        pareto_diversity_score += 0.5 * _normalized_spread(pareto_mat[nd_idx])

    infeasible = np.array([
        bool(d.get("infeasible_flag")) or bool(d.get("collision_flag"))
        for d in descriptors
    ])
    infeasibility_penalty = float(np.mean(infeasible))

    controls_near_zero = np.nanmean([
        float(d.get("max_abs_accel", np.nan)) < 0.05
        and float(d.get("max_abs_delta", np.nan)) < 1e-3
        for d in descriptors
    ])
    no_interaction = np.all(~np.isfinite(_matrix(descriptors, ["min_front_gap", "min_rear_gap"])))
    triviality_penalty = float(np.clip(
        0.5 * (risk_activation_score == 0.0)
        + 0.3 * (weight_sensitivity_score < 0.02)
        + 0.2 * controls_near_zero
        + 0.2 * no_interaction,
        0.0,
        1.0,
    ))

    usefulness_score = (
        1.0 * risk_activation_score
        + 2.0 * weight_sensitivity_score
        + 1.0 * pareto_diversity_score
        - 2.0 * infeasibility_penalty
        - 1.0 * triviality_penalty
    )
    notes = []
    if risk_activation_score == 0.0:
        notes.append("all risk channels below activation threshold")
    if weight_sensitivity_score < 0.02:
        notes.append("weight vectors produced nearly identical descriptors")
    if infeasibility_penalty > 0.5:
        notes.append("most rollouts failed or collided")

    return {
        "usefulness_score": float(usefulness_score),
        "risk_activation_score": float(risk_activation_score),
        "weight_sensitivity_score": float(weight_sensitivity_score),
        "pareto_diversity_score": float(pareto_diversity_score),
        "infeasibility_penalty": float(infeasibility_penalty),
        "triviality_penalty": float(triviality_penalty),
        "notes": notes,
    }


def evaluate_theta(
    theta: ScenarioTheta,
    weights_list: list[RiskWeights],
    *,
    t_end: float,
    dt: float,
    mpc_interval: int = 1,
    verbose: bool = False,
) -> dict:
    """Evaluate one scenario against every candidate risk-weight vector."""
    rollouts = [
        run_mpc_rollout(
            theta, weights, t_end=t_end, dt=dt,
            mpc_interval=mpc_interval, verbose=verbose,
        )
        for weights in weights_list
    ]
    descriptors = [extract_behavior_descriptor(result) for result in rollouts]
    return {
        "theta": theta,
        "weights_list": weights_list,
        "rollouts": rollouts,
        "descriptors": descriptors,
        "score": score_scenario_usefulness(descriptors),
    }


def sample_theta_random(rng: np.random.Generator) -> ScenarioTheta:
    """Sample one candidate scenario from bounded search ranges."""
    values = {
        name: float(rng.uniform(lo, hi))
        for name, (lo, hi) in THETA_BOUNDS.items()
    }
    return ScenarioTheta(**values)


def perturb_theta(
    theta: ScenarioTheta,
    rng: np.random.Generator,
    scale: float,
) -> ScenarioTheta:
    """Perturb theta while respecting valid bounds."""
    base = asdict(theta)
    child = {}
    for name, value in base.items():
        lo, hi = THETA_BOUNDS[name]
        width = hi - lo
        child[name] = float(np.clip(value + rng.normal(0.0, scale * width), lo, hi))
    return ScenarioTheta(**child)


def adaptive_theta_search(
    *,
    weights_list: list[RiskWeights],
    initial_samples: int = 32,
    refinement_rounds: int = 3,
    children_per_parent: int = 4,
    top_k: int = 8,
    t_end: float = 8.0,
    dt: float = 0.05,
    mpc_interval: int = 1,
    seed: int = 0,
    verbose: bool = False,
) -> list[dict]:
    """Search for informative scenarios using random sampling plus refinement."""
    rng = np.random.default_rng(seed)
    results = []
    thetas = [sample_theta_random(rng) for _ in range(initial_samples)]

    def evaluate_batch(batch):
        for i, theta in enumerate(batch, start=1):
            if verbose:
                print(f"evaluating theta {i}/{len(batch)}: {theta}")
            results.append(
                evaluate_theta(
                    theta, weights_list, t_end=t_end, dt=dt,
                    mpc_interval=mpc_interval, verbose=False,
                )
            )

    evaluate_batch(thetas)
    for round_idx in range(refinement_rounds):
        ranked = sorted(
            results,
            key=lambda r: r["score"]["usefulness_score"],
            reverse=True,
        )
        parents = ranked[:max(1, min(top_k, len(ranked)))]
        scale = 0.25 * (0.5 ** round_idx)
        children = [
            perturb_theta(parent["theta"], rng, scale)
            for parent in parents
            for _ in range(children_per_parent)
        ]
        if verbose:
            best = parents[0]["score"]["usefulness_score"]
            print(
                f"round {round_idx + 1}/{refinement_rounds}: "
                f"best={best:.3f}, children={len(children)}, scale={scale:.3f}"
            )
        evaluate_batch(children)

    return sorted(
        results,
        key=lambda r: r["score"]["usefulness_score"],
        reverse=True,
    )
