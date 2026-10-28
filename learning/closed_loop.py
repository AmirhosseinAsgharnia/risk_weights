"""
Closed-loop evaluation of an ego controller on one feasibility-arena
scenario, and the FIXED external performance cost J_eval used by the
minimax optimizer (see docs/minimax_design.md §3 for the definition).

Three different quantities must never be conflated:
  - PPO reward      (learning.env) -- shaping for PPO training only.
  - L_MPC(w)        (learning.mpc) -- the controller's internal objective; w varies.
  - J_eval          (this module)  -- the external yardstick; coefficients fixed
                                      per experiment (EvalCostConfig), w-independent.

Success / failure / timeout are read from EgoTrafficEnv's own terminal info
(never re-derived), so the outcome semantics are exactly the PPO pipeline's.

Realization randomness xi: realization j of scenario theta is the env built
from the family config with seed = realization_seeds[j] in mode="fixed" (so
reset() uses derive_seed(seed, 0, 0)). Evaluating different w against the
same seed list is common random numbers (CRN) by construction.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
import multiprocessing

import numpy as np

from learning.env import EgoTrafficEnv, DT, DELTA_MAX, FailureReason, Outcome
from learning.ego_control import EgoView, ConstantController, to_env_action
from learning.eval_batch import wilson_ci
from learning.feasibility_cutin import CutinConfig, CutinRuntime, THETA_BOUNDS_CUTIN, validate_cutin
from learning.feasibility_sandwich import SandwichConfig, SandwichRuntime, THETA_BOUNDS_SANDWICH, validate_sandwich
from learning.feasibility_common import SCHEMA_VERSION
from learning.mpc import SamplingMPC, MPCConfig, MPCWeights
from learning.risk_eval import CHANNELS, predicted_risks

EVALUATOR_VERSION = "closed-loop-v2"   # bump on any change that alters episode costs
                                        # v2: E_risk = all five PARALO channels (learning.risk_eval)

FAMILIES = {
    "cutin": (CutinConfig, CutinRuntime, THETA_BOUNDS_CUTIN, validate_cutin),
    "sandwich": (SandwichConfig, SandwichRuntime, THETA_BOUNDS_SANDWICH, validate_sandwich),
}
_LANE_NUM, _EGO_LANE = 3, 1   # matches learning.env (LANE_NUM, middle-lane ego) -- same as feasibility_sampling


# ── Scenarios ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Stratum:
    family: str
    blocker_side: int

    def __post_init__(self):
        assert self.family in FAMILIES and self.blocker_side in (-1, 1), self

    @property
    def label(self) -> str:
        return f"{self.family}_side{'pos1' if self.blocker_side == 1 else 'neg1'}"

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        return FAMILIES[self.family][2]


@dataclass(frozen=True)
class ScenarioSpec:
    """One nominal scenario theta within one stratum. Always realized in the
    family's "robust" mode (the same mode the PPO feasibility data uses)."""
    family: str
    blocker_side: int
    theta: tuple[tuple[str, float], ...]   # sorted (name, value) pairs -- hashable

    @staticmethod
    def make(stratum: Stratum, theta: dict) -> "ScenarioSpec":
        assert set(theta) == set(stratum.bounds), f"theta fields {sorted(theta)} != {stratum.family} schema"
        return ScenarioSpec(stratum.family, stratum.blocker_side, tuple(sorted((k, float(v)) for k, v in theta.items())))

    @property
    def stratum(self) -> Stratum:
        return Stratum(self.family, self.blocker_side)

    @property
    def theta_dict(self) -> dict[str, float]:
        return dict(self.theta)

    def config(self, seed: int = 0):
        cfg_cls = FAMILIES[self.family][0]
        return cfg_cls(blocker_side=self.blocker_side, seed=seed, mode="robust", **self.theta_dict)

    @property
    def scenario_id(self) -> str:
        return self.config().scenario_id()   # identical to the PPO dataset's scenario_id for this theta

    @property
    def key(self) -> str:
        return f"{self.stratum.label}/{self.scenario_id}"

    def validate(self) -> None:
        """Raises ScenarioFamilyConfigError if the family's own validator rejects it."""
        FAMILIES[self.family][3](self.config(), lane_num=_LANE_NUM, ego_lane=_EGO_LANE)

    def to_dict(self) -> dict:
        return {"family": self.family, "blocker_side": self.blocker_side, "theta": self.theta_dict,
                "scenario_id": self.scenario_id, "schema_version": SCHEMA_VERSION}

    @staticmethod
    def from_dict(d: dict) -> "ScenarioSpec":
        return ScenarioSpec.make(Stratum(d["family"], d["blocker_side"]), d["theta"])


# ── External cost ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EvalCostConfig:
    """J_eval coefficients and normalizers -- fixed for a whole experiment.
    Every per-step term E is dimensionless, clipped to [0, term_clip], and
    integrated as sum_k (DT / T_ep) * E_k, so the running part of any
    episode is bounded by sum(alpha) * term_clip. Safety penalties must
    exceed that bound (checked), so no failure can ever be cheaper than the
    worst survivable run, and a crash cannot be "cheap" because it ends the
    episode early."""
    alpha_tracking: float = 1.0
    alpha_comfort: float = 1.0
    alpha_control: float = 1.0
    alpha_risk: float = 1.0
    v_ref_mps: float = 20.0             # tracking: ((v - v_ref)/v_ref)^2
    lane_half_width_m: float = 2.0      # tracking: (e_y_nearest / half-width)^2
    jerk_ref_mps3: float = 10.0         # comfort:  (jerk/jerk_ref)^2
    lat_accel_ref_mps2: float = 4.0     # comfort:  (v*r / a_lat_ref)^2
    accel_ref_mps2: float = 4.0         # control:  (a/a_ref)^2 + (delta/DELTA_MAX)^2
    # risk: E_risk = sum_i beta_i * R_i over the five PARALO channels (learning.risk_eval)
    beta_c2c: float = 1.0
    beta_rollover: float = 1.0
    beta_slippage: float = 1.0
    beta_lane_departure: float = 1.0
    beta_road_departure: float = 1.0
    risk_measure: str = "prob"          # "prob": max-over-horizon probability; "score": PARALO R / (0.5 m v_ref^2)
    risk_horizon_s: float = 2.0         # constant-input prediction horizon for every channel
    risk_dt_s: float = 0.1
    risk_sigma_ego_m: float = 0.3       # position std (the risk modules' own default fallback)
    risk_sigma_obj_m: float = 0.3
    rollover_U_H_m: float = 0.1         # CG-height uncertainty (RolloverParams default)
    slip_mu_uncertainty: float = 0.05   # friction uncertainty (SlippageParams default); mu_max = scenario road mu
    term_clip: float = 10.0
    penalty_collision: float = 100.0
    penalty_rollover: float = 100.0
    penalty_off_road: float = 100.0
    penalty_timeout_base: float = 20.0       # a timeout is never a success
    penalty_timeout_shortfall: float = 30.0  # x (1 - progress / goal_distance), clipped to [0, 1]

    def __post_init__(self):
        alphas = [self.alpha_tracking, self.alpha_comfort, self.alpha_control, self.alpha_risk]
        assert all(a >= 0 for a in alphas) and sum(alphas) > 0
        running_max = sum(alphas) * self.term_clip
        assert min(self.penalty_collision, self.penalty_rollover, self.penalty_off_road) > running_max, (
            f"safety penalties must exceed the max running cost {running_max} (see class docstring)")
        assert self.penalty_timeout_base > 0 and self.penalty_timeout_shortfall >= 0
        assert all(b >= 0 for b in self.betas.values())
        assert self.risk_measure in ("prob", "score") and self.risk_horizon_s >= 0 and self.risk_dt_s > 0

    @property
    def betas(self) -> dict[str, float]:
        return {ch: getattr(self, f"beta_{ch}") for ch in CHANNELS}

    def terminal_penalty(self, info: dict, goal_distance_m: float) -> float:
        fr = info["failure_reason"]
        if fr == FailureReason.COLLISION.value:
            return self.penalty_collision
        if fr == FailureReason.ROLLOVER.value:
            return self.penalty_rollover
        if fr == FailureReason.OFF_ROAD.value:
            return self.penalty_off_road
        if info["outcome"] == Outcome.TIMEOUT.value:
            shortfall = float(np.clip(1.0 - info["progress_m"] / goal_distance_m, 0.0, 1.0))
            return self.penalty_timeout_base + self.penalty_timeout_shortfall * shortfall
        return 0.0


# ── Controllers by serializable spec (so worker processes can rebuild them) ─
def build_controller(spec: dict, w):
    name = spec["name"]
    if name == "mpc":
        return SamplingMPC(MPCConfig.from_dict(spec.get("config", {})), MPCWeights.from_vector(w))
    if name == "paralo_mpc":
        from learning.paralo_mpc import ParaloMPC, ParaloMPCConfig, ParaloWeights
        return ParaloMPC(ParaloMPCConfig.from_dict(spec.get("config", {})), ParaloWeights.from_vector(w))
    if name == "constant":   # test stub: w = (accel, delta)
        return ConstantController(float(w[0]), float(w[1]))
    raise ValueError(f"unknown controller {name!r}")


# ── One episode ─────────────────────────────────────────────────────────────
def run_episode(controller, scenario: ScenarioSpec, realization_seed: int, cost: EvalCostConfig,
                *, record_trajectory: bool = False) -> dict:
    cfg = scenario.config(seed=realization_seed)
    env = EgoTrafficEnv(arena=FAMILIES[scenario.family][1](cfg), mode="fixed")
    obs, _ = env.reset()
    controller.reset(EgoView(env, obs), {"scenario": scenario.to_dict(), "realization_seed": realization_seed})

    T_ep = cfg.max_episode_seconds
    sums = {"tracking": 0.0, "comfort": 0.0, "control": 0.0, "risk": 0.0}
    risk_sum = {ch: 0.0 for ch in CHANNELS}    # time-integrated, unweighted, per channel (diagnostic)
    risk_max = {ch: 0.0 for ch in CHANNELS}
    prev_accel, max_jerk, speed_sum, n = 0.0, 0.0, 0.0, 0
    traj = []
    done = False
    while not done:
        accel_cmd, delta_cmd = controller.act(EgoView(env, obs))
        obs, _, terminated, truncated, info = env.step(to_env_action(accel_cmd, delta_cmd))
        done = terminated or truncated

        st = env.ego_car.state
        a, d = env._prev_accel, st.delta            # APPLIED (post rate-limit) commands
        jerk = (a - prev_accel) / DT
        prev_accel = a
        _, e_y_near = env._ego_nearest_lane()
        clearance = env._ego_clearance()
        risks = predicted_risks(env, cost)
        clip = lambda x: min(max(x, 0.0), cost.term_clip)
        E = {
            "tracking": clip(((st.v_x - cost.v_ref_mps) / cost.v_ref_mps) ** 2 + (e_y_near / cost.lane_half_width_m) ** 2),
            "comfort": clip((jerk / cost.jerk_ref_mps3) ** 2 + (st.v_x * st.r / cost.lat_accel_ref_mps2) ** 2),
            "control": clip((a / cost.accel_ref_mps2) ** 2 + (d / DELTA_MAX) ** 2),
            "risk": clip(sum(cost.betas[ch] * risks[ch] for ch in CHANNELS)),
        }
        for k in sums:
            sums[k] += DT / T_ep * E[k]
        for ch in CHANNELS:
            risk_sum[ch] += DT / T_ep * risks[ch]
            risk_max[ch] = max(risk_max[ch], risks[ch])
        max_jerk = max(max_jerk, abs(jerk))
        speed_sum += st.v_x
        n += 1
        if record_trajectory:
            traj.append({"t": env.t, "s": st.s, "e_y_backbone": env._ego_backbone_e_y(), "v_x": st.v_x,
                         "accel": a, "delta": d, "clearance": clearance, **{f"risk_{ch}": risks[ch] for ch in CHANNELS}})

    running = (cost.alpha_tracking * sums["tracking"] + cost.alpha_comfort * sums["comfort"]
               + cost.alpha_control * sums["control"] + cost.alpha_risk * sums["risk"])
    terminal = cost.terminal_penalty(info, cfg.goal_distance_m)
    rec = {
        "realization_seed": realization_seed, "cost": running + terminal, "running_cost": running,
        "terminal_penalty": terminal, "cost_terms": sums,
        "risk_integrated": risk_sum, "risk_max": risk_max,
        "outcome": info["outcome"], "success": info["success"], "failure_reason": info["failure_reason"],
        "timeout": info["timeout"], "steps": n, "progress_m": info["progress_m"],
        "min_clearance_m": info["min_clearance_m"], "max_p_rollover": info["max_p_rollover"],
        "max_road_departure_m": info["max_road_departure_m"], "max_abs_jerk": max_jerk,
        "mean_speed_mps": speed_sum / max(n, 1),
        "solver_failures": getattr(controller, "solver_failures", 0),
    }
    if record_trajectory:
        rec["trajectory"] = traj
    return rec


def summarize_episodes(episodes: list[dict]) -> dict:
    costs = np.array([e["cost"] for e in episodes], dtype=float)
    n = len(episodes)
    k = sum(e["success"] for e in episodes)
    count = lambda fr: sum(e["failure_reason"] == fr for e in episodes)
    return {
        "n_episodes": n, "mean_cost": float(costs.mean()), "std_cost": float(costs.std()),
        "q50_cost": float(np.quantile(costs, 0.5)), "q90_cost": float(np.quantile(costs, 0.9)),
        "max_cost": float(costs.max()), "per_episode_costs": costs.tolist(),
        "n_success": k, "success_rate": k / n, "wilson_95ci": list(wilson_ci(k, n)),
        "n_collisions": count(FailureReason.COLLISION.value), "n_rollovers": count(FailureReason.ROLLOVER.value),
        "n_offroad": count(FailureReason.OFF_ROAD.value),
        "n_timeouts": sum(e["outcome"] == Outcome.TIMEOUT.value for e in episodes),
        "n_solver_failures": int(sum(e["solver_failures"] for e in episodes)),
        "mean_min_clearance_m": float(np.mean([e["min_clearance_m"] if e["min_clearance_m"] is not None else np.nan
                                               for e in episodes])),
        "max_p_rollover": float(max(e["max_p_rollover"] for e in episodes)),
        "realization_seeds": [e["realization_seed"] for e in episodes],
        "risk_integrated_mean": {ch: float(np.mean([e["risk_integrated"][ch] for e in episodes])) for ch in CHANNELS},
        "risk_max": {ch: float(max(e["risk_max"][ch] for e in episodes)) for ch in CHANNELS},
    }


def evaluate_controller(controller_spec: dict, w, scenario: ScenarioSpec, realization_seeds: list[int],
                        cost: EvalCostConfig) -> dict:
    """J_eval(w, theta) estimate = mean episode cost over `realization_seeds`
    (pass the same list for every w -> common random numbers)."""
    eps = [run_episode(build_controller(controller_spec, w), scenario, s, cost) for s in realization_seeds]
    return {**summarize_episodes(eps), "episodes": eps}


# ── Batch execution (process pool, single-threaded numerics per worker) ────
def _episode_job(args) -> dict:
    controller_spec, w, scenario_dict, seed, cost_dict = args
    return run_episode(build_controller(controller_spec, np.asarray(w)), ScenarioSpec.from_dict(scenario_dict),
                       seed, EvalCostConfig(**cost_dict))


def run_episode_jobs(jobs: list[tuple], n_workers: int = 1) -> list[dict]:
    """jobs: (controller_spec, w list, scenario dict, seed, cost dict) tuples.
    Results come back in job order. n_workers=1 runs in-process."""
    if n_workers <= 1 or len(jobs) <= 1:
        return [_episode_job(j) for j in jobs]
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("fork")) as ex:
        return list(ex.map(_episode_job, jobs, chunksize=max(1, len(jobs) // (4 * n_workers))))


def cost_config_dict(cost: EvalCostConfig) -> dict:
    return asdict(cost)


__all__ = ["Stratum", "ScenarioSpec", "EvalCostConfig", "build_controller", "run_episode", "summarize_episodes",
           "evaluate_controller", "run_episode_jobs", "EVALUATOR_VERSION", "FAMILIES"]
