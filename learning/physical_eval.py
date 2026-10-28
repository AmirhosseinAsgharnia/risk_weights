"""
Physical closed-loop outcome evaluation for the constrained minimax
(docs/constrained_design.md §4). No weights anywhere: an episode is reduced
to physical quantities, and scenarios to rates / peaks over realization
seeds.

Per episode (actual simulator states and APPLIED controls, never predictions):
    progress_shortfall = clip(1 - progress_m / goal_distance_m, 0, 1)
        progress_m is EgoTrafficEnv's own (Frenet arc length along the ego's lane
        from EGO_S0), so lateral motion is never counted as progress. A safety
        failure ends the episode: the shortfall is whatever progress was actually
        achieved, and the failure is recorded separately as an event.
    events (episode-level indicators; definitions in learning.vehicle_physics):
        collision       env's own COLLISION terminal condition
        rollover        env's own ROLLOVER terminal condition under the evaluator's event model:
                        "physical_v2" (default) |a_y realized| >= SSF g; "legacy_v1" P_roll(v, delta_cmd) > 0.5
                        (kept for reproducing historical results -- known false positives)
        road_departure  env's own OFF_ROAD terminal condition
        slippage        slip_lateral_saturation OR slip_friction_circle (does not end the episode)
        lane_departure  NOT a simulator event; the ego body crossing a lane marking
                        (|e_y to the nearest lane centre| > (l_w - CAR_WIDTH)/2). Every lane change
                        counts. Does not end the episode.
      separate indicators, always recorded whichever model terminates the episode:
        rollover_legacy_indicator    P_roll(v_x, delta_cmd) > 0.5 at some step
        rollover_physical_indicator  |a_y| >= SSF g at some step
        slip_lateral_saturation      some axle's requested C alpha >= mu F_z (tire-force saturation)
        slip_friction_circle         vehicle-level combined demand >= 1 (the simulator itself does NOT limit
                                     a_x by friction -- this flags e.g. 8 m/s^2 braking on mu = 0.4)
    peaks: max |applied accel|, max |realized lateral accel| ((F_yf + F_yr)/m), max |applied jerk|;
           diagnostics: max |a_y|/(SSF g), max friction demand, max |sideslip| [deg].

Per scenario (over the realization seeds, common random numbers across w):
    J_progress = mean shortfall;  event rate k/n with its Wilson 95% interval;
    peaks = max over seeds.

These are DIFFERENT from PARALO's instantaneous probabilities (recorded
separately by the MPC diagnostics); a rate here is an observed frequency over
n seeds, and k = 0 is not evidence of zero probability.
"""

import os
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict, field

import numpy as np

from initialization.traffic_init import CAR_WIDTH
from learning.vehicle_physics import (
    friction_demand, p_roll_legacy, rollover_accel_threshold, slip_friction_circle, slip_lateral_saturation,
    ROLLOVER_EVENT_MODELS, LEGACY_ROLLOVER_PROB_THRESHOLD,
)
from learning.closed_loop import FAMILIES, ScenarioSpec, build_controller
from learning.ego_control import EgoView, to_env_action
from learning.env import EgoTrafficEnv, DT, FailureReason
from learning.eval_batch import wilson_ci

PHYSICAL_EVALUATOR_VERSION = "physical-v2"   # bump on any change to an episode record's definitions
# physical-v1: rollover = legacy P_roll(v, delta_cmd) event only; slippage = lateral saturation only;
#              lateral accel peak by finite differences of v_y.

EVENTS = ("collision", "rollover", "road_departure", "slippage", "lane_departure",
          "rollover_legacy_indicator", "rollover_physical_indicator", "slip_lateral_saturation", "slip_friction_circle")
_STEP_EVENTS = ("lane_departure", "rollover_legacy_indicator", "rollover_physical_indicator",
                "slip_lateral_saturation", "slip_friction_circle")
PEAKS = ("peak_abs_accel", "peak_abs_lat_accel", "peak_abs_jerk")
DIAG_PEAKS = ("peak_ay_over_rollover_threshold", "peak_friction_demand", "peak_abs_sideslip_deg")
METRICS = {**{f"{e}_rate": ("event", e, "1/episode") for e in EVENTS},
           "peak_abs_accel": ("peak", "peak_abs_accel", "m/s^2"),
           "peak_abs_lat_accel": ("peak", "peak_abs_lat_accel", "m/s^2"),
           "peak_abs_jerk": ("peak", "peak_abs_jerk", "m/s^3"),
           "peak_ay_over_rollover_threshold": ("peak", "peak_ay_over_rollover_threshold", "1"),
           "peak_friction_demand": ("peak", "peak_friction_demand", "1"),
           "peak_abs_sideslip_deg": ("peak", "peak_abs_sideslip_deg", "deg")}


def run_physical_episode(controller, scenario: ScenarioSpec, realization_seed: int,
                         event_model: str = "physical_v2") -> dict:
    assert event_model in ROLLOVER_EVENT_MODELS, event_model
    cfg = scenario.config(seed=realization_seed)
    env = EgoTrafficEnv(arena=FAMILIES[scenario.family][1](cfg), mode="fixed", rollover_event=event_model)
    obs, _ = env.reset()
    controller.reset(EgoView(env, obs), {"scenario": scenario.to_dict(), "realization_seed": realization_seed})
    lane_h = (env.road.l_w - CAR_WIDTH) / 2.0
    vp = env.ego_car.vehicle_params
    ay_roll = rollover_accel_threshold(vp)

    ev_steps = dict.fromkeys(_STEP_EVENTS, 0)
    peaks = dict.fromkeys(PEAKS + DIAG_PEAKS, 0.0)
    prev_a = 0.0
    ctrl_s, n = 0.0, 0
    done = False
    while not done:
        t0 = time.perf_counter()
        a_cmd, d_cmd = controller.act(EgoView(env, obs))
        ctrl_s += time.perf_counter() - t0
        obs, _, terminated, truncated, info = env.step(to_env_action(a_cmd, d_cmd))
        done = terminated or truncated
        st = env.ego_car.state
        a = env._prev_accel
        ts = env.ego_tire_state()
        fd = friction_demand(ts, vp)
        peaks["peak_abs_accel"] = max(peaks["peak_abs_accel"], float(abs(a)))
        peaks["peak_abs_jerk"] = max(peaks["peak_abs_jerk"], float(abs(a - prev_a) / DT))
        peaks["peak_abs_lat_accel"] = max(peaks["peak_abs_lat_accel"], abs(ts.a_y))
        peaks["peak_ay_over_rollover_threshold"] = max(peaks["peak_ay_over_rollover_threshold"], abs(ts.a_y) / ay_roll)
        peaks["peak_friction_demand"] = max(peaks["peak_friction_demand"], fd)
        peaks["peak_abs_sideslip_deg"] = max(peaks["peak_abs_sideslip_deg"], abs(np.degrees(ts.sideslip_rad)))
        prev_a = a
        ev_steps["lane_departure"] += int(abs(env._ego_nearest_lane()[1]) > lane_h)
        ev_steps["rollover_legacy_indicator"] += int(p_roll_legacy(st.v_x, st.delta, vp) > LEGACY_ROLLOVER_PROB_THRESHOLD)
        ev_steps["rollover_physical_indicator"] += int(abs(ts.a_y) >= ay_roll)
        ev_steps["slip_lateral_saturation"] += int(slip_lateral_saturation(ts))
        ev_steps["slip_friction_circle"] += int(fd >= 1.0)
        n += 1

    fr = info["failure_reason"]
    events = {"collision": fr == FailureReason.COLLISION.value, "rollover": fr == FailureReason.ROLLOVER.value,
              "road_departure": fr == FailureReason.OFF_ROAD.value,
              "slippage": ev_steps["slip_lateral_saturation"] + ev_steps["slip_friction_circle"] > 0,
              **{k: ev_steps[k] > 0 for k in _STEP_EVENTS}}
    goal = cfg.goal_distance_m
    n_plans = getattr(controller, "n_plans", 0)
    return {
        "realization_seed": realization_seed, "event_model": event_model,
        "progress_m": float(info["progress_m"]), "goal_distance_m": goal,
        "progress_shortfall": float(np.clip(1.0 - info["progress_m"] / goal, 0.0, 1.0)),
        "outcome": info["outcome"], "success": info["success"], "failure_reason": fr, "timeout": info["timeout"],
        "terminated_by_safety_failure": fr is not None, "steps": n, "elapsed_s": env.t,
        "max_episode_seconds": float(env._max_steps * DT),
        "events": events, "event_steps": ev_steps, **{k: float(v) for k, v in peaks.items()},
        "min_clearance_m": info["min_clearance_m"], "max_p_rollover": info["max_p_rollover"],
        "solver_failures": getattr(controller, "solver_failures", 0), "n_plans": n_plans,
        "mpc_diagnostics": dict(getattr(controller, "diag", {}) or {}),
        # wall-clock timing is NOT deterministic -- excluded from every comparison/cache-equality check
        "timing": {"controller_s": ctrl_s, "ms_per_plan": 1000 * ctrl_s / n_plans if n_plans else None,
                   "ms_per_step": 1000 * ctrl_s / max(n, 1)},
    }


def summarize_scenario(episodes: list[dict]) -> dict:
    n = len(episodes)
    assert n > 0
    out = {"n_episodes": n, "realization_seeds": [e["realization_seed"] for e in episodes],
           "progress_shortfall": float(np.mean([e["progress_shortfall"] for e in episodes])),
           "progress_shortfall_per_seed": [e["progress_shortfall"] for e in episodes],
           "n_success": int(sum(e["success"] for e in episodes)),
           "n_timeouts": int(sum(e["timeout"] for e in episodes)),
           "n_solver_failures": int(sum(e["solver_failures"] for e in episodes))}
    out["completion_rate"] = out["n_success"] / n
    for ev in EVENTS:
        k = int(sum(e["events"][ev] for e in episodes))
        lo, hi = wilson_ci(k, n)
        out[f"{ev}_count"], out[f"{ev}_rate"], out[f"{ev}_wilson95"] = k, k / n, [lo, hi]
    for p in PEAKS + DIAG_PEAKS:
        out[p] = float(max(e[p] for e in episodes))
    # completion time: lexicographic, not weighted -- every success (t/T_max in [0, 1]) ranks
    # before every failure (1 + shortfall); see objective_value()
    out["completion_time"] = float(np.mean([e["elapsed_s"] / e["max_episode_seconds"] if e["success"]
                                            else 1.0 + e["progress_shortfall"] for e in episodes]))
    return out


OBJECTIVES = ("progress_shortfall", "completion_time")


def objective_value(summary: dict, objective: str) -> float:
    assert objective in OBJECTIVES, objective
    return float(summary[objective])


# ── Constraints ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ConstraintSpec:
    """R_i(w, theta) <= threshold. threshold None = MONITORED ONLY (reported,
    never enforced) -- an explicit, visible state, not a silent relaxation.
    statistic: "point" (observed rate / peak) or "wilson_upper" (event
    rates only: the upper Wilson 95% bound -- with threshold 0 this can never
    be satisfied, since k/n bounds are always > 0)."""
    name: str
    metric: str                     # key of METRICS
    threshold: float | None
    units: str
    statistic: str = "point"
    tolerance: float = 0.0          # violated iff value > threshold + tolerance
    status: str = "provisional"     # "provisional" research threshold | "validated" requirement
    note: str = ""

    def __post_init__(self):
        assert self.metric in METRICS, f"unknown metric {self.metric!r}; known: {sorted(METRICS)}"
        assert self.statistic in ("point", "wilson_upper")
        assert self.statistic == "point" or METRICS[self.metric][0] == "event", "wilson_upper is for event rates"
        assert self.status in ("provisional", "validated") and self.tolerance >= 0
        assert self.units == METRICS[self.metric][2], f"{self.name}: units {self.units!r} != {METRICS[self.metric][2]!r}"

    @property
    def enforced(self) -> bool:
        return self.threshold is not None

    def value(self, summary: dict) -> float:
        kind, key, _ = METRICS[self.metric]
        if kind == "event" and self.statistic == "wilson_upper":
            return float(summary[f"{key}_wilson95"][1])
        return float(summary[self.metric] if kind == "peak" else summary[f"{key}_rate"])

    def violation(self, value: float) -> float | None:
        """value - threshold in the constraint's own units (None if monitored only)."""
        return None if self.threshold is None else value - self.threshold

    def violated(self, value: float) -> bool:
        return self.threshold is not None and value > self.threshold + self.tolerance


DEFAULT_CONSTRAINTS: tuple[dict, ...] = (
    {"name": "collision", "metric": "collision_rate", "threshold": 0.0, "units": "1/episode",
     "note": "provisional: no observed collision over the evaluation seeds"},
    {"name": "rollover", "metric": "rollover_rate", "threshold": 0.0, "units": "1/episode",
     "note": "provisional; under physical_v2, quasi-static tipping onset |a_y| >= SSF g (reduced-order, "
             "no suspension roll / tripping) -- only reachable for mu >= SSF = 0.97 in this tire model"},
    {"name": "road_departure", "metric": "road_departure_rate", "threshold": 0.0, "units": "1/episode",
     "note": "provisional; env OFF_ROAD"},
    {"name": "slippage", "metric": "slippage_rate", "threshold": 0.0, "units": "1/episode",
     "note": "provisional; lateral tire-force saturation OR vehicle friction-circle exceedance "
             "(learning.vehicle_physics)"},
    {"name": "lane_departure", "metric": "lane_departure_rate", "threshold": None, "units": "1/episode",
     "note": "MONITORED ONLY: the event counts every lane-marking crossing, including deliberate lane "
             "changes, so any threshold < 1 forbids escaping a cut-in -- set one only with that in mind"},
    {"name": "lat_accel_peak", "metric": "peak_abs_lat_accel", "threshold": 4.0, "units": "m/s^2",
     "note": "provisional comfort/handling limit (~0.4 g); not a validated requirement"},
    {"name": "jerk_peak", "metric": "peak_abs_jerk", "threshold": None, "units": "m/s^3",
     "note": "MONITORED ONLY: emergency braking legitimately reaches the env's 40 m/s^3 actuator limit; "
             "no non-arbitrary threshold has been agreed"},
    {"name": "accel_peak", "metric": "peak_abs_accel", "threshold": None, "units": "m/s^2",
     "note": "MONITORED ONLY: full braking (8 m/s^2) is a legitimate safety response"},
)


def constraints_from_config(rows) -> list[ConstraintSpec]:
    specs = [ConstraintSpec(**r) for r in rows]
    names = [c.name for c in specs]
    assert len(set(names)) == len(names), f"duplicate constraint names: {names}"
    return specs


# ── Batch execution ─────────────────────────────────────────────────────────
def _job(args) -> dict:
    controller_spec, w, scenario_dict, seed, event_model = args
    return run_physical_episode(build_controller(controller_spec, np.asarray(w)), ScenarioSpec.from_dict(scenario_dict),
                                seed, event_model)


def run_physical_jobs(jobs: list[tuple], n_workers: int = 1) -> list[dict]:
    """jobs: (controller_spec, w list, scenario dict, seed, event_model); results in job order."""
    if n_workers <= 1 or len(jobs) <= 1:
        return [_job(j) for j in jobs]
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("fork")) as ex:
        return list(ex.map(_job, jobs, chunksize=max(1, len(jobs) // (4 * n_workers))))


def deterministic_part(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k != "timing"}


__all__ = ["PHYSICAL_EVALUATOR_VERSION", "EVENTS", "PEAKS", "DIAG_PEAKS", "METRICS", "OBJECTIVES", "objective_value", "run_physical_episode", "summarize_scenario",
           "ConstraintSpec", "DEFAULT_CONSTRAINTS", "constraints_from_config", "run_physical_jobs", "deterministic_part"]
