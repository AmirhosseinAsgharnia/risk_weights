"""
Controller-capability suite (spec §28): can the controllers execute the
maneuvers the minimax search will demand, BEFORE any weight tuning? Fixed,
hand-chosen theta inside the current bounds (none is PPO-certified as
admissible -- they probe the controller, not the oracle), evaluated under the
corrected physical events (physical_v2) for the original MPC and the PARALO
MPC at w0. Per episode: outcome, events, whether the ego changed lane
(nearest-lane index at the end != start), peak braking / steering / lateral
acceleration.

A failure here that persists at every w is a maneuver-generation limit
(e.g. the constant-absolute-steering candidate set), not a weight problem.

    python -m learning.capability_suite --out artifacts/physics_validation --seeds 1000 1001 1002
"""

import argparse
import json
from pathlib import Path

import numpy as np

from learning.closed_loop import ScenarioSpec, Stratum, build_controller
from learning.feasibility_cutin import CutinConfig
from learning.feasibility_sandwich import SandwichConfig
from learning.mpc import MPCWeights
from learning.paralo_mpc import ParaloWeights

_C, _S = CutinConfig().theta(), SandwichConfig().theta()
CASES = {
    "cutin_moderate": ("cutin", {**_C, "cutter_gap_m": 30.0, "cutter_relative_speed_mps": 0.0, "front_gap_m": 70.0}),
    "cutin_severe": ("cutin", {**_C, "cutter_gap_m": 9.0, "cutter_relative_speed_mps": -5.0, "front_gap_m": 30.0,
                               "front_relative_speed_mps": -8.0}),
    "cutin_curved": ("cutin", {**_C, "cutter_gap_m": 15.0, "road_kappa_max": 0.02}),
    "cutin_low_mu": ("cutin", {**_C, "cutter_gap_m": 15.0, "road_mu": 0.4}),
    "cutin_brake_and_steer": ("cutin", {**_C, "cutter_gap_m": 10.0, "cutter_relative_speed_mps": -5.0,
                                        "road_kappa_max": 0.015, "road_mu": 0.5, "escape_gap_length_m": 12.0}),
    "sandwich_lateral_escape": ("sandwich", {**_S, "escape_gap_length_m": 40.0, "escape_gap_center_m": 0.0,
                                             "front_gap_m": 15.0}),
    "sandwich_braking_required": ("sandwich", {**_S, "escape_gap_length_m": 8.0, "front_gap_m": 60.0}),
}
CONTROLLERS = {"original_mpc": ({"name": "mpc"}, MPCWeights().to_vector()),
               "paralo_mpc_w0": ({"name": "paralo_mpc"}, ParaloWeights().to_vector())}


def run_case(family: str, theta: dict, controller_spec: dict, w, seed: int) -> dict:
    from learning.closed_loop import FAMILIES
    from learning.ego_control import EgoView, to_env_action
    from learning.env import EgoTrafficEnv
    from learning.physical_eval import run_physical_episode
    spec = ScenarioSpec.make(Stratum(family, 1), theta)
    spec.validate()
    rec = run_physical_episode(build_controller(controller_spec, np.asarray(w)), spec, seed, "physical_v2")
    # lane change + peak steering need the trajectory: replay deterministically
    env = EgoTrafficEnv(arena=FAMILIES[family][1](spec.config(seed=seed)), mode="fixed", rollover_event="physical_v2")
    obs, _ = env.reset()
    ctl = build_controller(controller_spec, np.asarray(w))
    ctl.reset(EgoView(env, obs), {"scenario": spec.to_dict(), "realization_seed": seed})
    lane0, lanes, peak_d, done = env._ego_nearest_lane()[0], set(), 0.0, False
    while not done:
        obs, _, term, trunc, _ = env.step(to_env_action(*ctl.act(EgoView(env, obs))))
        done = term or trunc
        lanes.add(env._ego_nearest_lane()[0])
        peak_d = max(peak_d, abs(env.ego_car.state.delta))
    return {"outcome": rec["outcome"], "success": rec["success"], "failure_reason": rec["failure_reason"],
            "progress_shortfall": rec["progress_shortfall"], "changed_lane": lanes != {lane0},
            "events": {k: v for k, v in rec["events"].items() if v}, "peak_abs_accel": rec["peak_abs_accel"],
            "peak_abs_lat_accel": rec["peak_abs_lat_accel"], "peak_abs_delta": peak_d,
            "peak_friction_demand": rec["peak_friction_demand"], "min_clearance_m": rec["min_clearance_m"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="artifacts/physics_validation")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    a = ap.parse_args()
    res = {}
    for case, (family, theta) in CASES.items():
        for name, (spec, w) in CONTROLLERS.items():
            res.setdefault(case, {})[name] = [run_case(family, theta, spec, w, s) for s in a.seeds]
            eps = res[case][name]
            print(f"{case:28s} {name:14s} success {sum(e['success'] for e in eps)}/{len(eps)}  "
                  f"lane change {sum(e['changed_lane'] for e in eps)}/{len(eps)}  "
                  f"failures {[e['failure_reason'] or e['outcome'] for e in eps if not e['success']]}  "
                  f"events {sorted({k for e in eps for k in e['events']})}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "capability_suite.json").write_text(json.dumps({"cases": {c: {"family": f, "theta": t} for c, (f, t) in
                                                                      CASES.items()}, "seeds": a.seeds,
                                                            "results": res}, indent=2))


if __name__ == "__main__":
    main()
