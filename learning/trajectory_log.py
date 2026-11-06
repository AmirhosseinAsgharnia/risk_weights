"""
Rollout recording and physical validation (docs/description.md §23). Everything is read from the simulator:
EgoTrafficEnv's info dict, its states and its own helpers. Nothing is re-modelled, and a field the simulator
did not expose stays absent. Attached through learning.eval_batch.run_episodes(observer=...), so training
never pays for it and there is only one rollout loop.

Row 0 is the initial state (controls and forces NaN: nothing has been applied yet). Row k is the state after
step k, with the action and the tire forces at that state, sampled once per control step. Friction limits
inside the RK4 stages are model.car.model.tire_forces' own guarantee (tests/test_tire_forces.py).
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from model.collision import body_half_extent_e_y, lane_backbone_e_y
from learning.vehicle_physics import stability_loss, slip_lateral_saturation

TRAJECTORY_SCHEMA = "ppo-trajectory-v1"
FRICTION_TOLERANCE = 1e-6   # relative: eta <= 1 + this. The saturation code holds to rounding error (~1e-12 N).
MIN_LOAD_N = 1.0            # below this mu F_z, eta is undefined and is not computed
_TIRE_FIELDS = ("F_xf", "F_xr", "F_yf", "F_yr", "F_zf", "F_zr", "F_xf_req", "F_xr_req", "F_yf_req", "F_yr_req",
                "F_yf_max", "F_yr_max", "alpha_f", "alpha_r", "a_x", "a_y", "sideslip_rad")
_STEP_FIELDS = ("accel_requested", "accel_applied", "delta_requested", "delta_applied", "collided", "off_road",
                "rolled_over", "tire_saturated", "stability_loss", "event_triggered") + _TIRE_FIELDS
_STATE_FIELDS = ("t", "s", "e_y", "e_y_backbone", "e_psi", "v_x", "v_y", "r", "x", "y", "heading",
                 "clearance_m", "road_edge_clearance_m")
_SURR_FIELDS = ("s", "e_y_backbone", "e_psi", "v_x", "x", "y", "heading", "lane", "lane_changing", "crashed")


class TrajectoryRecorder:
    """One recorder per evaluation. keep(result) -> path or None decides, when an episode ends, whether its
    arrays are written; result["physical"] is filled in for every rollout, kept or not."""

    def __init__(self, keep=None):
        self.keep = keep

    def start(self, env) -> None:
        self.rows = {k: [] for k in _STATE_FIELDS + _STEP_FIELDS}
        self.surr = {k: [] for k in _SURR_FIELDS}
        self._record_state(env)
        for k in _STEP_FIELDS:
            self.rows[k].append(np.nan)

    def _record_state(self, env) -> None:
        st, road = env.ego_car.state, env.road
        backbone = float(env._ego_backbone_e_y())
        clearance = env._ego_clearance()
        values = dict(t=env.t, s=st.s, e_y=st.e_y, e_y_backbone=backbone, e_psi=st.e_psi, v_x=st.v_x, v_y=st.v_y, r=st.r,
                      x=st.x, y=st.y, heading=st.heading, clearance_m=clearance if math.isfinite(clearance) else np.nan,
                      road_edge_clearance_m=env._road_half_width - abs(backbone) - body_half_extent_e_y(st.e_psi))
        for k, v in values.items():
            self.rows[k].append(float(v))
        rows = {k: [] for k in _SURR_FIELDS}
        for a in env.agents:
            c = a.car.state
            e_y = float(lane_backbone_e_y(a, road))
            x, y, h = (c.x, c.y, c.heading) if c.x is not None else road.frenet_to_global(c.s, e_y, c.e_psi)
            for k, v in zip(_SURR_FIELDS, (c.s, e_y, c.e_psi, c.v_x, x, y, h, c.lane, a.lane_change_t0 is not None, a.crashed)):
                rows[k].append(float(v))
        for k in _SURR_FIELDS:
            self.surr[k].append(rows[k])

    def step(self, env, action, info: dict) -> None:
        self._record_state(env)
        ts = env.ego_tire_state()
        _, delta_requested = env._unscale_action(action)
        values = dict(accel_requested=info["accel_requested"], accel_applied=info["accel_actuator"],
                      delta_requested=delta_requested, delta_applied=env.ego_car.state.delta, collided=info["collided"],
                      off_road=info["off_road"], rolled_over=info["rolled_over"], tire_saturated=slip_lateral_saturation(ts),
                      stability_loss=stability_loss(ts), event_triggered=info.get("arena", {}).get("event_triggered", np.nan),
                      **{k: info["tire"][k] for k in _TIRE_FIELDS})
        for k, v in values.items():
            self.rows[k].append(float(v))

    def arrays(self, env, result: dict) -> dict:
        arrays = {f"ego__{k}": np.asarray(v) for k, v in self.rows.items()}
        arrays.update({f"surr__{k}": np.asarray(v) for k, v in self.surr.items()})
        arena = result.get("arena") or {}
        arrays["surr__car_id"] = np.array([a.car.car_id for a in env.agents])
        meta = {"schema": TRAJECTORY_SCHEMA, "mu": float(env.road.mu), "goal_s": float(env._goal_s),
                "road_half_width_m": float(env._road_half_width), "kappa_max": float(env.road.kappa_max),
                "s_max": float(env.road.s_max), "L_clothoid": float(env.road.L_clothoid),
                "lane_num": len(env.road.lanes), "seed_info": result.get("seed_info"), "arena": arena,
                "episode": result["episode"], "outcome": result["outcome"], "success": result["success"],
                "failure_reason": result["failure_reason"], "elapsed_s": result["elapsed_s"]}
        return {"arrays": arrays, "meta": meta}

    def end(self, env, result: dict) -> None:
        if env.arena is not None:
            result["arena"] = env.arena.extra_info()
            result["family"] = result["arena"].get("family")
        result["seed_info"] = env.get_realized_scenario()["seed_info"]
        packed = self.arrays(env, result)
        traj = Trajectory(packed["arrays"], packed["meta"])
        result["physical"] = physical_validation(traj)
        result["fingerprint"] = fingerprint(traj)
        path = self.keep(result) if self.keep else None
        if path is not None:
            save_trajectory(path, packed["arrays"], packed["meta"])


class Trajectory:
    """Loaded rollout. ego(name) / surr(name) return None for a field this file does not have."""

    def __init__(self, arrays: dict, meta: dict):
        self.arrays, self.meta = arrays, meta

    def ego(self, name: str):
        return self.arrays.get(f"ego__{name}")

    def surr(self, name: str):
        return self.arrays.get(f"surr__{name}")

    def __len__(self):
        return len(self.ego("t"))


def save_trajectory(path, arrays: dict, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, __meta__=np.array(json.dumps(meta, sort_keys=True, default=str)), **arrays)


def load_trajectory(path) -> Trajectory:
    with np.load(path) as z:
        return Trajectory({k: z[k] for k in z.files if k != "__meta__"}, json.loads(str(z["__meta__"])))


def fingerprint(traj: Trajectory) -> str:
    """Identical rollouts give identical values: how repeated exact-mode rollouts are detected."""
    data = np.stack([np.round(traj.ego(k), 6) for k in ("s", "e_y", "v_x")])
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


def friction_utilization(traj: Trajectory) -> dict | None:
    """eta_i(t) = sqrt(F_x,i^2 + F_y,i^2) / (mu F_z,i) from the logged realized forces, NaN where mu F_z,i is
    below MIN_LOAD_N. None if the forces were not logged."""
    needed = [traj.ego(k) for k in ("F_xf", "F_yf", "F_zf", "F_xr", "F_yr", "F_zr")]
    if any(a is None for a in needed):
        return None
    out = {}
    for axle, (fx, fy, fz) in (("front", needed[:3]), ("rear", needed[3:])):
        limit = traj.meta["mu"] * fz
        loaded = limit >= MIN_LOAD_N
        out[axle] = np.where(loaded, np.hypot(fx, fy) / np.where(loaded, limit, 1.0), np.nan)
        out[f"{axle}_unloaded_force_n"] = np.where(limit < MIN_LOAD_N, np.hypot(fx, fy), 0.0)
    return out


def _peak(a) -> float | None:
    return None if a is None or not np.any(np.isfinite(a)) else float(np.nanmax(a))


def _low(a) -> float | None:
    return None if a is None or not np.any(np.isfinite(a)) else float(np.nanmin(a))


def physical_validation(traj: Trajectory, tolerance: float = FRICTION_TOLERANCE) -> dict:
    """Checks on the simulator's own states and forces, never on PARALO's risk probabilities. Saturation
    (eta at 1) is normal braking at the limit and is reported, not failed. `status` is "valid" when every
    axle respects eta <= 1 + tolerance, "violation" otherwise, "unavailable" without forces. Loss of control
    is the separate stability_loss indicator."""
    util = friction_utilization(traj)
    a_x, a_y, delta = traj.ego("a_x"), traj.ego("a_y"), traj.ego("delta_applied")
    clearance, edge = traj.ego("clearance_m"), traj.ego("road_edge_clearance_m")
    out = {"status": "unavailable", "peak_eta_front": None, "peak_eta_rear": None, "friction_violations": None,
           "max_violation": None, "tolerance": tolerance,
           "peak_abs_lat_accel": _peak(np.abs(a_y)) if a_y is not None else None,
           "peak_long_decel": _peak(-a_x) if a_x is not None else None,
           "peak_abs_steering_rad": _peak(np.abs(delta)) if delta is not None else None,
           "min_clearance_m": _low(clearance), "min_road_edge_clearance_m": _low(edge),
           "duration_s": float(traj.ego("t")[-1]),
           "tire_saturated_steps": int(np.nansum(traj.ego("tire_saturated"))) if traj.ego("tire_saturated") is not None else None,
           "stability_loss": bool(np.nansum(traj.ego("stability_loss")) > 0) if traj.ego("stability_loss") is not None else None,
           "peak_abs_sideslip_deg": _peak(np.degrees(np.abs(traj.ego("sideslip_rad")))) if traj.ego("sideslip_rad") is not None else None,
           "peak_speed_mps": _peak(traj.ego("v_x"))}
    if util is None:
        return out
    front, rear = util["front"], util["rear"]
    excess = np.concatenate([front, rear]) - 1.0
    unloaded = np.concatenate([util["front_unloaded_force_n"], util["rear_unloaded_force_n"]]) > tolerance
    n = int(np.nansum(excess > tolerance) + unloaded.sum())
    out.update(peak_eta_front=_peak(front), peak_eta_rear=_peak(rear), friction_violations=n,
               max_violation=max(_peak(excess) or 0.0, 0.0), status="valid" if n == 0 else "violation")
    return out
