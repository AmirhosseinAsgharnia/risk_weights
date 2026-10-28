"""
PARALO risk-model validation against the simulator's physical events
(docs/physics_audit.md Part B, spec §9). Nothing is recalibrated here.

Part 1, steady state (real CarDynamics, straight road, constant delta held 8 s):
    grid over speed x commanded steering x friction, plus braking demand at
    the reached steady state. Per point: realized a_y, r, delta_eq, PARALO
    P_roll / P_slip given delta_cmd ("applied", legacy) and given delta_eq
    ("kinematic_equivalent"), and the physical indicators.
Part 2, closed loop: every simulator step of the original MPC and the PARALO
    MPC (w0) on the curated PPO-demonstrated scenarios (or, with no PPO data, the
    capability suite's hand-chosen cases) -- the same per-step
    probabilities against the same-step physical indicators: confusion counts
    at P > 0.5, Brier score and reliability bins.

    python -m learning.risk_validation --out artifacts/physics_validation --workers 10
"""

import os

for _env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import argparse
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from model.car.car import Car, CarState
from model.car.config import VehicleParameters, G
from learning.paralo_vec import p_rollover_vec, p_slip_vec
from learning.vehicle_physics import (
    friction_demand, paralo_steer_input, rollover_accel_threshold, rollover_physical_v2, slip_friction_circle,
    slip_lateral_saturation, tire_state,
)
from risks.slippage import SlippageParams

VP = VehicleParameters()
PAIRS = {  # predicted probability -> physical indicator it is checked against
    "p_roll_applied": "rollover_physical", "p_roll_equivalent": "rollover_physical",
    "p_slip_applied": "slip_any", "p_slip_equivalent": "slip_any",
}


def _probs(v, delta, r, a, mu):
    sp = SlippageParams(mu_max=mu)
    d_eq = float(paralo_steer_input("kinematic_equivalent", delta, v, r, VP))
    return {"delta_eq": d_eq,
            "p_roll_applied": float(p_rollover_vec(v, delta, VP, 0.1)), "p_roll_equivalent": float(p_rollover_vec(v, d_eq, VP, 0.1)),
            "p_slip_applied": float(p_slip_vec(v, delta, a, VP, sp)), "p_slip_equivalent": float(p_slip_vec(v, d_eq, a, VP, sp))}


def _physical(ts):
    sat, fc = slip_lateral_saturation(ts), slip_friction_circle(ts, VP)
    return {"a_y": ts.a_y, "friction_demand": friction_demand(ts, VP), "sideslip_deg": math.degrees(ts.sideslip_rad),
            "rollover_physical": rollover_physical_v2(ts, VP), "slip_lateral_saturation": sat,
            "slip_friction_circle": fc, "slip_any": sat or fc}


def steady_state_sweep() -> list[dict]:
    out = []
    for mu in (0.4, 0.7, 1.0):
        for v0 in (10.0, 15.0, 20.0, 25.0, 30.0):
            for delta in np.round(np.linspace(0.0, 0.3, 13), 4):
                car = Car(state=CarState(v_x=v0), vehicle_params=VP)
                for _ in range(800):
                    car.step(0.0, float(delta), 0.0, mu, 0.01)
                st = car.state
                stable = bool(st.v_x > 1.0 and abs(math.degrees(math.atan2(st.v_y, st.v_x))) < 20.0)
                for a in (0.0, -2.0, -4.0, -6.0, -8.0):
                    ts = tire_state(st.as_array(), a, float(delta), 0.0, mu, VP)
                    out.append({"mu": mu, "v0": v0, "v": st.v_x, "delta_cmd": float(delta), "a_x": a, "r": st.r,
                                "stable": stable, **_probs(st.v_x, float(delta), st.r, a, mu), **_physical(ts),
                                "kinematic_a_y": st.v_x ** 2 * math.tan(delta) / VP.L})
    return out


def _closed_loop_job(args) -> list[dict]:
    controller_spec, w, scen, seed = args
    from learning.closed_loop import FAMILIES, ScenarioSpec, build_controller
    from learning.ego_control import EgoView, to_env_action
    from learning.env import EgoTrafficEnv
    spec = ScenarioSpec.from_dict(scen)
    env = EgoTrafficEnv(arena=FAMILIES[spec.family][1](spec.config(seed=seed)), mode="fixed", rollover_event="physical_v2")
    obs, _ = env.reset()
    ctl = build_controller(controller_spec, np.asarray(w))
    ctl.reset(EgoView(env, obs), {"scenario": scen, "realization_seed": seed})
    rows, done = [], False
    while not done:
        obs, _, term, trunc, _ = env.step(to_env_action(*ctl.act(EgoView(env, obs))))
        done = term or trunc
        st, ts = env.ego_car.state, env.ego_tire_state()
        rows.append({"controller": controller_spec["name"], "mu": env.road.mu, "v": st.v_x, "delta_cmd": st.delta,
                     "r": st.r, "a_x": env._prev_accel,
                     **_probs(st.v_x, st.delta, st.r, env._prev_accel, env.road.mu), **_physical(ts)})
    return rows


def classification(rows: list[dict]) -> dict:
    out = {}
    for p, ev in PAIRS.items():
        prob = np.array([r[p] for r in rows])
        y = np.array([bool(r[ev]) for r in rows])
        pred = prob > 0.5
        bins = []
        for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            sel = (prob >= lo) & ((prob < hi) | (hi == 1.0))
            if sel.any():
                bins.append({"bin": [float(lo), float(hi)], "n": int(sel.sum()), "mean_p": float(prob[sel].mean()),
                             "event_rate": float(y[sel].mean())})
        out[p] = {"event": ev, "n": len(rows), "n_events": int(y.sum()), "tp": int((pred & y).sum()),
                  "fp": int((pred & ~y).sum()), "fn": int((~pred & y).sum()), "tn": int((~pred & ~y).sum()),
                  "false_positive_rate": float((pred & ~y).sum() / max((~y).sum(), 1)),
                  "false_negative_rate": float((~pred & y).sum() / y.sum()) if y.any() else None,
                  "brier": float(np.mean((prob - y) ** 2)), "reliability": bins}
    return out


def _plots(steady: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in steady if r["a_x"] == 0.0 and r["stable"]]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for mu, c in zip((0.4, 0.7, 1.0), ("tab:red", "tab:orange", "tab:blue")):
        sel = [r for r in rows if r["mu"] == mu]
        ay = [abs(r["a_y"]) for r in sel]
        ax[0, 0].scatter(ay, [r["p_roll_applied"] for r in sel], s=10, c=c, marker="x", label=f"applied δ, μ={mu}")
        ax[0, 0].scatter(ay, [r["p_roll_equivalent"] for r in sel], s=10, c=c, marker="o", label=f"δ_eq, μ={mu}")
        ax[0, 1].scatter(ay, [r["p_slip_applied"] for r in sel], s=10, c=c, marker="x")
        ax[0, 1].scatter(ay, [r["p_slip_equivalent"] for r in sel], s=10, c=c, marker="o")
        ax[1, 0].scatter([r["delta_cmd"] for r in sel], [r["delta_eq"] for r in sel], s=10, c=c, label=f"μ={mu}")
    ax[0, 0].axvline(rollover_accel_threshold(VP), ls="--", c="k")
    ax[0, 0].set(xlabel="realized |a_y| [m/s²]", ylabel="P_roll", title="rollover (dashed: SSF·g)")
    ax[0, 0].legend(fontsize=6)
    ax[0, 1].set(xlabel="realized |a_y| [m/s²]", ylabel="P_slip", title="slippage (x applied δ, o δ_eq)")
    ax[1, 0].plot([0, 0.3], [0, 0.3], "k:")
    ax[1, 0].set(xlabel="commanded δ [rad]", ylabel="δ_eq [rad]", title="understeer: δ_eq vs δ_cmd")
    ax[1, 0].legend(fontsize=7)
    br = [r for r in steady if r["delta_cmd"] == 0.1 and r["v0"] == 20.0 and r["stable"]]
    for mu, c in zip((0.4, 0.7, 1.0), ("tab:red", "tab:orange", "tab:blue")):
        sel = sorted([r for r in br if r["mu"] == mu], key=lambda r: r["a_x"])
        ax[1, 1].plot([-r["a_x"] for r in sel], [r["p_slip_equivalent"] for r in sel], "o-", c=c, label=f"P_slip δ_eq μ={mu}")
        ax[1, 1].plot([-r["a_x"] for r in sel], [min(r["friction_demand"], 2) for r in sel], "s--", c=c, alpha=.5)
    ax[1, 1].axhline(1.0, c="k", lw=.5)
    ax[1, 1].set(xlabel="braking [m/s²] (δ=0.1, 20 m/s)", ylabel="P_slip (o) / friction demand (s)",
                 title="combined braking + cornering")
    ax[1, 1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "risk_validation.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="artifacts/physics_validation")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001])
    ap.add_argument("--max-scenarios", type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    steady = steady_state_sweep()
    from learning.closed_loop import ScenarioSpec, Stratum
    from learning.feasibility_audit import curated_dataset
    from learning.minimax_feasibility import demonstrated
    from learning.mpc import MPCWeights
    from learning.paralo_mpc import ParaloWeights
    scen = [ScenarioSpec.make(Stratum("cutin", 1), r["theta"]).to_dict()
            for r in curated_dataset("cutin", 1)["scenarios"] if demonstrated(r["k"], r["n"], 0.8)][:a.max_scenarios]
    if not scen:   # no PPO data yet: the capability suite's hand-chosen cases
        from learning.capability_suite import CASES
        scen = [ScenarioSpec.make(Stratum(f, 1), t).to_dict() for f, t in CASES.values()]
    jobs = [(spec, w, s, seed) for spec, w in (({"name": "mpc"}, MPCWeights().to_vector().tolist()),
                                               ({"name": "paralo_mpc"}, ParaloWeights().to_vector().tolist()))
            for s in scen for seed in a.seeds]
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("fork")) as ex:
        steps = [r for rows in ex.map(_closed_loop_job, jobs) for r in rows]

    res = {"steady_state": {"n_points": len(steady), "classification": classification(steady),
                            "gentle_curve_example": next(r for r in steady if r["mu"] == 1.0 and r["v0"] == 20.0
                                                         and r["delta_cmd"] == 0.075 and r["a_x"] == 0.0)},
           "closed_loop": {"n_scenarios": len(scen), "seeds": a.seeds,
                           "by_controller": {c: classification([r for r in steps if r["controller"] == c])
                                             for c in ("mpc", "paralo_mpc")}},
           "notes": ["physical indicators from the simulator's own tire forces (learning.vehicle_physics)",
                     "rollover_physical = quasi-static |a_y| >= SSF g; reduced-order, not a validated roll model",
                     "probabilities are NOT recalibrated; reliability bins show their calibration as-is"]}
    (out / "risk_validation.json").write_text(json.dumps(res, indent=2))
    (out / "steady_state_sweep.jsonl").write_text("".join(json.dumps(r) + "\n" for r in steady))
    _plots(steady, out)
    print(json.dumps({k: {p: {m: v[m] for m in ("n_events", "tp", "fp", "fn", "false_positive_rate", "brier")}
                          for p, v in cl.items()} for k, cl in
                      [("steady", res["steady_state"]["classification"]),
                       *[(f"closed_loop:{c}", v) for c, v in res["closed_loop"]["by_controller"].items()]]}, indent=1))


if __name__ == "__main__":
    main()
