# Paralo_plus — Project Description

*As of 2026-11-04, commit 1b5d5ef on `main`.* This file describes what the project is for, how every part works, the cost functions, how the car is controlled, what has been measured, and what is still open. Commands are in [how_to_run.md](how_to_run.md).

**Contents**

1. Goal
2. Repository map
3. Architecture
4. Simulation core
5. Surrounding traffic
6. PARALO risk models
7. RL environment
8. Feasibility oracle
9. Artifacts and on-disk formats
10. Original MPC and J_eval minimax (baseline)
11. PARALO MPC: cost function and control
12. Physical evaluation and constraints
13. Constrained minimax
14. Reproducibility
15. Physics and data audit
16. Earlier results
17. Tests
18. Design decisions
19. How the evaluation design was chosen
20. Limitations
21. Open questions
22. Status
23. PPO evaluation and reporting

---

## 1. Goal

Paralo_plus tunes the cost weights of a model predictive controller (MPC) that drives an automated vehicle in curved, three-lane highway traffic. The aim is one weight vector that holds up in the worst scenario that is still solvable, not one that is good on average.

```
w* = argmin_{w ∈ W}  max_{θ ∈ Θ_f}  J_progress(w, θ)
     s.t.            max_{θ ∈ Θ_f}  R_i(w, θ) ≤ ε_i        for every enforced constraint i
```

| Symbol | Meaning |
|---|---|
| w | The MPC's 7 cost weights. The only thing tuned, shared across all scenarios (§11) |
| θ | One scenario: gaps, relative speeds, road curvature, friction μ, escape-gap size (§8.1) |
| Θ_f | The admissible scenarios: those a PPO policy demonstrably solves, or a surrogate is confident it would (§8) |
| J_progress | How far short of the goal the ego stops, 0 to 1 (§12) |
| R_i, ε_i | Physical event rates and peaks, and their limits (§12) |

The filter Θ_f exists so the adversary cannot win by choosing impossible scenarios.

A **stratum** is one (scenario family, blocker side) pair: `cutin +1`, `cutin −1`, `sandwich +1`, `sandwich −1`. Strata keep separate datasets and surrogates, but a run over several strata uses one shared w.

The end-to-end flow is:

1. Sample θ and train a PPO policy on each one.
2. Audit the data, fit a feasibility surrogate per stratum, and validate its admission decisions.
3. Run the exchange search for w: a master problem finds the best w on the current scenario set, and an adversary looks for an admitted scenario that breaks it.
4. Compare on held-out scenarios: original MPC under old and new events, PARALO MPC at w₀, and PARALO MPC at the tuned w.

> A PPO failure on θ is evidence about *this training procedure*, not proof that θ is physically unsolvable.

This is **research code**: the style favours short `assert`s over defensive `if`/`raise` handling.

---

## 2. Repository map

```
Paralo_plus/
├── solvers/rk4.py                 RK4 integrator
├── model/
│   ├── car/config.py              VehicleParameters (SUV), G
│   ├── car/model.py               CarDynamics: dynamic bicycle model in Frenet frame
│   ├── car/car.py                 Car, CarState: one vehicle, steps itself with RK4
│   ├── road/road.py               Road, Lane: clothoid road, per-lane geometry
│   ├── collision.py               overlap tests, plastic crash for surrounding cars, ego overlap
│   └── traffic_step.py            step_surr_agents(): ONE step of all surrounding traffic
├── controllers/
│   ├── idm.py                     IDM car-following + IDM_PRESETS
│   ├── mobil.py                   MOBIL lane-change decision
│   ├── far_near.py                two-point steering + FAR_NEAR_PRESETS (traffic and the original MPC)
│   └── stanley.py                 Stanley steering (unused)
├── initialization/traffic_init.py TrafficAgent, legacy 16-slot traffic template
├── risks/                         PARALO risk models: rollover, slippage, departure, c2c
├── learning/
│   ├── env.py                     EgoTrafficEnv (Gymnasium): the hub
│   ├── scenario.py                ScenarioConfig: 15-car compact scenario
│   ├── train.py, eval.py, eval_batch.py, animate_rollout.py    PPO training and inspection
│   ├── feasibility_common.py      shared arena plumbing, scenario_id hash, ArenaRuntime
│   ├── feasibility_cutin.py       cut-in family (θ: 8 dims)
│   ├── feasibility_sandwich.py    sandwich family (θ: 10 dims)
│   ├── feasibility_sampling.py    LHS/Sobol design of experiments + validity rejection
│   ├── feasibility_train_one.py   Runner A: train + evaluate one θ
│   ├── feasibility_pipeline.py    Runner B: pilot / run / fit-only
│   ├── feasibility_dataset.py     resumable JSONL dataset, single-writer
│   ├── feasibility_surrogate.py   classifier selection, bootstrap, SurrogateArtifact
│   ├── feasibility_interpret.py   2D heatmaps of feasible regions
│   ├── feasibility_audit.py       dataset audit, relabel, curation, admission validation
│   ├── vehicle_physics.py         steering/curvature conversions, tire state, event detectors, friction modes
│   ├── ego_control.py             EgoController protocol, EgoView, to_env_action
│   ├── mpc.py                     original SamplingMPC
│   ├── paralo_mpc.py              PARALO MPC
│   ├── paralo_vec.py              vectorized PARALO probabilities
│   ├── closed_loop.py             Stratum, ScenarioSpec, J_eval, episode runners
│   ├── risk_eval.py               PARALO channels on the live simulator state (inputs for J_eval)
│   ├── physical_eval.py           episode → progress, events, peaks; ConstraintSpec
│   ├── risk_validation.py         PARALO probabilities against simulator ground truth
│   ├── capability_suite.py        seven fixed maneuvers for each controller
│   ├── minimax_core.py, minimax_feasibility.py, minimax_oracle.py, minimax_adversary.py,
│   │   minimax_store.py, minimax_pipeline.py       original exchange search (J_eval)
│   └── constrained_core.py, constrained_adversary.py, constrained_pipeline.py   constrained search
├── configs/minimax/, configs/constrained/          experiment configs
├── tests/                         pytest suite (432 tests) + 3 demo scripts
├── docs/description.md, docs/how_to_run.md
├── requirements.txt               numpy, scipy, matplotlib, gymnasium, SB3, torch, sklearn, pytest
└── pyproject.toml                 pytest config only (pythonpath=".")
```

`artifacts/` holds all generated data. It is not in git, and it does not currently exist (§9).

---

## 3. Architecture

### 3.1 Layers (each layer imports only from the layers below it)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ MINIMAX SEARCH   constrained_* (current), minimax_* (baseline)           │
│   exchange loop ─► adversary ─► screen ─► closed-loop evaluation         │
├──────────────────────────────────────────────────────────────────────────┤
│ CONTROLLERS      paralo_mpc, mpc, PPO policies   (via ego_control)       │
│ EVALUATION       physical_eval, closed_loop, risk_eval                   │
├──────────────────────────────────────────────────────────────────────────┤
│ FEASIBILITY PIPELINE                                                     │
│  feasibility_pipeline (Runner B) ──► feasibility_sampling                │
│        │                        ──► feasibility_dataset                  │
│        │                        ──► feasibility_surrogate ◄── _interpret │
│        ▼                                                                 │
│  feasibility_train_one (Runner A) ──► eval_batch.run_episodes            │
├──────────────────────────────────────────────────────────────────────────┤
│ SCENARIO BUILDERS                                                        │
│  feasibility_cutin / feasibility_sandwich  (via feasibility_common)      │
│  scenario.ScenarioConfig                   traffic_init (legacy)         │
├──────────────────────────────────────────────────────────────────────────┤
│ RL ENVIRONMENT        learning/env.py :: EgoTrafficEnv                   │
│   ego: Car.step(action)   surr: step_surr_agents   rollover event        │
├──────────────────────────────────────────────────────────────────────────┤
│ TRAFFIC               model/traffic_step.py :: step_surr_agents          │
│   IDM ─ MOBIL ─ far-near   +   model/collision.py                        │
├──────────────────────────────────────────────────────────────────────────┤
│ PHYSICS               Car ─► CarDynamics ─► rk4      Road / Lane         │
└──────────────────────────────────────────────────────────────────────────┘
```

One exception to the layering: `feasibility_sandwich` imports `MAX_BRAKING` and `DT` from `learning.env`, so the two modules share one braking clamp.

### 3.2 One env step (`EgoTrafficEnv.step`)

PPO and both MPCs drive the car through this same step. A controller returns a physical (accel, δ), `to_env_action` maps it to the normalized action, and the env applies its own limits.

```
action [-1,1]² ─► unscale ─► rate-limit (steer 1 rad/s, jerk 40 m/s³)
      ─► ego Car.step(accel, δ, κ(s), μ)                     (ego moves first)
      ─► arena.pre_surr_step(t, ego, agents)  [arena path only]
             cut-in:   sets the cutter's target_lane / lane_change_t0
             sandwich: returns {front_1: −8 m/s²} accel override
      ─► step_surr_agents(agents, ego_car=ego, accel_override=…)
             MOBIL ─► IDM ─► far-near ─► Car.step ─► commit lane ─► pose
             then: surr–surr plastic collisions
      ─► checks: collision > rollover (§7) > off-road; goal reached; then reward
      ─► obs (7 + 5·N_surr), terminated / truncated, info
```

### 3.3 Data flow from θ to surrogate

```
feasibility_sampling.sample_cutin(n, side, seed)    ── LHS in θ bounds, reject invalid
      │  list[CutinConfig]
      ▼
Runner B run_campaign ── ProcessPoolExecutor(concurrency) ──┐
      │                                                     │ per θ, in a worker process:
      │                                                     ▼
      │                         train_and_evaluate(family, cfg, RunParams)
      │                           PPO(SubprocVecEnv(EgoTrafficEnv(arena=CutinRuntime(cfg))))
      │                           early stopping on held-out env, keep best checkpoint
      │                           run_episodes(..., episodes=100, distribution mode)
      │                           ─► policies/<fam>/<side>/<id>/model.zip + model.meta.json
      │                           ─► evaluations/<fam>/<side>/<id>.result.json   (worker-owned)
      ▼
coordinator appends ─► datasets/<fam>/<side>/scenarios.jsonl  (1 row per θ)
                    ─► datasets/<fam>/<side>/rollouts.jsonl   (1 row per episode, y∈{0,1})
      │
      ▼
feasibility_audit ─► curated rows (out-of-bounds and legacy-label rows left out)
      │
      ▼
fit_surrogate(rollouts) ── GroupKFold by scenario_id ─► pick lowest Brier
      │                  ── bootstrap over scenario_id groups (200×)
      ▼
SurrogateArtifact.joblib ─► .predict(θ) = {p_hat, epistemic_interval, warnings}
      │
      ▼
feasibility_interpret ─► heatmaps/<fam>_side<k>/heatmap_<a>_vs_<b>.png
```

---

## 4. Simulation core

Every vehicle is a dynamic bicycle model in **Frenet coordinates**, integrated with RK4. Coordinates are SAE J670: x forward, **y right**, z down. **Positive `e_y` and positive `δ` mean "to the right".** Most sign bugs come from forgetting this.

| Module | Contents | Notes |
|---|---|---|
| `solvers/rk4.py` | `rk4(f, t, x, dt)` | Classic 4th-order Runge–Kutta step. |
| `model/car/config.py` | `VehicleParameters` (frozen), `G` | SUV: m = 2000 kg, l_f = 1.2 m, l_r = 1.6 m, I_zz = 2875, C_f = 19 kN/rad, C_r = 33 kN/rad, h_CG = 0.80 m, track W_c = 1.55 m. `SSF = W_c/(2h_CG)` is the rollover onset in g. Also the tire settings of §4.1. `C_d` is declared but unused. |
| `model/car/model.py` | `tire_forces`, `CarDynamics` | State `[s, e_y, e_psi, v_x, v_y, r]`. Friction-circle tire forces on each axle with realized load transfer (§4.1). `step()` only computes `d_states`. |
| `model/car/car.py` | `CarState`, `Car` | `Car.step(accel, delta, kappa, mu, dt)` builds a fresh `CarDynamics` at every RK4 stage. The **caller** looks up κ and μ. `CarState` also stores x, y, heading, lane and the last δ. `behaviour` 1/2/3 means conservative/moderate/aggressive. |
| `model/road/road.py` | `Road`, `Lane` | Profile: straight → clothoid → 80 m constant arc → clothoid → straight. Grid ds = 0.1 m; 4 m lanes. Each `Lane` stores its own offset, κ and arclength. Helpers: `index_at(s)`, `frenet_to_global(s, e_y, e_psi)`. |
| `model/collision.py` | `bodies_overlap`, `resolve_surr_collisions`, `bleed_crashed`, `ego_overlaps_any`, `body_gap`, `body_half_extent_e_y` | Boxes in (s, e_y). Ego collision uses boxes turned by each car's e_psi (separating-axis test); surr–surr uses axis-aligned ones. Surr–surr crashes are perfectly plastic: shared momentum, lateral motion frozen, then the wreck bleeds speed. An ego overlap **never** goes through the crash model; it ends the episode. |
| `model/traffic_step.py` | `step_surr_agents(...)` | The **only** implementation of surrounding-traffic stepping. `tests/traffic_test.py` and the env both call it, so they cannot diverge. |

`step_surr_agents` does the following for each non-crashed agent each tick:

1. **MOBIL** lane-change decision. It is skipped while a change is in progress or during the 1 s cooldown.
2. **IDM** acceleration toward the target lane's leader, clamped at −8 m/s². `accel_override={car_id: a}` replaces the IDM value.
3. **Far-near steering** plus curvature-preview feedforward, rate-limited and held within the ±0.5 rad steering lock. During a lane change, the e_y target ramps over the lane-change duration.
4. Look up κ and μ, then call `Car.step`.
5. Commit the lane change once its duration has elapsed (the discrete `lane` flips at this point).
6. Store the global pose.

After every car has moved, it resolves surr–surr collisions once. Passing `ego_car=` wraps the ego in a temporary `TrafficAgent` proxy with id `EGO_CAR_ID = -1`. That makes the ego visible to every surrounding car's IDM and MOBIL.

### 4.1 Tire forces and vehicle dynamics

`model.car.model.tire_forces` is the single source of every force. It runs at every RK4 stage, for the ego and all surrounding cars. **`accel` is a request, not an acceleration:** the state only ever sees the saturated forces.

**Body dynamics (SAE: y right, +r clockwise from above).** The front forces act in the wheel frame, so they are rotated by δ:

```text
m (dv_x − r v_y) = F_xf cos δ − F_yf sin δ + F_xr
m (dv_y + r v_x) = F_xf sin δ + F_yf cos δ + F_yr
I_zz dr          = l_f (F_xf sin δ + F_yf cos δ) − l_r F_yr
```

**Slip angles, from wheel-frame velocity.** With v_fy = v_y + l_f r:

```text
α_f = −atan2(−v_x sin δ + v_fy cos δ, max(|v_x cos δ + v_fy sin δ|, v_tire_min))
α_r = −atan2(v_y − l_r r, max(|v_x|, v_tire_min))
F_y,req = C α
```

Above `v_tire_min` (1 m/s) this is exactly δ − atan2(v_y + l_f r, v_x) and −atan2(v_y − l_r r, v_x). Below it, the tire acts as a damper on wheel side-slip whose equilibrium is kinematic rolling, so a parked car stays parked and `v_x ≤ 0` never gives an angle near ±π. The floor also keeps RK4 stable: the stiffest mode is about −46 s⁻¹ here, and at dt = 0.05 s the amplification is 0.49. Below about 0.8 m/s it would leave the stability region.

**Longitudinal request and allocation.**

```text
F_req = m · accel                                         accel in the env's [−8, 4]
braking only: F_req ·= clip(v_x / (v_stop · max(1, |accel|/μg)), −1, 1)
F_xf,req = share · F_req,  F_xr,req = (1 − share) · F_req
```

The fade makes a brake force oppose motion and vanish at rest: **braking stops the car and never reverses it, and there is no reverse gear.** `brake_front_share` and `drive_front_share` set `share`. `None`, the default, means proportional to axle load, F_z,f/(mg): both axles reach their limit together, so the straight-line limit is exactly μg. For braking this is an ideal ABS/EBD assumption, for drive an idealized AWD. No drivetrain specification exists, and a fixed share gives a lower limit (one axle saturates first, with no redistribution).

**Combined saturation on each axle:**

```text
F_x = F_x,req · min(1, μF_z / √(F_x,req² + F_y,req²))
F_y,max = √(max(0, (μF_z)² − F_x²))
F_y = clip(F_y,req, ±F_y,max)
```

So F_x² + F_y² ≤ (μF_z)² holds exactly. Where the request leaves the circle, F_x and F_y shrink by the same factor, and the force keeps pointing along the slip direction.

*Why not give F_x the whole circle first?* That was tried. At full braking, F_y is then zero on both axles, nothing damps yaw, and the brake force along the steered wheel turns the car **against** its steering, in 647 of 1,944 braking turns and by up to 284°. Reserving 30% of the circle for lateral force still leaves 150 cases, because it only helps below δ ≈ 0.3 rad. Direction-preserving scaling leaves none in the same sweep.

**Normal loads** are quasi-static on the **realized** body acceleration:

```text
F_zf = clip((m g l_r − m h_CG a_x) / L, 0, mg),   F_zr = mg − F_zf
a_x = (F_xf cos δ − F_yf sin δ + F_xr) / m
```

Since a_x depends on F_z, the function does five fixed-point passes (worst residual ≈ 1e-3 m/s²). The circle holds exactly for the F_z it returns.

**Request to realized.** Four quantities must not be confused:

| Quantity | Meaning | Where |
|---|---|---|
| requested | the policy's raw command | `info["accel_requested"]` |
| actuator-limited | after the rate limits (steering 1 rad/s, jerk 40 m/s³) | `info["accel_actuator"]`, also the observation feature and the jerk limiter's state |
| tire request | `F_req`, per axle before saturation | `TireState.F_xf_req`, `F_xr_req` |
| realized | a_x from the saturated forces | `info["accel_realized"]`, `TireState.a_x` |

**Events keep four things apart** (`learning/vehicle_physics.py`):

| Indicator | Meaning |
|---|---|
| `slip_lateral_saturation` | **Tire-force saturation:** |F_y,req| exceeds the F_y,max left on some axle |
| `slip_friction_circle` | **Excess demand:** some axle's requested √(F_x,req² + F_y,req²) ≥ μF_z. It says what was asked for, not what happened |
| `alpha_f`, `alpha_r` | **Tire slip angles**, reported as they are |
| `stability_loss` | **Loss of stability:** |body sideslip| ≥ 10° while moving at ≥ 1 m/s. The threshold is a provisional rule of thumb |

A tire at its limit is not an unrecoverable skid, so saturation never sets `stability_loss`. None of the four ends an episode. `tire_utilization` is the realized counterpart of the demand, and it stays ≤ 1.

**Validation.** `tests/test_tire_forces.py` covers straight braking, cornering, combined braking and drive, low speed and load transfer, recording forces at every RK4 stage. Over 144,000 stages of random stress the largest excess over the circle was 2e-12 N. At dt = 0.05 s, final states on aggressive maneuvers match dt = 0.0025 s to within 1 mm in position and 8 mm/s in speed.

---

## 5. Surrounding traffic

These controllers drive the surrounding cars. The ego is driven by PPO or an MPC. One exception: the original ego MPC reuses far-near as its steering tracker (§10.1).

| Module | Role | Parameters |
|---|---|---|
| `controllers/idm.py` | `idm_accel()`. Not clamped internally; the caller clamps. | `IDM_PRESETS`: T = 2.0/1.4/0.8 s, s0 = 3/2/1 m, a_max = 0.8/1.5/2.5, b = 1.3/2.0/3.0. The desired speed `v0` is set per car. |
| `controllers/mobil.py` | `mobil_decision()`, `MobilParams` | politeness 0.2, threshold 0.2 m/s², plus a `b_safe` braking cap. The arena path sets `threshold=inf`, which disables discretionary lane changes. |
| `controllers/far_near.py` | Steering, curvature feedforward, `clip_steering_rate` | `FAR_NEAR_PRESETS`: shared gains (L_n = 5 m, T_f = 1 s, k = 2.5). Rate limit 0.6/0.8/1.0 rad/s; lane-change duration 5.5/4.0/2.5 s. |
| `controllers/stanley.py` | `stanley_steering()` | **Not used.** |
| `initialization/traffic_init.py` | `TrafficAgent`, `generate_traffic`, `steady_state_speed`, `CAR_LENGTH=4.0`, `CAR_WIDTH` | Legacy path only. Fixed 16-slot template (1 ego + 15 surrounding cars, 3 lanes). The seed changes only the headways and desired speeds. Initial speeds are IDM steady-state speeds. |

`TrafficAgent` = `Car` + controller runtime state: `v0`, `target_lane`, `lane_change_t0`, `prev_delta`, `last_lane_change_t`, `crashed`, `crash_t`, `crash_v`. **Every** surrounding vehicle on every path is a `TrafficAgent`. Scenario families script behaviour by editing these public fields; for example, a cut-in sets `target_lane` and `lane_change_t0`.

**Why far-near.** It is a two-point model of how human drivers steer. The near point keeps the car centred, and the far point mostly reflects heading, so it anticipates the road. Curvature feedforward, atan(κL), adds the steering a curve needs before any error builds up. The presets give human-like lane changes with a set duration, which is what lets the cut-in scenario time the cutter.

**Stanley** is not called anywhere. Only far-near's docstring mentions it, as a comparison.

---

## 6. PARALO risk models

All four modules are pure-function ports of the former PARALO ROS nodes. Each takes an explicit predicted trajectory and returns an **unweighted** (probability, score) pair; the caller applies its weight. The cumulative risk follows PARALO §5.2.5, eq. 5.51. Departure is used twice, against the lane and against the road, which gives five risk channels.

| Module | Physics | Used by |
|---|---|---|
| `risks/rollover.py` | §5.2.2, eqs. 5.2–5.6: v_safe from turn radius, track width and CG height; P_roll = Φ((v − v_safe)/(Δv/3)) | PARALO MPC, J_eval, and the legacy rollover event (P_roll > 0.5) |
| `risks/slippage.py` | §5.2.2, eqs. 5.9–5.19: speed-dependent μ, minimum safe speed, slip probability | PARALO MPC, J_eval |
| `risks/departure.py` | Gaussian clearance to a quadratic lane centreline with coefficient covariance | PARALO MPC (lane and road), J_eval. The env itself uses a geometric `_ego_off_road` |
| `risks/c2c.py` | Rectangle-overlap probability per axis × PARALO closure kinetic-energy severity | PARALO MPC, J_eval |

`learning/paralo_vec.py` holds vectorized versions for the MPC. Tests check they equal `risks/` exactly.

### 6.1 Two slippage models

The simulator decides when the car physically slips. PARALO predicts slip risk before the car drives a plan. They differ because they have different jobs and different information.

**Simulator (ground truth, `model/car/model.py`).** A dynamic bicycle model with states v_x, v_y and r.

- Slip angles come from real motion: α_f = δ − atan((v_y + l_f·r)/v_x) and α_r = −atan((v_y − l_r·r)/v_x).
- Tire force is linear, F_y = C·α, and shares a friction circle with the longitudinal force on each axle (§4.1). F_z includes load transfer from the realized acceleration.
- A slip event (`learning/vehicle_physics.py`) is either an axle's F_y,req exceeding the capacity left after its F_x, or an axle's requested combined force reaching μ·F_z. Both are separate from loss of stability.

**PARALO (prediction, `risks/slippage.py`).** It sees only (v, δ, a) per predicted step.

- The turning radius comes from a no-slip bicycle model: β = atan(l_r·tanδ/L), R = l_r/sin β.
- The safe speed is the smallest of v_SS = √(μgR), v_US = √(α_max·R·C_f·L/(m·l_r)) and v_OS = √(α_max·R·C_r·L/(m·l_f)), with α_max = 8.6°·μ.
- Cornering slip: P_CS = Φ((v − v_safe)/(Δv/3)).
- Braking slip: P_BS = 1 − Φ((a − a_min)/(g·U_μ/3)), with a_min = −μg·cos β.
- Combined: P_slip = 1 − (1 − P_CS)(1 − P_BS). The only uncertainty is U_μ = 0.05.

**Why the two disagree.**

1. **Calibration.** In a steady turn the simulator's axles saturate at a_y = μg ≈ 9.8·μ m/s². PARALO's understeer limit sets in at v²/R = α_max·C_f·L/(m·l_r) ≈ 2.5·μ m/s², about 4× lower. The 8.6° figure suits stiff real tires; the simulator's front axle (C_f = 19 kN/rad) saturates near 34°·μ. This is why P_slip(δ_eq) still has 60% false positives in steady state (§15, B7).
2. **Radius from geometry.** PARALO infers lateral demand from steering geometry. Given commanded δ on this strongly understeering car, it sees a turn far tighter than the real one. δ_eq fixes that input (§15).
3. **Braking and cornering are separate.** PARALO uses separate limits and combines the probabilities as if independent. The friction circle combines the forces, so moderate braking plus moderate cornering can exceed it while each stays under its PARALO limit.
4. **No transients.** PARALO uses a steady-state radius. A sharp steering input can saturate the front tire before yaw rate builds, and δ_eq, derived from yaw rate, lags behind.

Points 3 and 4 are the leading guesses, not yet confirmed, for why PARALO misses most real slip steps (§21, question 1).

---

## 7. RL environment (`learning/env.py`)

`EgoTrafficEnv` is the single Gymnasium env that every training, evaluation and animation script uses. It has **three mutually exclusive ways** to build the traffic:

| Path | Constructor argument | Surrounding cars | Obs dim | Builder |
|---|---|---|---|---|
| Legacy | neither | 15 | 82 | `traffic_init.generate_traffic` |
| Scenario | `scenario_config=ScenarioConfig` | 15 (3 critical + 12 background) | 82 | `scenario.generate_scenario` |
| **Arena** (feasibility) | `arena=CutinRuntime / SandwichRuntime` | 6 | 37 | `arena.reset()` + `arena.pre_surr_step()` each step |

`mode` is `"fixed"` or `"distribution"`. Fixed mode rebuilds the identical realization every reset. Distribution mode varies the realization per episode via `derive_seed(seed, worker_rank, episode_index)`.

**Constants:** DT = 0.05 s; 3 lanes; the ego starts in the middle lane at s = 50 m. The legacy road has s_max = 500 m and κ_max = 0.005. Episode length is 20 s by default; scenario and arena configs override it with `max_episode_seconds`.

**Action:** `Box[-1,1]²`, mapped to accel ∈ [−8, +4] m/s² and δ ∈ ±0.5 rad, then rate-limited by `MAX_STEERING_RATE = 1.0 rad/s` and `MAX_JERK = 40 m/s³`.

**Observation:** 7 ego features plus 5 per surrounding car, in fixed slot order with no padding.

- **Ego:** v_x, e_y relative to the nearest lane, e_psi, nearest-lane index, κ 20 m ahead, applied δ, applied accel.
- **Per surrounding car:** relative s, relative lane, relative lateral offset, e_psi, v_x.

**Outcome.** Computed from physics only, never from reward:

```
success = scenario_goal_reached and not (collision or rollover or off_road)
```

- Failure priority: COLLISION > ROLLOVER > OFF_ROAD.
- **OFF_ROAD** fires once any part of the body, turned by its heading, is more than `ROAD_EDGE_MARGIN_M` (0.5 m) past the paved edge. It is the same rule for PPO and the MPC, because both use this environment. Policy metadata records the margin as `off_road_margin_m`, and results with different margins are flagged as non-equivalent.
- **The ROLLOVER event is selectable** with `rollover_event`:
  - `physical_v2`: realized |a_y| ≥ SSF·g ≈ 9.5 m/s². This is what new PPO data and the constrained pipeline use.
  - `legacy_v1`: P_roll(v_x, δ_cmd) > 0.5. It fires at about 2 m/s² on the default curve (§15). It is the env's constructor default, so historical runs reproduce, and the baseline J_eval pipeline (`closed_loop`) still runs on it.
- Slippage and lane departure never end an episode. They are events defined in `physical_eval` (§12).
- Goal reached or a failure sets `terminated`. Hitting the time budget sets `truncated`, with outcome TIMEOUT. **A timeout is never a success.**
- `info` keys: `success`, `outcome`, `rollover_event_model`, `failure_reason`, `progress_m`, `distance_travelled_m`, `elapsed_s`, `min_clearance_m`, `max_p_rollover`, `max_road_departure_m`, plus the reward fields below. On the arena path it also has `arena`, with family diagnostics such as the trigger station and time.

**Reward (`goal-progress-v1`).** PPO is an empirical feasibility oracle, so the reward says only how far the ego got and how the attempt ended. It is never used to classify success.

r_t = r_progress + r_terminal − r_time

- **r_progress** = Δs⁺ / D_goal, where D_goal = goal_s − s_start and Δs⁺ = min(s_max,t, goal_s) − min(s_max,t−1, goal_s). s_max is the running maximum of s, starting at the ego's start. Retraced ground pays nothing, and an episode's sum is at most 1.
- **r_terminal** is a single value on the last step:

| Outcome | r_terminal |
|---|---:|
| Safe goal completion | +3 |
| Collision, rollover or road departure | −5 |
| Timeout | −1 |
| Nonterminal step | 0 |

- **r_time** = 0.0005 per executed step, including the last. That is 0.01 per simulated second at DT = 0.05 s. DT is one fixed module constant; if it changes, rescale the penalty to keep the per-second rate.

r_terminal is read from the outcome the env has already decided, so failure priority applies and a failure on the goal step pays −5, never +3. Two failures on one step pay once, by priority.

Nothing else is in the reward: no lane, heading, steering, acceleration, jerk, comfort or risk term, no survival bonus and no potential-based shaping. `RewardConfig` holds the coefficients. Penalties are positive magnitudes. `min_progress_m` no longer affects the reward; it only sets `lack_of_progress`.

`info` carries `reward_progress`, `reward_terminal`, `reward_time` (the positive magnitude), `reward_total`, and the running `episode_return`, `episode_length` and `episode_progress` (fraction of D_goal reached). Episode outcomes stay in the existing keys on the last step: `success`, `collided`, `rolled_over`, `off_road`, `timeout`.

**Discount.** `DEFAULT_PPO_GAMMA = 0.999` (`--gamma`). At 20 steps/s, a reward 20 s away is discounted to 0.018 at γ = 0.99 and 0.67 at γ = 0.999. Discounted return is not the probability of success: discounting, r_progress and r_time all move the policy's preference. On the cut-in defaults at γ = 0.999, stopping and timing out is worth more than attempting the maneuver unless P(success) ≥ about 0.47, taking a success at 20 s and a crash at 10 s. Undiscounted, success (≥ 3.65) still beats any timeout (< −0.35 on the 35 s budget), and any timeout beats any failure (≤ −4).

`REWARD_VERSION` and `ppo_gamma` are stamped into the meta files and dataset rows. Every result in §15 and §16 came from the earlier reward (progress boost, goal potential, lane tracking, survival bonus, γ = 0.99).

### 7.1 `learning/scenario.py` (scenario path)

`ScenarioConfig` has:

- global fields: `road_mu`, `road_kappa_max`, background speed mean and std, `aggressive_fraction`, `lane_change_tendency`, `traffic_spread`;
- three `CriticalActorConfig`s (`front`, `rear`, `blocker`), each relative to the ego;
- `goal_distance_m=150`, `max_episode_seconds=20`, `min_progress_m=50`, `seed`.

This path came before the feasibility families. It is still supported, but the pipeline does not use it.

### 7.2 Supporting scripts

| Script | Purpose |
|---|---|
| `learning/train.py` | PPO on the legacy or scenario path (`--scenario-config scenario.json --scenario-mode fixed\|distribution`). Writes `<out>.zip` + `<out>.meta.json`. |
| `learning/eval.py` | Matplotlib animation of one episode. Reads the meta to rebuild any of the three env paths. |
| `learning/eval_batch.py` | N episodes per model, success rate with a Wilson 95% CI, and a FEASIBLE / NOT_SOLVED / INCONCLUSIVE label across PPO seeds. Its `run_episodes()` is reused by Runner A. |
| `learning/animate_rollout.py` | Replays evaluation episode K of a feasibility policy exactly (distribution mode, rank 0, K+1 resets). Use it to inspect a specific failure from `rollouts.jsonl`. |

---

## 8. Feasibility oracle

### 8.1 Scenario families (the arena)

Both families use **six agents** in a fixed slot order `[front_1, front_2, rear_1, rear_2, blocker_1, blocker_2]` (car_ids 0–5). All six are "moderate" behaviour, and MOBIL is disabled for all of them. The scenario event is triggered **spatially**: it fires once, when `ego.s ≥` the realized trigger station.

| | **Cut-in** (`feasibility_cutin.py`) | **Sandwich** (`feasibility_sandwich.py`) |
|---|---|---|
| Event | `blocker_1` does a prescribed lane change into the ego's lane once the ego passes `s_cutin_m` | `front_1` does a prescribed emergency stop (−8 m/s², held at standstill) once the ego passes `s_stop_m` |
| Mechanism | `pre_surr_step` sets `target_lane` / `lane_change_t0`; the far-near ramp does the rest | `pre_surr_step` returns `accel_override` |
| θ dims | 8 | 10 |
| θ fields (bounds) | `road_mu` [0.4, 1.0], `road_kappa_max` [0, 0.02], `s_cutin_m` [100, 380], `cutter_gap_m` [8, 60], `cutter_relative_speed_mps` [−5, 15], `front_gap_m` [25, 100], `front_relative_speed_mps` [−12, 5], `escape_gap_length_m` [8, 40] | `road_mu`, `road_kappa_max`, `s_stop_m` [100, 380], `front_gap_m` [8, 60], `front_relative_speed_mps` [−12, 5], `rear_gap_m` [8, 60], `rear_relative_speed_mps` [−10, 12], `escape_gap_center_m` [−20, 30], `escape_gap_length_m` [8, 40], `blocker_relative_speed_mps` [−10, 10] |
| Settings outside θ | `cutin_duration_s` (4.0) | `emergency_deceleration_mps2` (−8) |

**Shared settings** (`feasibility_common.ArenaCommonConfig`):

| Setting | Default | Notes |
|---|---|---|
| `blocker_side` | — | ±1: the side of the ego's lane the blockers are on |
| `seed` | — | Realization seed |
| `mode` | `"robust"` | `"exact"` zeroes every perturbation; `"robust"` adds ±5% on gaps, ±0.5 m/s on speeds and ±2 m on the trigger station |
| `goal_distance_m` | 400 | |
| `max_episode_seconds` | 35 | |
| `min_progress_m` | 100 | |
| `nominal_ego_speed_mps` | 20 | All actor speeds are relative to this |

The bounds are versioned as `cutin-bounds-v1` and `sandwich-bounds-v1` (`THETA_BOUNDS_VERSION_*`). Every dataset row is stamped with its version, because old rows generated under other bounds were a real problem (§15, A3).

**`scenario_id`** is the first 16 hex characters of the SHA-256 of `{schema_version, family, blocker_side, mode, sorted θ}`. It excludes the seed, so all rollouts of one θ share an id. Bump `SCHEMA_VERSION` whenever the θ schema changes.

`ArenaRuntime` (a Protocol, duck-typed) is the contract a new family must satisfy:

- `reset(road, ego_s, ego_lane, rng, *, worker_rank, episode_index, scenario_seed) → (agents, ego_v0, realized_dict)`
- `pre_surr_step(t, ego_state, agents) → dict | None`
- `extra_info() → dict`
- attributes `goal_distance_m`, `max_episode_seconds`, `min_progress_m`.

### 8.2 Sampling (`feasibility_sampling.py`)

`sample_cutin` and `sample_sandwich(n, blocker_side, seed, method="lhs"|"sobol")` draw a space-filling design with scipy's `qmc` module inside the θ bounds. They reject and redraw any point the family validator rejects. Rejected points come back as a `RejectedSample` list for logging. The same inputs always give the same batch.

### 8.3 Runner A — `feasibility_train_one.py`

`train_and_evaluate(family, cfg, RunParams)` is the single implementation of "train and evaluate one θ". The CLI and Runner B both call it.

- PPO (SB3 `MlpPolicy`) with a `SubprocVecEnv` of `n_envs` arena envs.
- **Early stopping**: every `eval_freq_timesteps` (100k), it evaluates `patience_episodes` (20) on a held-out seed stream. It stops after `patience` (5) evaluations with no improvement, and only once `min_evals` (5) have run. `timesteps` (2M) is an upper bound.
- The **best** held-out checkpoint becomes `<out>.zip`. The raw final model is kept as `<out>_final.zip`. The meta's `evaluated_checkpoint` says which one `<out>.zip` is.
- Final evaluation: `episodes` (100) in distribution mode, deterministic by default.
- `RunParams.device="cpu"` and `torch_threads=1` are deliberate (see §18).
- `RunParams.rollover_event` defaults to `physical_v2`, and the value is stamped into the meta and the dataset rows.
- **Device history.** All three device bugs came from real runs. Without the CPU default, concurrent jobs fought over one GPU. Without the thread limit, every process claimed every core. And `PPO.load()` has its own `device` argument that defaulted to `"auto"`, so best-checkpoint reloads ignored `--device cpu` until every call site passed it explicitly.

### 8.4 Runner B — `feasibility_pipeline.py`

| Mode | Does |
|---|---|
| `pilot` | Benchmarks the given concurrency levels on this machine and recommends one. Writes only to `_pilot_bench/`, never to the real dataset. |
| `run` | Samples `n_thetas`, skips ids already done, reconciles finished-but-not-appended results, then trains the rest in a `ProcessPoolExecutor(concurrency)`. A failure in one θ does not stop the batch. `--max-wall-time-hours` stops launching new jobs after that budget. **Resumable**: re-run the same command. |
| `fit-only` | Fits the surrogate on the audit-curated rows of the current dataset and saves the `.joblib`, with the curation record stored as `provenance`. |

The BLAS/OMP thread environment variables are set to 1 at the top of the module, before numpy is imported. Keep it that way.

### 8.5 Dataset — `feasibility_dataset.py`

The dataset uses a single-writer design. Each worker writes only its own `evaluations/.../<id>.result.json`, atomically. Only the coordinator appends to the shared `scenarios.jsonl` and `rollouts.jsonl`. Path helpers: `dataset_dir`, `scenarios_path`, `rollouts_path`, `evaluations_dir`, `scenario_result_path`.

### 8.6 Surrogate — `feasibility_surrogate.py`

- **Training rows are rollouts** (one row per episode, `y ∈ {0,1}`), not one row per θ with a fractional target. θ is normalized to [0, 1] per field (not clipped).
- There are three candidates: `logistic_poly2`, `extra_trees` (300 trees, depth 6) and `hist_gb` (depth 4). They are compared by **GroupKFold over `scenario_id`**, and the lowest Brier score wins. Log-loss, ROC-AUC and calibration are also reported. If every rollout has the same label, `ConstantClassifier` is the fallback.
- **Epistemic uncertainty** comes from a 200-member bootstrap that resamples whole `scenario_id` groups.
- `SurrogateArtifact.predict(θ)` returns `{p_hat, epistemic_interval (2.5–97.5%), warnings}`. The warnings flag **PRELIMINARY** (fewer than 50 distinct θ) and **EXTRAPOLATION** (θ outside the bounds).
- One surrogate per (family, blocker_side). There is no surrogate across families or sides.
- **Runner C is not built.** It would train and evaluate one θ fresh and compare the empirical rate and Wilson interval with `SurrogateArtifact.predict(θ)`.

### 8.7 Interpretation — `feasibility_interpret.py`

It writes one PNG per pair of θ dimensions:

- **Empirical panel:** each θ's success rate, interpolated linearly inside the convex hull of the samples. This is a *marginal projection*, not a controlled slice.
- **Surrogate panel** (with `--surrogate`): p̂ on a grid, with the other dimensions held at the dataset median.

`--dims` restricts the plot to a subset of θ fields.

### 8.8 Audit and curation — `feasibility_audit.py`

The audit gives every dataset row a category and an inclusion decision. It never edits, clamps or deletes the original files.

| Category | Meaning | Used for fitting |
|---|---|---|
| `stamped` | Generated with a stamped, corrected event model | yes |
| `legacy_no_rollover` | Legacy row with no rollover episode, so labels are compatible | yes |
| `relabeled` | Legacy row re-evaluated under `physical_v2`, where the stored legacy label reproduced first | yes, with the new labels |
| `legacy_rollover_label` | Legacy row with false-rollover labels and no valid relabel | no |
| `relabel_not_reproducible` | Re-evaluation did not reproduce the stored legacy count | no |
| `out_of_bounds` | θ is outside the current bounds | no |

`--mode validate` runs grouped K-fold under the conservative rule and reports false admissions (admitted, but PPO could not demonstrate it), false rejections, and calibration. It has only run on synthetic data.

---

## 9. Artifacts and on-disk formats

```
artifacts/feasibility/
├── datasets/<family>/side_pos1|side_neg1/
│   ├── scenarios.jsonl        one line per θ
│   └── rollouts.jsonl         one line per evaluation episode
├── evaluations/<family>/<side>/<scenario_id>.result.json   worker output (for reconciliation)
├── policies/<family>/<side>/<scenario_id>/model.zip, model_final.zip, model.meta.json
├── surrogates/<family>_side<k>.joblib                      (created by fit-only)
├── heatmaps/<family>_side<k>/heatmap_<a>_vs_<b>.png
└── _pilot_bench/c<concurrency>_job<i>/                     pilot output
```

> **`artifacts/` is not in git, and all of it was deleted on 2026-10-28** (§15, B3). Any new policies will exist only on the machine that trained them, so back them up. `*.zip` is in `.gitignore`.

**`scenarios.jsonl` row:**

```json
{"scenario_id": "a836207f3c9fc314", "family": "cutin", "blocker_side": 1, "mode": "robust",
 "theta": {"road_mu": 0.81, "s_cutin_m": 269.4, "...": "..."},
 "k": 83, "n": 100, "success_rate": 0.83, "wilson_95ci": [0.745, 0.891],
 "n_collisions": 7, "n_rollovers": 0, "n_offroad": 0, "n_timeouts": 10,
 "mean_return": 116.7, "std_return": 14.8, "mean_success_time_s": 33.8,
 "checkpoint": "best", "stop_reason": "early stopping: ...", "timesteps_actual": 1100000, "ppo_seed": null,
 "rollover_event_model": "physical_v2", "theta_bounds_version": "cutin-bounds-v1", "git_commit": "..."}
```

**`rollouts.jsonl` row:**

```json
{"scenario_id": "a836207f3c9fc314", "episode": 0, "y": 1, "outcome": "finished", "failure_reason": null, "theta": {"...": "..."}}
```

**`model.meta.json`** holds the resolved scenario (family, id, side, mode, seed, θ, goal/time/progress settings) and the training facts (`timesteps_budget`, `timesteps_actual`, `stop_reason`, `n_envs`, `evaluated_checkpoint`, checkpoint paths). It also holds `evaluation.summary`. `eval.py` and `animate_rollout.py` rebuild the env from this file.

Constrained runs add `artifacts/constrained/experiments/<id>/`, and the baseline adds `artifacts/minimax/experiments/<id>/` (§14). Diagnostics go to `artifacts/physics_validation/`.

---

## 10. Original MPC and J_eval minimax (baseline)

This is the first tuning pipeline, kept as the baseline. It solves `w* = argmin_w max_{θ admitted} J_eval(w, θ)` with one shared w. It still uses the legacy rollover event (§7). It has only run on toy problems and small smoke budgets.

### 10.1 The original MPC and its cost

Every 2 env steps (0.1 s), `SamplingMPC` scores candidate plans of the form a_c ∈ {−8, −6, −4, −2, −1, 0, 0.5, 1, 2, 3} m/s² × target lane l_c ∈ {nearest − 1, nearest, nearest + 1}. Each is predicted over a 3 s horizon at 0.1 s steps.

- **Ego prediction:** v_k = max(0, v₀ + a_c·t_k). The lateral reference ramps toward the target lane's centre at one lane width per 4.0 s, the same rate moderate surrounding traffic uses. The tracking error about that reference decays with T_lat = 1.2 s. The implied steering is atan(L·(κ_SAE(s) + ÿ/v²)).
- **Surrounding traffic prediction:** constant speed and constant lateral velocity.
- **Lateral control:** the committed lane is tracked every step with the repository's far-near law plus curvature feedforward, using moderate-preset gains. A step reference saturated far-near and rolled the SUV over within 2 m; the ramp fixed that.

The internal cost, with w the only tuned quantity, is:

```
L_MPC(w) = w_speed·mean((v−v_ref)/v_ref)² + w_accel·[(a_c/8)² + ((a_c−a_prev)/12)²]
         + w_lane·[1(l_c≠nearest) + 1(l_c≠previous target)]
         + w_collision·[mean_k max_j exp(−ds⁺/d_safe)·exp(−dy⁺/0.5) + 10·overlap fraction],  d_safe = 2 + T_h·v
         + w_rollover·mean_k P_roll(v_k, δ_k)      (exact risks/rollover.py formulas, vectorized; tested equal)
         + w_road_edge·[mean_k exp(−margin⁺/0.5) + 10·off-road fraction]
```

| w component | Bounds | Search scale | Default w₀ |
|---|---|---|---|
| `speed` | [0.05, 20] | log | 1.0 |
| `accel` | [0.005, 5] | log | 0.1 |
| `lane_change` | [0.01, 10] | log | 0.5 |
| `collision` | [0.5, 500] | log | 20 |
| `rollover` | [0.5, 500] | log | 20 |
| `road_edge` | [0.5, 500] | log | 20 |
| `time_headway_s` (T_h) | [0.5, 3.0] s | linear | 1.5 |

The optimizer works in the unit cube [0,1]⁷ (`w_from_unit` / `w_to_unit`). Every safety weight has a strictly positive lower bound.

**Solver failure.** Enumeration always returns a plan. A non-finite cost counts as a solver failure: the MPC brakes at −8 m/s², holds the nearest lane, and the failure is counted in the episode record and the run summary.

**Consequence of reusing far-near.** The MPC can only steer as sharply as the moderate driver's lane change. That may be part of why it fails brake-and-steer and the sandwich escape (§15, B7).

### 10.2 External cost J_eval

J_eval is defined in `closed_loop.EvalCostConfig`. It never uses w or the PPO reward. Every per-step term is dimensionless and clipped to [0, `term_clip` = 10]:

| Term | Per-step expression E_k | Normalizers |
|---|---|---|
| tracking | ((v − v_ref)/v_ref)² + (e_y,nearest / 2 m)² | v_ref = 20 m/s |
| comfort | (jerk / 10 m/s³)² + (v·r / 4 m/s²)² | applied jerk; lateral accel ≈ v × yaw rate |
| control | (a / 4 m/s²)² + (δ / 0.5 rad)² | applied, post-rate-limit commands |
| risk | Σ_ch β_ch · R_ch over the five PARALO channels (below) | β = 1 each, configurable |

```
episode cost = Σ_k (DT / T_episode) · Σ_i α_i E_i,k  +  P_terminal
J_eval(w, θ) = mean over the experiment's fixed realization seeds ξ_j (common random numbers)

E_risk,k = β_c2c R_c2c + β_roll R_rollover + β_slip R_slippage + β_lane R_lane_dep + β_road R_road_dep
```

**Risk channels** (`learning/risk_eval.py`). The `risks/` modules are called unchanged; the adapter only builds their inputs from the live simulator state.

At each sim step, the ego and every surrounding car are extrapolated over `risk_horizon_s` = 2 s at `risk_dt_s` = 0.1 s. The ego holds its current applied a and δ; surrounding cars keep constant velocity. Everything is in the road's Frenet backbone frame (x = s, y = backbone e_y, SAE +right).

R_ch is the module's `prob` output, the maximum over the horizon, when `risk_measure = "prob"` (the default). With `"score"`, R_ch is the PARALO cumulative risk (eq. 5.51) divided by ½·m·v_ref².

| Channel | Module | Inputs from the simulator |
|---|---|---|
| c2c | `risks/c2c.py` | ego and 6 surrounding trajectories; 4 m × 2 m boxes (the simulator's footprint); position σ = 0.3 m for ego and objects. As σ → 0 this equals the simulator's collision test (tested). |
| rollover | `risks/rollover.py` | (v_n, δ), with δ the applied steering; U_H = 0.1 m |
| slippage | `risks/slippage.py` | (v_n, δ, a); μ_max = the scenario's road μ; U_μ = 0.05 |
| lane departure | `risks/departure.py` | lateral position vs the **nearest** lane's centre; lane width 4 m; vehicle width 2 m; exact map (zero centreline covariance); σ = 0.3 m |
| road departure | `risks/departure.py` | the same model against the whole paved road: centre = mean lane centre, width 12 m |

Per-channel time-integrated and maximum values are recorded for every episode (`risk_integrated`, `risk_max`) and aggregated per scenario.

- The running part is bounded by Σα·term_clip = 40.
- P_terminal is 100 for a collision, rollover or road departure. The config **asserts** that each of these exceeds 40, so a crash is never cheaper than any surviving run, even though it ends the episode early.
- A timeout costs 20 + 30·(1 − progress/goal). It is never a success.
- Success adds no terminal penalty.

Each episode record reports collisions, rollovers, departures, timeouts and solver failures separately.

The rollover and slippage channels here still receive applied δ, without the δ_eq correction in §15.

Zeroing the MPC's internal weights cannot game J_eval, for two reasons. The α coefficients are fixed, and w's lower bounds are positive. A reckless controller simply incurs the failure penalties.

### 10.3 Admissible set Θ_f

The admission rule is shared by both pipelines.

Θ_f is the set of θ that pass the family validator and have status **ADMITTED** (`minimax_feasibility.py`):

| Evidence | Rule |
|---|---|
| `oracle` | A PPO evaluation demonstrated success: Wilson 95% lower bound on k/n ≥ `tau_demonstrated`. This comes from either the existing dataset (`seed_pool_dataset`) or an oracle refinement. |
| `surrogate`, policy `conservative` (default) | Lower end of the bootstrap epistemic interval ≥ τ |
| `surrogate`, policy `point` | p̂ ≥ τ |
| `unscreened`, policy `none` | Everything valid. Ablation only. |

A candidate that is not admitted is either UNCERTAIN or REJECTED.

- **UNCERTAIN:** the interval straddles τ, the query extrapolates, no interval is available, or a PPO refinement failed to demonstrate success.
- **REJECTED:** the whole interval is below τ.

Neither status means physically infeasible. The bootstrap interval is an empirical spread over group resamples, not a guaranteed confidence bound. The screening policy, τ and each surrogate's sha256 are recorded in `config.json` and `manifest.json`.

### 10.4 Exchange algorithm

`minimax_core.run_exchange` keeps a scenario set S. On each iteration k:

1. **Master.** w_k = argmin over w in [0,1]⁷ of max over θ in S of J(w,θ). It starts from seeded LHS points (`master.n_init`) plus the incumbents w₀ and w_{k−1}, then runs `n_starts` bounded Nelder–Mead restarts. Each candidate w is evaluated on **all** of S with the same realization seeds.
2. **Adversary** (`ScenarioAdversary`), for each stratum in its own θ schema:
    1. An LHS batch from `feasibility_sampling`, seeded by (experiment seed, stratum, k).
    2. Screen every candidate (cheap).
    3. Evaluate up to `max_global_evals` admitted candidates, plus the seed pool.
    4. Locally perturb the top admitted candidates in normalized θ, re-validate and re-screen them, then evaluate.
    5. Optionally send uncertain candidates that cost more than the best admitted one to the PPO oracle.
    6. Return the costliest admitted candidate.
3. **Update.** Add θ_new if J(w_k, θ_new) > F_S(w_k) + `improvement_tol`. Earlier scenarios are never dropped, and duplicates (same `scenario_id`) are never re-added.
4. **Stop** on any of: no improving scenario, `max_iterations`, `max_oracle_evals` ((w, θ) pairs), `max_wall_s`, or stagnation (`stagnation_iterations`).

**Reported result.** For each iterate, the estimate is max(F_S(w_k), J(w_k, θ_new)), the worst cost actually observed at w_k. The returned w is the iterate with the lowest estimate. Its cost over the final set is also reported. This is an **approximate empirical minimax point**: both inner problems are solved approximately, and no global optimality or robustness certificate is claimed.

**Multiple strata.** Configure several entries in `strata`. Each keeps its own sampler, schema and surrogate, and they are compared only through scalar J_eval. Debug with one stratum first (`configs/minimax/cutin_side1_screened.json`), then expand (`configs/minimax/all_strata_screened.json`).

**Baselines** (`method`):

- `baseline`: w₀ only.
- `random_search`: the best of N random w (plus w₀) against a fixed admitted pool.
- `maximin`: max over θ in the pool of min over w of J. A secondary benchmark only, generally ≤ minimax.

"Minimax without screening" is `screening.policy = "none"`. "With refinement" is `oracle_refinement.enabled = true`.

### 10.5 Oracle refinement

Both pipelines use the same refiner.

`minimax_oracle.OracleRefiner` is off by default because it is expensive. When enabled:

1. Train PPO with `train_and_evaluate` for each seed in `ppo_seeds`, with `timesteps` and `n_envs` as configured.
2. Select the seed with the best selection-evaluation success rate.
3. Re-evaluate **only that policy** on an independent realization stream (`worker_rank = 2,000,000`) for `validation_episodes` episodes.
4. Apply the Wilson rule to the validation (k, n).

The max-over-seeds selection numbers are stored but labelled optimistic. Each result is one atomic JSON under `<experiment>/oracle/<stratum>/<scenario_id>.json`, next to its policies. A resume reloads it instead of retraining.

Refinement never writes to `artifacts/feasibility/`. The global surrogate is **not** refit automatically; refitting it is an explicit, separate step. To fold refinement data into a new surrogate, build a separate dataset version from the oracle JSONs, fit it, and start a **new** experiment. Resuming with a changed surrogate is refused (sha256 check).

### 10.6 Module wiring

```
                 ┌──────────────── minimax_core.run_exchange ────────────────┐
                 │  A. master: w_k = argmin_w max_{θ∈S_k} J(w,θ)             │
                 │  B. adversary.propose(w_k)                                │
                 │  C. S_{k+1} = S_k ∪ {θ_new} if J(w_k,θ_new) > F_S + tol   │
                 └───────┬──────────────────────────────┬────────────────────┘
                         │ J(w, [θ…])                   │ propose(w, k)
                         ▼                              ▼
     Experiment._J (episode cache) ◄──── ScenarioAdversary ──► FeasibilityScreen ──► SurrogateArtifact.predict
                         │                              │                   ▲
                         ▼                              └──► OracleRefiner ─┘ (train_and_evaluate ×seeds,
         run_episode_jobs ─► run_episode                                       independent validation)
                               └─► SamplingMPC.act ─► EgoTrafficEnv.step (shared physics + traffic)
```

**How a non-PPO controller is inserted.** The controller returns a physical (accel, δ). `to_env_action` clips that to the env's box and maps it to the normalized action. The env then applies its own steering-rate and jerk limits. The MPC reads the live `road`, `agents` and `ego_car` through `EgoView`, which is read-only by convention. `tests/test_mpc_controller.py` checks two things:

- the closed loop produces exactly the same states as stepping the env directly;
- outcomes and costs do not change when every PPO reward constant is changed.

### 10.7 What is confirmed and what is proposed

| Item | Status |
|---|---|
| Physics, traffic stepping, actuator limits | **Confirmed repository behaviour.** The MPC drives `EgoTrafficEnv.step()` like PPO does. |
| Scenario families, θ schemas, validators, `scenario_id`, samplers, surrogate artifacts | **Confirmed.** Reused unchanged. |
| The MPC itself (`learning/mpc.py`): sampling MPC over (accel, target lane), horizon 3 s | **Proposed default.** A reasonable first controller, not a validated research choice. |
| The 7 components of w and their bounds | **Proposed default.** |
| J_eval's terms, normalizers, coefficients α and penalties | **Proposed default**, configurable in the experiment JSON. |
| τ = 0.8, conservative screening, the Wilson rule for "demonstrated" | **Proposed default**, configurable. |
| The PARALO MPC's w bounds, w₀, fixed coefficients, candidate grid and constraint thresholds | **Proposed default / provisional.** |
| Which MPC structure and w the research should finally use | **Open research decision.** |

---

## 11. PARALO MPC: cost function and control

`learning/paralo_mpc.py`, version `paralo-mpc-v2`. This is the proposed controller. PARALO probabilities are the core of its own cost. It has no lane variable: lane changes emerge from replanning.

### 11.1 Control loop

Every 0.1 s (2 env steps) the MPC does the following:

1. It enumerates 7 accelerations {−8, −4, −2, −1, 0, 1, 2} m/s² × 13 absolute steering values. The steering values are the current δ + {0, ±0.002, ±0.005, ±0.01, ±0.02, ±0.04, ±0.08} rad, clipped to ±0.5. That gives 91 candidates.
2. It holds each candidate pair over a 3 s horizon at 0.1 s steps.
   - The env's jerk limit (40 m/s³) and steering-rate limit (1 rad/s) are modelled as ramps toward the held value.
   - The env still enforces both limits on the applied command (tested).
3. It predicts the motion:
   - **Ego:** the single-track model in §11.3, in the ego lane's Frenet frame, using the env's own sign conventions.
   - **Surrounding cars:** constant velocity.
4. It minimizes `Σ_{n=1..30} ℓ(x̂_n, u; w)` over the candidates (§11.2).
5. It commits to the cheapest pair. `to_env_action` maps that to the env's action, the env applies its rate limits, and the car moves. The whole search repeats 0.1 s later.

### 11.2 Cost function

```
ℓ = w_speed ((v−v_ref)/v_ref)² + w_accel (a/4)² + w_steer (δ/0.5)²
  + w_c2c P_c2c + w_roll P_roll + w_slip P_slip + w_lane P_lane
  + c_jerk (j/10)² + c_latacc (a_y/4)² + c_road P_road
```

| Symbol | Meaning |
|---|---|
| w (7, tuned, log-scale bounds) | speed [0.05, 20], accel [0.005, 5], steer [0.01, 100], c2c [0.5, 500], roll [0.5, 500], slip [0.01, 100], lane [0.01, 100]. w₀ = (1, 0.1, 1, 20, 20, 1, 1) |
| fixed controller parameters | c_jerk 0.1, c_latacc 1, c_road 20 |
| P_c2c | max over the 6 surrounding cars of the Gaussian rectangle-overlap probability (4 m × 2 m boxes, σ = 0.3 m) |
| P_roll, P_slip | PARALO, given (v, δ_eq, a), with δ_eq from the candidate's own predicted yaw rate. P_slip uses μ̂ and U_μ = 0.05 |
| P_lane | PARALO departure against the lane centre **nearest to each predicted state** |
| P_road | PARALO departure against the paved road |

The w bounds and w₀ are proposed defaults. The P terms are per-state probabilities summed as a **planning surrogate**. The sum is not treated as a probability of at least one event, and no independence between time steps is assumed.

### 11.3 Prediction model

The MPC predicts the ego with a steady-state single-track model:

```
s'     = v cos(e_psi) / (1 − κ e_y)
e_y'   = v sin(e_psi)
r      = v tan(δ) / (L + K v²)
e_psi' = r − κ s'
v'     = a
```

K = m/L·(l_r/C_f − l_f/C_r) = 0.034 rad·s²/m is the understeer gradient. A plain kinematic model (K = 0) over-predicts this SUV's steady yaw rate about 2.2× at 10 m/s and 5.9× at 20 m/s. With K it matches the simulator to within 3%. `understeer_gradient: 0.0` restores the kinematic model. Surrounding cars are constant velocity in the backbone frame.

### 11.4 Inputs to the PARALO models

- **Steering.** `paralo_steer_input` defaults to `"kinematic_equivalent"`: δ_eq = atan(L·r_pred/v) of the candidate's own predicted yaw rate, never simulator ground truth. `"applied"` passes the held δ and reproduces `paralo-mpc-v1`.
- **Friction.** The simulator always uses the scenario's `road_mu`. The MPC gets μ̂, fixed at reset, from `friction_mode`:
  - `oracle` (default): μ̂ = road_mu;
  - `estimated`: road_mu + bias + noise, with noise drawn once per episode from the realization seed;
  - `conservative`: the estimate minus a margin.

  U_μ = 0.05 is a separate, fixed PARALO parameter, not the estimation error. `friction_speed_decay_k` is 0.
- **Slip-angle limit.** `slip_alpha_max_deg_at_mu_1` stays at the paper value, 8.6° (§6.1). Change it only as a separately validated version.

### 11.5 Diagnostics

- **Solver failure.** A non-finite cost is counted, and the MPC brakes at −8 while holding the current steering.
- **Slippage-transition diagnostics** are kept per episode in `mpc_diagnostics`. Each counts replans out of all replans:

| Counter | Counts replans where… |
|---|---|
| `slip_active` | the slip term is non-zero |
| `slip_indistinct` | the slip term is active but equal across all candidates |
| `slip_saturated_all` | every candidate has P_slip ≈ 1 throughout the horizon |
| `slip_decisive` | the chosen plan changes when the slip term is removed |

---

## 12. Physical evaluation and constraints

`learning/physical_eval.py`, version `physical-v3`.

Every quantity below comes from actual simulator states and applied controls, never from predictions.

| Quantity | Definition |
|---|---|
| completion time (optional objective) | t/T_max for a success, 1 + shortfall otherwise. For use when shortfall sits at 0 and cannot separate weight vectors. |
| progress shortfall | `clip(1 − progress_m/goal, 0, 1)`. `progress_m` is the env's Frenet arc length along the ego lane, so lateral motion is never counted as progress. A safety failure ends the episode; the shortfall is then whatever progress was actually achieved, and the failure is recorded separately as an event. |
| collision | the env's COLLISION terminal condition |
| rollover | the env's ROLLOVER terminal condition under the run's `event_model`, default `physical_v2`: realized \|a_y\| ≥ SSF·g. The legacy and physical indicators are both recorded |
| road departure | the env's OFF_ROAD terminal condition |
| slippage | some step has an axle's F_y,req above the capacity left, or an axle's requested combined force at μ·F_z (§4.1). Each indicator is also recorded on its own. Does not end the episode. |
| stability_loss | \|body sideslip\| ≥ 10° while moving (§4.1). Recorded, not part of slippage, and has no constraint. |
| lane departure | the body crosses a lane marking: \|e_y to the nearest lane centre\| > (l_w − W)/2 = 1 m. Does not end the episode. |
| peaks | max \|realized a_x\|; max \|realized a_y\|; max \|applied jerk\|. Diagnostics add tire utilization, slip angle, the actuator-limited command, and requested vs. realized total F_x and F_y. |
| diagnostic peaks | peak a_y/(SSF·g), peak friction demand, peak \|sideslip\| |

**Per scenario:** J_progress is the mean shortfall. Each event is reported as k, n, the rate k/n and its Wilson 95% interval. Each peak is the maximum over seeds. A count of k = 0 is *not* evidence of zero probability; every report carries n and the interval.

**Constraints** (`ConstraintSpec`) have these fields:

| Field | Meaning |
|---|---|
| name, metric | the constraint and the metric it reads |
| threshold | the limit ε; `None` means monitored only |
| units | checked against the metric's units |
| statistic | `point` (the observed rate or peak) or `wilson_upper` (rates only) |
| tolerance | allowed margin above the threshold |
| status | `provisional` or `validated` |
| note | free text |

The `wilson_upper` statistic is already wired in, so confidence-bound feasibility needs no architectural change. With threshold 0 it can never be satisfied, because the upper bound is always > 0.

**Shipped thresholds.** Every one is *provisional*. None is a validated safety requirement.

| Constraint | Threshold | Status |
|---|---|---|
| collision rate | 0 | provisional |
| rollover rate | 0 | provisional, and nearly vacuous: only reachable for μ ≥ 0.97 (§15, A2) |
| road-departure rate | 0 | provisional |
| slippage rate | 0 | provisional |
| peak lateral acceleration | 4.0 m/s² | provisional |
| lane-departure rate | — | **monitored only**: the event counts every marking crossing, including deliberate lane changes, so any threshold below 1 forbids escaping a cut-in |
| peak jerk | — | **monitored only**: emergency braking legitimately hits the 40 m/s³ actuator limit |
| peak acceleration | — | **monitored only**: full braking is a legitimate safety response |

---

## 13. Constrained minimax

This is the current tuning method, replacing the weighted J_eval with a progress objective plus explicit constraints.

### 13.1 Formulation

```
w* = argmin_{w ∈ W}  max_{θ ∈ Θ_f}  J_progress(w, θ)
     s.t.            max_{θ ∈ Θ_f}  R_i(w, θ) ≤ ε_i        for every ENFORCED constraint i

J_progress(w, θ) = mean over realization seeds ξ of clip(1 − s_final/s_goal, 0, 1)   (or completion_time)
R_i(w, θ)        = an event rate (k/n over the seeds) or a physical peak (max over the seeds)
```

- There is **no evaluation weight vector**: the configuration has no `cost`/α section (a test checks this).
- The MPC uses weights internally, but only physical outcomes judge it.
- Θ_f is the PPO-admitted set (`learning/minimax_feasibility.py`), with admission semantics unchanged: a surrogate lower bound ≥ τ = 0.8, or direct PPO evidence with a Wilson lower bound ≥ 0.8.
- Admission is a filter only. It never enters the objective.

With several strata, one u is evaluated on every working-set scenario from all strata, and t is a max over them, not an average (tested).

### 13.2 Master and exchange (`learning/constrained_core.py`)

**Master** at iteration k, on the working set S_k:

```
min_{w,t} t   s.t.  J(w,θ_j) ≤ t,  R_i(w,θ_j) ≤ ε_i  for every θ_j ∈ S_k
```

It is solved with seeded LHS plus bounded Nelder–Mead on a **lexicographic order**. Nelder–Mead only ever compares function values, so the order is encoded exactly as a scalar `merit`:

| Candidate on S_k | merit |
|---|---|
| feasible | t, which lies in [0, OBJ_MAX] (OBJ_MAX = 2, which also covers `completion_time`) |
| infeasible | OBJ_MAX + 1 + (number of violated (constraint, scenario) pairs) + t/(OBJ_MAX + 1) |

So every feasible w ranks above every infeasible one, and violation magnitudes in different units are never added together. Feasibility is tracked as its own flag, and each assessment records the worst value, its scenario and the signed violation for every constraint in that constraint's own units.

**Why not COBYLA, COBYQA, SLSQP or trust-constr.** All available in scipy 1.18, they build linear or quadratic models of the constraints. Here every R_i is an event frequency over a few seeds, which is piecewise constant in w, so those models are flat almost everywhere and their steps carry no information.

**Exchange.**

1. Seed S₀ with the adversary's counterexamples at w₀.
2. Master.
3. If no feasible w was found on S_k, stop with `no_feasible_found`. The working set only grows, so further iterations cannot help.
4. Run the performance adversary and every constraint adversary at w_k.
5. Add the performance counterexample if J > t_S + `objective_tol`. Add every constraint counterexample with a violation beyond its tolerance, even when its progress shortfall is low.
6. Stop with `converged_feasible` when nothing was added. Otherwise repeat from step 2, subject to the budgets.

**Statuses.**

| Status | Meaning |
|---|---|
| `converged_feasible` | w_k is feasible on S_k, and the adversary found no worse objective and no violation |
| `no_feasible_found` | the master search found no feasible w. The last master records `scenarios_never_satisfied`: scenarios that *no* evaluated w satisfied on their own. These are candidates for "PPO-admitted but beyond this MPC architecture". |
| `budget_exhausted` | an iteration, evaluation or wall-clock budget was hit. Not convergence. |
| `no_admissible_scenarios` | the adversary found no admitted scenario |
| `master_infeasible_certified` | defined, but **never produced**: a sampling search cannot certify infeasibility |

Each iteration record has a `counterexample` flag for the case where w_k was feasible on S_k but violated a constraint on a new scenario.

**Reported solution** (`select_final`):
- After `converged_feasible`, the last iterate.
- Otherwise, every iterate is re-assessed on the final working set and the lexicographically best one is reported with its `feasible_on_final_set` flag.

An infeasible candidate is never labelled feasible (tested). No global optimality, robustness or safety certification is claimed.

### 13.3 Adversary (`learning/constrained_adversary.py`)

At a fixed w_k, for each stratum in its own θ schema:

1. Draw a seeded LHS batch plus the seed pool.
2. Screen every candidate. Only admitted ones are simulated.
3. Run one closed-loop evaluation per scenario, which yields the objective and every constraint value at once.
4. Run local Gaussian refinement **per criterion** (the objective and each enforced constraint) around that criterion's own top admitted candidates. Constraint refinement only happens around scenarios with a positive value.
5. Return the worst admitted scenario for the objective and for each constraint, separately, plus the per-stratum worst case for every criterion.

Rejected scenarios are never simulated, and uncertain ones are never simulated without a refiner (tested). The four strata keep their own schemas (tested).

**PPO refinement** is optional (`"oracle_refinement"`, §10.5). With it on, uncertain surrogate candidates are evaluated for referral only. A candidate is referred when it is worse than the admitted candidates on some criterion, ordered by violations and then objective. Budgets and the Wilson rule are unchanged, and training uses the run's event model. It is tested with stub trainers only.

---

## 14. Reproducibility

### 14.1 Constrained runs

The experiment directory is `artifacts/constrained/experiments/<id>/`.

| File | Contents |
|---|---|
| `config.json` | the full normalized config, including the constraint specs, and its scientific hash |
| `manifest.json` | git commit, surrogate sha256s and provenance, versions (evaluator, event model, vehicle physics, MPC, θ bounds), seeds, constraints, seed-pool rows skipped as invalid |
| `iterations.jsonl` | per iteration: w_k, working set, master assessment (t, worst value, scenario and violation per constraint), master candidate counts, adversary results per criterion, scenarios added and why, counterexample flag, status |
| `controller_candidates.jsonl` | **master** log: every (w, S_k) evaluation with per-scenario objectives and constraint values |
| `adversarial_candidates.jsonl` | **adversary** log: every candidate with its admission status and evidence, p̂ and interval, metrics |
| `checkpoints/state.json`, `summary.json`, `best_weights.json` | resume state, final result, best weights with their feasibility flag |
| `evaluations/<label>.json`, `plots/convergence.png` | held-out comparison, convergence plot |
| `cache/` | one episode per file |

**Episode cache key:** controller spec (with its full config), w at full precision, scenario key, seed, `PHYSICAL_EVALUATOR_VERSION`, the event model and the θ schema version. A test checks that a cached episode equals a direct re-simulation. Wall-clock timing is stored under `timing` and excluded from equality checks.

**Resume** follows the original pipeline's rules:
- The scientific config hash must match. Only `n_workers`, `artifacts_root` and the exchange budgets may change.
- The surrogate sha256s, evaluator version, event model, physics version and MPC version must match the manifest.
- Log rows from iterations that never reached a checkpoint are dropped.
- A `budget_exhausted` run continues when its budget is raised.

A test checks that a resumed run reproduces the uninterrupted run's w sequence.

### 14.2 Baseline runs

```
artifacts/minimax/experiments/<experiment_id>/
  config.json            full normalized config + scientific hash
  manifest.json          git commit (+dirty), surrogate path/sha256/n_scenarios/preliminary, evaluator & schema versions, seeds
  iterations.jsonl       per iteration: w_k, F_S, S keys and costs, adversary result, estimate, added?
  controller_candidates.jsonl   every J(w, set) oracle call (iteration, phase, w, per-scenario costs and successes)
  adversarial_candidates.jsonl  every screened candidate (θ, status, evidence, p̂/interval, warnings, cost)
  oracle_refinements.jsonl      PPO refinement provenance
  best_weights.json, summary.json, evaluations/<label>.json, plots/convergence.png
  checkpoints/state.json        S, k, w_{k−1}, iteration history, oracle evidence, counters
  cache/…/<key>.json            one closed-loop episode per file
  oracle/                       refinement results and policies
```

- **Determinism.** Everything is seeded: the LHS for master and adversary, local perturbations, realization seeds, and the MPC (which has no randomness). Re-running a config reproduces it exactly. `test_end_to_end_optimize_interrupt_resume_and_evaluate` checks that a resumed run gives the same w sequence as an uninterrupted one.
- **Episode cache key.** A hash of the controller spec and its full MPC config, w at full float precision, the scenario key, the realization seed, the whole cost config, `EVALUATOR_VERSION` and the θ `SCHEMA_VERSION`.
- **Resume.**
    - The scientific config hash must match. Only `n_workers`, `artifacts_root` and the exchange budgets may change.
    - Surrogate hashes and the evaluator version must match the manifest.
    - Log rows from iterations that never reached a checkpoint are dropped before replay.
    - If the run stopped on a budget, raising the budget continues it.
- **Single writer.** Workers return episode records; only the coordinator writes. Writes are atomic, appends are fsync'd single lines, and reads tolerate a truncated last line. These helpers are reused from `feasibility_dataset`.
- **Concurrency.** Episodes run in a fork `ProcessPoolExecutor` with BLAS/OMP pinned to one thread. PPO refinement keeps `device="cpu"` and `torch_threads=1`. Don't nest: during refinement, PPO's own `SubprocVecEnv` uses `n_envs` processes. Benchmark first with `--mode pilot --benchmark-workers …`.

---

## 15. Physics and data audit

This audit (2026-10-28) found why rollover and slippage were wrong, and what that did to the PPO data. Sections A2 and A3 were written before any code changed. Sections B1–B7 record what was then implemented and measured. Every number in this section was measured under `vehicle-physics-v1`, before the tire model of §4.1, and was not re-measured.

### A2. Root cause: rollover and slippage

**The reference value.** 9.5 m/s² is exactly SSF·g = W_c/(2h_CG)·g = 1.55/1.60·9.81 = 9.503 m/s²:
the quasi-static rigid-vehicle tipping condition m a_y h_CG = m g W_c/2. It ignores
suspension roll (which lowers the real threshold, typically by 5–10%), tire
lateral compliance and dynamic overshoot. PARALO's
v_safe = sqrt(g R W_c/(2h_CG)) encodes the same condition, with R taken from the
kinematic steering relation.

**Numerical confirmation.** These are steady-state constant-(v, δ) runs of the
real `CarDynamics` (RK4, 8 s, straight road). a_y is the realized body lateral acceleration (F_yf + F_yr)/m, and
δ_eq = atan(L r / v):

| v | δ_cmd | μ | r | a_y real | a_y kinematic | δ_eq | P_roll(δ_cmd) | P_roll(δ_eq) | P_slip(δ_cmd) | P_slip(δ_eq) | axle saturated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 0.082 | 1.0 | 0.101 | 1.96 | 10.9 | 0.015 | **0.99** | 0.00 | **1.00** | 0.00 | no |
| 20 | 0.082 | 0.5 | 0.101 | 1.96 | 10.9 | 0.015 | 0.99 | 0.00 | 1.00 | **1.00** | no |
| 20 | 0.200 | 1.0 | 0.271 | 4.50 | 19.4 | 0.046 | 1.00 | 0.00 | 1.00 | 1.00 | no |
| 25 | 0.300 | 1.0 | 0.408 | 6.67 | 27.5 | 0.072 | 1.00 | 0.00 | 1.00 | 1.00 | no |
| 30 | 0.400 | 1.0 | — spin-out — | | | | | | | | yes |

The understeer gradient is K = m/L (l_r/C_f − l_f/C_r) = 0.0342 rad·s²/m.

1. **Rollover: the input transformation is wrong; the PARALO model itself is
   consistent.** Commanded steering includes the understeer angle K·a_y
   (0.068 rad at 2 m/s²), so the kinematic radius is ~6× too small at 20 m/s.
   Fed δ_eq from the realized yaw rate, PARALO's rollover condition reduces to
   v·r ≥ SSF·g, the same quasi-static criterion.
2. **The simulator's ROLLOVER event inherits the wrong input.**
   `EgoTrafficEnv._ego_p_rollover` uses (v_x, δ_cmd), so it fires at a_y ≈ 2 m/s²
   on the default curve. That is a false positive by a factor of ~5 in
   lateral acceleration.
3. **Physical rollover is almost unreachable in this simulator.**
   F_zf + F_zr = mg and each axle's |F_y| ≤ μF_z, so |a_y| ≤ μg. Quasi-static
   tipping (|a_y| ≥ SSF·g) needs μ ≥ SSF = 0.969: the vehicle slides before it
   tips everywhere in the cut-in/sandwich μ range [0.4, 1.0] except
   μ ∈ [0.969, 1.0], and even there only with both axles saturated. This is the
   classical untripped-rollover result (SSF > μ ⇒ slide, not roll), not a bug.
   Tripped rollover (curb, soil) is not modelled.
4. **Slippage: input error plus a calibration mismatch.**
   - With δ_cmd, PARALO slippage reads 1 at 20% of the friction limit.
   - With δ_eq it is correct at μ = 1 but still reads 1 at μ = 0.5 and
     a_y = 0.4μg. The cause is PARALO's tire assumption: slip-angle limit
     α_max = 8.6°·μ.
   - The simulator's linear tire saturates at α = μF_z/C: **33.8°·μ front,
     14.6°·μ rear.** C_f = 19 kN/rad per axle is very compliant, and already
     6.8° of front slip at a_y = 2 m/s².
   - PARALO's understeer/oversteer limits therefore disagree with the simulator
     by construction. This is a calibration/model-validity problem, not an
     input problem.
5. **Simulator slippage semantics.** The tire force is *clipped* at μF_z, so
   `|Cα| ≥ μF_z` means "requested lateral force ≥ available" (saturation, the
   sliding regime of this tire model). There is **no combined slip**: the
   longitudinal command is applied without any tire force or friction limit,
   so the simulator lets the car brake at 8 m/s² on μ = 0.4. A friction-circle
   check is therefore *not* duplicated by the tire model and must be added as
   a separate (event-only) indicator. The dynamics stay unchanged. *(True of
   `vehicle-physics-v1`, when this was written. `v2` has a real friction
   circle, §4.1.)*

### A3. PPO dataset and label validity (`datasets/cutin/side_pos1`, 124 rows × 100 episodes)

Bound versions are recovered from git history, with no semantic change
between them:
- **cutin-bounds-v0**: commit baaeb16, `cutter_relative_speed_mps` ∈ [−12, 5], `front_gap_m` ∈ [8, 60].
- **cutin-bounds-v1**: commit eb2edbb, 2026-09-08 15:31, [−5, 15] and [25, 100].
- Rows 1–24 were committed in 75b33de (2026-09-08 11:26, before v1), so they are v0.
- Rows 25–124 were committed in be97cd3 (2026-09-11), so they are v1.
- No simulator/env/risk code changed between those commits.

| Bounds version | Out of current bounds | In bounds, ≥1 ROLLOVER-terminated episode | In bounds, 0 ROLLOVER episodes |
|---|---|---|---|
| v0 (24) | 14 (7 demonstrated) | 1 (1 demonstrated) | 9 (6 demonstrated) |
| v1 (100) | 0 | 36 (2 demonstrated) | 64 (33 demonstrated) |

- 748 of the 12,400 episodes (6.0%) ended in the legacy ROLLOVER event, spread
  over 39 rows. Those labels are **contaminated**: a false rollover is counted
  as a failure, and the episode is cut short.
- Rows with zero ROLLOVER episodes have trajectories that are identical under
  any corrected detector, up to the first step where the *new* detector fires.
  That needs |a_y| ≥ 9.5 m/s² (see A2.3). Their labels are **conditionally
  compatible**.
- Every policy was *trained* under the legacy event (terminal penalty, early
  termination). It learned to keep v²·tanδ/L below ~9.5 m/s², i.e. real a_y
  below ~2 m/s² at 20 m/s. These policies are therefore over-restricted: under
  corrected physics they are lower-bound evidence. The bias is toward false
  rejection, not false admission.
- Low-μ labels can be **optimistic**, because PPO could brake beyond μg (A2.5).
  This is measurable from re-evaluation (friction-circle indicator) for the
  policies we have.
- Policy weights are available locally only for the 24 v0 rows. The 100 v1
  policies are on abl-drivsim-iii, which was unreachable during this work.

### B1. Files

New: `learning/vehicle_physics.py`, `learning/feasibility_audit.py`, `learning/risk_validation.py`, `learning/capability_suite.py`, and their tests.

Modified (additive; legacy behaviour kept where history depends on it):

| File | Change |
|---|---|
| `env.py` | `rollover_event` option (default `legacy_v1`); `ego_tire_state()` |
| `physical_eval.py` | becomes `physical-v2`: event model option, separate legacy/physical rollover and saturation/friction-circle indicators, diagnostic peaks, `completion_time` |
| `paralo_mpc.py` | becomes `paralo-mpc-v2`: δ_eq from the *predicted* yaw rate by default, friction-information modes, PARALO α_max exposed |
| `feasibility_train_one.py`, `feasibility_pipeline.py` | `rollover_event` (default `physical_v2`); provenance stamp; fit-only uses the curated set |
| `feasibility_surrogate.py` | `provenance` field |
| `feasibility_cutin.py`, `feasibility_sandwich.py` | bound-version constants |
| `minimax_oracle.py` | event model threaded through refinement |
| `constrained_core.py` | per-stratum fields; merit generalized to OBJ_MAX |
| `constrained_adversary.py` | PPO oracle refinement; per-stratum worst cases |
| `constrained_pipeline.py` | event model, objective option, refinement, curated seed pool, version manifest, four-way compare |
| `minimax_pipeline.py` | seed pool skips out-of-bounds rows instead of crashing |



### B2. Physical inconsistencies corrected (numerically verified, `tests/test_vehicle_physics.py`)

- **Gentle curve** (20 m/s, δ = 0.082, a_y = 1.96 m/s²):
  - legacy rollover fires;
  - `physical_v2` does not;
  - PARALO P_roll(δ_eq) < 1e-6 and P_slip(δ_eq) < 1e-3 at μ = 1.
- **Recorded closed-loop false positive.** The original MPC on a cut-in (κ 0.0138, μ 0.76, seed 1000) was ended as a ROLLOVER at peak |a_y| = 2.1 m/s² (0.22·SSF·g). Under `physical_v2` it continues; the legacy indicator is still recorded.
- **Ceiling on lateral acceleration.** |a_y| ≤ μg is verified over δ sweeps at μ = 0.5, 0.8 and 0.95. Tipping is reached only with μ = 1 and both axles saturated.
- **MPC prediction.** The MPC's steady-state yaw model matches the simulator within 3% at 10–25 m/s; the kinematic model is more than 2× off.
- **Friction circle.** Braking at 8 m/s² on μ = 0.4 raises the friction-circle indicator without any lateral saturation. Combined braking and cornering raises the demand above pure cornering.
- **Low speed.** |v| < 1 m/s gives curvature 0 and δ_eq 0; there are no NaNs at standstill.

### B3. PPO dataset outcome

- The audit (§A3 table) was computed by `feasibility_audit --mode audit` and reproduced the hand count exactly, with bound versions recovered from git.
- A relabel of the 10 in-bounds rows with local policies was started. Before it finished, **the user decided (2026-10-28) to delete all artifacts**: the dataset, all 124 policies, the heatmaps, the remote surrogate and the minimax runs.
- **Labels regenerated: none. Rows retained: none.** Every stratum needs a fresh PPO campaign, which defaults to `physical_v2` and is stamped with its provenance.
- **Surrogate validation results: none** (no data). `feasibility_audit --mode validate` is ready and tested on synthetic data.

### B4. Oracle refinement

This is wired into the constrained adversary:
- Uncertain, surrogate-screened candidates are evaluated for referral only.
- A candidate is referred when it is worse than every admitted candidate on some criterion.
- Training and validation are budgeted, use the unchanged Wilson admission rule, and run under the evaluator's event model.
- Results are versioned per scenario in the experiment's `oracle/` directory, and the budget is restored on resume.
- It is tested with stub trainers. It has not been exercised with real PPO.

### B5. Minimax status

- **Shared w.** One shared w over the union of strata is enforced by construction, and tested: the master evaluates a single u on every working-set scenario from all strata, and t is a max, not an average.
- **Per-stratum reporting.** Each adversary call and the final summary report the per-stratum worst cases.
- **No optimization was run.** There is no surrogate and no PPO data. No weight vector is reported.

### B6. Remaining limitations

- `physical_v2` rollover is a reduced-order quasi-static criterion: no roll dynamics, suspension compliance or tripping. In this tire model it is reachable only for μ ≥ 0.97, so the rollover constraint is nearly vacuous by physics.
- PARALO slippage and rollover keep the paper calibration. The α_max mismatch with the simulator tire (8.6°·μ vs ~34°·μ) is unresolved. Recalibration must be a separately validated version.
- The capability suite and risk validation use hand-chosen θ, not PPO-certified ones (no PPO data).

### B7. Diagnostics (2026-10-28, local, hand-chosen θ)

**Risk-model validation** (`learning.risk_validation`). There were 975 steady-state points: v 10–30 m/s, δ 0–0.3 rad, μ ∈ {0.4, 0.7, 1.0}, braking 0–8 m/s². The closed loop ran the 7 capability cases × 3 seeds for each controller. Step-level counts at P > 0.5 are against the simulator's physical indicators:

| Set | Probability | Physical events | False positives (FPR) | Missed events | Brier |
|---|---|---|---|---|---|
| steady state | P_roll(δ_cmd) | 0 rollovers | 405 (0.42) | — | 0.41 |
| steady state | P_roll(δ_eq) | 0 rollovers | 0 | — | 0.00 |
| steady state | P_slip(δ_cmd) | 380 slip | 499 (0.84) | 60 | 0.57 |
| steady state | P_slip(δ_eq) | 380 slip | 356 (0.60) | 58 | 0.42 |
| closed loop, original MPC (13,219 steps) | P_roll(δ_cmd) | 0 | 261 (0.02) | — | 0.015 |
| | P_roll(δ_eq) | 0 | 0 | — | 0.000 |
| | P_slip(δ_cmd) | 555 slip | 4,905 (0.39) | 504 | 0.40 |
| | P_slip(δ_eq) | 555 slip | 1,436 (0.11) | 504 | 0.14 |
| closed loop, PARALO MPC w0 (10,498 steps) | P_slip(δ_cmd) | 0 | 4,706 (0.45) | — | 0.45 |
| | P_slip(δ_eq) | 0 | 69 (0.007) | — | 0.005 |

Gentle-curve check point (20 m/s, δ = 0.075, μ = 1): a_y = 1.80 m/s²; P_roll(δ_cmd) = 0.86 and P_slip(δ_cmd) = 1.0; with δ_eq both are 0.

- **The input correction removes every rollover false positive.** No physical rollover occurred anywhere, as A2.3 predicts.
- **For slippage, δ_eq cuts false positives sharply** (closed loop 0.39 → 0.11; PARALO-MPC trajectories 0.45 → 0.007).
- **New, unresolved: PARALO misses most real slipping.** With δ_eq, P_slip stays below 0.5 on 504 of the 555 steps where the original MPC's trajectories physically slip (91% missed). These occur in the brake-and-steer and sandwich cases. This is an open model-validity question: the braking term vs. the vehicle-level friction circle, and lateral saturation during transients. It needs investigating before slip probabilities are trusted.
- Probabilities were **not** recalibrated.

**Controller capability** (`learning.capability_suite`, `physical_v2`, 3 seeds each; successes / 3):

| Case | Original MPC | PARALO MPC w0 |
|---|---|---|
| cut-in, moderate | 3 (lane change) | 3 (lane change) |
| cut-in, severe | 2 (1 timeout) | 3 (stays in lane, brakes) |
| cut-in, curved (κ 0.02) | 3 (legacy-rollover indicator raised, no physical rollover) | 3 |
| cut-in, μ = 0.4 | 3 | 3 |
| cut-in, brake + steer (κ 0.015, μ 0.5) | 0 (timeouts; tire saturation + friction circle) | 3 |
| sandwich, lateral escape available | 0 (timeouts; slip) | **0 (3 collisions)** |
| sandwich, braking required | 2 (1 timeout; slip) | 3 |

**The PARALO MPC (constant absolute (a, δ) candidates, replanned every 0.1 s) does change lanes and brakes.** Its one systematic failure is the sandwich with a lateral escape gap: it collides every time, which needs diagnosis before tuning. Weights should not be expected to fix it if the candidate set cannot represent the escape maneuver. None of these θ are PPO-certified as admissible.

---

## 16. Earlier results

Everything here came from data that has since been deleted, under the **legacy** event definitions and `vehicle-physics-v1`. It is kept for context only.

### 16.1 First PPO campaign (cut-in, side +1, LHS seed 0)

The first campaign used 24 θ, `--concurrency 8 --n-envs 4 --timesteps 2000000 --episodes 100`, for 2,400 rollouts in total.

| Result | Value |
|---|---|
| Success rate 1.0 (solved) | 10 θ |
| Success rate 0.97–0.99 | 4 θ |
| Success rate 0.07–0.83 (mixed) | 4 θ |
| Success rate 0.0 | 6 θ |
| Failure counts (all rollouts) | 585 timeouts, 138 collisions, 93 off-road, 6 rollovers |
| Training length | 1.1M–2.0M timesteps; 9 runs used the full 2M budget, 15 stopped early |

Timeouts are the largest failure mode. They may mean under-trained policies rather than infeasible scenarios, so check them before trusting a 0% θ.

Throughput on abl-drivsim-iii is in [how_to_run.md](how_to_run.md), Step 3.

### 16.2 Original MPC vs. PARALO MPC at w₀, before the physics corrections

The scenario set was the 42 cut-in(+1) scenarios that PPO demonstrably solved and that pass the current validator, with 2 held-out seeds each (84 episodes per controller).

| Metric | Original MPC (default w) | PARALO MPC (w₀) |
|---|---|---|
| Completion rate | 0.845 | 0.940 |
| Collision rate | 0 | 0 |
| Rollover rate (env event) | **0.095** | 0 |
| Road-departure rate | 0 | 0 |
| Slippage rate (tire saturation) | 0.036 | 0 |
| Lane-departure rate (any marking crossing) | 1.0 | 0.774 |
| Worst-case progress shortfall | 0.669 | 0.074 |
| Mean progress shortfall | 0.080 | 0.002 |
| Peak \|accel\| | 8.0 m/s² | 4.0 m/s² |
| Peak \|lateral accel\| | 2.63 m/s² | 1.28 m/s² |
| Peak \|jerk\| | 40 m/s³ | 40 m/s³ |
| Controller time per replan | 2.6 ms | 15.9 ms |

The original MPC's failures were all rollover events. The PARALO MPC's 6% non-completions were all timeouts. Timing was measured with 12 parallel workers on 12 cores and is contended; run alone, the PARALO MPC takes about 3.7 ms per replan.

**Slippage diagnostic** over all 23,722 PARALO replans:

| Counter | Count |
|---|---|
| `slip_active` | 23,722 (every replan) |
| `slip_indistinct` | 0 |
| `slip_saturated_all` | 0 |
| `slip_decisive` | 11,793 (50%) |

So the slip term always separates the candidates and decides about half of all plans. It is not flat, but it strongly steers the controller toward slowing down. That is consistent with §16.3.

### 16.3 PARALO on commanded δ, before the correction

This table is what started the audit in §15.

| Road curvature, speed | Steady δ | P_roll | P_slip | True lateral accel |
|---|---|---|---|---|
| κ = 0.005 (default road), 20 m/s | 0.082 | 1.00 | 1.00 | 2.0 m/s² |
| κ = 0.005, 15 m/s | 0.052 | 0.00 | 1.00 | 1.1 m/s² |
| κ = 0.010, 20 m/s | 0.165 | 1.00 | 1.00 | 4.0 m/s² |
| κ = 0.020 (max in θ), 20 m/s | 0.329 | 1.00 | 1.00 | 8.0 m/s² |

With the original baseline, slippage also dominated E_risk: 0.28 of the 0.36 risk term in the baseline cut-in episode.

---

## 17. Tests

432 tests, all passing. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is needed where ROS is installed. `dynamics_test.py`, `road_test.py` and `traffic_test.py` are **visual demo scripts**, not pytest tests.

| File | Tests | Covers |
|---|---|---|
| `test_tire_forces.py` | 113 | Combined tire model: braking, cornering, drive, low speed, load transfer, RK4-stage friction circle, wrong-way spin sweep, event definitions, oriented footprints, env and MPC smoke tests |
| `test_ppo_reporting.py` | 94 | Records, seed streams, exact vs. robust, Wilson and admission, seed selection, tire utilization, Frenet to world, footprints, maps, comparisons, training metrics, figures, animation, reports |
| `test_reward.py` | 45 | Reward decomposition, running-maximum progress, oscillation, goal cap, terminal values and priority, timeout, outcome independence, Gymnasium and PPO compatibility, γ and version stamping |
| `test_feasibility_arena.py` | 29 | Both families: placement, triggers, prescribed events, env integration |
| `test_minimax_feasibility.py` | 16 | Screening policies; uncertain, rejected and extrapolation cases; Wilson rule; real artifacts; seeding from the dataset |
| `test_scenario.py` | 12 | `ScenarioConfig` validation and realization |
| `test_vehicle_physics.py` | 11 | Rollover regressions with real dynamics, the recorded false positive, slip across μ, friction modes, low speed |
| `test_env.py` | 10 | Env API, outcomes, observation |
| `test_feasibility_surrogate.py` | 10 | Grouped CV, constant fallback, bootstrap, predict warnings |
| `test_feasibility_dataset.py` | 9 | JSONL I/O, atomic writes, resumability |
| `test_feasibility_sampling.py` | 9 | Determinism, bounds, rejection |
| `test_minimax_core.py` | 9 | Known minimax solutions, minimax vs. maximin, budgets, resume equivalence, caching |
| `test_minimax_pipeline.py` | 9 | Cache keys, scientific hash, strata kept separate, rejected never simulated, oracle refinement, end-to-end resume |
| `test_mpc_controller.py` | 9 | Vectorized P_roll, action mapping, rate bounds, env equivalence, reward independence, solver fallback, J_eval guards |
| `test_traffic_init.py` | 8 | Legacy template invariants |
| `test_risk_eval.py` | 7 | All five channels, c2c against the collision test, departure at the line, weighting |
| `test_paralo_mpc.py` | 6 | w round-trip, yaw model against the simulator, candidates, all risks react, actuator limits, determinism |
| `test_physical_eval.py` | 6 | Event recording, shortfall bounds, rates and Wilson intervals, constraints, slippage indicators |
| `test_paralo_vec.py` | 5 | Vectorized risks equal `risks/` |
| `test_constrained_core.py` | 5 | Toy solution, infeasible never reported feasible, budgets, merit order, separate counterexamples |
| `test_constrained_refinement.py` | 5 | Refinement admission and failure, per-stratum worst cases, one shared w |
| `test_feasibility_audit.py` | 3 | Categories, curation, unreproducible relabels, provenance, admission validation |
| `test_constrained_pipeline.py` | 3 | No evaluation weights, inadmissible never simulated, end-to-end resume and four-way comparison |

---

## 18. Design decisions

These were each settled deliberately, several after a real bug or an explicit correction. Change one only with a strong new reason.

1. **Cut-in and sandwich stay strictly separate.** Separate configs, θ schemas, samplers, datasets and surrogates. They are never combined, not even as a categorical feature.
2. **Blocker sides +1 and −1 are separate strata**, never one continuous axis. Run the pipeline once per side.
3. **Events trigger spatially** (`ego.s ≥ s_event`, fired once), not by elapsed time.
4. **Success is physical** and independent of reward. A timeout is never a success. Reward changes must not touch the success logic.
5. **Rollover event: `physical_v2`, pending confirmation.** The original decision was the probabilistic `P_roll` model, with an LTR approach tried and rejected. The audit showed that model fires on commanded steering at 2 m/s². The spec of 2026-10-28 asked for a criterion on realized lateral acceleration, and |a_y| ≥ SSF·g is mathematically the LTR = 1 condition. See §21, question 5.
6. **The surrogate trains on rollout-level binary rows**, grouped by `scenario_id` for CV and bootstrap. Not one fractional row per θ with sample weights.
7. **CPU, not GPU.** A small MLP over NumPy physics is CPU-bound, so `device="cpu"`. Use `torch_threads=1` and BLAS threads = 1 per process. Scale with **concurrent processes** (`--concurrency`), not a larger `--n-envs`. Run `pilot` on new hardware.
8. **Single-writer dataset.** Workers write only their own result file; only the coordinator appends.
9. **One traffic-step implementation** (`model/traffic_step.py`), shared by the demos and the env.
10. **The ego never goes through the plastic-crash model.** An ego collision ends the episode.
11. **The reward has only progress, terminal outcome and time terms.** Do not add shaping, including potential-based shaping, to fix a training problem. Treat a reward change as a new `REWARD_VERSION`.

**Pitfalls**

- **Sign convention:** e_y is positive to the right (SAE). The road κ is negated when passed to `Car.step` (`kappa = -lane.kappa[idx]`).
- **The ego's `CarState.lane` never updates**, because the ego has no lane-commit logic. Use `_ego_nearest_lane()`.
- **The remote training machine is read-only for editing.** Edit and commit locally; pull and run there.

**Decisions on the MPC cost:**

| Question | Decision |
|---|---|
| PARALO risks in the MPC cost | **Yes**, this is the core purpose |
| MPC input model | Hold (a, δ) constant over the horizon |
| Steering held as | **Absolute** δ, not relative to road curvature |
| Lateral risk horizon / lane reference | **Full horizon**, **nearest lane** (no target-lane modification) |
| Risk measure | **Probability**, not energy score |
| How coefficients are set | The MPC cost keeps weights; evaluation uses a physical objective and constraints, with no evaluation weights (§19) |

---

## 19. How the evaluation design was chosen

This is why J_eval was replaced. The owner wanted PARALO at the core of the MPC cost, with the same cost structure in the MPC and the evaluation. That raised one objection:

**The objection:** *"The whole project is about finding w. Now we have to choose α. We have just moved the problem."*

### 19.1 Why α cannot be tuned, or tied to w

- If the optimizer could change α, or if α had to equal w, it would lower the score by shrinking coefficients rather than by driving better. With a c2c weight near zero, near-misses become free in both the MPC and the score. A fixed α is the ruler that makes it meaningful to compare two values of w.
- w and α can legitimately differ because the MPC is myopic:
  - it sees only 3 s;
  - it assumes constant inputs and constant-velocity traffic;
  - it never sees the crash penalty.

  The w that gives the best actual driving under α is therefore generally ≠ α. The gap w* − α measures how much the planner's simplifications must be compensated.

### 19.2 Why the objection is still valid

Even so, the answer w* depends on α. Choosing α is a value judgment: how much one unit of slippage risk is worth compared with one unit of lost speed. This judgment existed before the redesign too, since J_eval always had fixed α and β. The redesign only made it visible, because the terms in J_eval and in the MPC are now identical. No algorithm removes the need for *someone* to say what "good driving" means. What can be done is to make that statement **smaller, more interpretable, or less influential** on the result.

### 19.3 Candidate resolutions considered

1. **Make J_eval mostly free of weights by using outcomes.** Grade on physical outcomes only, for example `J = 1(failure) + λ·(time-to-goal shortfall)`, and keep risk probabilities and comfort only inside the MPC. One scalar λ replaces about 10 coefficients. Risk then matters to the evaluation only through whether the car actually crashed, so w is tuned for real outcomes, not proxies.
2. **Turn trade-offs into constraints (ε-constraint).** For example: minimize worst-case time-to-goal or discomfort subject to worst-case failure rate ≤ ε, or worst-case integrated risk ≤ ε. Thresholds like "failure probability ≤ 10⁻³" are easier to justify from requirements or standards than weights are.
3. **Multi-objective minimax (a Pareto front).** Do not scalarize. Compute w for the worst-case (safety, progress, comfort) trade-offs and present the front, then pick or justify a point afterwards. This is more compute, but the value judgment is made after seeing the trade-off.
4. **Use PARALO's physical units.** PARALO scores are probability × energy (joules), so the five risks are already in one common unit. Use scores with all risk weights = 1, which removes 5 arbitrary numbers. Only one exchange rate between "risk energy" and progress/comfort remains. (The owner currently prefers probabilities, so this would need a reason.)
5. **Robustness to α.** Show that w* is stable over a plausible range of α, or make the evaluator itself adversarial: `min_w max_{θ, α ∈ A} J(w, θ; α)` over a simplex A of reasonable weightings. The answer is then robust to the designer's uncertainty about the weights, not tuned to one choice.
6. **Normalize α by reference scales and set them all to 1.** This is the current approach: each term is divided by a physically meaningful reference (v_ref, a comfort jerk of 10 m/s³, a lateral-acceleration limit of 4 m/s², the steering limit). With equal α, the only choices are the reference scales, which can be cited from literature or standards (ISO 2631, ride-comfort thresholds).
7. **Infer α from preferences or demonstrations.** Use inverse RL or preference learning from expert driving, or from the PPO policy's behaviour. This is heavier, and it brings its own assumptions.

**Outcome (2026-10-27):** resolution 2 (ε-constraint) was adopted, as §13 describes.

---

## 20. Limitations

**Scientific**

1. The w bounds, w₀, c_jerk, c_latacc, c_road, the candidate grid and every constraint threshold are proposed or provisional. τ = 0.8 is a proposed default.
2. Both inner problems are solved approximately by sampling and local search. Results are "best found under budget", with no global optimality, robustness or safety certificate.
3. The true worst case over Θ_f can be higher than the worst case found over finitely many scenarios and seeds.
4. Event rates are point estimates over few seeds (8 in the shipped configs). k = 0 is not zero probability. `wilson_upper` is available but cannot be met with ε = 0.
5. `no_feasible_found` is not proof of infeasibility, and PPO admission does not imply this MPC architecture can solve a scenario.
6. The bootstrap interval is an empirical spread, not a guaranteed confidence bound. PPO failures may mean under-training rather than infeasibility.
7. J_eval remains undecided on its α weights, mean vs. CVaR over ξ, and `improvement_tol`. It is now only a baseline.

**Physical**

8. `physical_v2` rollover is reduced-order: no roll dynamics, suspension compliance or tripping.
9. The tire model is linear up to a hard limit: no peak-and-falloff, no separate μ for sliding, no tire relaxation, no temperature or wear. Saturation is exact, but near the limit forces are approximate.
   - Load transfer is quasi-static and longitudinal only. There is no lateral transfer, suspension, roll or pitch dynamics, as this is a bicycle model with one force per axle.
   - Allocation is load-proportional by assumption, so limits are upper bounds for a real ABS and drivetrain. There is no engine power limit, aerodynamic drag or rolling resistance (`C_d` is unused).
   - There is no stability control and no reverse gear. Steady hard steering at speed under light braking can still spin the car in the steered direction (322 of the 1,944 sweep cases, 872 under the old model).
   - The env's lane spacing is 3 m while `l_w` is 4 m, so lanes overlap by 1 m (`Road.lane_calc`). Road edges follow from that spacing. This predates the tire model and was left alone.
10. PARALO keeps the paper calibration (§6.1). Probabilities were not recalibrated.
11. The lane-departure event cannot separate intended from unintended crossings, so it is monitored only.

**Implementation**

12. The original MPC's prediction is deliberately simple: constant-velocity traffic, an idealized lateral response, and ramp corners neglected in the rollover prediction. A different structure would change what w means and needs a new `controller.name`.
13. With the baseline w, the sandwich times out on the default θ: the ego waits behind the stopped car, so J_eval there is dominated by the timeout penalty.
14. The master is non-smooth: LHS plus bounded Nelder–Mead on a max of noisy simulations, so restarts matter. Budgets still need tuning.
15. The risk adapter adds about 1.3 ms per step, roughly doubling J_eval episode time to about 2.4 s. `risk_dt_s = 0.2` halves the overhead. The PARALO MPC is about 5× slower per replan than the original, but still a few milliseconds.
16. Controllers and evaluators read private env helpers (`_ego_nearest_lane`, `_ego_clearance`, `_ego_p_rollover`, `_prev_accel`). A rename in `env.py` would break them, and tests would catch it.
17. The capability suite and risk validation use hand-chosen θ, not PPO-certified ones.

---

## 21. Open questions

**Risk models**

1. **Why does PARALO miss 91% of real slip steps?** It misses 504 of 555 in the brake-and-steer and sandwich cases. The missed steps need splitting by cause (§6.1).
2. **Should PARALO's tire limit match this simulator?** Needs a decision, and would be a new versioned model.
3. **Should braking and cornering slip share one friction circle?** Tied to question 1.
4. **Is it acceptable that rollover is almost never reachable?** Rollover risk would hardly affect any result.
5. **Is `physical_v2` acceptable?** It equals the LTR criterion that was once rejected (§18).
6. **Are the tire assumptions right?** Load-proportional allocation is an assumption, and the `slippage` constraint still counts any saturation even though a tire at its limit is not a skid. Should the constraint use `stability_loss`, and is 10° the right threshold?

**Controllers**

7. **Why does the PARALO MPC collide in the sandwich escape?** It does on all 3 seeds. If the constant-input candidates cannot represent the maneuver, weight tuning cannot fix it.
8. **Why does the original MPC fail brake-and-steer and the sandwich escape?** Is it under-powered, or are the scenarios unrealistic?

**Data and oracle**

9. **New PPO campaign.** Size, family order and machine time are undecided.
10. **Where should new policies be backed up?**
11. **How good is the surrogate?** Unknown until there is data.
12. **Refinement budget.** Not chosen, and refinement has not run with real PPO.

**Optimization**

13. **Does tuned PARALO beat the original MPC?** No run has happened.
14. **Does the progress objective plateau?** If so, `completion_time` may need to be the primary objective.
15. **Should friction be uncertain to the MPC?** The other modes exist, but no study is planned.

**Reward**

16. **Should PPO bootstrap at timeouts?** SB3 adds γ·V(final state) on a truncated step, so a timeout's −1 is not a final −1: a moving policy with V above about 1 sees a positive target. The observation has no clock, so the critic cannot foresee the cutoff either.
17. **Does the sparser reward stall at "stop and wait"?** The old progress boost was added because policies did that. A stopped timeout (−1.35) beats a crash (about −4.8), so an early crash-prone policy may settle there and never find the +3.

18. **Should PPO train on one realization?** Settled: no. In `robust` mode every reset draws a new realization, so training and evaluation use the same perturbation ranges. The ranges are still ±5%, ±0.5 m/s and ±2 m.

Questions 1, 2, 5, 7 and 16 should be settled before the PPO campaign. Labels depend on those choices, and changing them later means relabeling.

---

## 22. Status

**Built and tested:** simulator with friction-circle tire forces (`vehicle-physics-v2`), traffic, risk models, env, both scenario families, Runners A and B, surrogate, audit, physics corrections, both MPCs, physical evaluator, both minimax pipelines, refinement (stub-tested), diagnostics.

**Not built:** Runner C.

**Not run:** a new PPO campaign, surrogate fitting or validation on real data, and any minimax optimization. All PPO data, policies and run outputs were deleted on 2026-10-28, so **there is no tuned w and no result comparing the controllers yet.**

**Reporting pipeline (§23):** built and tested on fixtures and a tiny smoke run. It has produced no scientific result.

**Machines:** training runs on abl-drivsim-iii, 32 cores and an RTX 5070, used for running only. Edit and commit locally, then pull there.

---

## 23. PPO evaluation and reporting

This pipeline turns saved PPO policies into reproducible evaluation records, figures, tables and reports. It changes nothing about training, the reward, the physics, the risk models, the surrogate or minimax. Commands are in Part D of [how_to_run.md](how_to_run.md).

### 23.1 What was reused

| Need | Existing code used |
|---|---|
| Rollouts and outcomes | `eval_batch.run_episodes`, `summarize_seed`, the env's `info` |
| Wilson interval | `eval_batch.wilson_ci` |
| Admission rule | `minimax_feasibility.demonstrated` (Wilson lower ≥ 0.8) |
| Scenario sampling and validation | `feasibility_sampling`, `validate_cutin`, `validate_sandwich` |
| Scenario identity | `feasibility_common.scenario_id` |
| Training | `feasibility_train_one.train_and_evaluate` |
| Road geometry | `Road.frenet_to_global`, `Road.offset_curve` |

`run_episodes` gained an optional `observer`, and `RunParams` gained `metrics_path`. With neither set, behaviour is unchanged. A test confirms that logging leaves trained parameters identical. `meta.json` now also records the physics version, PPO hyperparameters, observation and action sizes, and DT.

### 23.2 Modules

| Module | Role |
|---|---|
| `learning/ppo_experiment.py` | Experiment layout, manifest, seed streams, sampling, training jobs |
| `learning/ppo_metrics.py` | Training and validation scalars (`metrics.jsonl`) |
| `learning/trajectory_log.py` | Rollout recorder, compressed storage, friction utilization, physical validation |
| `learning/evaluate_ppo.py` | Independent evaluation; writes one JSON record per policy and mode |
| `learning/ppo_analysis.py` | Scenario rows, admission, seed selection, Tables A to C, robustness, comparison checks |
| `learning/plot_ppo_results.py` | Static figures (PDF and PNG) |
| `learning/animate_ppo_results.py` | Optional replay from saved rollouts (MP4, GIF fallback) |
| `learning/report_ppo_results.py` | Markdown report, CSV and LaTeX tables, optional HTML |

### 23.3 Seed streams

A realization is `derive_seed(cfg.seed, worker_rank, episode_index)`, so worker ranks act as streams.

| Rank | Stream | Used for |
|---|---|---|
| 0 | Built-in evaluation | `train_and_evaluate` and `animate_rollout`; held out from training |
| 1,000,000 | Early stopping | Best checkpoint, and stopping unless `no_early_stop` is set |
| 1,100,000 | Validation | Learning curves only |
| 1,200,000 | Selection | Choosing one PPO seed per scenario |
| 1,300,000 | Test | Every reported rate and every admission decision |
| 2,000,000 + i | Training | Worker i, a new realization every reset in `robust` mode |

**What training sees.**

- In `robust` mode, training uses env mode `distribution`, so every reset of every worker draws a new realization. Evaluation streams never overlap it (tested).
- In `exact` mode, training uses env mode `fixed`, because there is one realization.
- The metadata and the manifest record this as `training_realization`.
- `train_and_evaluate`'s built-in evaluation stays on rank 0, which is now held out. Experiments here still pass `episodes=0`, because selection and admission need their own streams.

### 23.4 Evaluation protocol

For one policy and one evaluation mode, the evaluator runs the selection stream, then the test stream with recording. When a scenario has several PPO seeds, the seed is chosen on the selection stream only. Taking the best test result across seeds would inflate p̂.

**Sideslip is reported separately from success.** A rollout with body sideslip of 10° or more still counts as a success, because feasibility and stability are distinct and 10° is provisional. Tables A and B report these rollouts separately, and Table B's `friction_status` covers friction limits only. When seeds tie on the selection stream, the one with fewer stability losses there is chosen.

**Speed limit.** `max_ego_speed_mps` in `ExperimentConfig` cuts off acceleration at that speed, and braking and coasting stay free. It is stored in the policy metadata, so training and evaluation share it. The default is 30 m/s, on `ArenaCommonConfig` as well, so it also applies to the MPC's arena runs. Policies trained before the limit existed keep no limit when re-evaluated.

**Traffic crashes are logged, not interpreted.** Each rollout records `traffic_crashes` and `first_traffic_crash_s`, and Table A reports both the crash rate and the rate of timeouts after a crash. A crash does not show the road was blocked, and those timeouts still count as failures. Older records show these as unknown, not zero.

**Exact mode is one outcome.** It has no perturbation, so a deterministic policy repeats itself. It runs 3 times only to confirm that, then counts as one outcome: no Wilson interval and no admission status. If the repeats disagree, the report says so. Robust rollouts, and rollouts of a stochastic policy, are independent draws.

**Admission** applies to independent evaluations only, with p₀ = 0.8 and the two-sided 95% Wilson interval:

| Status | Rule |
|---|---|
| Admitted | Wilson lower ≥ p₀ (`demonstrated`) |
| Not admitted (`rejected`) | Wilson upper < p₀ |
| Uncertain | Otherwise |

| N | Admitted needs k ≥ | Not admitted if k ≤ |
|---|---:|---:|
| 16 | 16 | 9 |
| 20 | 20 | 12 |
| 50 | 46 | 34 |
| 100 | 88 | 72 |

N below 16 can never admit. The upper-bound test is new here: `FeasibilityScreen.record_oracle_evidence` still treats every undemonstrated scenario as UNCERTAIN, because for refinement that is the right reading. "Not admitted" is a statement about this evidence. It never means physically impossible.

### 23.5 Trajectories and physical validation

The recorder stores every field from simulator state and `info`. The first row is the initial state, with controls and forces NaN. Tire forces are sampled once per control step, and the RK4-stage limit is covered by `test_tire_forces.py`.

Friction utilization η_i = √(F_x,i² + F_y,i²)/(μF_z,i) comes from the logged realized forces, and the tolerance is ε = 10⁻⁶. Below μF_z = 1 N, η is undefined and not computed; any force there counts as a violation. Saturation at η = 1 is reported and is not a failure. Stability loss is a separate indicator. A file without forces reports `unavailable`.

Rollouts are recorded in memory and validated for every test episode, but only `trajectories_per_outcome` per outcome class are saved. Recording cost 7% of evaluation throughput (328 to 306 steps/s). A saved rollout is about 150 kB. Training records none.

### 23.6 Figures

Every value plotted is a measurement. No surrogate output or interpolated surface appears.

A feasibility map is a **slice** only when every other parameter is held. Use `--fix`, or the `grid` design, which makes exact slices. Latin hypercube data is a **projection**, and its caption says which parameters vary. Complete grids draw as labelled cells and irregular data as scatter.

Stratum-level intervals bootstrap over scenarios, because rollouts of one scenario are not independent evidence about the stratum. "Scenario difficulty" ranking uses mean peak tire utilization, a measured proxy, because the project has no difficulty metric.

A data class other than `experiment` puts a notice on every figure and at the top of the report. Missing categories are skipped and listed in `figures/index.json`.

### 23.7 Reward comparison

`table_c` reads each policy's reward from its metadata, never its file name. It marks a comparison non-equivalent if training dynamics, rollover events, spaces, DT, evaluation protocol or scenario sets differ, or if any policy lacks reward metadata.

**The legacy reward can't be compared yet.** It was removed from `env.py`. The last commit containing it also has the old physics, and no policy trained with it survives. A valid comparison needs legacy-reward policies trained on `vehicle-physics-v2`, which requires a separate branch.

### 23.8 Limitations

- No PPO result exists. Everything generated so far is a demonstration.
- Existing `feasibility_pipeline` datasets aren't imported. Any made before training moved to its own ranks also trained on one realization, and their evaluations include it. None exists locally.
- Early-stopped runs leave the learning-curve average, so late steps may cover fewer runs.
- Robustness intervals need at least two scenarios per stratum.
- Lane boundaries are drawn midway between lane centres, because lanes 4 m wide sit 3 m apart (§20).
