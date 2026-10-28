# Paralo_plus — Developer Manual

*As of 2026-10-27, branch `rl_branch`.* This manual describes what the repository contains, how the pieces connect, and where the work stands. For copy-paste commands with every flag explained, see the companion guide [feasibility_pipeline.md](feasibility_pipeline.md).

---

## 1. Purpose

Paralo_plus simulates one RL-controlled **ego vehicle** in curved, three-lane highway traffic. Its current goal is to learn a **feasibility surrogate**:

```
p̂(θ) ≈ P(scenario is solvable | θ)
```

θ is a compact parameter vector for a dangerous scenario, such as a car cutting in or a front car emergency-braking.

The surrogate is for a later **`min_w max_θ` search over controller weights w**. That search should only optimize against scenarios that some policy can actually solve. The surrogate tells it which θ values to skip.

The pipeline runs these steps for each θ:

1. Sample a nominal θ for one scenario family (cut-in or sandwich).
2. Train a dedicated PPO policy π_θ on that scenario.
3. Evaluate π_θ over N randomized realizations and record each episode's binary success.
4. Accumulate (θ, episode outcome) rows over many θ and fit a small tabular classifier.

> A PPO failure on θ is evidence about *this training procedure*, not proof that θ is physically unsolvable. Read the surrogate with that caveat in mind.

This is **research code**: the style favours short `assert`s over defensive `if`/`raise` handling. The codebase is about 9,000 lines of Python, including tests.

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
│   ├── far_near.py                two-point steering + FAR_NEAR_PRESETS
│   └── stanley.py                 Stanley steering (unused)
├── initialization/traffic_init.py TrafficAgent, legacy 16-slot traffic template
├── risks/                         PARALO risk models (pure functions)
│   ├── rollover.py                used by the env (rollover termination)
│   ├── slippage.py, departure.py, c2c.py   not wired in yet
├── learning/
│   ├── env.py                     EgoTrafficEnv (Gymnasium) — the hub
│   ├── scenario.py                ScenarioConfig: 15-car compact scenario
│   ├── train.py                   legacy PPO training CLI
│   ├── eval.py                    single-episode animation
│   ├── eval_batch.py              batch eval, Wilson CI, feasibility labels
│   ├── animate_rollout.py         replay evaluation episode K of a feasibility policy
│   ├── feasibility_common.py      shared arena plumbing, scenario_id hash, ArenaRuntime
│   ├── feasibility_cutin.py       cut-in family (θ: 8 dims)
│   ├── feasibility_sandwich.py    sandwich family (θ: 10 dims)
│   ├── feasibility_sampling.py    LHS/Sobol design of experiments + validity rejection
│   ├── feasibility_train_one.py   Runner A: train + evaluate one θ
│   ├── feasibility_pipeline.py    Runner B: pilot / run / fit-only
│   ├── feasibility_dataset.py     resumable JSONL dataset, single-writer
│   ├── feasibility_surrogate.py   classifier selection, bootstrap, SurrogateArtifact
│   └── feasibility_interpret.py   2D heatmaps of feasible regions
├── tests/                         pytest suite (87 tests) + 3 demo scripts
├── artifacts/feasibility/         campaign outputs (see §9)
├── docs/feasibility_pipeline.md   operating guide with commands
├── requirements.txt               numpy, scipy, matplotlib, gymnasium, SB3, torch, sklearn, pytest
└── pyproject.toml                 pytest config only (pythonpath=".")
```

---

## 3. Architecture: how the pieces connect

### 3.1 Layers (each layer imports only from the layers below it)

```
┌──────────────────────────────────────────────────────────────────────────┐
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
│   ego: Car.step(PPO action)   surr: step_surr_agents   risk: rollover    │
├──────────────────────────────────────────────────────────────────────────┤
│ TRAFFIC               model/traffic_step.py :: step_surr_agents          │
│   IDM ─ MOBIL ─ far-near   +   model/collision.py                        │
├──────────────────────────────────────────────────────────────────────────┤
│ PHYSICS               Car ─► CarDynamics ─► rk4      Road / Lane         │
└──────────────────────────────────────────────────────────────────────────┘
```

One exception to the layering: `feasibility_sandwich` imports `MAX_BRAKING` and `DT` from `learning.env`, so the two modules share one braking clamp.

### 3.2 One env step (`EgoTrafficEnv.step`)

```
PPO action [-1,1]² ─► unscale ─► rate-limit (steer 1 rad/s, jerk 40 m/s³)
      ─► ego Car.step(accel, δ, κ(s), μ)                     (ego moves first)
      ─► arena.pre_surr_step(t, ego, agents)  [arena path only]
             cut-in:   sets the cutter's target_lane / lane_change_t0
             sandwich: returns {front_1: −8 m/s²} accel override
      ─► step_surr_agents(agents, ego_car=ego, accel_override=…)
             MOBIL ─► IDM ─► far-near ─► Car.step ─► commit lane ─► pose
             then: surr–surr plastic collisions
      ─► reward, then checks: collision > rollover (P_roll>0.5) > off-road; goal reached
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
| `model/car/config.py` | `VehicleParameters` (frozen), `G` | SUV: m = 2000 kg, l_f = 1.2 m, l_r = 1.6 m, I_zz = 2875, C_f = 19 kN/rad, C_r = 33 kN/rad, h_CG = 0.80 m, track W_c = 1.55 m. `SSF = W_c/(2h_CG)` is the rollover onset in g. |
| `model/car/model.py` | `CarDynamics` | State `[s, e_y, e_psi, v_x, v_y, r]`. Linear tyres saturated at ±μF_z. Axle loads include longitudinal load transfer. `step()` only computes `d_states`. |
| `model/car/car.py` | `CarState`, `Car` | `Car.step(accel, delta, kappa, mu, dt)` builds a fresh `CarDynamics` at every RK4 stage. The **caller** looks up κ and μ. `CarState` also stores x, y, heading, lane and the last δ. `behaviour` 1/2/3 means conservative/moderate/aggressive. |
| `model/road/road.py` | `Road`, `Lane` | Profile: straight → clothoid → 80 m constant arc → clothoid → straight. Grid ds = 0.1 m; 4 m lanes. Each `Lane` stores its own offset, κ and arclength. Helpers: `index_at(s)`, `frenet_to_global(s, e_y, e_psi)`. |
| `model/collision.py` | `bodies_overlap`, `resolve_surr_collisions`, `bleed_crashed`, `ego_overlaps_any` | Axis-aligned boxes in (s, e_y). Surr–surr crashes are perfectly plastic: shared momentum, lateral motion frozen, then the wreck bleeds speed. An ego overlap **never** goes through the crash model; it ends the episode. |
| `model/traffic_step.py` | `step_surr_agents(...)` | The **only** implementation of surrounding-traffic stepping. `tests/traffic_test.py` and the env both call it, so they cannot diverge. |

`step_surr_agents` does the following for each non-crashed agent each tick:

1. **MOBIL** lane-change decision. It is skipped while a change is in progress or during the 1 s cooldown.
2. **IDM** acceleration toward the target lane's leader, clamped at −8 m/s². `accel_override={car_id: a}` replaces the IDM value.
3. **Far-near steering** plus curvature-preview feedforward, rate-limited. During a lane change, the e_y target ramps over the lane-change duration.
4. Look up κ and μ, then call `Car.step`.
5. Commit the lane change once its duration has elapsed (the discrete `lane` flips at this point).
6. Store the global pose.

After every car has moved, it resolves surr–surr collisions once. Passing `ego_car=` wraps the ego in a temporary `TrafficAgent` proxy with id `EGO_CAR_ID = -1`. That makes the ego visible to every surrounding car's IDM and MOBIL.

---

## 5. Traffic controllers and initialization

Only surrounding cars use these controllers. The ego is always RL-driven; the controller docstrings mention a future MPC for the ego.

| Module | Role | Parameters |
|---|---|---|
| `controllers/idm.py` | `idm_accel()`. Not clamped internally; the caller clamps. | `IDM_PRESETS`: T = 2.0/1.4/0.8 s, s0 = 3/2/1 m, a_max = 0.8/1.5/2.5, b = 1.3/2.0/3.0. The desired speed `v0` is set per car. |
| `controllers/mobil.py` | `mobil_decision()`, `MobilParams` | politeness 0.2, threshold 0.2 m/s², plus a `b_safe` braking cap. The arena path sets `threshold=inf`, which disables discretionary lane changes. |
| `controllers/far_near.py` | Steering, curvature feedforward, `clip_steering_rate` | `FAR_NEAR_PRESETS`: shared gains (L_n = 5 m, T_f = 1 s, k = 2.5). Rate limit 0.6/0.8/1.0 rad/s; lane-change duration 5.5/4.0/2.5 s. |
| `controllers/stanley.py` | `stanley_steering()` | **Not used.** |
| `initialization/traffic_init.py` | `TrafficAgent`, `generate_traffic`, `steady_state_speed`, `CAR_LENGTH=4.0`, `CAR_WIDTH` | Legacy path only. Fixed 16-slot template (1 ego + 15 surrounding cars, 3 lanes). The seed changes only the headways and desired speeds. Initial speeds are IDM steady-state speeds. |

`TrafficAgent` = `Car` + controller runtime state: `v0`, `target_lane`, `lane_change_t0`, `prev_delta`, `last_lane_change_t`, `crashed`, `crash_t`, `crash_v`. **Every** surrounding vehicle on every path is a `TrafficAgent`. Scenario families script behaviour by editing these public fields; for example, a cut-in sets `target_lane` and `lane_change_t0`.

---

## 6. Risk models

All four are pure-function ports of the former PARALO ROS nodes. Each takes an explicit predicted trajectory and returns an **unweighted** (probability, score) pair; the caller applies its weight w. The cumulative risk follows PARALO §5.2.5, eq. 5.51.

| Module | Physics | Used by |
|---|---|---|
| `risks/rollover.py` | §5.2.2, eqs. 5.2–5.6: v_safe from turn radius, track width and CG height; P_roll = Φ((v − v_safe)/(Δv/3)) | **`EgoTrafficEnv`**: P_roll > 0.5 means ROLLOVER |
| `risks/slippage.py` | §5.2.2, eqs. 5.9–5.19: speed-dependent μ, minimum safe speed, slip probability | not used yet |
| `risks/departure.py` | Gaussian clearance to a quadratic lane centreline with coefficient covariance | not used yet (the env uses its geometric `_ego_off_road`) |
| `risks/c2c.py` | Rectangle-overlap probability per axis × PARALO closure kinetic-energy severity | not used yet |

The three unused models are the natural building blocks for the weighted cost in the later `min_w max_θ` search.

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
- Goal reached or a failure sets `terminated`. Hitting the time budget sets `truncated`, with outcome TIMEOUT. **A timeout is never a success.**
- `info` keys: `success`, `outcome`, `failure_reason`, `progress_m`, `distance_travelled_m`, `elapsed_s`, `min_clearance_m`, `max_p_rollover`, `max_road_departure_m`. On the arena path it also has `arena`, with family diagnostics such as the trigger station and time.

**Reward.** It exists only to make PPO learn. It is the sum of:

- progress × `PROGRESS_REWARD_SCALE` (boost = 100);
- potential-based goal shaping, totalling `FINISH_BONUS`;
- a lane-tracking cost, 0.05·((e_y/2)² + e_psi²);
- penalties for collision, rollover and off-road;
- `SURVIVAL_REWARD = 1` at truncation, only if `min_progress_m` was reached.

The penalties and `FINISH_BONUS` are pegged to the discounted progress over one effective horizon (γ = 0.99, about 5 s). **If you retune the boost, these move with it automatically. If you change PPO's γ, update `_ASSUMED_GAMMA` too.**

### `learning/scenario.py` (scenario path)

`ScenarioConfig` has:

- global fields: `road_mu`, `road_kappa_max`, background speed mean and std, `aggressive_fraction`, `lane_change_tendency`, `traffic_spread`;
- three `CriticalActorConfig`s (`front`, `rear`, `blocker`), each relative to the ego;
- `goal_distance_m=150`, `max_episode_seconds=20`, `min_progress_m=50`, `seed`.

This path came before the feasibility families. It is still supported, but the pipeline does not use it.

### Supporting scripts

| Script | Purpose |
|---|---|
| `learning/train.py` | PPO on the legacy or scenario path (`--scenario-config scenario.json --scenario-mode fixed\|distribution`). Writes `<out>.zip` + `<out>.meta.json`. |
| `learning/eval.py` | Matplotlib animation of one episode. Reads the meta to rebuild any of the three env paths. |
| `learning/eval_batch.py` | N episodes per model, success rate with a Wilson 95% CI, and a FEASIBLE / NOT_SOLVED / INCONCLUSIVE label across PPO seeds. Its `run_episodes()` is reused by Runner A. |
| `learning/animate_rollout.py` | Replays evaluation episode K of a feasibility policy exactly (distribution mode, rank 0, K+1 resets). Use it to inspect a specific failure from `rollouts.jsonl`. |

---

## 8. Feasibility pipeline

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
- `RunParams.device="cpu"` and `torch_threads=1` are deliberate (see §11).

### 8.4 Runner B — `feasibility_pipeline.py`

| Mode | Does |
|---|---|
| `pilot` | Benchmarks the given concurrency levels on this machine and recommends one. Writes only to `_pilot_bench/`, never to the real dataset. |
| `run` | Samples `n_thetas`, skips ids already done, reconciles finished-but-not-appended results, then trains the rest in a `ProcessPoolExecutor(concurrency)`. A failure in one θ does not stop the batch. `--max-wall-time-hours` stops launching new jobs after that budget. **Resumable**: re-run the same command. |
| `fit-only` | Fits the surrogate on the current dataset and saves the `.joblib`. |

The BLAS/OMP thread environment variables are set to 1 at the top of the module, before numpy is imported. Keep it that way.

### 8.5 Dataset — `feasibility_dataset.py`

The dataset uses a single-writer design. Each worker writes only its own `evaluations/.../<id>.result.json`, atomically. Only the coordinator appends to the shared `scenarios.jsonl` and `rollouts.jsonl`. Path helpers: `dataset_dir`, `scenarios_path`, `rollouts_path`, `evaluations_dir`, `scenario_result_path`.

### 8.6 Surrogate — `feasibility_surrogate.py`

- **Training rows are rollouts** (one row per episode, `y ∈ {0,1}`), not one row per θ with a fractional target. θ is normalized to [0, 1] per field (not clipped).
- There are three candidates: `logistic_poly2`, `extra_trees` (300 trees, depth 6) and `hist_gb` (depth 4). They are compared by **GroupKFold over `scenario_id`**, and the lowest Brier score wins. Log-loss, ROC-AUC and calibration are also reported. If every rollout has the same label, `ConstantClassifier` is the fallback.
- **Epistemic uncertainty** comes from a 200-member bootstrap that resamples whole `scenario_id` groups.
- `SurrogateArtifact.predict(θ)` returns `{p_hat, epistemic_interval (2.5–97.5%), warnings}`. The warnings flag **PRELIMINARY** (fewer than 50 distinct θ) and **EXTRAPOLATION** (θ outside the bounds).
- One surrogate per (family, blocker_side). There is no surrogate across families or sides.

### 8.7 Interpretation — `feasibility_interpret.py`

It writes one PNG per pair of θ dimensions:

- **Empirical panel:** each θ's success rate, interpolated linearly inside the convex hull of the samples. This is a *marginal projection*, not a controlled slice.
- **Surrogate panel** (with `--surrogate`): p̂ on a grid, with the other dimensions held at the dataset median.

`--dims` restricts the plot to a subset of θ fields.

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

> `*.zip` is in `.gitignore`. Trained **policy weights are not in git**; only `model.meta.json` is. The weights live on the training machine (`abl-drivsim-iii`). No surrogate `.joblib` is committed yet.

**`scenarios.jsonl` row:**

```json
{"scenario_id": "a836207f3c9fc314", "family": "cutin", "blocker_side": 1, "mode": "robust",
 "theta": {"road_mu": 0.81, "s_cutin_m": 269.4, "...": "..."},
 "k": 83, "n": 100, "success_rate": 0.83, "wilson_95ci": [0.745, 0.891],
 "n_collisions": 7, "n_rollovers": 0, "n_offroad": 0, "n_timeouts": 10,
 "mean_return": 116.7, "std_return": 14.8, "mean_success_time_s": 33.8,
 "checkpoint": "best", "stop_reason": "early stopping: ...", "timesteps_actual": 1100000, "ppo_seed": null}
```

**`rollouts.jsonl` row:**

```json
{"scenario_id": "a836207f3c9fc314", "episode": 0, "y": 1, "outcome": "finished", "failure_reason": null, "theta": {"...": "..."}}
```

**`model.meta.json`** holds the resolved scenario (family, id, side, mode, seed, θ, goal/time/progress settings) and the training facts (`timesteps_budget`, `timesteps_actual`, `stop_reason`, `n_envs`, `evaluated_checkpoint`, checkpoint paths). It also holds `evaluation.summary`. `eval.py` and `animate_rollout.py` rebuild the env from this file.

### Current results (cut-in, side +1, LHS seed 0)

The first campaign used 24 θ, `--concurrency 8 --n-envs 4 --timesteps 2000000 --episodes 100`, for 2,400 rollouts in total.

| Result | Value |
|---|---|
| Success rate 1.0 (solved) | 10 θ |
| Success rate 0.97–0.99 | 4 θ |
| Success rate 0.07–0.83 (mixed) | 4 θ |
| Success rate 0.0 | 6 θ |
| Failure counts (all rollouts) | 585 timeouts, 138 collisions, 93 off-road, 6 rollovers |
| Training length | 1.1M–2.0M timesteps; 9 runs used the full 2M budget, 15 stopped early |

The heatmaps in `heatmaps/cutin_side1/` are from this dataset (empirical panel only). Timeouts are the largest failure mode. They may mean under-trained policies rather than infeasible scenarios, so check them before trusting a 0% θ.

---

## 10. How to run

Run everything from the repo root inside `.venv`. See [feasibility_pipeline.md](feasibility_pipeline.md) for the full flag list.

```bash
# Tests (the ROS install on this machine injects pytest plugins; disable autoload)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q

# Runner A: one θ
python -m learning.feasibility_train_one --scenario-family cutin --blocker-side 1 --mode robust --timesteps 2000000 --n-envs 4 --episodes 100 --s-cutin-m 200 --cutter-gap-m 25 --out learning/ppo_cutin_theta0

# Runner B: benchmark, campaign, fit
python -m learning.feasibility_pipeline --mode pilot --scenario-family cutin --blocker-side 1 --pilot-concurrencies 1,2,4,8,16 --pilot-envs-per-job 4
python -m learning.feasibility_pipeline --mode run --scenario-family cutin --blocker-side 1 --n-thetas 24 --sampling-seed 0 --concurrency 8 --n-envs 4 --timesteps 2000000 --episodes 100
python -m learning.feasibility_pipeline --mode fit-only --scenario-family cutin --blocker-side 1 --surrogate-out artifacts/feasibility/surrogates/cutin_side1.joblib

# Heatmaps
python -m learning.feasibility_interpret --scenario-family cutin --blocker-side 1 --out-dir artifacts/feasibility/heatmaps/cutin_side1
python -m learning.feasibility_interpret --scenario-family cutin --blocker-side 1 --surrogate artifacts/feasibility/surrogates/cutin_side1.joblib --out-dir artifacts/feasibility/heatmaps/cutin_side1

# Inspect a policy
python -m learning.eval_batch --model artifacts/feasibility/policies/cutin/side_pos1/<id>/model --episodes 100
python -m learning.eval --model artifacts/feasibility/policies/cutin/side_pos1/<id>/model
python -m learning.animate_rollout --model artifacts/feasibility/policies/cutin/side_pos1/<id>/model --episode-index 37 --save out.mp4
```

**Tests** (87 passing on 2026-10-27):

| File | Covers |
|---|---|
| `test_feasibility_arena.py` (25) | Both families: placement, triggers, prescribed events, env integration |
| `test_scenario.py` (12) | `ScenarioConfig` validation and realization |
| `test_env.py` (10) | Env API, outcomes, reward, observation |
| `test_feasibility_surrogate.py` (10) | Grouped CV, constant fallback, bootstrap, predict warnings |
| `test_feasibility_dataset.py` (9) | JSONL I/O, atomic writes, resumability |
| `test_feasibility_sampling.py` (9) | Determinism, bounds, rejection |
| `test_traffic_init.py` (8) | Legacy template invariants |

`dynamics_test.py`, `road_test.py` and `traffic_test.py` are **visual demo scripts**, not pytest tests; they don't match `test_*.py`.

---

## 11. Design decisions to keep

These were each settled deliberately, several after a real bug or an explicit correction. Change one only with a strong new reason.

1. **Cut-in and sandwich stay strictly separate.** Separate configs, θ schemas, samplers, datasets and surrogates. They are never combined, not even as a categorical feature.
2. **Blocker sides +1 and −1 are separate strata**, never one continuous axis. Run the pipeline once per side.
3. **Events trigger spatially** (`ego.s ≥ s_event`, fired once), not by elapsed time.
4. **Success is physical** and independent of reward. A timeout is never a success. Reward changes must not touch the success logic.
5. **Rollover uses the probabilistic `P_roll` model.** An LTR approach was tried and rejected.
6. **The surrogate trains on rollout-level binary rows**, grouped by `scenario_id` for CV and bootstrap. Not one fractional row per θ with sample weights.
7. **CPU, not GPU.** A small MLP over NumPy physics is CPU-bound, so `device="cpu"`. Use `torch_threads=1` and BLAS threads = 1 per process. Scale with **concurrent processes** (`--concurrency`), not a larger `--n-envs`. Run `pilot` on new hardware.
8. **Single-writer dataset.** Workers write only their own result file; only the coordinator appends.
9. **One traffic-step implementation** (`model/traffic_step.py`), shared by the demos and the env.
10. **The ego never goes through the plastic-crash model.** An ego collision ends the episode.
11. Retune the reward penalties together with the progress boost; they are pegged to the same horizon value.

**Pitfalls**

- **Sign convention:** e_y is positive to the right (SAE). The road κ is negated when passed to `Car.step` (`kappa = -lane.kappa[idx]`).
- **The ego's `CarState.lane` never updates**, because the ego has no lane-commit logic. Use `_ego_nearest_lane()`.
- **The remote training machine is read-only for editing.** Edit and commit locally; pull and run there.

---

## 12. Status and next steps

**Built and tested**

- Physics, traffic and risk layers.
- `EgoTrafficEnv` with legacy, scenario and arena paths.
- Both scenario families.
- Runner A, with early stopping and best-checkpoint promotion.
- Runner B (pilot, run, fit-only), the resumable dataset and the surrogate.
- The heatmap interpreter.
- One real campaign: cut-in, side +1, 24 θ.

**Not built yet**

1. **Runner C:** for one θ, train and evaluate it fresh, then compare the empirical rate and Wilson CI with `SurrogateArtifact.predict(θ)`. It would reuse `train_and_evaluate`. The planned shape is in `feasibility_pipeline.md`, Case 6.
2. ~~The `min_w max_θ` controller-weight search.~~ **Now implemented** (with a *proposed* sampling MPC, a fixed external cost J_eval and surrogate screening). See [minimax_design.md](minimax_design.md). The MPC design and the J_eval coefficients are still open research decisions.
   - **Redesign (2026-10-27):** a PARALO-based MPC tuned by *constrained* minimax, with worst-case progress subject to physical event-rate and peak constraints and no evaluation weights. See [constrained_design.md](constrained_design.md). The original pipeline is kept as the baseline.

**Suggested immediate work**

- Run `fit-only` on the existing cut-in data. It will be marked PRELIMINARY because 24 θ is below the 50 threshold.
- Extend cut-in side +1 to 50 θ or more. Re-running with a larger `--n-thetas` and the same seed is not guaranteed to reproduce the first 24 θ, because the design depends on n. Use a new `--sampling-seed` to add points.
- Run cut-in side −1, then sandwich on both sides.
- Investigate the 585 timeouts. They may be under-training, not infeasibility; try a longer `--timesteps` or `animate_rollout` on those episodes.
- Build Runner C.
