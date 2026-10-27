# Surrogate-assisted minimax MPC tuning — design and operating guide

*As of 2026-10-27, branch `rl_branch`.* This document defines the minimax tuning framework, says which parts are established repository behaviour versus proposed defaults, and gives the commands to run it. It is the companion to [developer_manual.md](developer_manual.md) and [feasibility_pipeline.md](feasibility_pipeline.md).

The framework solves, approximately and empirically,

```
w* = argmin_{w ∈ W}  max_{(f, b, θ) admitted}  J_eval(w, θ)
```

One shared MPC parameter vector w is used for every scenario and every selected stratum (family f × blocker side b).

> **Status.**
> - Milestones 1–6 are implemented and tested: 50 new tests, 137 in total, all passing.
> - It has run only on toy problems and small smoke budgets. No full campaign has been run.
> - No surrogate artifact exists in the repo yet (`fit-only` has never been run), so screened experiments need one fitted first (§7.1).

---

## 1. What is confirmed, proposed, or open

| Item | Status |
|---|---|
| Physics, traffic stepping, actuator limits, success/failure semantics | **Confirmed repository behaviour.** Reused unchanged: the MPC drives `EgoTrafficEnv.step()` like PPO does. |
| Scenario families, θ schemas, validators, `scenario_id`, samplers, surrogate artifacts | **Confirmed.** Reused unchanged. |
| The MPC itself (`learning/mpc.py`): sampling MPC over (accel, target lane), horizon 3 s | **Proposed default.** A reasonable first controller, not a validated research choice. |
| The 7 components of w and their bounds | **Proposed default.** |
| J_eval's terms, normalizers, coefficients α and penalties | **Proposed default**, configurable in the experiment JSON. |
| τ = 0.8, conservative screening, the Wilson rule for "demonstrated" | **Proposed default**, configurable. |
| Which MPC structure and w the research should finally use | **Open research decision.** |

No existing module was modified. All new code is in new files.

---

## 2. Architecture

```
learning/
├── ego_control.py         EgoController protocol, EgoView, to_env_action, PPOController, ConstantController
├── mpc.py                 PROPOSED SamplingMPC; MPCWeights (= w), W_SPEC bounds, MPCConfig (fixed)
├── closed_loop.py         Stratum, ScenarioSpec, EvalCostConfig (= J_eval), run_episode, evaluate_controller,
│                          run_episode_jobs (process pool)
├── risk_eval.py           the five PARALO risk channels on the live sim state (inputs for J_eval)
├── minimax_core.py        pure exchange algorithm, master solver, random search, maximin, toy plug-ins
├── minimax_feasibility.py FeasibilityScreen: admitted / uncertain / rejected; oracle evidence registry
├── minimax_oracle.py      OracleRefiner: PPO refinement via train_and_evaluate (unchanged)
├── minimax_adversary.py   ScenarioAdversary: sample → screen → evaluate → local refine → (refine oracle)
├── minimax_store.py       experiment directory, config hash, episode cache, log truncation on resume
└── minimax_pipeline.py    CLI: pilot / optimize / resume / evaluate / report
configs/minimax/           example experiment configs
tests/test_minimax_core.py, test_mpc_controller.py, test_minimax_feasibility.py, test_minimax_pipeline.py
```

Each concern lives in its own module. Numerical optimization (`minimax_core`) knows nothing about vehicles. Scenario selection is in `minimax_adversary`, controller execution in `closed_loop`, and feasibility in `minimax_feasibility` and `minimax_oracle`.

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

---

## 3. Definitions

### 3.1 The MPC and w (proposed)

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

### 3.2 External cost J_eval (fixed per experiment)

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
| rollover | `risks/rollover.py` | (v_n, δ); U_H = 0.1 m |
| slippage | `risks/slippage.py` | (v_n, δ, a); μ_max = the scenario's road μ; U_μ = 0.05 |
| lane departure | `risks/departure.py` | lateral position vs the **nearest** lane's centre; lane width 4 m; vehicle width 2 m; exact map (zero centreline covariance); σ = 0.3 m |
| road departure | `risks/departure.py` | the same model against the whole paved road: centre = mean lane centre, width 12 m |

Per-channel time-integrated and maximum values are recorded for every episode (`risk_integrated`, `risk_max`) and aggregated per scenario.

- The running part is bounded by Σα·term_clip = 40.
- P_terminal is 100 for a collision, rollover or road departure. The config **asserts** that each of these exceeds 40, so a crash is never cheaper than any surviving run, even though it ends the episode early.
- A timeout costs 20 + 30·(1 − progress/goal). It is never a success.
- Success adds no terminal penalty.

Each episode record reports collisions, rollovers, departures, timeouts and solver failures separately.

Zeroing the MPC's internal weights cannot game J_eval, for two reasons. The α coefficients are fixed, and w's lower bounds are positive. A reckless controller simply incurs the failure penalties.

### 3.3 Admissible set Θ_f

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

---

## 4. Algorithm

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

---

## 5. Oracle refinement

`minimax_oracle.OracleRefiner` is off by default because it is expensive. When enabled:

1. Train PPO with `train_and_evaluate` for each seed in `ppo_seeds`, with `timesteps` and `n_envs` as configured.
2. Select the seed with the best selection-evaluation success rate.
3. Re-evaluate **only that policy** on an independent realization stream (`worker_rank = 2,000,000`) for `validation_episodes` episodes.
4. Apply the Wilson rule to the validation (k, n).

The max-over-seeds selection numbers are stored but labelled optimistic. Each result is one atomic JSON under `<experiment>/oracle/<stratum>/<scenario_id>.json`, next to its policies. A resume reloads it instead of retraining.

Refinement never writes to `artifacts/feasibility/`. The global surrogate is **not** refit automatically; refitting it is an explicit, separate step. To fold refinement data into a new surrogate, build a separate dataset version from the oracle JSONs, fit it, and start a **new** experiment. Resuming with a changed surrogate is refused (sha256 check).

---

## 6. Persistence and reproducibility

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

## 7. Commands

Every command runs from the repo root in `.venv` and fits on one line.

```bash
# Tests (all 137)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q

# Toy minimax / maximin check (seconds) and closed-loop throughput benchmark
.venv/bin/python -m learning.minimax_pipeline --mode pilot
.venv/bin/python -m learning.minimax_pipeline --mode pilot --benchmark-workers 1,4,8,16,24,32 --benchmark-episodes 64

# Tiny real-simulation smoke run (no surrogate needed, ~20 s)
.venv/bin/python -m learning.minimax_pipeline --mode optimize --config configs/minimax/smoke_unscreened.json
.venv/bin/python -m learning.minimax_pipeline --mode resume --experiment smoke_unscreened --set exchange.max_iterations=4
.venv/bin/python -m learning.minimax_pipeline --mode evaluate --experiment smoke_unscreened --label heldout
.venv/bin/python -m learning.minimax_pipeline --mode report --experiment smoke_unscreened
```

### 7.1 A real screened run (on abl-drivsim-iii)

```bash
# 1. Fit the surrogate (existing Runner B; writes only artifacts/feasibility/surrogates/)
python -m learning.feasibility_pipeline --mode fit-only --scenario-family cutin --blocker-side 1 --surrogate-out artifacts/feasibility/surrogates/cutin_side1.joblib

# 2. Optimize, then evaluate and report
python -m learning.minimax_pipeline --mode optimize --config configs/minimax/cutin_side1_screened.json
python -m learning.minimax_pipeline --mode evaluate --experiment cutin_side1_screened_v1 --label heldout
python -m learning.minimax_pipeline --mode report --experiment cutin_side1_screened_v1
```

### 7.2 Comparison protocol (§13 of the task)

1. Run the same config five times with a different `experiment_id` each, changing one setting per run:

    | Variant | Override |
    |---|---|
    | Baseline | `--set method='"baseline"'` |
    | Random search | `--set method='"random_search"'` |
    | Minimax, no screening | `--set screening.policy='"none"'` |
    | Minimax, screened | the config as is |
    | Minimax, with refinement | `--set oracle_refinement.enabled=true` |

    Optionally add `--set method='"maximin"'` as the secondary benchmark.

2. Evaluate every variant on **the same** held-out definition by passing the same file to each, for example `--heldout-config configs/minimax/cutin_side1_screened.json`.
3. Tabulate:

```bash
python -m learning.minimax_pipeline --mode report --compare artifacts/minimax/experiments/*/evaluations/heldout.json
```

`compare` warns if the held-out pools or seeds differ.

**What each evaluation reports:**

- worst, mean, q90 and q95 of the per-scenario mean cost;
- success, collision, rollover, departure and timeout rates;
- the number of episodes and runtime.

Each `summary.json` adds simulations run, cache hits, surrogate calls, PPO refinements, solver failures and optimization runtime. Held-out scenarios exclude every scenario the adversary ever saw. Held-out realization seeds must not overlap the optimization seeds (asserted).

---

## 8. Tests added (50)

| File | Covers |
|---|---|
| `test_minimax_core.py` (9) | J = (w−θ)² gives w* = 0, J* = 1; a discrete case where minimax 2 ≠ maximin 1; set max; same w across the set; earlier scenarios kept and no duplicates; a new scenario moves w; iteration, eval and wall-clock budgets; resume equivalence; caching; bounds |
| `test_mpc_controller.py` (9) | Vectorized P_roll matches `risks/rollover.py`; action mapping; w bounds; command and rate bounds; dynamics identical to direct env stepping; determinism; w changes behaviour; independence from PPO reward; solver-failure fallback; J_eval guards and penalties |
| `test_minimax_feasibility.py` (16) | point vs conservative screening; uncertain, rejected and extrapolation cases; missing interval; preliminary flag; oracle override; Wilson rule; real artifacts (identity, missing file, family/side mismatch, bounds mismatch, preliminary refused, single-class fallback); seeding from the dataset |
| `test_minimax_pipeline.py` (9) | Cache key sensitivity; scientific hash; log truncation; strata kept separate; rejected candidates never simulated; oracle refinement (best seed, independent stream, never retrains, failure stays unverified); seed pool admitted on oracle evidence; end-to-end optimize → interrupt → resume equals uninterrupted → held-out evaluation |
| `test_risk_eval.py` (7) | All five channels present and in [0,1]; c2c equals the simulator's collision test as σ → 0; c2c is predictive; lane/road departure = 0.5 exactly at the lane line and road edge (both signs); rollover equals the env's P_roll; slippage uses the scenario's μ; score normalization; β weighting and per-channel records |

---

## 9. Known limitations and open research decisions

1. **The MPC is a proposal.** Its prediction model is deliberately simple: constant-velocity traffic, an idealized lateral response, and ramp corners neglected in the rollover prediction. Discrete acceleration levels limit its resolution. A different controller structure (for example a QP-based linear MPC on the bicycle model) would change what w means and needs a new `controller.name`.
2. **Sandwich with the baseline w times out on the default θ.** The ego waits behind the stopped car instead of taking the escape gap. This is expected behaviour for a minimax search to expose, not a bug, but it does mean J_eval there is dominated by the timeout penalty.
3. **The current surrogate is weak.** On the 24-θ cut-in dataset, a scratch fit was PRELIMINARY. Under conservative τ = 0.8, about 85% of LHS candidates were UNCERTAIN, and the admitted ones were all easy. A meaningful screened campaign needs more PPO data first: 50 or more θ per stratum, and the −1 side and sandwich families.
4. **No robustness certificate.** The result is the worst cost observed over finitely many sampled scenarios and seeds. The true worst case over Θ_f can be higher. J_eval is a mean over a small seed set, so per-scenario noise affects the max. Use more `realization_seeds` for final runs.
5. **The master is non-smooth.** It uses LHS plus bounded Nelder–Mead on a max of noisy simulations, so restarts matter. Budgets need tuning once the throughput on abl-drivsim-iii is measured.
6. **Undecided scientific definitions:**
    - the α weights and penalty magnitudes;
    - whether J_eval should use a mean or a risk measure (for example CVaR) over ξ;
    - the value of τ and whether point or conservative screening is the headline result;
    - what counts as a "meaningful" adversarial improvement (`improvement_tol`, in cost units).
7. **Surrogate path for side −1.** No naming convention exists yet; `all_strata_screened.json` assumes `…_side-1.joblib`.
8. **Slippage dominates E_risk.** With the current `VehicleParameters` (C_f = 19 kN/rad) and PARALO's α_max = 8.6° at μ = 1, the understeer safe speed falls below 20 m/s once δ exceeds about 0.018 rad. The default curve already needs about 0.014 rad. Because U_μ = 0.05 is small, P_CS jumps from 0 to 1 almost immediately. In the baseline cut-in episode, slippage contributed 0.28 of the 0.36 risk term. The simulator has no longitudinal tyre limit and does not end episodes on slip, so this is a risk *indicator*, not a modelled failure. Whether the parameters, U_μ or β_slippage should change is an open calibration question.
9. **Lane departure includes deliberate lane changes.** It is measured against the nearest lane, so every lane change is charged while the car straddles the line. That penalizes lane changes evenly for every controller.
10. **Evaluation cost.** The risk adapter adds about 1.3 ms per step (half of it in c2c), roughly doubling episode time to about 2.4 s. `risk_dt_s = 0.2` halves the overhead.
11. **Private env helpers.** The evaluator reads `EgoTrafficEnv` helpers (`_ego_nearest_lane`, `_ego_clearance`, `_ego_p_rollover`, `_prev_accel`) rather than duplicating them. A rename in `env.py` would break `closed_loop.py`, and the tests would catch it.
