# Constrained minimax tuning of the PARALO MPC — design, status and findings

This document covers the redesign specified in "Implementation Task: Constrained Minimax Optimization for PARALO-Based MPC" (2026-10-27). It replaces the weighted evaluation cost J_eval with a progress objective plus explicit physical constraints, and replaces the original MPC with one whose cost is built on PARALO risks. The original MPC and J_eval pipeline ([minimax_design.md](minimax_design.md), `learning/minimax_pipeline.py`) remain unchanged as the baseline.

Commands are in [how_to_run.md](how_to_run.md), Part B. The design discussion that led here is in [proposal_paralo_mpc.md](proposal_paralo_mpc.md).

---

## 1. Formulation

```
w* = argmin_{w ∈ W}  max_{θ ∈ Θ_f}  J_progress(w, θ)
     s.t.            max_{θ ∈ Θ_f}  R_i(w, θ) ≤ ε_i        for every ENFORCED constraint i

J_progress(w, θ) = mean over realization seeds ξ of clip(1 − s_final/s_goal, 0, 1)
R_i(w, θ)        = an event rate (k/n over the seeds) or a physical peak (max over the seeds)
```

- There is **no evaluation weight vector**: the configuration has no `cost`/α section (a test checks this).
- The MPC uses weights internally, but only physical outcomes judge it.
- Θ_f is the PPO-admitted set (`learning/minimax_feasibility.py`), with admission semantics unchanged: a surrogate lower bound ≥ τ = 0.8, or direct PPO evidence with a Wilson lower bound ≥ 0.8.
- Admission is a filter only. It never enters the objective.

---

## 2. Mapping to the existing code (§2 and §17 of the task)

### 2.1 Reused unchanged

| Component | Module |
|---|---|
| PARALO risk models | `risks/c2c.py`, `risks/rollover.py`, `risks/slippage.py`, `risks/departure.py` |
| Simulator, outcome logic, actuator limits | `learning/env.py`, `model/` |
| PPO oracle admission | `learning/minimax_feasibility.py` (`FeasibilityScreen`, Wilson rule) |
| Scenario families, samplers, validators | `feasibility_cutin.py`, `feasibility_sandwich.py`, `feasibility_sampling.py` |
| Scenario identity, the four strata | `closed_loop.ScenarioSpec`, `Stratum` |
| LHS + bounded Nelder–Mead master machinery | `minimax_core.solve_master` |
| Adversary sampling / screening / local perturbation | `minimax_adversary.ScenarioAdversary` (subclassed) |
| Persistence, cache, config hash, resume, log truncation | `minimax_store.ExperimentStore` |

### 2.2 New modules

| Module | Responsibility |
|---|---|
| `learning/paralo_vec.py` | per-state PARALO probabilities, vectorized; exact equality with `risks/` is tested |
| `learning/paralo_mpc.py` | `ParaloMPC`: candidate selection, prediction, stage cost; `ParaloWeights` (w), `ParaloMPCConfig` (fixed) |
| `learning/physical_eval.py` | closed-loop episode → progress, events, peaks; per-scenario rates and Wilson intervals; `ConstraintSpec` |
| `learning/constrained_core.py` | vehicle-agnostic constrained master, exchange loop, statuses, final selection |
| `learning/constrained_adversary.py` | separate performance and per-constraint adversarial searches |
| `learning/constrained_pipeline.py` | config, cached physical oracle, checkpoint/resume, held-out comparison, report, CLI |

### 2.3 Small changes to existing files

- `closed_loop.build_controller` gains `"paralo_mpc"`.
- `minimax_store.NON_SCIENTIFIC_EXCHANGE_KEYS` gains `max_scenario_evals`, so this budget can be raised when resuming.

### 2.4 Contradictions and deviations, stated before implementation

1. **The "kinematic bicycle" prediction is materially wrong for this vehicle, so it is corrected.**
   - The simulated SUV understeers strongly. Its understeer gradient is K = m/L·(l_r/C_f − l_f/C_r) = 0.034 rad·s²/m.
   - Holding δ, the plain kinematic model r = v·tanδ/L over-predicts the simulator's steady yaw rate by about **2.2× at 10 m/s and 5.9× at 20 m/s**.
   - The MPC therefore predicts with the steady-state single-track form `r = v·tanδ/(L + K·v²)`, which matches the simulator within 2% (test: `test_prediction_yaw_model_matches_the_simulator_steady_state`).
   - The model keeps the kinematic structure (no lateral-velocity state). Setting `understeer_gradient: 0.0` restores the plain kinematic model.
   - Steering stays **absolute**, as decided.
2. **The simulator has no slippage event and no lane-departure event.** Both are defined in `physical_eval` and documented there (§4).
   - The lane-departure event counts every lane-marking crossing, including deliberate lane changes. The MPC has no target-lane variable, so a deliberate change cannot be told apart from an unintended one.
   - Any threshold below 1 therefore forbids escaping a cut-in. This constraint is shipped as **monitored only** and shown as such in every report. It is not silently dropped.
3. **Nelder–Mead does not handle general constraints** (§5 has the method chosen and the reason).
4. **PPO oracle refinement is not wired into the constrained adversary yet.**
   - Uncertain candidates are logged and never evaluated.
   - The original pipeline's refiner (`minimax_oracle.py`) is unchanged and still works there.
5. **The PPO dataset mixes two θ-bound definitions.**
   - 14 of the first 24 cut-in(+1) rows lie outside the current `THETA_BOUNDS_CUTIN` (`front_gap_m` < 25, `cutter_relative_speed_mps` outside [−5, 15]). The 100 newer rows are all inside.
   - The constrained pipeline skips seed-pool scenarios that fail the current validator and lists them in `manifest.json`.
   - The original pipeline's seed pool does not skip them and will raise `ScenarioFamilyConfigError` with `configs/minimax/cutin_side1_screened.json`.
   - Refitting the surrogate on this mixed dataset is a separate decision for you; see `docs/feasibility_pipeline.md` on archiving bound-inconsistent data.

---

## 3. The PARALO MPC (`learning/paralo_mpc.py`)

Every 0.1 s (2 env steps) the MPC does the following:

1. It enumerates 7 accelerations {−8, −4, −2, −1, 0, 1, 2} m/s² × 13 absolute steering values. The steering values are the current δ + {0, ±0.002, ±0.005, ±0.01, ±0.02, ±0.04, ±0.08} rad, clipped to ±0.5. That gives 91 candidates.
2. It holds each candidate pair over a 3 s horizon at 0.1 s steps.
   - The env's jerk limit (40 m/s³) and steering-rate limit (1 rad/s) are modelled as ramps toward the held value.
   - The env still enforces both limits on the applied command (tested).
3. It predicts the motion:
   - **Ego:** the corrected single-track model above, in the ego lane's Frenet frame, using the env's own sign conventions.
   - **Surrounding cars:** constant velocity.
4. It minimizes `Σ_{n=1..30} ℓ(x̂_n, u; w)` over the candidates:

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
| P_roll, P_slip | PARALO, given (v, held δ, a). P_slip uses the scenario's road μ and U_μ = 0.05 |
| P_lane | PARALO departure against the lane centre **nearest to each predicted state** |
| P_road | PARALO departure against the paved road |

The w bounds and w₀ are proposed defaults. The P terms are per-state probabilities summed as a **planning surrogate**. The sum is not treated as a probability of at least one event, and no independence between time steps is assumed.

- **Solver failure.** A non-finite cost is counted, and the MPC brakes at −8 while holding the current steering.
- **Slippage-transition diagnostics** are kept per episode in `mpc_diagnostics`. Each counts replans out of all replans:

| Counter | Counts replans where… |
|---|---|
| `slip_active` | the slip term is non-zero |
| `slip_indistinct` | the slip term is active but equal across all candidates |
| `slip_saturated_all` | every candidate has P_slip ≈ 1 throughout the horizon |
| `slip_decisive` | the chosen plan changes when the slip term is removed |

- **`paralo_steer_input`.** The default `"applied"` passes the held δ to PARALO, the same input the env's rollover event uses. The option `"kinematic_equivalent"` passes atan(L·r/v) instead. §8 explains why this choice matters.

---

## 4. Physical evaluation (`learning/physical_eval.py`)

Every quantity below comes from actual simulator states and applied controls, never from predictions.

| Quantity | Definition |
|---|---|
| progress shortfall | `clip(1 − progress_m/goal, 0, 1)`. `progress_m` is the env's Frenet arc length along the ego lane, so lateral motion is never counted as progress. A safety failure ends the episode; the shortfall is then whatever progress was actually achieved, and the failure is recorded separately as an event. |
| collision | the env's COLLISION terminal condition |
| rollover | the env's ROLLOVER terminal condition (P_roll(v, δ) > 0.5) |
| road departure | the env's OFF_ROAD terminal condition |
| slippage (new) | some step's front or rear lateral tire force reaches its friction limit, \|C·α\| ≥ μ·F_z, in the simulator's own tire model. Does not end the episode. |
| lane departure (new) | the body crosses a lane marking: \|e_y to the nearest lane centre\| > (l_w − W)/2 = 1 m. Does not end the episode. |
| peaks | max \|applied accel\|; max \|lateral accel\| (body frame: dv_y/dt + r·v_x); max \|applied jerk\| |

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
| rollover rate | 0 | provisional |
| road-departure rate | 0 | provisional |
| slippage rate | 0 | provisional |
| peak lateral acceleration | 4.0 m/s² | provisional |
| lane-departure rate | — | **monitored only**: every lane change counts (§2.4) |
| peak jerk | — | **monitored only**: emergency braking legitimately hits the 40 m/s³ actuator limit |
| peak acceleration | — | **monitored only**: full braking is a legitimate safety response |

---

## 5. Constrained master and exchange (`learning/constrained_core.py`)

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
| `no_feasible_found` | the master search found no feasible w. The last master records `scenarios_never_satisfied`: scenarios that *no* evaluated w satisfied on their own. These are candidates for "PPO-admitted but beyond this MPC architecture" (§3 of the task). |
| `budget_exhausted` | an iteration, evaluation or wall-clock budget was hit. Not convergence. |
| `no_admissible_scenarios` | the adversary found no admitted scenario |
| `master_infeasible_certified` | defined, but **never produced**: a sampling search cannot certify infeasibility |

Each iteration record has a `counterexample` flag for the case where w_k was feasible on S_k but violated a constraint on a new scenario.

**Reported solution** (`select_final`):
- After `converged_feasible`, the last iterate.
- Otherwise, every iterate is re-assessed on the final working set and the lexicographically best one is reported with its `feasible_on_final_set` flag.

An infeasible candidate is never labelled feasible (tested). No global optimality, robustness or safety certification is claimed.

---

## 6. Adversary (`learning/constrained_adversary.py`)

At a fixed w_k, for each stratum in its own θ schema:

1. Draw a seeded LHS batch plus the seed pool.
2. Screen every candidate. Only admitted ones are simulated.
3. Run one closed-loop evaluation per scenario, which yields the objective and every constraint value at once.
4. Run local Gaussian refinement **per criterion** (the objective and each enforced constraint) around that criterion's own top candidates. Constraint refinement only happens around scenarios with a positive value.
5. Return the worst admitted scenario for the objective and for each constraint, separately.

Rejected and uncertain scenarios are never simulated (tested). The four strata stay independent and keep their own schemas (tested).

---

## 7. Reproducibility and logs

The experiment directory is `artifacts/constrained/experiments/<id>/`.

| File | Contents |
|---|---|
| `config.json` | the full normalized config, including the constraint specs, and its scientific hash |
| `manifest.json` | git commit, surrogate sha256s, evaluator version, seeds, constraints, seed-pool rows skipped as invalid |
| `iterations.jsonl` | per iteration: w_k, working set, master assessment (t, worst value, scenario and violation per constraint), master candidate counts, adversary results per criterion, scenarios added and why, counterexample flag, status |
| `controller_candidates.jsonl` | **master** log: every (w, S_k) evaluation with per-scenario objectives and constraint values |
| `adversarial_candidates.jsonl` | **adversary** log: every candidate with its admission status and evidence, p̂ and interval, metrics |
| `checkpoints/state.json`, `summary.json`, `best_weights.json` | resume state, final result, best weights with their feasibility flag |
| `evaluations/<label>.json`, `plots/convergence.png` | held-out comparison, convergence plot |
| `cache/` | one episode per file |

**Episode cache key:** controller spec (with its full config), w at full precision, scenario key, seed, `PHYSICAL_EVALUATOR_VERSION` and the θ schema version. A test checks that a cached episode equals a direct re-simulation. Wall-clock timing is stored under `timing` and excluded from equality checks.

**Resume** follows the original pipeline's rules:
- The scientific config hash must match. Only `n_workers`, `artifacts_root` and the exchange budgets may change.
- The surrogate sha256s and the evaluator version must match the manifest.
- Log rows from iterations that never reached a checkpoint are dropped.
- A `budget_exhausted` run continues when its budget is raised.

A test checks that a resumed run reproduces the uninterrupted run's w sequence.

---

## 8. Findings so far

The runs so far were tests plus one diagnostic. **No constrained optimization campaign has been run.**

### 8.1 Diagnostic: original MPC vs. PARALO MPC at w₀

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

So the slip term always separates the candidates and decides about half of all plans. It is not flat, but it strongly steers the controller toward slowing down. That is consistent with §8.2.

### 8.2 PARALO rollover and slippage versus this simulator: the key open issue

PARALO's rollover and slippage models compute the turning radius from δ with a **kinematic** relation, R = l_r/sin β. In this understeering SUV the δ needed for a given path is about 6× larger at 20 m/s.

| Road curvature, speed | Steady δ | P_roll | P_slip | True lateral accel |
|---|---|---|---|---|
| κ = 0.005 (default road), 20 m/s | 0.082 | 1.00 | 1.00 | 2.0 m/s² |
| κ = 0.005, 15 m/s | 0.052 | 0.00 | 1.00 | 1.1 m/s² |
| κ = 0.010, 20 m/s | 0.165 | 1.00 | 1.00 | 4.0 m/s² |
| κ = 0.020 (max in θ), 20 m/s | 0.329 | 1.00 | 1.00 | 8.0 m/s² |

Consequences:
- **The env's rollover event (P_roll > 0.5) fires on moderate curves at highway speed**, at a true lateral acceleration of about 2 m/s². The physical rollover threshold is SSF·g ≈ 9.5 m/s². This is why the original MPC "rolled over" in 9.5% of episodes, and why the unscreened smoke run found a rollover counterexample at κ = 0.019.
- Meanwhile **the simulator's own tires saturate only rarely**: the original MPC had a 3.6% slippage-event rate, the PARALO MPC 0%. In this simulator, P_slip ≈ 1 does not mean tire slip.
- The PPO feasibility data was generated under the same env semantics, so admission is consistent with this rollover definition.

These are modelling decisions for the project owner. The code changes nothing about them; it only exposes options. Possible directions:
- Keep everything as is (the current default).
- Feed PARALO the kinematic-equivalent steering (`paralo_steer_input: "kinematic_equivalent"`) inside the MPC. The env's rollover event would still use the applied δ.
- Recalibrate the models' vehicle parameters.
- Revisit the env's rollover event definition. This changes simulator semantics, so it is outside this task.

---

## 9. Limitations

1. The w bounds, w₀, c_jerk, c_latacc, c_road, the candidate grid and every constraint threshold are proposed or provisional.
2. Event rates are point estimates over few seeds (8 in the shipped configs). k = 0 is not zero probability. `wilson_upper` is available but cannot be met with ε = 0.
3. The lane-departure event cannot separate intended from unintended crossings (§2.4.2), so it is monitored only.
4. Both inner problems are solved approximately by sampling and local search. Results are "best found under budget", with no global or robustness guarantee.
5. `no_feasible_found` is not proof of infeasibility.
6. PPO admission does not imply that this MPC architecture can solve a scenario. The `scenarios_never_satisfied` list points to such cases but proves nothing.
7. Oracle refinement is not wired into the constrained adversary (§2.4.4).
8. The cut-in(+1) dataset mixes θ-bound definitions (§2.4.5), and no surrogate has been fitted yet, so screened configs need step 4 of the run guide first.
9. The PARALO MPC is about 5× slower per replan than the original, but still a few milliseconds.
10. The MPC reads private env attributes (`env._prev_accel`), as `learning/risk_eval.py` already does.

---

## 10. Tests added (24)

| File | Tests | What they cover |
|---|---|---|
| `test_paralo_vec.py` | 5 | vectorized c2c (σ > 0 and σ → 0), slippage (two values of μ) and departure (lane and road width) equal the `risks/` modules |
| `test_paralo_mpc.py` | 6 | w round-trip and bounds; prediction yaw model against the simulator; candidates absolute, within bounds, horizon and step; cost uses w and all five risks react; actuator limits and weights kept across replans; determinism |
| `test_physical_eval.py` | 5 | collision recorded separately with actual-progress shortfall; shortfall bounds, success = 0; rates, Wilson intervals and peaks; constraint independence, monitored-only, `wilson_upper`, unit checks; slippage = tire saturation |
| `test_constrained_core.py` | 5 | toy problem reaches its known solution (w* = J* = 0.7); infeasible never reported feasible; budget exhaustion ≠ convergence; worst-case aggregation and lexicographic merit; adversary returns separate counterexamples per criterion and keeps strata schemas apart |
| `test_constrained_pipeline.py` | 3 | no evaluation weights in the config; inadmissible scenarios never simulated; end-to-end optimize, interrupt and resume equivalence, cache = direct re-simulation, held-out comparison of all three controllers, seed-overlap guard |

The full suite is 161 tests, all passing.
