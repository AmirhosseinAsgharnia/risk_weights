# Physics, Oracle-Integrity and Constrained-Minimax Audit (2026-10-28)

Part A was written BEFORE any code change (spec §33). Part B records what
was implemented and measured afterwards.

---------------------------------------------------------------------------

## Part A — Pre-implementation report

### A1. Architecture map (modules that matter here)

| Concern | Module | Role |
|---|---|---|
| Vehicle dynamics | `model/car/model.py` (`CarDynamics`), `model/car/car.py` (RK4), `model/car/config.py` (`VehicleParameters`) | Dynamic single-track model in Frenet coordinates; linear tires, each axle's lateral force clipped at ±μF_z; longitudinal load transfer in F_z; **no longitudinal tire force** (commanded `accel` enters v̇_x directly); no roll degree of freedom |
| Simulator events | `learning/env.py` (`EgoTrafficEnv.step`, `_ego_p_rollover`) | COLLISION / ROLLOVER / OFF_ROAD terminate; ROLLOVER = `P_roll(v_x, δ_cmd) > 0.5` |
| PARALO models | `risks/rollover.py`, `risks/slippage.py`, `risks/c2c.py`, `risks/departure.py`; vectorized ports in `learning/paralo_vec.py` | Turning radius from the **kinematic** relation R = l_r / sin(atan(l_r tanδ / L)) |
| PARALO MPC | `learning/paralo_mpc.py` | Ego prediction with understeer-corrected yaw rate; PARALO input `paralo_steer_input` ∈ {applied (default), kinematic_equivalent} |
| Physical evaluator | `learning/physical_eval.py` | Episode → progress shortfall, events, peaks; rollover = the env's event; slippage = lateral-axle saturation |
| PPO oracle | `learning/feasibility_train_one.py` (train + evaluate), `learning/feasibility_pipeline.py` (DoE, dataset), `learning/feasibility_dataset.py`, `learning/feasibility_surrogate.py` | Policies are trained and evaluated in `EgoTrafficEnv`, so **PPO rewards, termination and success labels all use the env's ROLLOVER event** |
| Admission | `learning/minimax_feasibility.py` (`FeasibilityScreen`, `demonstrated`) | τ = 0.8 lower bound (surrogate) or Wilson ≥ 0.8 (PPO evidence) |
| Refinement | `learning/minimax_oracle.py` (`OracleRefiner`) | Wired into the original adversary only |
| Constrained minimax | `learning/constrained_core.py`, `constrained_adversary.py`, `constrained_pipeline.py` | Master over the union working set (one shared w already); no refinement; no per-stratum report |

### A2. Root cause: rollover and slippage

**The reference value.** 9.5 m/s² is exactly SSF·g = W_c/(2h_CG)·g = 1.55/1.60·9.81 = 9.503 m/s²:
the quasi-static rigid-vehicle tipping condition m a_y h_CG = m g W_c/2. It ignores
suspension roll (which lowers the real threshold, typically by 5–10%), tire
lateral compliance and dynamic overshoot. PARALO's
v_safe = sqrt(g R W_c/(2h_CG)) encodes the same condition, with R taken from the
kinematic steering relation.

**Numerical confirmation.** These are steady-state constant-(v, δ) runs of the
real `CarDynamics` (RK4, 8 s, straight road; script in the conversation
scratchpad). a_y is the realized body lateral acceleration (F_yf + F_yr)/m, and
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
   a separate (event-only) indicator. The dynamics stay unchanged.

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

### A4. Minimal implementation plan

1. `learning/vehicle_physics.py` (new) is the single place for these
   conversions:
   - understeer gradient;
   - realized curvature r/v, with a low-speed policy;
   - δ_eq;
   - realized tire forces and lateral acceleration;
   - the friction-circle demand;
   - versioned event detectors: `rollover_legacy_v1`, `rollover_physical_v2`
     (|a_y| ≥ SSF·g), lateral saturation, friction-circle exceedance;
   - the PARALO risk-input adapter;
   - the friction-information modes.
2. `EgoTrafficEnv(rollover_event=...)`:
   - The default stays `legacy_v1`, so historical runs reproduce exactly.
   - The model name is reported in `info`.
3. `RunParams.rollover_event`:
   - The default is `physical_v2` for **new** PPO data.
   - It is stamped into meta and scenario records.
   - It is threaded through `OracleRefiner`.
4. `physical_eval` → `physical-v2`:
   - Uses the configurable event model.
   - Keeps legacy and new indicators side by side (legacy rollover; lateral
     saturation; friction circle).
   - Adds peaks for a_y/(SSF·g), friction demand and sideslip.
   - An option evaluates under `legacy_v1`, for baseline 1.
5. `ParaloMPC`:
   - The risk adapter becomes the default input (`kinematic_equivalent`,
     derived from the candidate's own predicted yaw rate, never from
     simulator ground truth).
   - Friction information becomes μ̂ (oracle / estimated / conservative modes;
     oracle by default).
   - PARALO's α_max is exposed as an explicitly versioned calibration knob,
     unchanged by default.
6. `learning/feasibility_audit.py` (new), to:
   - write a per-row audit (provenance, validity, inclusion/exclusion reason);
   - re-evaluate available policies under `physical_v2` into a separate
     versioned relabel file;
   - build a curated dataset;
   - fit and validate the surrogate on it (grouped OOF accuracy, calibration,
     false admission and false rejection under the conservative rule).

   The original jsonl files are never modified.
7. Constrained pipeline:
   - PPO oracle refinement inside the constrained adversary;
   - per-stratum worst-case reporting;
   - an optional `completion_time` objective, lexicographic and not weighted,
     for the plateau diagnosis;
   - version stamps in the manifest.
8. The original minimax pipeline's seed pool skips out-of-bounds rows instead
   of crashing.
9. Diagnostic scripts:
   - risk-model validation (PARALO probability vs. a_y, v, κ, μ, δ, δ_eq);
   - a controller-capability suite (7 maneuver types).

### A5. Reuse

- `risks/*` and `learning/paralo_vec.py` (unchanged formulas).
- `CarDynamics` properties (`alpha_f`, `F_yf`, `F_zf`, …) for realized forces.
- `wilson_ci`, `demonstrated`, `FeasibilityScreen`, `OracleRefiner`,
  `SurrogateArtifact`, `fit_surrogate`, `_grouped_oof_predictions`.
- `ScenarioAdversary._local` and the samplers; `minimax_core.solve_master`.
- `ExperimentStore` and the episode cache (keys include the evaluator version,
  so v1 and v2 records never mix).

### A6. Proposed code changes

- New modules: `vehicle_physics.py`, `feasibility_audit.py`,
  `risk_validation.py`, `capability_suite.py`.
- Edits (small, additive, defaults preserving legacy behaviour wherever
  historical results depend on it): `env.py`, `physical_eval.py`,
  `paralo_mpc.py`, `feasibility_train_one.py`, `feasibility_pipeline.py`
  (stamp), `minimax_oracle.py`, `constrained_adversary.py`,
  `constrained_core.py` (per-stratum fields only), `constrained_pipeline.py`,
  `minimax_pipeline.py` (seed-pool skip).

### A7. Validation plan

- Unit tests for the rollover regression cases §6.1–6.8, using real dynamics
  and a recorded closed-loop false positive.
- Slippage across μ, combined braking and cornering, and low-speed stability.
- Adapter consistency (MPC δ_eq equals simulator δ_eq at steady state).
- Friction modes.
- Env default unchanged (legacy) and v2 wiring.
- Dataset audit counts.
- Relabel determinism.
- Adversary refinement with a stub trainer.
- Per-stratum reporting and the shared w.
- Then the numerical diagnostics: risk validation, capability suite,
  four-way baseline, and the PPO relabel of the local policies.

---------------------------------------------------------------------------

## Part B — What was implemented and measured

### B1. Files

New:
- `learning/vehicle_physics.py`: conversions, realized tire state, versioned detectors, PARALO input adapter, friction modes.
- `learning/feasibility_audit.py`: audit, relabel, curation, admission validation.
- `learning/risk_validation.py`
- `learning/capability_suite.py`
- Tests: `tests/test_vehicle_physics.py` (11), `tests/test_feasibility_audit.py` (3), `tests/test_constrained_refinement.py` (5).

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
- The simulator has no longitudinal tire force, so braking beyond μg is physically allowed and only flagged.
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
