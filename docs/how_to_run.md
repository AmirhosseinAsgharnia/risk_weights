# How to run: from a fresh checkout to a tuned MPC

This guide lists the steps in order. Each step says what it does, the command to run, and what to check before moving on. For *why* things work this way, see [description.md](description.md).

Run every command from the repository root. All commands fit on one line. Long runs belong on abl-drivsim-iii: edit and commit locally, then pull and run there.

There are two tuning pipelines. Both share steps 0 and 3–5 (setup, PPO data, surrogates):

- **Part A** (steps 1–2 and 6–10): the original MPC, scored by one weighted cost J_eval. It is kept as the baseline.
- **[Part B](#part-b--constrained-minimax-paralo-mpc)**: the PARALO MPC, tuned for worst-case progress subject to physical safety and comfort constraints, with no evaluation weights. Its design is in [description.md §13](description.md).
- **[Part C](#part-c--single-scenarios-and-inspection)**: train one scenario, evaluate or watch a policy, and draw heatmaps.

```
 0. Setup               install, run the tests
 1. Pilot               toy check + choose n_workers                   (minutes)
 2. Smoke run           tiny end-to-end minimax run, no surrogate       (~20 s)
 3. PPO data            train PPO on many θ, per stratum                (hours, big machine)
 4. Fit surrogate       one .joblib per stratum                         (seconds)
 5. Inspect (optional)  feasibility heatmaps
 6. Optimize            screened minimax search for w*                  (hours)
 7. Resume              only if step 6 was interrupted
 8. Evaluate            w* on held-out scenarios
 9. Report              plots + summary
10. Compare             baseline / random / unscreened / screened / refined
```

A **stratum** is one (scenario family, blocker side) pair: `cutin +1`, `cutin -1`, `sandwich +1`, `sandwich -1`. Strata are never merged. Each stratum gets its own dataset and its own surrogate.

---

## Step 0 — Setup

Make a virtual environment once, then activate it in every new shell:

```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

On a machine without a GPU, install the CPU build of torch first. See the comment in `requirements.txt`.

Run the tests. They take a few minutes and do not train anything substantial:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

**Check:** every test passes. You need `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` when ROS is installed: without it, ROS's pytest plugins load and fail with `ModuleNotFoundError: yaml`.

---

## Step 1 — Pilot: toy check and throughput benchmark

First run a quick check with toy problems. It needs no simulation:

```bash
python -m learning.minimax_pipeline --mode pilot
```

**Check:** it prints `w*≈0, J*≈1` for the toy problem and shows that minimax and maximin differ on the matrix example.

Next, measure how many closed-loop episodes per second the machine runs at different worker counts:

```bash
python -m learning.minimax_pipeline --mode pilot --benchmark-workers 1,4,8,16,24,32 --benchmark-episodes 64
```

**Check:** the last line suggests an `n_workers`. Use that value in step 6, either in the config or with `--set n_workers=<N>`. One episode takes about 2.4 s on one core.

---

## Step 2 — Smoke run (end to end, no surrogate)

This is a very small minimax run on the cut-in family with screening turned off. It tests the whole chain: optimize, interrupt and resume, evaluate, and report.

```bash
python -m learning.minimax_pipeline --mode optimize --config configs/minimax/smoke_unscreened.json
```
```bash
python -m learning.minimax_pipeline --mode resume --experiment smoke_unscreened --set exchange.max_iterations=4
```
```bash
python -m learning.minimax_pipeline --mode evaluate --experiment smoke_unscreened --label heldout
```
```bash
python -m learning.minimax_pipeline --mode report --experiment smoke_unscreened
```

**Check:** `artifacts/minimax/experiments/smoke_unscreened/` contains `best_weights.json`, `summary.json`, `evaluations/heldout.json` and `plots/convergence.png`.

The numbers mean nothing because the budget is tiny. Before running the smoke test again, delete the folder: `rm -rf artifacts/minimax/experiments/smoke_unscreened`.

---

## Step 3 — Collect PPO feasibility data (per stratum)

The surrogate learns *P(PPO can solve θ)* from PPO training results. This is the expensive step, so run it on the big machine (32 cores). Repeat 3a–3b for every stratum you plan to optimize over. The commands below are for `cutin +1`.

**3a. Benchmark concurrency.** Run this on new hardware. It recommends a `--concurrency` and does not write to the real dataset:

```bash
python -m learning.feasibility_pipeline --mode pilot --scenario-family cutin --blocker-side 1 --pilot-concurrencies 1,2,4,8,16 --pilot-envs-per-job 4
```

Measured on `abl-drivsim-iii` (32 cores, RTX 5070) with `--pilot-envs-per-job 4`:

| Concurrency | Scenarios/hour | Env workers |
|---:|---:|---:|
| 1 | 268.6 | 4 |
| 2 | 413.0 | 8 |
| 4 | 632.5 | 16 |
| 8 | 932.3 | 32 (physical core count) |
| 16 | 1079.7 | 64 (2× oversubscribed) |

`pilot` picks the highest number, but that comes from a 53-second burst. Concurrency 8 is the safer choice for a long run, because it sits at the physical core count and captures most of the gain.

**3b. Sample θ and train PPO on each one** (resumable; re-running the same command skips scenarios that are already finished, and picks up finished-but-unrecorded results without retraining):

```bash
python -m learning.feasibility_pipeline --mode run --scenario-family cutin --blocker-side 1 --n-thetas 100 --sampling-seed 1 --concurrency 8 --n-envs 4 --timesteps 2000000 --episodes 100
```

Things to know:

- `--n-thetas` **adds** that many new scenarios. It is not a target total. Use a **new `--sampling-seed`** for each extra batch.
- With fewer than 50 scenarios, the surrogate is marked `PRELIMINARY`. With so few, a conservative screen leaves most candidates `UNCERTAIN`. Aim for at least 100 per stratum.
- **There is no PPO data at the moment.** The old `cutin +1` dataset was deleted on 2026-10-28; its labels came from the legacy rollover detector (see [description.md §15](description.md)). Every stratum starts from scratch.
- New runs train and evaluate with the corrected rollover event (`--rollover-event physical_v2`, the default). Every dataset row records its `rollover_event_model`, `theta_bounds_version` and `git_commit`. Never append `legacy_v1` rows to a `physical_v2` dataset.
- Results are written to `artifacts/feasibility/datasets/<family>/side_pos1|side_neg1/`.
- Run both blocker sides separately. They get independent designs, datasets and surrogates.
- If you change `THETA_BOUNDS_CUTIN` or `THETA_BOUNDS_SANDWICH`, also bump the bounds version and treat the old data as a separate experiment. Rows are stamped, so the audit excludes old out-of-bounds rows, but archiving keeps the dataset honest.
- **Back up the policies after a campaign.** They are not in git, so they exist only on that machine.

For the other strata, change `--scenario-family` and `--blocker-side`, for example `--scenario-family sandwich --blocker-side -1`.

---

## Step 4 — Audit, fit and validate the surrogate (per stratum)

**4a. Audit the dataset.** This writes `audit_dataset-audit-v1.jsonl` next to the data and never edits the data itself. It marks every row as included or excluded, and says why:

```bash
python -m learning.feasibility_audit --mode audit --scenario-family cutin --blocker-side 1
```

**4b. Fit.** The fit only uses rows the audit included. Rows outside the current θ bounds, or whose labels came from the legacy rollover event, are left out. The curation record is stored inside the artifact as `provenance`.

```bash
python -m learning.feasibility_pipeline --mode fit-only --scenario-family cutin --blocker-side 1 --surrogate-out artifacts/feasibility/surrogates/cutin_side1.joblib
```

**4c. Validate the admission decisions.** This uses grouped cross-validation and reports how often the surrogate admits scenarios PPO could not demonstrate (false admissions) and rejects ones it could (false rejections), together with calibration:

```bash
python -m learning.feasibility_audit --mode validate --scenario-family cutin --blocker-side 1
```

**Check:** look at the number of scenarios, the cross-validated metrics, whether the fit is `PRELIMINARY`, and the false-admission and false-rejection counts.

The file name must match the `surrogate` path in your minimax config. The shipped configs expect these names:

| Stratum | Surrogate path |
|---|---|
| cutin +1 | `artifacts/feasibility/surrogates/cutin_side1.joblib` |
| cutin −1 | `artifacts/feasibility/surrogates/cutin_side-1.joblib` |
| sandwich +1 | `artifacts/feasibility/surrogates/sandwich_side1.joblib` |
| sandwich −1 | `artifacts/feasibility/surrogates/sandwich_side-1.joblib` |

---

## Step 5 — Inspect feasibility (optional)

Draw heatmaps of the regions that are feasible and infeasible for PPO. The first command uses only the data. The second also draws the surrogate's prediction:

```bash
python -m learning.feasibility_interpret --scenario-family cutin --blocker-side 1 --out-dir artifacts/feasibility/heatmaps/cutin_side1
```
```bash
python -m learning.feasibility_interpret --scenario-family cutin --blocker-side 1 --surrogate artifacts/feasibility/surrogates/cutin_side1.joblib --out-dir artifacts/feasibility/heatmaps/cutin_side1
```

To restrict the plots to some θ dimensions instead of every pair:

```bash
python -m learning.feasibility_interpret --scenario-family cutin --blocker-side 1 --dims front_gap_m,cutter_relative_speed_mps,s_cutin_m --out-dir artifacts/feasibility/heatmaps/cutin_side1
```

It writes one PNG per pair of θ dimensions (28 for cut-in, 45 for sandwich).

- **Empirical panel** (always): every sampled scenario's success rate, projected onto that pair and interpolated inside the convex hull of the samples. It is a marginal projection, not a controlled slice. Gray means no data.
- **Surrogate panel** (with `--surrogate`): p̂ on a grid, with the other dimensions held at the dataset median. This one is a controlled slice.

This lets you judge the surrogate before trusting it to screen scenarios.

---

## Step 6 — Optimize w (screened minimax)

Pick a config:

| Config | Strata | Use |
|---|---|---|
| `configs/minimax/cutin_side1_screened.json` | cutin +1 | first real run; seeds the adversary from the PPO dataset |
| `configs/minimax/all_strata_screened.json` | all four | needs all four surrogates from step 4 |

```bash
python -m learning.minimax_pipeline --mode optimize --config configs/minimax/cutin_side1_screened.json --set n_workers=24
```

The run prints `summary.json` when it finishes. The best weights are written to `artifacts/minimax/experiments/<experiment_id>/best_weights.json`.

Change settings with `--set key.sub=<json>`. Strings need inner quotes. Some examples:

- `--set exchange.max_iterations=25`
- `--set screening.tau=0.7`
- `--set cost.beta_slippage=0.2`
- `--set experiment_id='"cutin_side1_tau07"'`

**An experiment ID cannot be reused.** If the ID already exists, `optimize` refuses to start. Either change `experiment_id` or use `resume`.

To keep a long run going after you log out, start it inside `tmux` or `screen`, or prefix it with `nohup` and add `> run.log 2>&1 &` at the end.

---

## Step 7 — Resume (only if interrupted)

```bash
python -m learning.minimax_pipeline --mode resume --experiment cutin_side1_screened_v1
```

The run continues from the last checkpoint. Finished episodes are read back from the cache. If the run stopped because it hit a budget, you can raise the budget when you resume, for example with `--set exchange.max_iterations=30`.

Resume refuses to continue if any of the following changed:

- the scientific config (anything except `n_workers`, `artifacts_root` and the exchange budgets);
- the surrogate files;
- the evaluator version.

When that happens, start a new experiment instead.

---

## Step 8 — Evaluate w* on held-out scenarios

```bash
python -m learning.minimax_pipeline --mode evaluate --experiment cutin_side1_screened_v1 --label heldout
```

The held-out set is fresh: it contains new admitted scenarios and new realization seeds that do not overlap the optimization seeds. The command writes `evaluations/heldout.json` with:

- the worst, mean, q90 and q95 scenario cost;
- success, collision, rollover, off-road and timeout rates.

To evaluate a hand-picked w instead of `best_weights.json`, write a JSON file of natural-unit weights. Any name you leave out keeps its default:

```bash
echo '{"speed": 1.0, "accel": 0.1, "lane_change": 0.5, "collision": 20, "rollover": 20, "road_edge": 20, "time_headway_s": 1.5}' > my_w.json
```
```bash
python -m learning.minimax_pipeline --mode evaluate --experiment cutin_side1_screened_v1 --w-json my_w.json --label my_w
```

---

## Step 9 — Report

```bash
python -m learning.minimax_pipeline --mode report --experiment cutin_side1_screened_v1
```

This prints the summary and how many candidates fell into each status (admitted, uncertain, rejected). It also writes `plots/convergence.png`, which shows the worst-case estimate and w over the iterations.

---

## Step 10 — Compare methods (for the paper)

Run the same config five times, each with its own `experiment_id`:

```bash
python -m learning.minimax_pipeline --mode optimize --config configs/minimax/cutin_side1_screened.json --set method='"baseline"' --set experiment_id='"c1_baseline"'
```
```bash
python -m learning.minimax_pipeline --mode optimize --config configs/minimax/cutin_side1_screened.json --set method='"random_search"' --set experiment_id='"c1_random"'
```
```bash
python -m learning.minimax_pipeline --mode optimize --config configs/minimax/cutin_side1_screened.json --set screening.policy='"none"' --set experiment_id='"c1_unscreened"'
```
```bash
python -m learning.minimax_pipeline --mode optimize --config configs/minimax/cutin_side1_screened.json --set experiment_id='"c1_screened"'
```
```bash
python -m learning.minimax_pipeline --mode optimize --config configs/minimax/cutin_side1_screened.json --set oracle_refinement.enabled=true --set experiment_id='"c1_refined"'
```

The last variant trains new PPO policies during the search, which takes hours. Refinement is limited by `oracle_refinement.max_refinements`, which defaults to 2.

Evaluate each run on the **same** held-out definition (repeat for each experiment ID):

```bash
python -m learning.minimax_pipeline --mode evaluate --experiment c1_baseline --label heldout --heldout-config configs/minimax/cutin_side1_screened.json
```

Then tabulate:

```bash
python -m learning.minimax_pipeline --mode report --compare artifacts/minimax/experiments/c1_*/evaluations/heldout.json
```

`compare` prints a warning if the held-out pools or seeds differ between runs.

---

## Where things end up

```
artifacts/feasibility/
  datasets/<family>/side_pos1|side_neg1/   scenarios.jsonl, rollouts.jsonl       (step 3)
  policies/<family>/<side>/<scenario_id>/ model.zip, model_final.zip, model.meta.json
  evaluations/<family>/<side>/<scenario_id>.result.json                          (worker output)
  surrogates/<family>_side<±1>.joblib                                            (step 4)
  heatmaps/<family>_side<±1>/                                                    (step 5)
artifacts/minimax/experiments/<experiment_id>/
  config.json, manifest.json             what was run, git commit, surrogate hashes
  iterations.jsonl                       one row per exchange iteration
  adversarial_candidates.jsonl           every θ the adversary screened, with its status
  best_weights.json, summary.json        result of step 6
  evaluations/<label>.json               step 8
  plots/convergence.png                  step 9
  cache/                                 one file per simulated episode (safe to keep; makes resume cheap)
```

## Common problems

| Symptom | Cause / fix |
|---|---|
| `experiment … already exists -- use --mode resume` | Change `experiment_id`, resume, or delete the folder. |
| Resume refuses: config hash mismatch | You changed a setting that affects results. Start a new experiment. |
| Assertion about a missing surrogate, or family/side mismatch | Run step 4 for that stratum, or fix the `surrogate` path in the config. |
| Almost every candidate is `UNCERTAIN` | The surrogate has too little data. Add more PPO data (step 3), or for exploration only, use `--set screening.policy='"point"'`. |
| Risk cost is mostly slippage | This is expected with the current vehicle parameters. See [description.md §6.1](description.md). Lower it with `--set cost.beta_slippage=…`. |
| Tests fail with `ModuleNotFoundError: yaml` | Prefix the test command with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. |

---

# Part B — Constrained minimax (PARALO MPC)

This pipeline finds one PARALO-MPC weight vector w that minimizes the worst-case **progress shortfall** over PPO-admitted scenarios. It is subject to explicit **physical constraints**: event rates and peak accelerations, each with its own threshold. No evaluation weights are involved.

Do steps 0 and 3–5 of Part A first. You need a fitted and validated surrogate for every stratum you screen.

Every evaluation uses the corrected physics ([description.md §15](description.md)):

- `"event_model": "physical_v2"`: rollover means realized |a_y| ≥ SSF·g.
- Slippage means a tire reaches its friction limit sideways, or the combined braking-plus-cornering demand exceeds the friction circle.
- The PARALO MPC passes PARALO the steering angle that matches its *predicted* yaw rate (δ_eq).
- The MPC knows the true road μ (`friction_mode: "oracle"`). Other friction modes exist for later sensitivity studies.
- `"objective": "completion_time"` is an optional alternative objective. Use it if progress shortfall sits at 0 for most scenarios and cannot tell weight vectors apart.
- `"oracle_refinement": {"enabled": true, ...}` sends uncertain but informative candidates to PPO for direct evidence.

**B0 (optional): check the risk models and controller capability first.**

```bash
python -m learning.risk_validation --out artifacts/physics_validation --workers 12
```

```bash
python -m learning.capability_suite --out artifacts/physics_validation
```

## B1 — Smoke run (no surrogate, ~25 s)

```bash
python -m learning.constrained_pipeline --mode optimize --config configs/constrained/smoke_unscreened.json
```

**Check:** the run prints a summary with a `status`. With this tiny budget, `no_feasible_found` is a normal result. Delete `artifacts/constrained/experiments/c_smoke` before running it again.

## B2 — Look at the constraints before a real run

The constraints are listed in the config under `"constraints"`, or taken from the defaults in `learning/physical_eval.py` (`DEFAULT_CONSTRAINTS`). Every shipped threshold is **provisional**. `"threshold": null` means *monitored only*: the value is reported but never enforced.

| Constraint | Default |
|---|---|
| collision, rollover, road-departure, slippage rate | 0 |
| peak lateral acceleration | 4 m/s² |
| lane-departure rate, peak jerk, peak acceleration | monitored only |

The reasons are in [description.md §12](description.md).

To use your own constraints, put a `"constraints"` list in the config. The list **replaces** the defaults rather than merging with them. Each entry looks like this:

```json
{"name": "collision", "metric": "collision_rate", "threshold": 0.0, "units": "1/episode", "statistic": "point", "status": "provisional"}
```

## B3 — Optimize

```bash
python -m learning.constrained_pipeline --mode optimize --config configs/constrained/cutin_side1_screened.json --set n_workers=24
```

The run ends with one of these statuses:

| Status | Meaning |
|---|---|
| `converged_feasible` | A w satisfied every enforced constraint on the working set, and the adversary found nothing worse |
| `no_feasible_found` | No feasible w was found. `summary.json` lists the violated constraints, their worst scenarios, and `scenarios_never_satisfied`. **The constraints are never relaxed automatically.** |
| `budget_exhausted` | No conclusion yet. Resume with a larger budget. |

## B4 — Resume

```bash
python -m learning.constrained_pipeline --mode resume --experiment c_cutin_side1_v1 --set exchange.max_iterations=30
```

## B5 — Compare with the original MPC on held-out scenarios

```bash
python -m learning.constrained_pipeline --mode compare --experiment c_cutin_side1_v1 --label heldout
```

This evaluates four controllers with the same physical metrics, on the same new admitted scenarios and new seeds:

- the original MPC under the **legacy** event definitions. This row is historical, has `primary_comparison: false`, and is not a safety result;
- the original MPC under the corrected definitions;
- the PARALO MPC at w₀;
- the PARALO MPC at the best w found.

It prints completion and event rates, worst and mean progress shortfall, peaks, and milliseconds per replan. The results are written to `evaluations/heldout.json`.

## B6 — Report

```bash
python -m learning.constrained_pipeline --mode report --experiment c_cutin_side1_v1
```

This prints a per-iteration table and writes `plots/convergence.png`, which shows the worst shortfall and the worst value of each constraint over the iterations.

Results go to `artifacts/constrained/experiments/<experiment_id>/`, with the same layout as Part A.

---

# Part C — Single scenarios and inspection

These commands work on one scenario or one policy. Paths below use a feasibility policy; a model from C1 works the same way with its own `--out` path.

## C1 — Train one scenario

This trains one dedicated PPO policy for one hand-specified θ. Any field you leave out keeps the family's default:

```bash
python -m learning.feasibility_train_one --scenario-family cutin --timesteps 2000000 --n-envs 4 --blocker-side 1 --mode robust --scenario-seed 0 --ppo-seed 0 --s-cutin-m 200 --cutter-gap-m 25 --episodes 100 --out learning/ppo_cutin_theta0
```
```bash
python -m learning.feasibility_train_one --scenario-family sandwich --timesteps 2000000 --n-envs 4 --blocker-side -1 --mode exact --scenario-seed 0 --ppo-seed 0 --s-stop-m 220 --front-gap-m 25 --episodes 100 --out learning/ppo_sandwich_thetaA
```

It prints the resolved scenario, trains, evaluates on `--episodes` held-out seeds, and saves `<out>.zip` and `<out>.meta.json`. The rollover event defaults to `physical_v2`; `--rollover-event` changes it. The PPO discount defaults to 0.999; `--gamma` changes it here and in `feasibility_pipeline --mode run` and `learning.train`.

- `--mode exact` freezes every perturbation at 0 ("can PPO solve this precise scenario?").
- `--mode robust` applies seeded perturbations across resets ("can PPO solve a local distribution around it?").
- `--stochastic` samples actions during evaluation instead of using the deterministic mean.
- `--out` refuses to overwrite a *different* scenario at that path unless you pass `--force`.

Field ranges are `THETA_BOUNDS_CUTIN` in `learning/feasibility_cutin.py` and `THETA_BOUNDS_SANDWICH` in `learning/feasibility_sandwich.py`. Check the source for current values.

## C2 — Evaluate a saved policy

Every C1 run already evaluates after training. To re-evaluate a policy without retraining:

```bash
python -m learning.eval_batch --model artifacts/feasibility/policies/cutin/side_pos1/<id>/model --episodes 100
```

Or evaluate it on a scenario you set by hand. Copy `cfg` from `<out>.meta.json`'s `"theta"` field, and keep `device='cpu'`:

```bash
python -c "from stable_baselines3 import PPO; from learning.env import EgoTrafficEnv; from learning.feasibility_cutin import CutinConfig, CutinRuntime; from learning.eval_batch import run_episodes, summarize_seed; model = PPO.load('learning/ppo_cutin_theta0', device='cpu'); cfg = CutinConfig(seed=0, blocker_side=1, mode='robust', s_cutin_m=200.0, cutter_gap_m=25.0); env = EgoTrafficEnv(arena=CutinRuntime(cfg), mode='distribution', worker_rank=0, rollover_event='physical_v2'); print(summarize_seed(run_episodes(model, env, episodes=200, deterministic=True), mode='distribution', success_threshold=1.0))"
```

## C3 — Watch an episode

The model's own canonical realization (always the same one). It reads the meta file, so it needs no other flags:

```bash
python -m learning.eval --model learning/ppo_cutin_theta0
```

`--stochastic` samples actions. `--seed` does **not** select an evaluated episode, because this animator always runs in `mode="fixed"`.

To replay one specific evaluated episode, use its index from `rollouts.jsonl`:

```bash
python -m learning.animate_rollout --model artifacts/feasibility/policies/cutin/side_pos1/<scenario_id>/model --episode-index 37 --save out.mp4
```

`--save` writes a file, which is the practical choice on the headless training machine. Omit it to open a window.

To list a scenario's failed episodes:

```bash
python3 -c "import json; sid = 'PUT_A_SCENARIO_ID_HERE'; [print(r['episode'], r['failure_reason']) for r in map(json.loads, open('artifacts/feasibility/datasets/cutin/side_pos1/rollouts.jsonl')) if r['scenario_id'] == sid and r['y'] == 0]"
```

## C4 — Archive a dataset

Do this before resampling after a bounds change. Replace `<label>` with a name:

```bash
mkdir -p artifacts/feasibility/_archive
```
```bash
mv artifacts/feasibility/datasets/cutin/side_pos1 artifacts/feasibility/_archive/datasets_side_pos1_<label>
```
```bash
mv artifacts/feasibility/policies/cutin/side_pos1 artifacts/feasibility/_archive/policies_side_pos1_<label>
```
```bash
mv artifacts/feasibility/evaluations/cutin/side_pos1 artifacts/feasibility/_archive/evaluations_side_pos1_<label>
```

## C5 — Not built yet

Runner C, which would compare one θ's fresh empirical success rate with the surrogate's prediction, does not exist.

---

# Part D — PPO evaluation and reporting

This takes PPO from sampled scenarios to figures, tables and a report (description §23). Nothing here starts on its own. `init` trains nothing, and `train` is the campaign.

## D1 — Write a config

`experiment.json` takes any `ExperimentConfig` field. Only `name` is required:

```json
{"name": "cutin_pilot", "strata": [["cutin", 1], ["cutin", -1]], "design": "lhs", "n_thetas": 10, "ppo_seeds": [0, 1, 2], "timesteps": 2000000, "n_envs": 4, "test_episodes": 100}
```

For a true two-dimensional slice, use the grid design. Fields you leave out keep the family default:

```json
{"name": "cutin_slice", "strata": [["cutin", 1]], "design": "grid", "grid_levels": {"road_mu": [0.4, 0.6, 0.8, 1.0], "cutter_gap_m": [15, 30, 45, 60]}, "grid_fixed": {"front_gap_m": 50}, "ppo_seeds": [0]}
```

## D2 — Create and train

```bash
python -m learning.ppo_experiment init --config experiment.json
```
```bash
python -m learning.ppo_experiment train --experiment results/ppo_feasibility/cutin_pilot --concurrency 4
```

`init` refuses an existing name. `train` skips finished jobs, so rerun it after an interruption. Add `--data-class smoke` to `init` for a test run: its outputs are marked as not experimental.

## D3 — Evaluate

```bash
python -m learning.evaluate_ppo --experiment results/ppo_feasibility/cutin_pilot
```

This evaluates every policy in the config's `eval_modes`. To evaluate one saved policy instead, add `--checkpoint path/to/model.zip`; its `model.meta.json` must sit beside it. The experiment directory is created if needed. An existing record is never replaced without `--force`.

## D4 — Figures, report, animations

None of these reruns PPO:

```bash
python -m learning.plot_ppo_results --experiment results/ppo_feasibility/cutin_pilot
```
```bash
python -m learning.report_ppo_results --experiment results/ppo_feasibility/cutin_pilot --html
```
```bash
python -m learning.animate_ppo_results --experiment results/ppo_feasibility/cutin_pilot
```

Useful plot options:

- `--width single` for IEEE single-column size.
- `--rank lower`, or `difficulty`, or `family`, for the Wilson plot.
- `--pair cutin:road_mu,cutter_gap_m --fix front_gap_m=50` for a specific slice.

To replay one rollout, take its scenario and episode from the evaluation record:

```bash
python -m learning.animate_ppo_results --experiment results/ppo_feasibility/cutin_pilot --scenario <id> --eval-mode robust --episode 12
```

Only rollouts with a saved trajectory can be replayed.

## D5 — Where things end up

```text
results/ppo_feasibility/<name>/
├── manifest.json          identity, provenance, definitions
├── configs/               experiment.json, scenarios.jsonl
├── training/<family>/side_<pos1|neg1>/<scenario_id>/seed_<n>/    model.zip, model.meta.json, metrics.jsonl
├── evaluations/<same path>/<mode>_<protocol>.json
├── trajectories/<same path>/<mode>/ep_<i>.npz
├── figures/               PDF, PNG, index.json
├── animations/
└── reports/               report.md, report.html, tables/
```

`results/` is git-ignored. To see a labelled demonstration without any training:

```bash
python -m tests.ppo_fixtures results/ppo_feasibility
```
