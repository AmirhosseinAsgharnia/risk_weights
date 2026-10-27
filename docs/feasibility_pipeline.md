# Operating the feasibility-surrogate pipeline

**Status**: the simulation layer (two scenario families + `EgoTrafficEnv`
integration), **Runner A** (single-scenario train/evaluate,
`learning/feasibility_train_one.py`), and **Runner B** (the database +
surrogate pipeline, `learning/feasibility_pipeline.py`) are all built and
tested (87/87 tests passing). Runner B's `pilot` mode and a full `run`
campaign have both now been executed for real on the target 32-core / RTX
5070 machine (`abl-drivsim-iii`) — see §3's measured numbers below. A
result-interpretation tool (heatmaps of feasible/non-feasible regions, §5)
is also built. Runner C (numerically compare one theta's empirical vs.
surrogate probability) is the one piece still not built.

| Case | Command exists? |
|---|---|
| 1. Train a single scenario | Yes |
| 2. Evaluate a trained model | Yes (two ways) |
| 3. Train the approximator (surrogate) | Yes — Runner B (§3 below) |
| 4. Watch an episode animate | Yes (two ways) |
| 5. Visualize feasible/non-feasible regions (heatmaps) | Yes (§5 below) |
| 6. Compare one theta: empirical vs. surrogate | Not built yet (Runner C) |

---

## Case 1: Train a single scenario

One dedicated PPO policy `pi_theta` for one manually-specified nominal
scenario. Family, blocker side, mode, and every theta field are CLI flags;
anything you don't pass keeps that family's default.

```bash
python -m learning.feasibility_train_one --scenario-family cutin \
    --timesteps 2000000 --n-envs 4 --blocker-side 1 --mode robust \
    --scenario-seed 0 --ppo-seed 0 \
    --s-cutin-m 200 --cutter-gap-m 25 \
    --episodes 100 --out learning/ppo_cutin_theta0
```

```bash
python -m learning.feasibility_train_one --scenario-family sandwich \
    --timesteps 2000000 --n-envs 4 --blocker-side -1 --mode exact \
    --scenario-seed 0 --ppo-seed 0 \
    --s-stop-m 220 --front-gap-m 25 \
    --episodes 100 --out learning/ppo_sandwich_thetaA
```

What happens: prints the fully resolved scenario -> trains with PPO
(`SubprocVecEnv`, same pattern as `learning/train.py`) -> evaluates on
`--episodes` held-out seeds (deterministic by default; `--stochastic` for the
diagnostic sampling mode) -> saves `<out>.zip` + `<out>.meta.json` (the
resolved scenario + training args + evaluation summary — everything Cases 2
and 4 need to find this scenario again automatically).

`--mode exact` freezes every perturbation at 0 ("can PPO solve this
*precise* scenario?"); `--mode robust` applies each family's small seeded
nuisance perturbations across resets ("can PPO solve a local distribution
around it?"). `--out` refuses to silently overwrite a *different* scenario
already saved at that path — pass `--force` to do it anyway.

Full field list and valid ranges: `THETA_BOUNDS_CUTIN` in
`learning/feasibility_cutin.py`, `THETA_BOUNDS_SANDWICH` in
`learning/feasibility_sandwich.py` — these are a moving target as the
project's understanding of "physically sensible" ranges evolves (e.g.
`front_gap_m`/`cutter_relative_speed_mps` were both widened this session),
so always check the source rather than trusting an old copy of this table.
**If you change these bounds, treat any dataset sampled under the old
bounds as a separate experiment** — `learning.feasibility_pipeline --mode
run` will happily mix old- and new-bound scenarios into the same
`scenarios.jsonl` (nothing tags a record with which bounds generated it),
which skews coverage without erroring. Archiving the old dataset directory
aside before resampling (see §3) keeps this honest.

Device: every job here runs on CPU by default (`--device cpu`) — see the
GPU note in §3, which also covers a real bug (fixed this session) where
`PPO.load()` ignored this setting.

---

## Case 2: Evaluate a trained model

**2a — quick, built into Case 1.** Every Case 1 run already evaluates on
`--episodes N` held-out seeds right after training and prints the summary —
if that's enough, there's nothing further to run.

**2b — re-evaluate an existing model later, without retraining:**

```bash
python -c "
from stable_baselines3 import PPO
from learning.env import EgoTrafficEnv
from learning.feasibility_cutin import CutinConfig, CutinRuntime
from learning.eval_batch import run_episodes, summarize_seed

model = PPO.load('learning/ppo_cutin_theta0', device='cpu')
cfg = CutinConfig(seed=0, blocker_side=1, mode='robust', s_cutin_m=200.0, cutter_gap_m=25.0)
env = EgoTrafficEnv(arena=CutinRuntime(cfg), mode='distribution', worker_rank=0)

results = run_episodes(model, env, episodes=200, deterministic=True)
print(summarize_seed(results, mode='distribution', success_threshold=1.0))
"
```

`cfg` must match the theta you actually trained with — copy it straight out
of `<out>.meta.json`'s `"theta"` field rather than retyping it if you don't
remember the values. This is the same `run_episodes`/`summarize_seed`
(from `learning/eval_batch.py`) that both Case 1 and Runner B (§3) already
use internally for their own evaluation.

`device='cpu'` on the `PPO.load(...)` call is deliberate, not optional —
see §3's GPU note.

---

## Case 3: Train the approximator (surrogate) — Runner B

Three steps, each a mode of the same command. **Run `pilot` first, on the
real machine** — it benchmarks concurrency levels and recommends one,
rather than you guessing (see the GPU note below).

```bash
# 3a. Benchmark concurrency on this machine (writes nothing to the real dataset).
python -m learning.feasibility_pipeline --mode pilot --scenario-family cutin \
    --blocker-side 1 --pilot-concurrencies 1,2,4,8,16 --pilot-envs-per-job 4

# 3b. Sample a batch of theta via Latin Hypercube Sampling, train+evaluate each
#     one, accumulate into artifacts/feasibility/datasets/cutin/side_pos1/.
#     Safely resumable -- re-running this exact command skips every scenario
#     that already has a result (see "Resuming" below).
python -m learning.feasibility_pipeline --mode run --scenario-family cutin \
    --blocker-side 1 --n-thetas 100 --sampling-seed 1 \
    --concurrency 8 --n-envs 4 --timesteps 2000000 --episodes 100

# 3c. Fit the surrogate on whatever's in the dataset so far.
python -m learning.feasibility_pipeline --mode fit-only --scenario-family cutin \
    --blocker-side 1 --surrogate-out artifacts/feasibility/surrogates/cutin_side1.joblib
```

**Run both blocker sides separately** (`--blocker-side 1` and `--blocker-side
-1`) — they get independent Latin Hypercube designs, independent datasets,
and independent surrogates, never combined, per the original design
decision that the two sides are separate experimental strata.

**Why `pilot` first, and why the GPU isn't part of it**: PPO with a small
`MlpPolicy` over this plain NumPy/Python physics environment is CPU-bound
end to end — putting the network on a GPU adds transfer overhead for no
benefit (the same warning SB3 prints if you pass `--device cuda`). Every job
in this pipeline runs on CPU (`--device cpu`, the default) with PyTorch/BLAS
restricted to 1 thread per process (`--torch-threads 1`, also the default)
-- both matter, and were bugs caught by actually running this on real
hardware: without the CPU default, concurrent jobs fight over one shared
GPU; without the thread restriction, every process's own tensor math
independently tries to claim every core, so concurrent jobs starve each
other even on CPU. The real lever is how many *cores* go to one job's
`--n-envs` vs. how many *scenarios* train **concurrently** -- `pilot`
measures scenarios/hour at a few concurrency levels on your actual hardware
and tells you which to use for step 3b's `--concurrency`.

**A third device bug, found and fixed after a real 24-scenario campaign
still showed GPU warnings despite the two fixes above**: `--device cpu`
only ever flowed into the *training-phase* `PPO(...)` construction.
`PPO.load(...)` has its own, independent `device` parameter that defaults
to `"auto"` — so reloading the best early-stopping checkpoint (which is
what's actually used for final evaluation, and what gets saved as the
deliverable `<out>.zip`, for nearly every scenario) silently ignored
`--device cpu` and grabbed CUDA whenever available. Fixed by passing
`device=` explicitly at every `PPO.load(...)` call site in the codebase
(`feasibility_train_one.py`, `eval.py`, `eval_batch.py`,
`animate_rollout.py`, `tests/traffic_test.py`). If you're on an older
checkout, `git pull` before running a new campaign.

**Measured on `abl-drivsim-iii` (32 cores, RTX 5070), `--pilot-envs-per-job 4`:**

| Concurrency | Scenarios/hour | Environment workers (concurrency x 4) |
|---:|---:|---:|
| 1  |  268.6 |  4 |
| 2  |  413.0 |  8 |
| 4  |  632.5 | 16 |
| 8  |  932.3 | 32 (= physical core count) |
| 16 | 1079.7 | 64 (2x oversubscribed) |

`pilot` itself picks the raw-highest number (16 here), but that's only a
53-second burst -- at 64 environment-worker processes plus 16 main
processes (~80 total) on 32 cores, a real multi-hour campaign might not
hold up as cleanly (memory pressure, disk contention when several jobs
checkpoint at once). `concurrency=8` sits at exactly the physical core
count and captured most of the gain (632 -> 932/hr was the biggest single
jump; 932 -> 1080/hr was much smaller) -- the safer choice for a long run,
which is what step 3b above uses. Use `--concurrency 16` instead if you'd
rather chase the extra ~16% and keep an eye on memory while it runs.

**Resuming**: every worker durably writes its own result file under
`artifacts/feasibility/evaluations/<family>/side_*/<scenario_id>.result.json`
*before* the coordinator ever touches the shared dataset. Re-running the
same `--mode run` command later — after a crash, a `Ctrl-C`, or just to
pick up where you left off — reconciles any already-finished-but-not-yet-
recorded scenarios for free (no retraining) and only launches new work for
what's actually missing.

**`--n-thetas` is additive, not a target total.** Each `--mode run` call
draws a *fresh* LHS design of that size and appends the results on top of
whatever's already in `scenarios.jsonl`/`rollouts.jsonl` — it does not "top
up" an existing dataset to that count, and it doesn't deduplicate against a
differently-seeded design. Running `--n-thetas 100` against a dataset that
already has 24 in it ends with ~124, not 100. If you've changed
`THETA_BOUNDS_CUTIN`/`THETA_BOUNDS_SANDWICH` since the existing data was
sampled, archive it first rather than mixing bound-inconsistent data
silently:

```bash
mkdir -p artifacts/feasibility/_archive
mv artifacts/feasibility/datasets/cutin/side_pos1    artifacts/feasibility/_archive/datasets_side_pos1_<label>
mv artifacts/feasibility/policies/cutin/side_pos1     artifacts/feasibility/_archive/policies_side_pos1_<label>
mv artifacts/feasibility/evaluations/cutin/side_pos1  artifacts/feasibility/_archive/evaluations_side_pos1_<label>
```

**Preliminary datasets**: below 50 distinct scenarios, `fit-only`'s printed
report and the saved artifact are both explicitly marked `PRELIMINARY` —
worth having as a real pipeline-correctness check, never a claim of accuracy.
An all-one-outcome dataset (every rollout succeeds, or every one fails) is
handled too — the surrogate falls back to a constant predictor rather than
erroring, since a classifier can't discriminate what it's never seen vary.

---

## Case 4: Watch an episode animate

**4a — the model's own canonical realization** (episode-index 0 of that
scenario, always the same realization every time you run this):

```bash
python -m learning.eval --model learning/ppo_cutin_theta0
```

Opens a matplotlib window: the road, all six agents (colored by behaviour —
"moderate"/steelblue for every feasibility-family actor today), ego in
black. Prints the outcome (`success`/`collision`/`rollover`/`off_road`/
`timeout`) once the episode ends. Reads `<model>.meta.json` to reconstruct
the *exact* scenario automatically — no extra flags needed, works for any
model saved by Case 1.

```bash
python -m learning.eval --model learning/ppo_cutin_theta0 --stochastic
```

`--stochastic` samples actions instead of using the policy's deterministic
mean. **Note**: `learning.eval`'s `--seed` flag exists but does **not**
select which of the scenario's evaluated episodes gets replayed — the
animator always reconstructs the environment in `mode="fixed"`, which is
hardcoded to episode-index 0 regardless of `--seed`. Use 4b instead if you
specifically want to watch one of the 100 evaluated realizations (e.g. one
that's known to have collided).

**4b — a specific evaluated episode** (e.g. one you found via
`rollouts.jsonl` — see §5's "find failing episodes" snippet):

```bash
python -m learning.animate_rollout --model artifacts/feasibility/policies/cutin/side_pos1/<scenario_id>/model \
    --episode-index 37 --save out.mp4
```

`--episode-index N` (0-based) reconstructs the environment in
`mode="distribution"` and replays `reset()` N+1 times to land on exactly
the realization episode N saw during the real evaluation (`run_episodes`'s
own episode-counter bookkeeping) — the realization actually recorded in
`rollouts.jsonl` for that episode, not an approximation.

`--save PATH.mp4` writes the animation to a file instead of opening a
window — the practical choice on `abl-drivsim-iii`, which is headless over
SSH; `scp` the file back afterward, or use `ssh -X` and drop `--save` to
watch it live. Omit `--save` to open an (auto-maximized, where the backend
supports it) interactive window instead.

---

## Case 5: Visualize feasible/non-feasible regions (heatmaps)

```bash
# Works right now with just the accumulated dataset -- no fitted surrogate needed.
python -m learning.feasibility_interpret --scenario-family cutin --blocker-side 1 \
    --out-dir artifacts/feasibility/heatmaps/cutin_side1

# Once a surrogate is fit (Case 3, step 3c), add the model-based panel:
python -m learning.feasibility_interpret --scenario-family cutin --blocker-side 1 \
    --surrogate artifacts/feasibility/surrogates/cutin_side1.joblib \
    --out-dir artifacts/feasibility/heatmaps/cutin_side1

# Restrict to specific dimensions instead of every pair (default: all of them):
python -m learning.feasibility_interpret --scenario-family cutin --blocker-side 1 \
    --dims front_gap_m,cutter_relative_speed_mps,s_cutin_m \
    --out-dir artifacts/feasibility/heatmaps/cutin_side1
```

Writes one PNG per pair of theta dimensions (28 for cutin's 8 fields, 45 for
sandwich's 10, unless `--dims` restricts it) to `--out-dir`. Each PNG has
one or two panels:

- **Empirical** (always present): every sampled scenario's `success_rate`,
  projected onto that 2D pair and smoothed via linear interpolation between
  points (red = infeasible, green = feasible). This is a **marginal
  projection** — a point's color reflects its true full-dimensional theta,
  including whatever the other 6-8 fields happened to be, not a controlled
  slice — and the fill is only shown inside the convex hull of actually-
  sampled points (gray elsewhere, never fabricated).
- **Surrogate** (only with `--surrogate`): the fitted model's `p_hat` on a
  dense grid, with every *other* theta dimension held fixed at the
  dataset's own median value — an actual controlled 2D slice, and (unlike
  the empirical panel) defined everywhere in-bounds, not just inside the
  sampled convex hull.

**Finding a specific failing episode to animate** (for Case 4b), e.g. from
a scenario_id you spotted as low-`success_rate` in a heatmap:

```bash
python3 -c "
import json
sid = 'PUT_A_SCENARIO_ID_HERE'
for l in open('artifacts/feasibility/datasets/cutin/side_pos1/rollouts.jsonl'):
    r = json.loads(l)
    if r['scenario_id'] == sid and r['y'] == 0:
        print(r['episode'], r['failure_reason'])
"
```

---

## Case 6: Compare one theta's empirical vs. surrogate probability

Not built yet (Runner C) — the plan is a small CLI that takes one manually
specified theta, trains+evaluates it fresh (reusing `feasibility_train_one.
train_and_evaluate`) for a genuine empirical success rate, and compares it
against `SurrogateArtifact.predict(theta)`'s `p_hat` + epistemic interval,
to sanity-check the surrogate against ground truth at a point it may or may
not have seen during training.
