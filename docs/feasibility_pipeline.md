# Operating the feasibility-surrogate pipeline

**Status**: the simulation layer (two scenario families + `EgoTrafficEnv`
integration), **Runner A** (single-scenario train/evaluate,
`learning/feasibility_train_one.py`), and **Runner B** (the cutin-database +
surrogate pipeline, `learning/feasibility_pipeline.py`) are all built and
tested (87/87 tests passing). Runner B is validated end to end (pilot,
resumable `run`, `fit-only`) on tiny synthetic budgets — it has not yet been
run for a real campaign on the target 32-core machine. Runner C (compare one
theta's empirical vs. surrogate probability) is the one piece still not
built.

| Case | Command exists? |
|---|---|
| 1. Train a single scenario | Yes |
| 2. Evaluate a trained model | Yes (two ways) |
| 3. Train the approximator (surrogate) | Yes — Runner B (§3 below) |
| 4. Watch an episode animate | Yes |
| 5. Compare one theta: empirical vs. surrogate | Not built yet (Runner C) |

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
`learning/feasibility_sandwich.py`.

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

model = PPO.load('learning/ppo_cutin_theta0')
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
    --blocker-side 1 --n-thetas 24 --sampling-seed 0 \
    --concurrency 6 --n-envs 4 --timesteps 2000000 --episodes 100

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
in this pipeline runs on CPU. The real lever is how many *cores* go to one
job's `--n-envs` vs. how many *scenarios* train **concurrently** — `pilot`
measures scenarios/hour at a few concurrency levels on your actual hardware
and tells you which to use for step 3b's `--concurrency`, rather than
assuming more concurrency (or more envs per job) is automatically better.

**Resuming**: every worker durably writes its own result file under
`artifacts/feasibility/evaluations/<family>/side_*/<scenario_id>.result.json`
*before* the coordinator ever touches the shared dataset. Re-running the
same `--mode run` command later — after a crash, a `Ctrl-C`, or just to
pick up where you left off — reconciles any already-finished-but-not-yet-
recorded scenarios for free (no retraining) and only launches new work for
what's actually missing.

**Preliminary datasets**: below 50 distinct scenarios, `fit-only`'s printed
report and the saved artifact are both explicitly marked `PRELIMINARY` —
worth having a real pipeline-correctness check, never a claim of accuracy.
An all-one-outcome dataset (every rollout succeeds, or every one fails) is
handled too — the surrogate falls back to a constant predictor rather than
erroring, since a classifier can't discriminate what it's never seen vary.

---

## Case 4: Watch an episode animate

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
python -m learning.eval --model learning/ppo_cutin_theta0 --seed 7 --stochastic
```

`--seed N` replays a specific realization seed; `--stochastic` samples
actions instead of using the policy's deterministic mean.

*(This command didn't actually work for feasibility-arena models until this
session — `learning/eval.py` only knew how to reconstruct the older
15-vehicle `ScenarioConfig` models before now; fixed and verified.)*
