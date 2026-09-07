# Operating the feasibility-surrogate pipeline

**Status**: the simulation layer (two scenario families + `EgoTrafficEnv`
integration) and **Runner A** (single-scenario train/evaluate) are built and
tested (59/59 tests passing). The dataset/surrogate/sampling layer and
Runners B/C are **not built yet**. Every case below has a real, tested
command except Case 3, which is marked as not yet available rather than
faked.

| Case | Command exists? |
|---|---|
| 1. Train a single scenario | Yes |
| 2. Evaluate a trained model | Yes (two ways) |
| 3. Train the approximator (surrogate) | **No — not built yet** |
| 4. Watch an episode animate | Yes |

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
remember the values. This is also the exact function (`run_episodes`/
`summarize_seed`, from `learning/eval_batch.py`) Runner B will call
internally once it exists.

---

## Case 3: Train the approximator (surrogate) — not built yet

There is no command for this today. It needs, in order:

1. **Runner B** (not built) — sample a batch of theta values (Sobol/Latin
   hypercube), run Case 1's train+evaluate for each one, append every result
   to a durable dataset.
2. Fit a small classifier (logistic/random-forest/histogram-gradient-boosting)
   on the accumulated *rollout-level* binary outcomes, grouped by
   `scenario_id` so no scenario's data leaks across train/validation.

You can approximate step 1 by hand today — run Case 1 several times with
different `--s-cutin-m`/`--cutter-gap-m`/etc. values and different `--out`
paths, and keep your own table of (theta, success_rate) from each run's
printed summary — but there's no automated dataset file or fitting command
yet. Say the word and I'll build Runner B next.

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
