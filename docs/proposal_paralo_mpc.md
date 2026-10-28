# Proposal: PARALO risks inside the MPC, and the "α problem"

*Brainstorming brief. It is meant to stand on its own, so a reader without access to the repository can follow it.*

> **Outcome (2026-10-27):** resolution 2 (ε-constraint) was adopted and implemented. See [constrained_design.md](constrained_design.md).

---

## 1. Context

### 1.1 The project

The goal is to tune **one shared parameter vector w** for a model predictive controller (MPC) that drives an automated vehicle on a highway. The tuning is robust (minimax):

```
w* = argmin_w  max_{θ ∈ Θ_f}  J_eval(w, θ)
```

- **θ** is a traffic scenario from one of two families:
  - **cut-in** (8 parameters): a car cuts in front of the ego vehicle;
  - **sandwich** (10 parameters): the ego is boxed in.

  Each family is run separately for blocker side ±1, which gives four strata that are never merged.
- **Θ_f** is the set of *admissible* scenarios, meaning scenarios a competent driver can solve. It is estimated by a surrogate classifier trained on PPO results. For many θ, a PPO agent was trained and evaluated, giving success counts k/n. The surrogate predicts P(PPO solves θ) with a bootstrap interval. A θ is admitted if the interval's lower bound is ≥ τ = 0.8, or if PPO demonstrably solved it (Wilson lower bound ≥ 0.8). The point is that the adversary should not win by choosing physically impossible scenarios.
- **J_eval(w, θ)** is the closed-loop cost of the MPC with parameters w driving scenario θ in the simulator, averaged over fixed random seeds.
- **The search** is an exchange (cutting-set) algorithm:
  1. **Master:** w_k = argmin over w of the worst cost over the scenario set S. It uses Latin-hypercube starts plus bounded Nelder–Mead.
  2. **Adversary:** search for the admitted θ that costs the most at w_k. It samples, screens with the surrogate, simulates, and refines locally.
  3. **Update:** add that θ to S if it beats the current worst case, then go back to step 1.

  The search stops when no scenario beats the current worst case, or when the budget runs out.

### 1.2 Simulator facts

| Item | Value |
|---|---|
| Vehicle model | Frenet-frame dynamic bicycle, RK4 integration |
| Steps | 0.05 s per sim step; episode up to 35 s; route 400 m |
| Ego limits | accel [−8, 4] m/s², steering ±0.5 rad, steering rate 1 rad/s, jerk 40 m/s³ |
| Wheelbase | L = 2.8 m |
| Traffic | 6 surrounding cars with scripted behaviour |
| Episode outcomes | success, collision, rollover, off-road, timeout |

### 1.3 PARALO risk models (existing code, used unchanged)

There are five closed-form risk models. Each returns a probability and an energy-weighted score (probability × ½·m·v², summed into a cumulative risk as in PARALO eq. 5.51).

| Risk | Inputs | Notes |
|---|---|---|
| **c2c** (car-to-car) | ego and object poses, 4 m × 2 m boxes, position σ | Gaussian rectangle-overlap probability |
| **rollover** | v, δ | onset around δ ≈ 0.07 rad at 20 m/s |
| **slippage** | v, δ, a, road friction μ | Close to a step function: at 20 m/s, P goes 0 → 1 around δ ≈ 0.018 rad (cornering stiffness C_f 19 kN/rad, slip-angle limit 8.6°, friction uncertainty U_μ 0.05). The default curve needs δ ≈ 0.014 rad; the sharpest curves need ≈ 0.056 rad. |
| **lane departure** | lateral position vs the nearest lane centre, lane width 4 m | |
| **road departure** | lateral position vs road centre, paved width 12 m | |

### 1.4 What is built today (before this proposal)

**MPC.** A sampling MPC chooses among 10 accelerations × 3 target lanes. Lateral motion uses a ramped lane-change reference tracked by a far-near steering law. Its internal cost uses w (7 numbers):
- speed, accel and lane-change weights;
- a heuristic proximity term with time headway T_h;
- rollover probability;
- a road-edge proximity term.

**J_eval.** A separate fixed-coefficient cost:

```
J_ep = Σ_k (Δt/T) [α₁ E_track + α₂ E_comfort + α₃ E_control + α₄ E_risk]_k + P_terminal
E_risk = Σ_ch β_ch · R_ch     (the five PARALO probabilities, max over a 2 s prediction)
P_terminal = 100 for collision, rollover or off-road; 20 + 30·(unfinished fraction) for a timeout; 0 for success
```

Each E term is clipped to [0, 10], so the running cost is at most 40 and any crash costs more than any surviving run. All α and β are currently 1.

---

## 2. The proposed redesign

The project owner wants PARALO to be the core of the MPC's own cost, with a simpler MPC and the *same* cost structure used for both the MPC and the evaluation.

### 2.1 One per-step cost

ℓ is a function of a single state x and input u:

```
ℓ(x, u; c) = c_speed·((v − v_ref)/v_ref)²
           + c_jerk·(jerk/10)²  + c_latacc·(a_y/4)²
           + c_accel·(a/4)²     + c_steer·(δ/0.5)²
           + c_c2c·P_c2c + c_roll·P_roll + c_slip·P_slip + c_lane·P_lane + c_road·P_road
```

- The P terms are PARALO **probabilities** evaluated at that single state. The owner chose probabilities over energy scores.
- Lane departure is measured against the **nearest** lane, unmodified.

### 2.2 MPC: coasting inputs

- Every 0.1 s, the MPC chooses one **absolute** pair (a, δ) and assumes it is held constant over the whole 3 s horizon. The owner chose absolute steering, not steering relative to the road's curvature.
- The candidate set is a grid. For example, 7 accelerations × 13 steering values: current δ + {0, ±0.001, ±0.002, ±0.004, ±0.008, ±0.016, ±0.03} rad. The grid is fine near the current δ because holding a steering error Δδ for 3 s at 20 m/s shifts the car sideways by about 643·Δδ m.
- Prediction:
  - ego: kinematic bicycle in the Frenet frame, with a rate-limited ramp to the held δ;
  - others: constant velocity.
- Objective: `min_{a,δ} Σ_{n=1..N} ℓ(x̂_n, u; w)`, summed over the predicted states.
- All risks use the full horizon. The MPC has no lane variable, so lane changes emerge from replanning every 0.1 s.

### 2.3 Evaluation: the same ℓ on what actually happened

```
J_eval(w, θ) = E_ξ [ Σ_k (Δt/T) ℓ(x_k, u_k; α) + P_terminal ]      (x_k = the states the simulator actually produced)
```

These are the textbook pair: the MPC minimizes Σℓ over *predicted* states with coefficients w, and the evaluation sums the same ℓ over *actual* states with coefficients α.

The vector w becomes about 7 tuned coefficients: c_speed, c_accel, c_steer, c_c2c, c_roll, c_slip and c_lane. The remaining coefficients are fixed at α. The search starts at w₀ = α.

### 2.4 Expected behaviour and known issues

- **Lane changes.** Nearest-lane departure acts as a barrier that peaks at the lane line. The MPC changes lanes only when c2c and speed costs outweigh that barrier, so c_lane controls how readily it escapes a cut-in. Expect fewer lane changes than with the current MPC.
- **Curves.** Absolute δ works where curvature is roughly constant over the horizon. Entering and leaving curves relies on replanning.
- **Slippage.** It saturates at 1 on sharp curves at 20 m/s. Plans that brake bring it down later in the horizon, so the MPC should slow for curves, but the response switches abruptly because the probability is close to a step.
- **Computation.** All five risks are closed-form expressions (erf and Gaussian overlaps), so they can be vectorized over 91 candidates × 30 steps × 6 objects. This should take milliseconds per replan.

---

## 3. The open problem: the "α problem"

**The objection:** *"The whole project is about finding w. Now we have to choose α. We have just moved the problem."*

### 3.1 Why α cannot be tuned, or tied to w

- If the optimizer could change α, or if α had to equal w, it would lower the score by shrinking coefficients rather than by driving better. With a c2c weight near zero, near-misses become free in both the MPC and the score. A fixed α is the ruler that makes it meaningful to compare two values of w.
- w and α can legitimately differ because the MPC is myopic:
  - it sees only 3 s;
  - it assumes constant inputs and constant-velocity traffic;
  - it never sees the crash penalty.

  The w that gives the best actual driving under α is therefore generally ≠ α. The gap w* − α measures how much the planner's simplifications must be compensated.

### 3.2 Why the objection is still valid

Even so, the answer w* depends on α. Choosing α is a value judgment: how much one unit of slippage risk is worth compared with one unit of lost speed. This judgment existed before the redesign too, since J_eval always had fixed α and β. The redesign only made it visible, because the terms in J_eval and in the MPC are now identical. No algorithm removes the need for *someone* to say what "good driving" means. What can be done is to make that statement **smaller, more interpretable, or less influential** on the result.

### 3.3 Candidate resolutions to brainstorm

1. **Make J_eval mostly free of weights by using outcomes.** Grade on physical outcomes only, for example `J = 1(failure) + λ·(time-to-goal shortfall)`, and keep risk probabilities and comfort only inside the MPC. One scalar λ replaces about 10 coefficients. Risk then matters to the evaluation only through whether the car actually crashed, so w is tuned for real outcomes, not proxies.
2. **Turn trade-offs into constraints (ε-constraint).** For example: minimize worst-case time-to-goal or discomfort subject to worst-case failure rate ≤ ε, or worst-case integrated risk ≤ ε. Thresholds like "failure probability ≤ 10⁻³" are easier to justify from requirements or standards than weights are.
3. **Multi-objective minimax (a Pareto front).** Do not scalarize. Compute w for the worst-case (safety, progress, comfort) trade-offs and present the front, then pick or justify a point afterwards. This is more compute, but the value judgment is made after seeing the trade-off.
4. **Use PARALO's physical units.** PARALO scores are probability × energy (joules), so the five risks are already in one common unit. Use scores with all risk weights = 1, which removes 5 arbitrary numbers. Only one exchange rate between "risk energy" and progress/comfort remains. (The owner currently prefers probabilities, so this would need a reason.)
5. **Robustness to α.** Show that w* is stable over a plausible range of α, or make the evaluator itself adversarial: `min_w max_{θ, α ∈ A} J(w, θ; α)` over a simplex A of reasonable weightings. The answer is then robust to the designer's uncertainty about the weights, not tuned to one choice.
6. **Normalize α by reference scales and set them all to 1.** This is the current approach: each term is divided by a physically meaningful reference (v_ref, a comfort jerk of 10 m/s³, a lateral-acceleration limit of 4 m/s², the steering limit). With equal α, the only choices are the reference scales, which can be cited from literature or standards (ISO 2631, ride-comfort thresholds).
7. **Infer α from preferences or demonstrations.** Use inverse RL or preference learning from expert driving, or from the PPO policy's behaviour. This is heavier, and it brings its own assumptions.

### 3.4 Questions for the brainstorm

- Is the honest framing that α is a **specification**, not a parameter, so the contribution is "given a specification, find the robust w"? Or does the project need α to come from somewhere principled?
- Which of resolutions 1–6 gives the most defensible research claim for a paper with the least extra compute?
- If J_eval is outcome-based (option 1), does putting PARALO in the MPC still matter? Yes, as the planner's model, but then w tunes how much PARALO-predicted risk the MPC should fear in order to minimize real failures. Is that a cleaner story?
- With probabilities and a near-step slippage model, does the MPC cost give usable differences between candidate plans? Should probabilities be summed over the horizon (as proposed) or combined as `1 − Π(1 − P)`?
- Is holding absolute δ for 3 s compatible with nearest-lane departure risk, or will the MPC almost never change lanes? Is that acceptable for cut-in and sandwich scenarios, where escaping may require a lane change?
- How many coefficients should w contain? Nelder–Mead degrades above about 8 dimensions when every evaluation runs simulations.

---

## 4. Decisions already made by the project owner

| Question | Decision |
|---|---|
| PARALO risks in the MPC cost | **Yes**, this is the core purpose |
| MPC input model | Hold (a, δ) constant over the horizon |
| Steering held as | **Absolute** δ, not relative to road curvature |
| Lateral risk horizon / lane reference | **Full horizon**, **nearest lane** (no target-lane modification) |
| Risk measure | **Probability**, not energy score |
| Same cost structure for MPC and evaluation | Yes; the open question is how its coefficients are set (§3) |
