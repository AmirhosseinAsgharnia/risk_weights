"""
Gymnasium environment for training the ego vehicle with reinforcement
learning: the ego directly commands (accel, delta) each step; surrounding
("surr") traffic runs the existing IDM/MOBIL/far-near stack unchanged (see
model.traffic_step) on the same Road/curve scenario tests/traffic_test.py
demos.

Episode = one attempt at the fixed scenario (Road(s_max=500, kappa_max=
0.005, L_clothoid=60, mu=1.0, lane_num=3), 15 surr cars, same as
tests/traffic_test.py) up to EPISODE_SECONDS of sim time (or
ScenarioConfig.max_episode_seconds, when one is given -- see __init__).

Episode outcome (see step()'s own docstring for the exact rules, and
Outcome/FailureReason below): this is deliberately independent of the
shaped training reward (progress/finish/lane-tracking/survival) -- reward
exists to make PPO learnable, not to define whether the scenario was
solved. The physical, reward-independent terminal conditions are:
  - FINISHED:        ego reached this scenario's configured goal
                      (_ego_goal_reached) without triggering any of the
                      failures below.
  - SAFETY_FAILURE:  COLLISION (ego's body overlaps ANY surr body, crashed
                      or not -- model.collision.ego_overlaps_any; ego is
                      never run through the surr plastic-crash model, a
                      collision just ends the episode), ROLLOVER
                      (risks.rollover.rollover's P_roll for ego's current
                      (v, delta) exceeds ROLLOVER_PROB_THRESHOLD), or
                      OFF_ROAD (ego's body has drifted entirely off the
                      paved road).
  - TIMEOUT:         the episode's time budget was reached without either
                      of the above -- Gymnasium `truncated`, not
                      `terminated` (a time-limit cutoff on an otherwise-
                      ongoing task, not an MDP failure). A safe timeout is
                      NOT success -- see `success`'s own definition below.
FINISHED and SAFETY_FAILURE are `terminated`; TIMEOUT is `truncated`.

    success = scenario_goal_reached and not (collision or rollover or off_road)

PPO failing to reach `success` on this scenario is evidence about this
training procedure, not proof the scenario is physically unsolvable -- see
learning.eval_batch's FEASIBLE/NOT_SOLVED/INCONCLUSIVE labeling.
"""

import math
from enum import Enum
from typing import Literal

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from model.road.road import Road
from model.car.car import Car, CarState
from model.car.config import VehicleParameters
from model.traffic_step import step_surr_agents
from model.collision import ego_overlaps_any, lane_backbone_e_y
from controllers.mobil import MobilParams
from controllers.far_near import clip_steering_rate
from initialization.traffic_init import generate_traffic, CAR_LENGTH, CAR_WIDTH, TEMPLATE_EGO_LANE
from risks.rollover import rollover, TrajectoryStep as RolloverStep
from learning.scenario import ScenarioConfig, generate_scenario, derive_seed


class FailureReason(str, Enum):
    COLLISION = "collision"
    ROLLOVER = "rollover"
    OFF_ROAD = "off_road"


class Outcome(str, Enum):
    FINISHED = "finished"           # scenario_goal_reached, no safety failure -- see `success`
    SAFETY_FAILURE = "safety_failure"   # see FailureReason for which one
    TIMEOUT = "timeout"              # time budget reached, unharmed, goal NOT reached -- never `success`

# ── Fixed scenario (matches tests/traffic_test.py) ──────────────────────────
N_SURR = 15
LANE_NUM = 3
EGO_S0 = 50.0
DT = 0.05
EPISODE_SECONDS = 20.0
MAX_STEPS = int(EPISODE_SECONDS / DT)

ROAD_KWARGS = dict(s_max = 500, kappa_max = 0.005, L_clothoid = 60, mu = 1.0, lane_num = LANE_NUM)
MAX_BRAKING = 8.0            # [m/s^2] same clamp used on surr IDM in model.traffic_step
LANE_CHANGE_COOLDOWN = 1.0   # [s] see model.traffic_step
CRASH_BLEED_K = 1.0          # [-] see model.collision.bleed_crashed
EGO_CAR_ID = -1   # sentinel car_id, distinct from every surr slot id (0..N_SURR-1 in
                  # both initialization.traffic_init and learning.scenario) -- required
                  # now that model.traffic_step.step_surr_agents treats ego as a
                  # visible leader/follower candidate: find_leader/find_follower
                  # exclude a car by matching car_id against "itself", so a colliding
                  # id would wrongly hide ego from, or wrongly self-exclude, whichever
                  # surr car happened to share it.

# ── Action space: accel in [ACCEL_MIN, ACCEL_MAX], delta in +/-DELTA_MAX ───
ACCEL_MIN = -8.0   # [m/s^2] matches MAX_BRAKING -- ego can brake as hard as surr cars are clamped to
ACCEL_MAX = 4.0    # [m/s^2] typical passenger-car acceleration ceiling
DELTA_MAX = 0.5    # [rad] front-wheel steering lock, ~29 deg

# ── Reward ───────────────────────────────────────────────────────────────
# Per-step shaping is proportional to forward progress (delta-s this step);
# NOMINAL_SPEED/EPISODE_SECONDS calibrate a *base* scale so a full, clean,
# roughly-nominal-speed episode's shaping totals ~1.0 on its own, on the
# same order as SURVIVAL_REWARD. PROGRESS_REWARD_BOOST multiplies that
# further: PPO's gamma=0.99 only "sees" ~1/(1-gamma) = 100 steps (~5s) of
# future reward at any moment, so within that effective horizon shaping
# needs to clearly outweigh a single terminal penalty (COLLISION_PENALTY/
# ROLLOVER_PENALTY/OFFROAD_PENALTY, all 1.0) landing just a few seconds
# out, or an under-trained policy finds "stop and never risk it" locally
# optimal (observed: ego progress stalling to ~0, never reaching s_max).
# Boost history: 1 (unboosted) -> 4 -> 10 -> 100, each still observed to
# undershoot -- at boost=100, discounted progress over that ~5s horizon is
# ~25x a single terminal penalty, making driving forward the overwhelming
# favorite as soon as the policy is even moderately safe.
NOMINAL_SPEED = 20.0   # [m/s] rough cruising speed used only for this calibration
PROGRESS_REWARD_BOOST = 100.0
PROGRESS_REWARD_SCALE = PROGRESS_REWARD_BOOST / (NOMINAL_SPEED * EPISODE_SECONDS)
SURVIVAL_REWARD = 1.0   # added once, at truncation (reaching the episode's time budget unharmed), but
                        # ONLY if net progress at truncation is >= the effective min-progress requirement
                        # (ScenarioConfig.min_progress_m on the scenario path, _LEGACY_MIN_PROGRESS_M on
                        # the legacy path -- see step()) -- an idling/crawling-but-safe episode must not
                        # out-earn a genuine attempt at the scenario (see FailureReason/Outcome's module
                        # docstring: this is a training-reward knob only, it never makes a timeout
                        # `success`). Deliberately NOT rescaled with the terminal penalties below: it's a
                        # bonus for a good outcome, not a cost the agent needs to be scared away from, so
                        # it stays on its own small scale.
_LEGACY_MIN_PROGRESS_M = 100.0   # [m] min-progress-for-SURVIVAL_REWARD floor used only when this env has
                                  # no ScenarioConfig (no min_progress_m field to read) -- an assumed
                                  # fraction of the legacy path's full-road goal, not derived from a spec.
# Terminal penalties (collision/rollover/road-departure) must be re-balanced every time
# PROGRESS_REWARD_BOOST changes, or the ratio between them silently drifts: at boost=100 with these left
# at the old flat 1.0, discounted progress over PPO's ~5s effective horizon (gamma=0.99) reached ~25x a
# single penalty -- "crash is basically free" territory (observed: ego stopped avoiding traffic).
# Pegged here to that same discounted-horizon progress value so a crash costs "one full effective horizon
# of clean driving" regardless of how PROGRESS_REWARD_BOOST is retuned next.
_ASSUMED_GAMMA = 0.99   # must match learning.train's PPO(gamma=...) (SB3 default, not overridden there) --
                        # not importable from here since PPO isn't constructed until train.py.
_DISCOUNTED_HORIZON_PROGRESS = PROGRESS_REWARD_SCALE * NOMINAL_SPEED * DT / (1.0 - _ASSUMED_GAMMA)
COLLISION_PENALTY = _DISCOUNTED_HORIZON_PROGRESS   # subtracted once, at collision
ROLLOVER_PENALTY = _DISCOUNTED_HORIZON_PROGRESS    # subtracted once, at rollover
OFFROAD_PENALTY = _DISCOUNTED_HORIZON_PROGRESS     # subtracted once, at road departure (ego's body entirely
                          # off the paved road -- see _ego_off_road; without LANE_TRACKING_SCALE below,
                          # drifting off the road (e.g. failing to steer through a curve) is never
                          # discouraged until it's this severe or also causes a collision)
LANE_TRACKING_SCALE = 0.05   # dense per-step cost on (e_y, e_psi) against the nearest lane centerline --
                             # without this, progress reward only rewards steering through a curve
                             # *indirectly* (via its delayed effect on forward speed along s), which wasn't
                             # enough signal in practice: ego was observed cutting curves without steering,
                             # drifting until either OFFROAD_PENALTY/collision fired or it simply never
                             # covered enough ground to reach s_max in EPISODE_SECONDS. Kept well below
                             # PROGRESS_REWARD_SCALE's typical per-step contribution (~NOMINAL_SPEED * DT *
                             # PROGRESS_REWARD_SCALE = 0.25 at nominal speed) so correcting drift never
                             # outweighs making forward progress -- e_psi in particular is in radians, so
                             # even a modest steering error squares into a real cost at this scale.
# FINISH_BONUS: the total value of reaching this scenario's goal (self._goal_s -- see _ego_goal_reached
# and reset()), delivered as dense, INSTANTANEOUS potential-based shaping every step (_goal_potential/see
# step()) rather than as one lump sum on the terminal step. A pure terminal bonus gives zero learning
# signal until the policy first stumbles into finishing at all -- rare while it's still crash-prone -- so
# this needed to be dense from step one instead. Phi(s) here is FINISH_BONUS * (fraction of the way from
# EGO_S0 to self._goal_s), clipped to [0, FINISH_BONUS]; each step's reward gets Phi(s') - Phi(s) added
# (see step() for why this is deliberately UNdiscounted, unlike the textbook gamma*Phi(s')-Phi(s) form) --
# this telescopes exactly to Phi(s_final) - Phi(s_initial) over any trajectory length or early termination,
# i.e. ego collects exactly FINISH_BONUS times whatever fraction of the way to the goal it actually
# covered, paid out incrementally as that progress is made rather than withheld until (and unless) it
# reaches the very end. Pegged to the same discounted-horizon scale as the terminal penalties (not
# SURVIVAL_REWARD's small scale) so actually finishing is unambiguously the most valuable thing ego can do,
# not a rounding error on top of merely surviving.
# On the LEGACY (no-ScenarioConfig) path, self._goal_s is road.s_max, same as this env's original
# behaviour -- and NOMINAL_SPEED * EPISODE_SECONDS = 400m is itself short of the 450m that path needs
# (s_max=500 - EGO_S0=50), i.e. "nominal" cruising alone was never enough to finish in time. On the
# ScenarioConfig path this is exactly the unreachable-horizon problem ScenarioConfig.goal_distance_m's own
# comment addresses -- its default (150m) is well inside reach.
FINISH_BONUS = _DISCOUNTED_HORIZON_PROGRESS
ROLLOVER_PROB_THRESHOLD = 0.5   # P_roll above this counts as "rolled over" this step

# ── Actuator rate limits ────────────────────────────────────────────────────
# PPO's raw action is a *requested* command -- applying it instantaneously (delta jumping the full
# -DELTA_MAX..+DELTA_MAX range, or accel the full ACCEL_MIN..ACCEL_MAX range, in one DT=0.05s step) would
# let the policy exploit an actuator no real vehicle has. Surr cars already go through exactly this kind
# of limiting for steering (controllers.far_near.clip_steering_rate, via FAR_NEAR_PRESETS'
# steer_rate_limit=0.6-1.0 rad/s depending on behaviour) -- ego had no equivalent. Both defaults below are
# ASSUMPTIONS, not a real vehicle spec -- easy to override per-instance (see EgoTrafficEnv.__init__).
MAX_STEERING_RATE = 1.0   # [rad/s] matches the fastest (aggressive) surr preset already in this codebase
                          # (FAR_NEAR_PRESETS[3].steer_rate_limit) -- reusing a value this codebase already
                          # treats as a plausible upper bound, rather than inventing a new one.
MAX_JERK = 40.0   # [m/s^3] deliberately generous (permissive), NOT a comfort-jerk limit (typical
                  # passenger-car comfort jerk is often cited around 2-5 m/s^3, which would make hard
                  # collision-avoidance braking itself infeasible) -- this only rules out a literal
                  # instantaneous full-range accel swing (ACCEL_MIN..ACCEL_MAX spans 12 m/s^2; at 40 m/s^3
                  # that swing takes >= 0.3s, 6 steps, rather than one). Confirm/tighten against a real
                  # actuator spec if one becomes available.

# ── Observation normalization (fixed scales, not learned -- see _get_obs) ──
_V_SCALE = 30.0     # [m/s]
_EY_SCALE = CAR_WIDTH   # [m] lane half-width is 2.0 m; keeps e_y roughly O(1)
_S_SCALE = 100.0   # [m] relative longitudinal distance to a surr car
_KAPPA_SCALE = 100.0   # 1/kappa_max-ish, brings curvature into an O(1) range
_KAPPA_PREVIEW_DIST = 20.0   # [m] fixed lookahead for the curvature-ahead feature


class EgoTrafficEnv(gym.Env):
    """See module docstring. Observation is a fixed-size full traffic
    snapshot (N_SURR is constant -- no padding/masking): 7 ego features
    (v_x, e_y against the nearest lane, e_psi, nearest-lane index,
    curvature ahead, applied steering, applied acceleration) followed by 5
    features per surr car (relative s, relative lane, relative lateral
    (backbone) offset, heading error, v_x), in `agents`' list order (a
    stable, documented rule -- role order front/rear/blocker at indices
    0/1/2 then background in generation order, see learning.scenario.
    generate_scenario -- identical across resets/instances for the same
    scenario+seed in "fixed" mode; an MLP policy just treats this as 15
    fixed "slots"). See _get_obs for the exact layout."""

    metadata = {"render_modes": []}

    def __init__(
            self,
            scenario_config: ScenarioConfig | None = None,
            mode: Literal["fixed", "distribution"] = "distribution",
            worker_rank: int = 0,
            max_steering_rate: float = MAX_STEERING_RATE,
            max_jerk: float = MAX_JERK,
    ):
        """
        scenario_config: if given, every reset() instead realizes this
        ScenarioConfig via learning.scenario.generate_scenario -- exactly
        15 surr cars (3 explicit critical actors + 12 stochastic
        background), ego's speed drawn from Normal(background_mean_speed,
        background_speed_std) (that path's own config fields), and the
        road is built from scenario_config.road_mu/road_kappa_max rather
        than ROAD_KWARGS. See mode below for how the background
        realization varies (or doesn't) across resets. This scenario's
        goal_distance_m/max_episode_seconds/min_progress_m fields (see
        ScenarioConfig) drive this env's success/timeout/reward-shaping --
        see reset() and step(). If omitted, every reset() instead realizes
        initialization.traffic_init's fixed 16-slot traffic template (see
        generate_traffic) -- ego's speed there is the IDM steady-state
        speed resolved for its fixed slot, not an independently drawn
        quantity, and the goal/episode-length fall back to this module's
        EPISODE_SECONDS/road.s_max (i.e. the original full-road-completion
        behaviour).

        mode: only meaningful when scenario_config is given.
          "fixed":        every reset() reconstructs the *exact* same
                           realization (background included) -- for
                           training/evaluating whether one specific
                           scenario is solvable at all. worker_rank is
                           ignored so parallel envs training on the same
                           fixed scenario all see the same realization.
          "distribution": the critical actors and globals stay fixed, but
                           the background realization varies across resets
                           (a new episode index each time) -- for
                           estimating a success probability over the
                           scenario family. See derive_seed for how
                           worker_rank/episode_index keep parallel
                           SubprocVecEnv workers from producing identical
                           sequences while staying reproducible.

        worker_rank: this env's rank among parallel envs (only used to
        seed distribution-mode background realizations differently per
        worker -- see learning.train's SubprocVecEnv construction).

        max_steering_rate/max_jerk: [rad/s]/[m/s^3] actuator rate limits
        applied to ego's own commanded (delta, accel) each step -- see
        MAX_STEERING_RATE/MAX_JERK's own comments for the defaults'
        rationale. Override here (not via ScenarioConfig: these are a
        vehicle/actuator property, not a scenario parameter) if a real
        vehicle spec is available.
        """
        super().__init__()
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (2,), dtype = np.float32)
        obs_dim = 7 + 5 * N_SURR
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (obs_dim,), dtype = np.float32)

        self.scenario_config = scenario_config
        self.mode = mode
        self.worker_rank = worker_rank
        self.max_steering_rate = max_steering_rate
        self.max_jerk = max_jerk
        self._episode_counter = 0
        self._realized_scenario: dict | None = None

        self.mobil_params = MobilParams()
        self.road: Road | None = None
        self.agents = None
        self.ego_car: Car | None = None
        self.t = 0.0
        self.step_count = 0
        self._prev_s = 0.0
        self._prev_accel = 0.0
        self._goal_s = 0.0
        self._max_steps = MAX_STEPS
        self._min_progress_m = 0.0
        self._distance_travelled = 0.0
        self._min_clearance = math.inf
        self._max_p_rollover = 0.0
        self._max_departure = 0.0

    # ── Gymnasium API ────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed = seed)   # sets self.np_random; reseeds only if seed is not None

        if self.scenario_config is None:
            # Legacy path: initialization.traffic_init's fixed 16-slot
            # traffic template (see generate_traffic) -- ego_lane must
            # match TEMPLATE_EGO_LANE, the lane the template's fixed ego
            # slot actually lives in, not an independently-assumed value.
            self.road = Road(**ROAD_KWARGS)
            self.agents, ego_v0 = generate_traffic(ego_s = EGO_S0, rng = self.np_random)
            ego_lane = TEMPLATE_EGO_LANE
        else:
            # Compact-scenario path (see __init__'s own docstring). `seed`
            # passed to this reset() call only restarts the episode
            # counter (a reproducible restart point for a distribution-
            # mode sequence) -- it never overrides scenario_config.seed,
            # so an unrelated Gym seed can't silently change the scenario.
            if seed is not None:
                self._episode_counter = 0

            if self.mode == "fixed":
                episode_index, worker_rank = 0, 0
            else:
                episode_index, worker_rank = self._episode_counter, self.worker_rank
                self._episode_counter += 1

            scenario_seed = derive_seed(self.scenario_config.seed, worker_rank, episode_index)
            scenario_rng = np.random.default_rng(scenario_seed)   # dedicated stream -- never self.np_random

            self.road = Road(s_max = ROAD_KWARGS["s_max"], kappa_max = self.scenario_config.road_kappa_max,
                              L_clothoid = ROAD_KWARGS["L_clothoid"], mu = self.scenario_config.road_mu,
                              lane_num = LANE_NUM)
            ego_lane = LANE_NUM // 2
            self.agents, ego_v0, self._realized_scenario = generate_scenario(
                self.scenario_config, self.road, ego_s = EGO_S0, ego_lane = ego_lane, rng = scenario_rng,
                worker_rank = worker_rank, episode_index = episode_index, scenario_seed = scenario_seed,
            )
            # tendency > 1 => lower threshold => more lane changes (see ScenarioConfig.lane_change_tendency)
            self.mobil_params = MobilParams(threshold = MobilParams().threshold / self.scenario_config.lane_change_tendency)

        # Total paved half-width, symmetric about the backbone (lane offsets
        # are laid out symmetrically -- see Road.lane_calc) -- used by
        # _ego_off_road to terminate once ego's body is entirely off the
        # road, not just out of its current lane. Derived from self.road
        # rather than hard-coded so it can never drift out of sync with it.
        self._road_half_width = self.road.lanes[-1].offset + self.road.lanes[-1].width / 2   # type: ignore

        self.ego_car = Car(car_id = EGO_CAR_ID,
                            state = CarState(s = EGO_S0, e_y = 0.0, e_psi = 0.0, v_x = ego_v0, lane = ego_lane),
                            vehicle_params = VehicleParameters())
        self._update_ego_pose()

        self.t = 0.0
        self.step_count = 0
        self._prev_s = self.ego_car.state.s
        self._prev_accel = 0.0
        self._distance_travelled = 0.0
        self._min_clearance = math.inf
        self._max_p_rollover = 0.0
        self._max_departure = abs(self._ego_backbone_e_y())

        # Goal/episode-length/reward config: from the ScenarioConfig when one is given, else this
        # module's original legacy-path constants (full-road completion, EPISODE_SECONDS) -- see
        # __init__'s own docstring and ScenarioConfig.goal_distance_m's comment.
        if self.scenario_config is not None:
            self._goal_s = EGO_S0 + self.scenario_config.goal_distance_m
            self._max_steps = max(1, int(round(self.scenario_config.max_episode_seconds / DT)))
            self._min_progress_m = self.scenario_config.min_progress_m
        else:
            self._goal_s = self.road.s_max
            self._max_steps = MAX_STEPS
            self._min_progress_m = _LEGACY_MIN_PROGRESS_M

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        accel_cmd, delta_cmd = self._unscale_action(action)

        # Actuator rate limiting (see MAX_STEERING_RATE/MAX_JERK's own comments): clip_steering_rate is a
        # dimension-agnostic "limit how far cmd can move from prev in one dt" rate limiter (see its own
        # docstring) -- reused here for jerk too rather than duplicating the same three-line clamp.
        # CarState.delta already persists the last *applied* delta across steps (Car.step sets it), so
        # that alone is the correct "previous applied command" without any extra bookkeeping; there's no
        # equivalent persisted field for accel, hence self._prev_accel.
        prev_delta = self.ego_car.state.delta
        delta = clip_steering_rate(delta_cmd, prev_delta, self.max_steering_rate, DT)
        accel = clip_steering_rate(accel_cmd, self._prev_accel, self.max_jerk, DT)
        self._prev_accel = accel

        idx = self.road.index_at(self.ego_car.state.s)
        kappa = -self.road.lanes[self.ego_car.state.lane].kappa[idx]   # type: ignore
        self.ego_car.step(accel, delta, kappa, self.road.mu, DT)
        self._update_ego_pose()

        step_surr_agents(self.agents, self.road, self.t, DT,
                          mobil_params = self.mobil_params, lane_num = LANE_NUM,
                          ego_car = self.ego_car,
                          max_braking = MAX_BRAKING, lane_change_cooldown = LANE_CHANGE_COOLDOWN,
                          crash_bleed_k = CRASH_BLEED_K)

        self.t += DT
        self.step_count += 1

        s_before = self._prev_s
        progress = self.ego_car.state.s - s_before
        self._prev_s = self.ego_car.state.s
        self._distance_travelled += abs(progress)
        reward = PROGRESS_REWARD_SCALE * progress
        # FINISH_BONUS's potential-based shaping -- instantaneous, every step
        # (see that constant's own comment for why, and _goal_potential for Phi).
        # Deliberately UNdiscounted (Phi(s') - Phi(s), no gamma factor): the
        # textbook gamma*Phi(s') - Phi(s) form only stays well-behaved over
        # a horizon short relative to 1/(1-gamma) (~100 steps, ~5s) -- over
        # this env's full 400-step episodes the (1-gamma)*sum(intermediate
        # Phi) drag term dominates and drives the *total* shaping negative
        # for any realistic (non-suicidally-fast) crossing, the opposite of
        # the intended effect (verified numerically before settling on
        # this). The undiscounted form telescopes exactly to
        # Phi(s_final) - Phi(s_initial) with no drag, for any trajectory
        # length or early termination -- simpler, and correct here.
        reward += self._goal_potential(self.ego_car.state.s) - self._goal_potential(s_before)

        # Dense lane-tracking cost -- see LANE_TRACKING_SCALE's own comment for why this is needed
        # alongside progress/finish shaping (both blind to *how* ego covers ground, only that it does).
        _, e_y_nearest = self._ego_nearest_lane()
        reward -= LANE_TRACKING_SCALE * ((e_y_nearest / _EY_SCALE) ** 2 + self.ego_car.state.e_psi ** 2)

        collided = self._ego_collided()
        p_roll = self._ego_p_rollover()
        rolled_over = p_roll > ROLLOVER_PROB_THRESHOLD
        off_road = self._ego_off_road()
        scenario_goal_reached = self._ego_goal_reached()

        # Diagnostics -- reuse the exact authoritative quantities collision/rollover/off-road detection
        # above already computed (never a second, possibly-inconsistent calculation of the same thing).
        self._max_p_rollover = max(self._max_p_rollover, p_roll)
        self._max_departure = max(self._max_departure, abs(self._ego_backbone_e_y()))
        self._min_clearance = min(self._min_clearance, self._ego_clearance())

        if collided:
            failure_reason = FailureReason.COLLISION
        elif rolled_over:
            failure_reason = FailureReason.ROLLOVER
        elif off_road:
            failure_reason = FailureReason.OFF_ROAD
        else:
            failure_reason = None
        safety_failure = failure_reason is not None

        # Physical terminal conditions only -- see module docstring's `success` formula. This is computed
        # independently of the shaped reward below: reward makes PPO learnable, it never decides outcome.
        terminated = safety_failure or scenario_goal_reached
        success = scenario_goal_reached and not safety_failure

        if collided:
            reward -= COLLISION_PENALTY
        if rolled_over:
            reward -= ROLLOVER_PENALTY
        if off_road:
            reward -= OFFROAD_PENALTY

        progress_m = self.ego_car.state.s - EGO_S0
        truncated = False
        if not terminated and self.step_count >= self._max_steps:
            truncated = True
            # Timeout reward is conditional on meaningful forward progress (self._min_progress_m -- see
            # ScenarioConfig.min_progress_m/SURVIVAL_REWARD's own comments) so an unharmed-but-idling
            # episode can't out-earn a genuine attempt -- this NEVER makes a timeout `success` (see above).
            if progress_m >= self._min_progress_m:
                reward += SURVIVAL_REWARD

        if safety_failure:
            outcome = Outcome.SAFETY_FAILURE
        elif scenario_goal_reached:
            outcome = Outcome.FINISHED
        elif truncated:
            outcome = Outcome.TIMEOUT
        else:
            outcome = None   # episode still ongoing

        info = {
            # Legacy keys -- preserved as-is for tests/traffic_test.py, learning.eval, etc. `finished`
            # now means "reached THIS scenario's configured goal", not necessarily the full road.
            "collided": collided, "rolled_over": rolled_over, "off_road": off_road,
            "finished": scenario_goal_reached, "s": self.ego_car.state.s,
            # Canonical outcome reporting -- physical terminal conditions, not accumulated reward.
            "outcome": outcome.value if outcome is not None else None,
            "success": success,
            "failure_reason": failure_reason.value if failure_reason is not None else None,
            "timeout": truncated,
            "progress_m": progress_m,
            "distance_travelled_m": self._distance_travelled,
            "elapsed_s": self.t,
            "min_clearance_m": None if math.isinf(self._min_clearance) else self._min_clearance,
            "max_p_rollover": self._max_p_rollover,
            "max_road_departure_m": self._max_departure,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def get_realized_scenario(self) -> dict:
        """The JSON-serializable record of every realized initial
        condition from the most recent reset() -- see
        learning.scenario._build_realized for the exact structure. Only
        populated when this env was constructed with a scenario_config;
        raises otherwise (there's nothing to report on the fully-random
        path -- initialization.traffic_init.generate_traffic doesn't
        produce one)."""
        if self._realized_scenario is None:
            raise RuntimeError(
                "get_realized_scenario() has nothing to return -- either reset() hasn't been called yet, "
                "or this env has no scenario_config (the default fully-random-traffic path doesn't "
                "produce a realized-scenario record).")
        return self._realized_scenario

    # ── Internals ────────────────────────────────────────────────────────

    def _update_ego_pose(self) -> None:
        """Global (x, y, heading), stored on ego_car.state -- same
        convention model.traffic_step uses for surr cars (needed here for
        anything downstream that wants to render a rollout)."""
        state = self.ego_car.state
        lane_obj = self.road.lanes[state.lane]
        backbone_e_y = -lane_obj.offset + state.e_y   # type: ignore
        x, y, heading = self.road.frenet_to_global(state.s, backbone_e_y, state.e_psi)
        state.x, state.y, state.heading = x, y, heading

    def _unscale_action(self, action: np.ndarray) -> tuple[float, float]:
        action = np.clip(np.asarray(action, dtype = np.float64), -1.0, 1.0)
        accel = ACCEL_MIN + (action[0] + 1.0) * 0.5 * (ACCEL_MAX - ACCEL_MIN)
        delta = action[1] * DELTA_MAX
        return float(accel), float(delta)

    def _ego_backbone_e_y(self) -> float:
        state = self.ego_car.state
        lane_obj = self.road.lanes[state.lane]
        return -lane_obj.offset + state.e_y   # type: ignore

    def _ego_nearest_lane(self) -> tuple[int, float]:
        """(index, e_y) of whichever lane centerline is currently closest
        to ego, in the shared backbone frame -- NOT necessarily
        ego_car.state.lane, which never updates for ego (no MOBIL/lane-
        commit controller drives it the way surr cars have). Using the
        nearest lane instead keeps both this e_y and, below, surr cars'
        relative-lane feature meaningful after ego drifts across a lane
        boundary, rather than staying pinned to whatever lane it spawned
        in for the whole episode."""
        backbone_e_y = self._ego_backbone_e_y()
        lane_centers = [-lane.offset for lane in self.road.lanes]   # type: ignore
        idx = min(range(len(lane_centers)), key = lambda i: abs(backbone_e_y - lane_centers[i]))
        return idx, backbone_e_y - lane_centers[idx]

    def _ego_collided(self) -> bool:
        return ego_overlaps_any(self.ego_car.state.s, self._ego_backbone_e_y(), self.agents, self.road)

    def _ego_p_rollover(self) -> float:
        """P_roll (risks.rollover.rollover's eq. 5.6) for ego's current
        (v, delta) -- the single authoritative rollover measure, used both
        for ROLLOVER_PROB_THRESHOLD's pass/fail check (step()) and as this
        episode's running max_p_rollover diagnostic (info)."""
        state = self.ego_car.state
        p_roll, _ = rollover([RolloverStep(v = state.v_x, delta = state.delta)],
                              vehicle_params = self.ego_car.vehicle_params)
        return p_roll

    def _ego_off_road(self) -> bool:
        """True once ego's body has drifted entirely off the paved road --
        not just out of its current lane -- in the shared backbone_e_y
        frame (see _ego_backbone_e_y/_ego_nearest_lane), compared against
        self._road_half_width (the true outer edge, computed once per
        reset() from self.road). Wrapped in bool(): both operands are
        numpy floats, so the raw comparison is numpy.bool_ -- Gymnasium's
        own env checker requires `terminated` to be a genuine Python bool."""
        return bool(abs(self._ego_backbone_e_y()) > self._road_half_width)

    def _ego_clearance(self) -> float:
        """Distance from ego's body to the nearest surr body, in the same
        (s, e_y) CAR_LENGTH x CAR_WIDTH box metric model.collision.
        ego_overlaps_any/bodies_overlap use for collision detection --
        exactly 0.0 at the moment ego_overlaps_any would return True.
        Diagnostic only (info's min_clearance_m); never a substitute for
        _ego_collided's own authoritative check."""
        ego_s = self.ego_car.state.s
        ego_e_y = self._ego_backbone_e_y()
        best = math.inf
        for agent in self.agents:
            ds = abs(ego_s - agent.car.state.s)
            dey = abs(ego_e_y - lane_backbone_e_y(agent, self.road))
            best = min(best, math.hypot(max(ds - CAR_LENGTH, 0.0), max(dey - CAR_WIDTH, 0.0)))
        return best

    def _ego_goal_reached(self) -> bool:
        """True once ego has reached (or passed) this scenario's
        configured goal (self._goal_s -- EGO_S0 + ScenarioConfig.
        goal_distance_m, or road.s_max on the legacy no-ScenarioConfig
        path -- see reset()). A distinct, physical-state-only condition
        from `success` (step()), which also requires no safety failure."""
        return bool(self.ego_car.state.s >= self._goal_s)

    def _goal_potential(self, s: float) -> float:
        """Phi(s) for FINISH_BONUS's potential-based shaping (see that
        constant's own comment) -- FINISH_BONUS times how far ego has
        gotten from EGO_S0 to self._goal_s, clipped to [0, FINISH_BONUS] so
        it saturates rather than extrapolating past either end."""
        frac = (s - EGO_S0) / (self._goal_s - EGO_S0)
        return FINISH_BONUS * float(np.clip(frac, 0.0, 1.0))

    def _scale_accel(self, accel: float) -> float:
        """Map an applied acceleration in [ACCEL_MIN, ACCEL_MAX] to
        [-1, 1] -- the same convention _unscale_action uses for the raw
        action, inverted, so the observation's applied-accel feature is
        O(1) like every other feature here."""
        return 2.0 * (accel - ACCEL_MIN) / (ACCEL_MAX - ACCEL_MIN) - 1.0

    def _get_obs(self) -> np.ndarray:
        """7 ego features + 5 features per surr car (fixed N_SURR slots,
        `agents`' list order). Ego's lane feature is the *nearest* lane
        index (see _ego_nearest_lane), not the raw ego_car.state.lane --
        that never updates on its own (no MOBIL/lane-commit controller for
        ego), so it would stay pinned to ego's spawn lane forever.

        The last two ego features (applied steering/acceleration) are
        included because the actuator rate limits in step() make the next
        reachable state depend on what was actually applied last step, not
        just the raw action -- without exposing that applied value, this
        would no longer be Markov in the observation alone. state.delta is
        exactly that (Car.step persists the last *applied* delta on
        CarState); self._prev_accel is this env's own equivalent
        bookkeeping for accel (CarState has no persisted field for it).

        The surr lateral-offset and heading-error features exist because a
        lane change is not instantaneous (model.traffic_step ramps a
        changing car's e_y continuously over its lane_change_duration and
        only flips its discrete `lane` at commit -- see step_surr_agents)
        -- relative s/lane/v_x alone make an in-progress lane change
        invisible until that abrupt commit instant; the continuous
        backbone-frame lateral offset and e_psi are both already-computed
        state (no privileged/future information) that reveal it develop
        ing. Relative velocity was considered and deliberately left out:
        it's a linear combination of two features already in this vector
        (this v_x and ego's own), which a single linear layer can already
        reconstruct, so adding it wouldn't expand what's representable. No
        presence/validity flag either: N_SURR is fixed and always fully
        populated (3 critical + 12 background every episode) -- there is
        no empty-slot case in this environment.

        No crashed flag for surr cars: a crash instantly changes that
        car's v_x to the momentum-conserved value (see model.collision),
        and step() always resolves collisions before building this
        observation, so the speed itself is already the tell."""
        state = self.ego_car.state
        ego_lane_idx, e_y_nearest = self._ego_nearest_lane()
        ego_backbone_e_y = self._ego_backbone_e_y()

        idx_preview = self.road.index_at(state.s + _KAPPA_PREVIEW_DIST)
        kappa_preview = -self.road.lanes[state.lane].kappa[idx_preview]   # type: ignore

        ego_feats = [
            state.v_x / _V_SCALE,
            e_y_nearest / _EY_SCALE,
            state.e_psi,
            ego_lane_idx / (LANE_NUM - 1),
            kappa_preview * _KAPPA_SCALE,
            state.delta / DELTA_MAX,
            self._scale_accel(self._prev_accel),
        ]

        surr_feats: list[float] = []
        for agent in self.agents:
            car_state = agent.car.state
            surr_backbone_e_y = -self.road.lanes[car_state.lane].offset + car_state.e_y   # type: ignore
            surr_feats.extend([
                (car_state.s - state.s) / _S_SCALE,
                (car_state.lane - ego_lane_idx) / (LANE_NUM - 1),
                (surr_backbone_e_y - ego_backbone_e_y) / _EY_SCALE,
                car_state.e_psi,
                car_state.v_x / _V_SCALE,
            ])

        return np.asarray(ego_feats + surr_feats, dtype = np.float32)
