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
training reward (RewardConfig: progress, terminal outcome, time) -- reward
exists to make PPO learnable, not to define whether the scenario was
solved. The physical, reward-independent terminal conditions are:
  - FINISHED:        ego reached this scenario's configured goal
                      (_ego_goal_reached) without triggering any of the
                      failures below.
  - SAFETY_FAILURE:  COLLISION (ego's body, turned by its heading error,
                      overlaps ANY surr body, crashed or not --
                      model.collision.ego_overlaps_any; ego is
                      never run through the surr plastic-crash model, a
                      collision just ends the episode), ROLLOVER
                      (risks.rollover.rollover's P_roll for ego's current
                      (v, delta) exceeds ROLLOVER_PROB_THRESHOLD), or
                      OFF_ROAD (any part of ego's body, turned by its
                      heading error, is more than ROAD_EDGE_MARGIN_M past
                      the paved road's edge).
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
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from model.road.road import Road
from model.car.car import Car, CarState
from model.car.config import VehicleParameters
from model.traffic_step import step_surr_agents
from model.collision import _BODY_RADIUS, body_gap, body_half_extent_e_y, ego_overlaps_any, lane_backbone_e_y
from controllers.mobil import MobilParams
from controllers.far_near import clip_steering_rate
from initialization.traffic_init import generate_traffic, CAR_WIDTH, TEMPLATE_EGO_LANE
from risks.rollover import rollover, TrajectoryStep as RolloverStep
from learning.vehicle_physics import (
    ROLLOVER_EVENT_MODELS, friction_demand, rollover_physical_v2, slip_lateral_saturation, stability_loss, tire_state,
    tire_utilization,
)
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
ARENA_N_SURR = 6   # learning.feasibility_common's six-slot arena families (cutin/sandwich) -- see
                   # EgoTrafficEnv's `arena` constructor parameter
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
# accel is a tire-force REQUEST (F = m accel), not an applied acceleration: the realized one is limited by
# the road's friction (model.car.model.tire_forces) and is what info["accel_realized"] reports.
ROAD_EDGE_MARGIN_M = 0.5   # [m] how far the body may pass the paved edge before OFF_ROAD
ACCEL_MIN = -8.0   # [m/s^2] matches MAX_BRAKING -- ego can brake as hard as surr cars are clamped to
ACCEL_MAX = 4.0    # [m/s^2] typical passenger-car acceleration ceiling
DELTA_MAX = VehicleParameters.delta_max   # [rad] front-wheel steering lock

# ── Reward (docs/description.md §7) ─────────────────────────────────────────
#     r_t = r_progress + r_terminal - r_time
# r_progress: the new ground gained toward the goal this step, as a fraction of the goal distance. Measured
#             on a running maximum of s, so retraced ground pays nothing and an episode's sum is at most 1.
# r_terminal: one value on the step the episode ends, chosen by the outcome flags in step().
# r_time:     a constant per executed step, the last one included.
# Nothing else: no lane, heading, comfort or risk term, and no potential-based shaping. PPO is an empirical
# feasibility oracle here, so the reward says only how far ego got and how the attempt ended. It never
# decides the outcome: `success` comes from physics, whatever the return was.
REWARD_VERSION = "goal-progress-v1"
DEFAULT_PPO_GAMMA = 0.999   # initial value for the training scripts, overridable with --gamma. At 20 steps/s,
                            # 0.99 discounts a reward 20 s away to 0.018 and 0.999 to 0.67. Discounted return
                            # is still not the probability of success: discounting, r_progress and r_time all
                            # move the policy's preference away from that.


@dataclass(frozen=True)
class RewardConfig:
    """Penalties are positive magnitudes, subtracted from the reward. The time penalty is per executed step
    at this module's single fixed DT = 0.05 s, i.e. 0.01 per simulated second; if DT ever changes, rescale
    it to keep that per-second rate."""
    progress_reward_scale: float = 1.0
    success_reward: float = 3.0
    collision_penalty: float = 5.0
    rollover_penalty: float = 5.0
    road_departure_penalty: float = 5.0
    timeout_penalty: float = 1.0
    time_penalty_per_step: float = 0.0005


DEFAULT_REWARD_CONFIG = RewardConfig()

NOMINAL_SPEED = 20.0   # [m/s] cruising speed the MPC controllers track (learning.mpc, learning.paralo_mpc)
_LEGACY_MIN_PROGRESS_M = 100.0   # [m] info["lack_of_progress"]'s floor when this env has no ScenarioConfig
                                  # (no min_progress_m field to read) -- assumed, not derived from a spec.
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
            arena=None,
            rollover_event: str = "legacy_v1",
            reward_config: RewardConfig | None = None,
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
        ScenarioConfig) drive this env's goal, timeout and progress reward --
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

        arena: mutually exclusive with scenario_config -- a feasibility-
        pipeline scenario-family runtime (learning.feasibility_cutin.
        CutinRuntime / learning.feasibility_sandwich.SandwichRuntime, or
        anything else satisfying learning.feasibility_common.ArenaRuntime's
        structural contract; duck-typed, EgoTrafficEnv never imports either
        family module or isinstance-checks this). When given, every
        reset() instead realizes exactly ARENA_N_SURR=6 agents via
        `arena.reset(...)`, discretionary MOBIL is disabled for all of them
        (MobilParams(threshold=inf) -- see reset()), and every step()
        calls `arena.pre_surr_step(...)` right before advancing surr
        traffic (so a family can script one agent's behaviour -- a
        prescribed lane change or emergency stop -- via
        model.traffic_step.step_surr_agents' accel_override parameter or
        by mutating a TrafficAgent's public fields directly) and merges
        `arena.extra_info()` into info["arena"]. goal_distance_m/
        max_episode_seconds/min_progress_m are read off `arena` exactly
        like ScenarioConfig's own fields (see reset()).

        rollover_event: which ROLLOVER detector terminates the episode
        (learning.vehicle_physics). "legacy_v1" (default, so every historical
        run reproduces exactly): P_roll(v_x, delta_cmd) > ROLLOVER_PROB_THRESHOLD
        -- known to fire at ~2 m/s^2 real lateral acceleration on the default
        curve (docs/description.md §A2). "physical_v2": realized lateral
        body acceleration from tire forces |a_y| >= SSF g (quasi-static tipping onset).
        info["rollover_event_model"] reports which one ran.

        reward_config: the training reward's coefficients (DEFAULT_REWARD_CONFIG
        if omitted). They only change the reward: outcomes never depend on them.
        """
        if scenario_config is not None and arena is not None:
            raise ValueError("EgoTrafficEnv: scenario_config and arena are mutually exclusive.")

        super().__init__()
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (2,), dtype = np.float32)
        n_surr = ARENA_N_SURR if arena is not None else N_SURR
        obs_dim = 7 + 5 * n_surr
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (obs_dim,), dtype = np.float32)

        self.scenario_config = scenario_config
        self.arena = arena
        self.mode = mode
        self.worker_rank = worker_rank
        self.max_steering_rate = max_steering_rate
        self.max_jerk = max_jerk
        assert rollover_event in ROLLOVER_EVENT_MODELS, rollover_event
        self.rollover_event = rollover_event
        self.reward_config = DEFAULT_REWARD_CONFIG if reward_config is None else reward_config
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
        self._s_start = 0.0
        self._s_reached = 0.0
        self._goal_distance = 0.0
        self._episode_return = 0.0
        self._max_steps = MAX_STEPS
        self._min_progress_m = 0.0
        self._distance_travelled = 0.0
        self._min_clearance = math.inf
        self._max_p_rollover = 0.0
        self._max_departure = 0.0
        self._reset_tire_diagnostics()

    # ── Gymnasium API ────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed = seed)   # sets self.np_random; reseeds only if seed is not None

        if self.arena is not None:
            # Feasibility-pipeline arena path (see __init__'s own docstring). Mirrors the compact-
            # scenario path's seed/episode-index bookkeeping exactly (fixed vs distribution mode,
            # worker_rank, derive_seed) -- arena.reset() is the only thing that differs.
            if seed is not None:
                self._episode_counter = 0
            if self.mode == "fixed":
                episode_index, worker_rank = 0, 0
            else:
                episode_index, worker_rank = self._episode_counter, self.worker_rank
                self._episode_counter += 1

            scenario_seed = derive_seed(self.arena.cfg.seed, worker_rank, episode_index)
            scenario_rng = np.random.default_rng(scenario_seed)

            self.road = Road(s_max = ROAD_KWARGS["s_max"], kappa_max = self.arena.cfg.road_kappa_max,
                              L_clothoid = ROAD_KWARGS["L_clothoid"], mu = self.arena.cfg.road_mu,
                              lane_num = LANE_NUM)
            ego_lane = LANE_NUM // 2
            self.agents, ego_v0, self._realized_scenario = self.arena.reset(
                self.road, EGO_S0, ego_lane, scenario_rng,
                worker_rank = worker_rank, episode_index = episode_index, scenario_seed = scenario_seed,
            )
            # Discretionary MOBIL is disabled for every arena actor: mobil_decision's
            # `incentive > threshold` can never fire with an infinite threshold, so IDM/steering/
            # dynamics stay fully live but no agent ever *initiates* a lane change on its own --
            # only this env's arena.pre_surr_step (see step()) can prescribe one. No change to
            # controllers.mobil/model.traffic_step needed for this.
            self.mobil_params = MobilParams(threshold = math.inf)
        elif self.scenario_config is None:
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
        # _ego_off_road to terminate once ego's body passes the road edge by
        # more than ROAD_EDGE_MARGIN_M, not just its current lane. Derived from self.road
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
        self._reset_tire_diagnostics()

        # Goal/episode-length/reward config: from whichever config is active (ScenarioConfig and
        # ArenaCommonConfig -- see learning.feasibility_common -- expose the identical field names,
        # so one branch covers both without either config type needing a shared base class), else
        # this module's original legacy-path constants (full-road completion, EPISODE_SECONDS) --
        # see __init__'s own docstring and ScenarioConfig.goal_distance_m's comment.
        active_cfg = self.arena.cfg if self.arena is not None else self.scenario_config
        if active_cfg is not None:
            self._goal_s = EGO_S0 + active_cfg.goal_distance_m
            self._max_steps = max(1, int(round(active_cfg.max_episode_seconds / DT)))
            self._min_progress_m = active_cfg.min_progress_m
        else:
            self._goal_s = self.road.s_max
            self._max_steps = MAX_STEPS
            self._min_progress_m = _LEGACY_MIN_PROGRESS_M

        # Distance from where ego actually starts, never the absolute station s.
        self._s_start = self._s_reached = self.ego_car.state.s
        self._goal_distance = self._goal_s - self._s_start
        assert self._goal_distance > 0, (self._goal_s, self._s_start)
        self._episode_return = 0.0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        accel_cmd, delta_cmd = self._unscale_action(action)

        # Actuator rate limiting (see MAX_STEERING_RATE/MAX_JERK's own comments): clip_steering_rate is a
        # dimension-agnostic "limit how far cmd can move from prev in one dt" rate limiter (see its own
        # docstring) -- reused here for jerk too rather than duplicating the same three-line clamp.
        # CarState.delta already persists the last *applied* delta across steps (Car.step sets it), so
        # that alone is the correct "previous applied command" without any extra bookkeeping; there's no
        # equivalent persisted field for accel, hence self._prev_accel.
        speed_limit = self.arena.cfg.max_ego_speed_mps if self.arena is not None else None
        if speed_limit is not None:
            accel_cmd = min(accel_cmd, max((speed_limit - self.ego_car.state.v_x) / DT, 0.0))
        prev_delta = self.ego_car.state.delta
        delta = clip_steering_rate(delta_cmd, prev_delta, self.max_steering_rate, DT)
        accel = clip_steering_rate(accel_cmd, self._prev_accel, self.max_jerk, DT)
        self._prev_accel = accel

        idx = self.road.index_at(self.ego_car.state.s)
        kappa = -self.road.lanes[self.ego_car.state.lane].kappa[idx]   # type: ignore
        self.ego_car.step(accel, delta, kappa, self.road.mu, DT)
        self._update_ego_pose()

        # Arena scripted-event hook (see learning.feasibility_common.ArenaRuntime) -- may mutate
        # self.agents in place (e.g. a cut-in's prescribed target_lane/lane_change_t0 assignment)
        # and/or return a {car_id: accel} override for step_surr_agents' own accel_override
        # parameter (e.g. a sandwich's prescribed emergency stop). None for every non-arena env.
        accel_override = (self.arena.pre_surr_step(self.t, self.ego_car.state, self.agents)
                           if self.arena is not None else None)

        step_surr_agents(self.agents, self.road, self.t, DT,
                          mobil_params = self.mobil_params, lane_num = LANE_NUM,
                          ego_car = self.ego_car,
                          max_braking = MAX_BRAKING, lane_change_cooldown = LANE_CHANGE_COOLDOWN,
                          crash_bleed_k = CRASH_BLEED_K, accel_override = accel_override)

        self.t += DT
        self.step_count += 1

        self._distance_travelled += abs(self.ego_car.state.s - self._prev_s)
        self._prev_s = self.ego_car.state.s

        collided = self._ego_collided()
        p_roll = self._ego_p_rollover()
        ts = self.ego_tire_state()
        rolled_over = (p_roll > ROLLOVER_PROB_THRESHOLD if self.rollover_event == "legacy_v1"
                       else rollover_physical_v2(ts, self.ego_car.vehicle_params))
        off_road = self._ego_off_road()
        scenario_goal_reached = self._ego_goal_reached()

        # Diagnostics -- reuse the exact authoritative quantities collision/rollover/off-road detection
        # above already computed (never a second, possibly-inconsistent calculation of the same thing).
        self._max_p_rollover = max(self._max_p_rollover, p_roll)
        self._max_departure = max(self._max_departure, abs(self._ego_backbone_e_y()))
        self._min_clearance = min(self._min_clearance, self._ego_clearance())
        self._update_tire_diagnostics(ts)

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
        # independently of the reward below: reward makes PPO learnable, it never decides outcome.
        terminated = safety_failure or scenario_goal_reached
        success = scenario_goal_reached and not safety_failure

        progress_m = self.ego_car.state.s - EGO_S0
        truncated = not terminated and self.step_count >= self._max_steps

        r_progress = self._progress_reward()
        r_terminal = self._terminal_reward(failure_reason, success, truncated)
        r_time = self.reward_config.time_penalty_per_step
        reward = r_progress + r_terminal - r_time
        self._episode_return += reward

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
            "rollover_event_model": self.rollover_event,
            # Requested -> actuator-limited -> realized (see learning.vehicle_physics): separates a policy
            # asking for too much from what the tires delivered. `tire` is this step's forces, the rest
            # are running episode values. lack_of_progress tells an idling timeout from a genuine one.
            "accel_requested": accel_cmd, "accel_actuator": accel, "accel_realized": ts.a_x,
            "tire": ts.to_dict(), **self._tire_diag,
            "lack_of_progress": truncated and progress_m < self._min_progress_m,
            # Training signal, kept apart from the outcome fields above. reward_time is the positive magnitude:
            # reward_total = reward_progress + reward_terminal - reward_time. Episode outcomes are the keys
            # above on the last step (success, collided, rolled_over, off_road, timeout).
            "reward_progress": r_progress, "reward_terminal": r_terminal, "reward_time": r_time,
            "reward_total": reward, "episode_return": self._episode_return, "episode_length": self.step_count,
            "episode_progress": (min(self._s_reached, self._goal_s) - self._s_start) / self._goal_distance,
        }
        if self.arena is not None:
            # Family-specific diagnostics (event trigger station/time, phase classification, etc.)
            # -- namespaced under "arena" so it can never collide with a key above.
            info["arena"] = self.arena.extra_info()
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
        return ego_overlaps_any(self.ego_car.state.s, self._ego_backbone_e_y(), self.agents, self.road,
                                ego_e_psi = self.ego_car.state.e_psi)

    def _ego_p_rollover(self) -> float:
        """P_roll (risks.rollover.rollover's eq. 5.6) for ego's current
        (v, delta) -- the single authoritative rollover measure, used both
        for ROLLOVER_PROB_THRESHOLD's pass/fail check (step()) and as this
        episode's running max_p_rollover diagnostic (info)."""
        state = self.ego_car.state
        p_roll, _ = rollover([RolloverStep(v = state.v_x, delta = state.delta)],
                              vehicle_params = self.ego_car.vehicle_params)
        return p_roll

    def ego_tire_state(self):
        """Realized tire forces / lateral acceleration at ego's current state (learning.vehicle_physics.
        tire_state, read off CarDynamics with the last APPLIED accel/delta and the road at ego's s)."""
        st = self.ego_car.state
        kappa = -self.road.lanes[st.lane].kappa[self.road.index_at(st.s)]   # type: ignore
        return tire_state(st.as_array(), self._prev_accel, st.delta, kappa, self.road.mu, self.ego_car.vehicle_params)

    def _reset_tire_diagnostics(self) -> None:
        self._tire_diag = {"max_friction_demand": 0.0, "max_tire_utilization": 0.0, "max_abs_sideslip_deg": 0.0,
                           "tire_saturated_steps": 0, "stability_loss": False}

    def _update_tire_diagnostics(self, ts) -> None:
        d = self._tire_diag
        d["max_friction_demand"] = max(d["max_friction_demand"], friction_demand(ts, self.ego_car.vehicle_params))
        d["max_tire_utilization"] = max(d["max_tire_utilization"], tire_utilization(ts))
        d["max_abs_sideslip_deg"] = max(d["max_abs_sideslip_deg"], abs(math.degrees(ts.sideslip_rad)))
        d["tire_saturated_steps"] += int(slip_lateral_saturation(ts))
        d["stability_loss"] = d["stability_loss"] or stability_loss(ts)

    def _ego_off_road(self) -> bool:
        """True once any part of ego's body is more than ROAD_EDGE_MARGIN_M
        past the paved road's edge -- not just out of its current lane -- in
        the shared backbone_e_y frame (see _ego_backbone_e_y/_ego_nearest_lane):
        the footprint's outer side, turned by e_psi, against
        self._road_half_width (the true outer edge, computed once per reset()
        from self.road). Wrapped in
        bool(): both operands are numpy floats, so the raw comparison is
        numpy.bool_ -- Gymnasium's own env checker requires `terminated`
        to be a genuine Python bool."""
        outer_edge = abs(self._ego_backbone_e_y()) + body_half_extent_e_y(self.ego_car.state.e_psi)
        return bool(outer_edge > self._road_half_width + ROAD_EDGE_MARGIN_M)

    def _ego_clearance(self) -> float:
        """Distance from ego's body to the nearest surr body, using the same
        oriented (s, e_y) boxes as model.collision.ego_overlaps_any -- exactly
        0.0 at the moment it would return True. Diagnostic only (info's
        min_clearance_m); never a substitute for _ego_collided's own
        authoritative check."""
        state = self.ego_car.state
        ego_e_y = self._ego_backbone_e_y()
        best = math.inf
        for agent in self.agents:
            s, e_y = agent.car.state.s, lane_backbone_e_y(agent, self.road)
            if math.hypot(state.s - s, ego_e_y - e_y) - 2.0 * _BODY_RADIUS < best:   # else it cannot beat best
                best = min(best, body_gap(state.s, ego_e_y, state.e_psi, s, e_y, agent.car.state.e_psi))
        return best

    def _ego_goal_reached(self) -> bool:
        """True once ego has reached (or passed) this scenario's
        configured goal (self._goal_s -- EGO_S0 + ScenarioConfig.
        goal_distance_m, or road.s_max on the legacy no-ScenarioConfig
        path -- see reset()). A distinct, physical-state-only condition
        from `success` (step()), which also requires no safety failure."""
        return bool(self.ego_car.state.s >= self._goal_s)

    def _progress_reward(self) -> float:
        """The newly reached part of [s_start, goal_s] as a fraction of the goal distance. Only the running
        maximum of s counts, capped at the goal, so the sum over an episode cannot pass progress_reward_scale."""
        s_reached = max(self._s_reached, float(self.ego_car.state.s))
        gained = min(s_reached, self._goal_s) - min(self._s_reached, self._goal_s)
        self._s_reached = s_reached
        return self.reward_config.progress_reward_scale * gained / self._goal_distance

    def _terminal_reward(self, failure_reason: FailureReason | None, success: bool, truncated: bool) -> float:
        """Read off the outcome step() already decided, so exactly one value applies and a failure always
        outranks reaching the goal on the same step."""
        c = self.reward_config
        if failure_reason is FailureReason.COLLISION:
            return -c.collision_penalty
        if failure_reason is FailureReason.ROLLOVER:
            return -c.rollover_penalty
        if failure_reason is FailureReason.OFF_ROAD:
            return -c.road_departure_penalty
        if success:
            return c.success_reward
        return -c.timeout_penalty if truncated else 0.0

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
