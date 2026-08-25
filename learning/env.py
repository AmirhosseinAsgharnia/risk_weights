"""
Gymnasium environment for training the ego vehicle with reinforcement
learning: the ego directly commands (accel, delta) each step; surrounding
("surr") traffic runs the existing IDM/MOBIL/far-near stack unchanged (see
model.traffic_step) on the same Road/curve scenario tests/traffic_test.py
demos.

Episode = one lap of the fixed scenario (Road(s_max=500, kappa_max=0.005,
L_clothoid=60, mu=1.0, lane_num=3), 15 surr cars, same as
tests/traffic_test.py) up to EPISODE_SECONDS of sim time.

Terminal states (see the constructor's own docstring for the reward this
produces):
  - collision: ego's body overlaps ANY surr body (crashed or not) --
    model.collision.ego_overlaps_any. Ego is never run through the surr
    plastic-crash model (see that module) -- a collision just ends the
    episode.
  - rollover: risks.rollover.rollover's P_roll for ego's current (v, delta)
    exceeds ROLLOVER_PROB_THRESHOLD.
Both are `terminated`. Reaching EPISODE_SECONDS without either is
`truncated` (Gymnasium's distinction: a true MDP failure vs. a time-limit
cutoff on an otherwise-ongoing task) -- see step()'s docstring for why the
survival bonus is attached there rather than to `terminated`.
"""

from typing import Literal

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from model.road.road import Road
from model.car.car import Car, CarState
from model.car.config import VehicleParameters
from model.traffic_step import step_surr_agents
from model.collision import ego_overlaps_any
from controllers.mobil import MobilParams
from initialization.traffic_init import generate_traffic, CAR_WIDTH, TEMPLATE_EGO_LANE
from risks.rollover import rollover, TrajectoryStep as RolloverStep
from learning.scenario import ScenarioConfig, generate_scenario, derive_seed

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
SURVIVAL_REWARD = 1.0   # added once, at truncation (reaching EPISODE_SECONDS unharmed) -- deliberately
                        # NOT rescaled with the terminal penalties below: it's a bonus for a good outcome,
                        # not a cost the agent needs to be scared away from, so it stays on its own small scale.
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
                          # off the paved road -- see _ego_off_road; nothing else in the reward penalizes
                          # drifting off the road, e.g. failing to steer through a curve, so without this
                          # it's simply never discouraged as long as it doesn't also cause a collision)
# FINISH_BONUS: the total value of reaching s_max (see _ego_finished), delivered as dense, INSTANTANEOUS
# potential-based shaping every step (_goal_potential/see step()) rather than as one lump sum on the
# terminal step. A pure terminal bonus gives zero learning signal until the policy first stumbles into
# finishing at all -- rare while it's still crash-prone -- so this needed to be dense from step one instead.
# Phi(s) here is FINISH_BONUS * (fraction of the way from EGO_S0 to s_max), clipped to [0, FINISH_BONUS];
# each step's reward gets Phi(s') - Phi(s) added (see step() for why this is deliberately UNdiscounted,
# unlike the textbook gamma*Phi(s')-Phi(s) form) -- this telescopes exactly to Phi(s_final) - Phi(s_initial)
# over any trajectory length or early termination, i.e. ego collects exactly FINISH_BONUS times whatever
# fraction of the road it actually covered, paid out incrementally as that progress is made rather than
# withheld until (and unless) it reaches the very end. Pegged to the same discounted-horizon scale as the
# terminal penalties (not SURVIVAL_REWARD's small scale) so actually finishing is unambiguously the most
# valuable thing ego can do, not a rounding error on top of merely surviving.
# Also note NOMINAL_SPEED * EPISODE_SECONDS = 400m is itself short of the 450m ego actually needs to cover
# (s_max=500 - EGO_S0=50) -- "nominal" cruising alone was never enough to finish in time.
FINISH_BONUS = _DISCOUNTED_HORIZON_PROGRESS
ROLLOVER_PROB_THRESHOLD = 0.5   # P_roll above this counts as "rolled over" this step

# ── Observation normalization (fixed scales, not learned -- see _get_obs) ──
_V_SCALE = 30.0     # [m/s]
_EY_SCALE = CAR_WIDTH   # [m] lane half-width is 2.0 m; keeps e_y roughly O(1)
_S_SCALE = 100.0   # [m] relative longitudinal distance to a surr car
_KAPPA_SCALE = 100.0   # 1/kappa_max-ish, brings curvature into an O(1) range
_KAPPA_PREVIEW_DIST = 20.0   # [m] fixed lookahead for the curvature-ahead feature


class EgoTrafficEnv(gym.Env):
    """See module docstring. Observation is a fixed-size full traffic
    snapshot (N_SURR is constant -- no padding/masking): 5 ego features
    (v_x, e_y against the nearest lane, e_psi, nearest-lane index,
    curvature ahead) followed by 3 features per surr car (relative s,
    relative lane, v_x), in
    `agents`' list order (that order is fixed for the life of an episode,
    but which physical car ends up at which index varies episode to
    episode -- an MLP policy just treats it as 15 fixed "slots", which is
    what was asked for here). See _get_obs for the exact layout."""

    metadata = {"render_modes": []}

    def __init__(
            self,
            scenario_config: ScenarioConfig | None = None,
            mode: Literal["fixed", "distribution"] = "distribution",
            worker_rank: int = 0,
    ):
        """
        scenario_config: if given, every reset() instead realizes this
        ScenarioConfig via learning.scenario.generate_scenario -- exactly
        15 surr cars (3 explicit critical actors + 12 stochastic
        background), ego's speed drawn from Normal(background_mean_speed,
        background_speed_std) (that path's own config fields), and the
        road is built from scenario_config.road_mu/road_kappa_max rather
        than ROAD_KWARGS. See mode below for how the background
        realization varies (or doesn't) across resets. If omitted, every
        reset() instead realizes initialization.traffic_init's fixed
        16-slot traffic template (see generate_traffic) -- ego's speed
        there is the IDM steady-state speed resolved for its fixed slot,
        not an independently drawn quantity.

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
        """
        super().__init__()
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (2,), dtype = np.float32)
        obs_dim = 5 + 3 * N_SURR
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (obs_dim,), dtype = np.float32)

        self.scenario_config = scenario_config
        self.mode = mode
        self.worker_rank = worker_rank
        self._episode_counter = 0
        self._realized_scenario: dict | None = None

        self.mobil_params = MobilParams()
        self.road: Road | None = None
        self.agents = None
        self.ego_car: Car | None = None
        self.t = 0.0
        self.step_count = 0
        self._prev_s = 0.0

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

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        accel, delta = self._unscale_action(action)

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

        collided = self._ego_collided()
        rolled_over = self._ego_rolled_over()
        off_road = self._ego_off_road()
        finished = self._ego_finished()
        terminated = collided or rolled_over or off_road or finished
        if collided:
            reward -= COLLISION_PENALTY
        if rolled_over:
            reward -= ROLLOVER_PENALTY
        if off_road:
            reward -= OFFROAD_PENALTY

        truncated = False
        if not terminated and self.step_count >= MAX_STEPS:
            truncated = True
            reward += SURVIVAL_REWARD

        info = {"collided": collided, "rolled_over": rolled_over, "off_road": off_road,
                "finished": finished, "s": self.ego_car.state.s}
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

    def _ego_rolled_over(self) -> bool:
        state = self.ego_car.state
        p_roll, _ = rollover([RolloverStep(v = state.v_x, delta = state.delta)],
                              vehicle_params = self.ego_car.vehicle_params)
        return p_roll > ROLLOVER_PROB_THRESHOLD

    def _ego_off_road(self) -> bool:
        """True once ego's body has drifted entirely off the paved road --
        not just out of its current lane -- in the shared backbone_e_y
        frame (see _ego_backbone_e_y/_ego_nearest_lane), compared against
        self._road_half_width (the true outer edge, computed once per
        reset() from self.road). Wrapped in bool(): both operands are
        numpy floats, so the raw comparison is numpy.bool_ -- Gymnasium's
        own env checker requires `terminated` to be a genuine Python bool."""
        return bool(abs(self._ego_backbone_e_y()) > self._road_half_width)

    def _ego_finished(self) -> bool:
        """True once ego has reached (or passed) the end of the modeled
        road -- a distinct success condition from merely surviving to
        MAX_STEPS (see FINISH_BONUS's own comment for why that distinction
        needed its own reward, not just SURVIVAL_REWARD)."""
        return bool(self.ego_car.state.s >= self.road.s_max)

    def _goal_potential(self, s: float) -> float:
        """Phi(s) for FINISH_BONUS's potential-based shaping (see that
        constant's own comment) -- FINISH_BONUS times how far ego has
        gotten from EGO_S0 to road.s_max, clipped to [0, FINISH_BONUS] so
        it saturates rather than extrapolating past either end."""
        frac = (s - EGO_S0) / (self.road.s_max - EGO_S0)
        return FINISH_BONUS * float(np.clip(frac, 0.0, 1.0))

    def _get_obs(self) -> np.ndarray:
        """5 ego features + 3 features per surr car (fixed N_SURR slots,
        `agents`' list order). Ego's lane feature is the *nearest* lane
        index (see _ego_nearest_lane), not the raw ego_car.state.lane --
        that never updates on its own (no MOBIL/lane-commit controller for
        ego), so it would stay pinned to ego's spawn lane forever. No
        crashed flag for surr cars: a crash instantly changes that car's
        v_x to the momentum-conserved value (see model.collision), and
        step() always resolves collisions before building this
        observation, so the speed itself is already the tell."""
        state = self.ego_car.state
        ego_lane_idx, e_y_nearest = self._ego_nearest_lane()

        idx_preview = self.road.index_at(state.s + _KAPPA_PREVIEW_DIST)
        kappa_preview = -self.road.lanes[state.lane].kappa[idx_preview]   # type: ignore

        ego_feats = [
            state.v_x / _V_SCALE,
            e_y_nearest / _EY_SCALE,
            state.e_psi,
            ego_lane_idx / (LANE_NUM - 1),
            kappa_preview * _KAPPA_SCALE,
        ]

        surr_feats: list[float] = []
        for agent in self.agents:
            car_state = agent.car.state
            surr_feats.extend([
                (car_state.s - state.s) / _S_SCALE,
                (car_state.lane - ego_lane_idx) / (LANE_NUM - 1),
                car_state.v_x / _V_SCALE,
            ])

        return np.asarray(ego_feats + surr_feats, dtype = np.float32)
