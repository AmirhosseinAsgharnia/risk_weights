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
from initialization.traffic_init import generate_traffic, CAR_WIDTH
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

# ── Action space: accel in [ACCEL_MIN, ACCEL_MAX], delta in +/-DELTA_MAX ───
ACCEL_MIN = -8.0   # [m/s^2] matches MAX_BRAKING -- ego can brake as hard as surr cars are clamped to
ACCEL_MAX = 4.0    # [m/s^2] typical passenger-car acceleration ceiling
DELTA_MAX = 0.5    # [rad] front-wheel steering lock, ~29 deg

# ── Reward ───────────────────────────────────────────────────────────────
# Per-step shaping is proportional to forward progress (delta-s this step);
# NOMINAL_SPEED/EPISODE_SECONDS calibrate PROGRESS_REWARD_SCALE so a full,
# clean, roughly-nominal-speed episode accumulates shaping reward on the
# same order as the terminal survival bonus (~1.0) -- if shaping dominated
# the return, the agent would have little incentive to actually avoid the
# terminal states, and if it were negligible it wouldn't shape anything.
NOMINAL_SPEED = 20.0   # [m/s] rough cruising speed used only for this calibration
PROGRESS_REWARD_SCALE = 1.0 / (NOMINAL_SPEED * EPISODE_SECONDS)
SURVIVAL_REWARD = 1.0   # added once, at truncation (reaching EPISODE_SECONDS unharmed)
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
            ego_speed_std: float = 2.5,
            scenario_config: ScenarioConfig | None = None,
            mode: Literal["fixed", "distribution"] = "distribution",
            worker_rank: int = 0,
    ):
        """
        ego_speed_std: [m/s] ego is the road speed, not an independently
        set quantity: its initial v_x each episode ~ Normal(mean(realized
        surr v_x), ego_speed_std) -- see generate_traffic. Only used when
        scenario_config is None (see below) -- this is the original,
        fully-random-traffic path. Default 2.5 matches
        learning.scenario.ScenarioConfig's default background_speed_std,
        for consistency between the two paths.

        scenario_config: if given, every reset() instead realizes this
        ScenarioConfig via learning.scenario.generate_scenario -- exactly
        15 surr cars (3 explicit critical actors + 12 stochastic
        background), ego_speed_std is ignored (ego's speed is drawn from
        Normal(background_mean_speed, background_speed_std) instead --
        same idea, just that path's own config fields), and the road is
        built from scenario_config.road_mu/road_kappa_max rather than
        ROAD_KWARGS. See mode below for how the background realization
        varies (or doesn't) across resets.

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

        self.ego_speed_std = ego_speed_std
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
            # Original path: fully-random traffic, unchanged.
            self.road = Road(**ROAD_KWARGS)
            self.agents, ego_v0 = generate_traffic(N_SURR, ego_s = EGO_S0, lanes = tuple(range(LANE_NUM)),
                                                    ego_speed_std = self.ego_speed_std, rng = self.np_random)
            ego_lane = LANE_NUM // 2
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

        self.ego_car = Car(state = CarState(s = EGO_S0, e_y = 0.0, e_psi = 0.0, v_x = ego_v0, lane = ego_lane),
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
                          max_braking = MAX_BRAKING, lane_change_cooldown = LANE_CHANGE_COOLDOWN,
                          crash_bleed_k = CRASH_BLEED_K)

        self.t += DT
        self.step_count += 1

        progress = self.ego_car.state.s - self._prev_s
        self._prev_s = self.ego_car.state.s
        reward = PROGRESS_REWARD_SCALE * progress

        collided = self._ego_collided()
        rolled_over = self._ego_rolled_over()
        terminated = collided or rolled_over

        truncated = False
        if not terminated and self.step_count >= MAX_STEPS:
            truncated = True
            reward += SURVIVAL_REWARD

        info = {"collided": collided, "rolled_over": rolled_over, "s": self.ego_car.state.s}
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
