"""
Cut-in scenario family for the feasibility-surrogate pipeline (see
learning.feasibility_common for shared plumbing, docs/feasibility_pipeline.md
for the pipeline overview).

Six agents, fixed semantic order [front_1, front_2, rear_1, rear_2,
blocker_1, blocker_2] (car_ids 0-5): blocker_1 starts in the lane adjacent to
ego (ego_lane + blocker_side) and, once ego's road position crosses a
realized (nominal + small seeded perturbation) trigger station s_cutin,
performs a *prescribed* continuous lane change into ego's lane -- not a
MOBIL decision (discretionary MOBIL is disabled for every agent in this
family, see generate_cutin_arena). blocker_2 stays in the adjacent lane
throughout, used only to size how much room blocker_1 had in its own lane
before merging (see THETA_BOUNDS_CUTIN's escape_gap_length_m). front_1/
front_2/rear_1/rear_2 are ordinary IDM-following traffic in ego's own lane,
in a fixed, documented formation -- not surrogate inputs (see
CutinConfig's own docstring).

The prescribed lane change reuses model.traffic_step.step_surr_agents' own
MOBIL/far-near state machine unmodified: setting a TrafficAgent's
target_lane/lane_change_t0 directly (both already public fields) is exactly
the state a normal MOBIL-triggered change already produces, so the existing
far-near ramp + eventual discrete-lane commit (step_surr_agents' step 5)
picks it up with no changes to that module.
"""

from dataclasses import dataclass

import numpy as np

from model.road.road import Road
from model.car.car import Car, CarState
from controllers.far_near import FAR_NEAR_PRESETS
from initialization.traffic_init import TrafficAgent, CAR_LENGTH

from learning.feasibility_common import (
    ArenaCommonConfig, ScenarioFamilyConfigError, MODERATE_BEHAVIOUR,
    place_relative, perturb, check_no_overlap, scenario_id,
)

FAMILY = "cutin"
N_ARENA_SURR = 6
SLOT_NAMES = ("front_1", "front_2", "rear_1", "rear_2", "blocker_1", "blocker_2")

# -- Fixed secondary-actor formation (documented constants, NOT theta -- see
# CutinConfig's own docstring for why): front_2 a further gap ahead of
# front_1, rear_1 a fixed default gap behind ego (mirrors
# learning.scenario.ScenarioConfig.rear's own default of 20m), rear_2 a
# further gap behind rear_1.
FRONT2_EXTRA_GAP_M = 30.0
REAR1_GAP_M = 20.0
REAR2_EXTRA_GAP_M = 30.0
SECONDARY_RELATIVE_SPEED_MPS = 0.0   # front_2/rear_1/rear_2 nominal speed == ego's own

# Validated event-location domain -- see feasibility_common.road_segment_bounds for the
# five named segments this range spans on the project's standard 500m road.
S_CUTIN_MIN_M = 100.0
S_CUTIN_MAX_M = 380.0

# Small seeded nuisance perturbations (robust mode only; exact mode forces all of these to 0).
_PCT_PERTURB = 0.05       # +/-5% on critical longitudinal positions/gaps
_SPEED_PERTURB_MPS = 0.5  # +/-0.5 m/s on critical relative speeds
_EVENT_PERTURB_M = 2.0    # +/-2m on the realized event station (s_cutin)

THETA_BOUNDS_CUTIN: dict[str, tuple[float, float]] = {
    "road_mu": (0.4, 1.0),
    "road_kappa_max": (0.0, 0.02),
    "s_cutin_m": (S_CUTIN_MIN_M, S_CUTIN_MAX_M),
    "cutter_gap_m": (8.0, 60.0),
    "cutter_relative_speed_mps": (-12.0, 5.0),
    "front_gap_m": (8.0, 60.0),
    "front_relative_speed_mps": (-12.0, 5.0),
    "escape_gap_length_m": (8.0, 40.0),
}
THETA_FIELDS_CUTIN = tuple(THETA_BOUNDS_CUTIN)


@dataclass(frozen=True)
class CutinConfig(ArenaCommonConfig):
    """theta_cutin (8 fields, see THETA_BOUNDS_CUTIN) + cutin-specific
    experiment settings. All theta fields have defaults so CutinConfig()
    alone is already a valid, usable scenario -- same convention as
    learning.scenario.ScenarioConfig.

    s_cutin_m:                  nominal ego road station that triggers the
                                 cut-in (part of theta -- see feasibility_
                                 common.road_segment_bounds for the named
                                 segments this can fall in).
    cutter_gap_m/cutter_relative_speed_mps:
                                 blocker_1 (the cutter)'s initial bumper gap
                                 ahead of ego and speed relative to ego,
                                 while still in the adjacent lane -- same
                                 gap convention as front_gap_m (positive =
                                 ahead of ego).
    front_gap_m/front_relative_speed_mps:
                                 front_1's initial gap/relative speed.
    escape_gap_length_m:        usable bumper-to-bumper spacing from
                                 blocker_1 to blocker_2 (blocker_2 placed
                                 that far ahead of blocker_1, same lane) --
                                 how much room the cutter had in its own
                                 lane before merging.
    cutin_duration_s:           wall-clock duration of the prescribed lane
                                 change -- an experiment setting, not theta
                                 (see module docstring). Defaults to exactly
                                 FAR_NEAR_PRESETS[moderate].lane_change_duration
                                 (4.0s) so the default produces zero special
                                 handling; overriding it re-times the
                                 existing linear progress ramp (see
                                 _prescribe_lane_change) without touching
                                 model.traffic_step.
    """
    road_mu: float = 1.0
    road_kappa_max: float = 0.005
    s_cutin_m: float = 240.0
    cutter_gap_m: float = 20.0
    cutter_relative_speed_mps: float = 0.0
    front_gap_m: float = 30.0
    front_relative_speed_mps: float = 0.0
    escape_gap_length_m: float = 20.0
    cutin_duration_s: float = FAR_NEAR_PRESETS[MODERATE_BEHAVIOUR].lane_change_duration

    def theta(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in THETA_FIELDS_CUTIN}

    def scenario_id(self) -> str:
        return scenario_id(FAMILY, self.blocker_side, self.mode, self.theta())


def validate_cutin(cfg: CutinConfig, lane_num: int, ego_lane: int) -> None:
    """Raises ScenarioFamilyConfigError on the first problem found -- mirrors
    learning.scenario.validate_scenario's style (never silently corrects)."""
    for name, (lo, hi) in THETA_BOUNDS_CUTIN.items():
        v = getattr(cfg, name)
        if not (lo <= v <= hi):
            raise ScenarioFamilyConfigError(f"{name}={v!r} outside validated bounds [{lo}, {hi}]")
    if cfg.cutin_duration_s <= 0:
        raise ScenarioFamilyConfigError(f"cutin_duration_s must be > 0, got {cfg.cutin_duration_s!r}")
    if cfg.blocker_side not in (-1, 1):
        raise ScenarioFamilyConfigError(f"blocker_side must be -1 or +1, got {cfg.blocker_side!r}")

    blocker_lane = ego_lane + cfg.blocker_side
    if not (0 <= blocker_lane < lane_num):
        raise ScenarioFamilyConfigError(
            f"blocker_side={cfg.blocker_side} from ego_lane={ego_lane} implies lane {blocker_lane}, "
            f"outside [0, {lane_num}) for this {lane_num}-lane road.")

    # Enough road must remain after the (perturbed) trigger to observe the maneuver + some recovery.
    max_realized_s_cutin = cfg.s_cutin_m + (0.0 if cfg.mode == "exact" else _EVENT_PERTURB_M)
    if max_realized_s_cutin + cfg.cutin_duration_s * 5.0 > cfg.goal_distance_m + 50.0:
        # 50.0m: same order-of-magnitude slack used elsewhere in this codebase's own generators
        # (e.g. learning.scenario's ego_gap_multiplier berth) -- not a hard physical limit, just
        # "obviously not enough road left to see the maneuver play out and recover."
        raise ScenarioFamilyConfigError(
            f"s_cutin_m={cfg.s_cutin_m} leaves too little road before the goal "
            f"(EGO_S0 + goal_distance_m={cfg.goal_distance_m}) to observe the cut-in and any recovery.")


def _build_realized(cfg: CutinConfig, ego_s: float, ego_lane: int, ego_v0: float,
                     agents: list[TrafficAgent], s_cutin_realized: float,
                     worker_rank: int, episode_index: int, scenario_seed: int | None) -> dict:
    def car_record(agent: TrafficAgent, role: str) -> dict:
        s = agent.car.state
        return {"id": int(agent.car.car_id), "role": role, "s": float(s.s), "lane": int(s.lane),
                "v_x": float(s.v_x)}

    return {
        "family": FAMILY,
        "ego": {"s": float(ego_s), "lane": int(ego_lane), "v_x": float(ego_v0)},
        "agents": [car_record(a, role) for a, role in zip(agents, SLOT_NAMES)],
        "s_cutin_nominal_m": float(cfg.s_cutin_m),
        "s_cutin_realized_m": float(s_cutin_realized),
        "cutin_config": {"blocker_side": cfg.blocker_side, "mode": cfg.mode, **cfg.theta()},
        "scenario_id": cfg.scenario_id(),
        "seed_info": {"base_seed": int(cfg.seed), "worker_rank": int(worker_rank),
                      "episode_index": int(episode_index),
                      "scenario_seed": int(scenario_seed) if scenario_seed is not None else None},
    }


def generate_cutin_arena(
        cfg: CutinConfig, road: Road, ego_s: float, ego_lane: int, rng: np.random.Generator,
        *, worker_rank: int = 0, episode_index: int = 0, scenario_seed: int | None = None,
) -> tuple[list[TrafficAgent], float, dict]:
    """Realize a CutinConfig into exactly 6 TrafficAgents, in the fixed
    [front_1, front_2, rear_1, rear_2, blocker_1, blocker_2] order (car_ids
    0-5), ego's own initial speed, and a JSON-safe realized-scenario record.
    `rng` is the only source of randomness (small seeded perturbations only
    -- see perturb()), always a np.random.Generator the caller derived
    deterministically, never global RNG state. ego_v0 is drawn here (from
    cfg.nominal_ego_speed_mps -- see ArenaCommonConfig), same role as
    learning.scenario.generate_scenario's own ego_v0 draw -- every other
    actor's speed is relative to it."""
    validate_cutin(cfg, lane_num=road.lane_num, ego_lane=ego_lane)

    ego_v0 = perturb(rng, cfg.nominal_ego_speed_mps, _SPEED_PERTURB_MPS, cfg.mode)
    g_f = perturb(rng, cfg.front_gap_m, cfg.front_gap_m * _PCT_PERTURB, cfg.mode)
    dv_f = perturb(rng, cfg.front_relative_speed_mps, _SPEED_PERTURB_MPS, cfg.mode)
    g_c = perturb(rng, cfg.cutter_gap_m, cfg.cutter_gap_m * _PCT_PERTURB, cfg.mode)
    dv_c = perturb(rng, cfg.cutter_relative_speed_mps, _SPEED_PERTURB_MPS, cfg.mode)
    l_escape = perturb(rng, cfg.escape_gap_length_m, cfg.escape_gap_length_m * _PCT_PERTURB, cfg.mode)
    s_cutin_realized = perturb(rng, cfg.s_cutin_m, _EVENT_PERTURB_M, cfg.mode)

    blocker_lane = ego_lane + cfg.blocker_side

    front_1 = place_relative(ego_s, ego_lane, ego_v0, g_f, dv_f, ego_lane, car_id=0)
    front_2 = place_relative(ego_s, ego_lane, ego_v0, g_f + FRONT2_EXTRA_GAP_M, dv_f, ego_lane, car_id=1)
    rear_1 = place_relative(ego_s, ego_lane, ego_v0, -REAR1_GAP_M, SECONDARY_RELATIVE_SPEED_MPS, ego_lane, car_id=2)
    rear_2 = place_relative(ego_s, ego_lane, ego_v0, -(REAR1_GAP_M + REAR2_EXTRA_GAP_M),
                             SECONDARY_RELATIVE_SPEED_MPS, ego_lane, car_id=3)
    blocker_1 = place_relative(ego_s, ego_lane, ego_v0, g_c, dv_c, blocker_lane, car_id=4)
    # blocker_2 is anchored off blocker_1 (l_escape is the bumper-to-bumper gap BETWEEN the two
    # blockers, not a second ego-relative gap -- see place_relative's own docstring for why
    # anchoring both ends off ego independently would double-count CAR_LENGTH here).
    blocker_2 = place_relative(blocker_1.state.s, blocker_lane, blocker_1.state.v_x, l_escape, 0.0,
                                blocker_lane, car_id=5)

    cars = [front_1, front_2, rear_1, rear_2, blocker_1, blocker_2]
    check_no_overlap(cars)

    agents = [TrafficAgent(car=c, v0=c.state.v_x, target_lane=c.state.lane) for c in cars]
    realized = _build_realized(cfg, ego_s, ego_lane, ego_v0, agents, s_cutin_realized,
                                worker_rank, episode_index, scenario_seed)
    realized["s_cutin_realized_m"] = s_cutin_realized
    return agents, ego_v0, realized


def _prescribe_lane_change(agent: TrafficAgent, target_lane: int, t_trigger: float, duration_s: float) -> None:
    """Commands `agent` to begin a continuous lane change into target_lane,
    reusing model.traffic_step.step_surr_agents' own far-near ramp +
    discrete-commit machinery unmodified (see module docstring) -- this
    function only ever sets the same public TrafficAgent fields a normal
    MOBIL-triggered change already sets.

    lane_change_t0 is backdated/postdated so that the existing linear
    progress formula (progress = (t - lane_change_t0) / fn_p.
    lane_change_duration, see step_surr_agents step 3) reaches 1.0 exactly
    duration_s after the real trigger time t_trigger, without needing
    duration_s to equal this agent's behaviour-class preset value. e_y
    itself is still continuously *integrated* by CarDynamics regardless
    (this only re-times the far-near steering *target*, itself still rate-
    limited by clip_steering_rate) -- so this never teleports position, and
    is a no-op exactly when duration_s already equals the preset (the
    default -- see CutinConfig.cutin_duration_s)."""
    fn_p = FAR_NEAR_PRESETS[agent.car.behaviour]
    agent.target_lane = target_lane
    agent.lane_change_t0 = t_trigger + duration_s - fn_p.lane_change_duration


class CutinRuntime:
    """Implements learning.feasibility_common.ArenaRuntime for the cut-in
    family -- owns one episode's latched trigger state and diagnostics.
    A fresh instance (or .reset()) is required per episode; EgoTrafficEnv
    calls .reset() every reset() and .pre_surr_step()/.extra_info() every
    step()."""

    CUTTER_CAR_ID = 4   # blocker_1 -- see SLOT_NAMES
    CUTTER_ORIGINAL_LANE_OFFSET = None   # set in reset()

    def __init__(self, cfg: CutinConfig):
        self.cfg = cfg
        self.goal_distance_m = cfg.goal_distance_m
        self.max_episode_seconds = cfg.max_episode_seconds
        self.min_progress_m = cfg.min_progress_m
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self.triggered = False
        self.trigger_time_s: float | None = None
        self.trigger_ego_s: float | None = None
        self.cutter_valid_at_trigger: bool | None = None
        self.cutter_crashed_before_trigger = False
        self.completed = False
        self.completion_time_s: float | None = None
        self._was_mid_change = False
        self._realized: dict | None = None

    def reset(self, road: Road, ego_s: float, ego_lane: int, rng: np.random.Generator,
              *, worker_rank: int, episode_index: int,
              scenario_seed: int | None) -> tuple[list[TrafficAgent], float, dict]:
        self._reset_episode_state()
        agents, ego_v0, realized = generate_cutin_arena(
            self.cfg, road, ego_s, ego_lane, rng,
            worker_rank=worker_rank, episode_index=episode_index, scenario_seed=scenario_seed)
        self._realized = realized
        self._s_cutin_realized = realized["s_cutin_realized_m"]
        self._blocker_lane = ego_lane + self.cfg.blocker_side
        self._target_lane = ego_lane
        return agents, ego_v0, realized

    def pre_surr_step(self, t: float, ego_state: CarState, agents: list[TrafficAgent]) -> dict[int, float] | None:
        cutter = next(a for a in agents if a.car.car_id == self.CUTTER_CAR_ID)

        if not self.triggered:
            if cutter.crashed:
                self.cutter_crashed_before_trigger = True
            if ego_state.s >= self._s_cutin_realized:
                self.triggered = True
                self.trigger_time_s = t
                self.trigger_ego_s = ego_state.s
                self.cutter_valid_at_trigger = (not cutter.crashed) and (cutter.car.state.lane == self._blocker_lane)
                if self.cutter_valid_at_trigger:
                    _prescribe_lane_change(cutter, self._target_lane, t, self.cfg.cutin_duration_s)
        else:
            mid_change = cutter.lane_change_t0 is not None
            if self._was_mid_change and not mid_change and not self.completed:
                self.completed = True
                self.completion_time_s = t
            self._was_mid_change = mid_change

        return None   # cut-in needs no accel override, only the target_lane/lane_change_t0 mutation above

    def extra_info(self) -> dict:
        return {
            "family": FAMILY,
            "event_triggered": self.triggered,
            "event_trigger_time_s": self.trigger_time_s,
            "event_trigger_ego_s": self.trigger_ego_s,
            "cutter_valid_at_trigger": self.cutter_valid_at_trigger,
            "cutter_crashed_before_trigger": self.cutter_crashed_before_trigger,
            "cutin_completed": self.completed,
            "cutin_completion_time_s": self.completion_time_s,
        }
