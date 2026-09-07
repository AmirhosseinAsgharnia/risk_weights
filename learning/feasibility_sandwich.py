"""
Sandwich scenario family for the feasibility-surrogate pipeline (see
learning.feasibility_common for shared plumbing, docs/feasibility_pipeline.md
for the pipeline overview).

Six agents, fixed semantic order [front_1, front_2, rear_1, rear_2,
blocker_1, blocker_2] (car_ids 0-5): ego starts boxed in by front_1 (ahead,
in ego's lane), rear_1 (behind, closing), and blocker_1/blocker_2 (adjacent
lane, defining an escape gap). Once ego's road position crosses a realized
trigger station s_stop, front_1 begins a *prescribed* emergency stop --
model.traffic_step.step_surr_agents' new (see that module) accel_override
parameter replaces front_1's IDM-computed acceleration with a fixed
emergency deceleration every step from the trigger onward, until it reaches
a standstill, where it is then held (never handed back to ordinary IDM).
rear_1/rear_2 remain ordinary IDM-following traffic throughout -- whether
they brake in time is genuine scenario dynamics, never prescribed.
"""

from dataclasses import dataclass

import numpy as np

from model.road.road import Road
from model.car.car import Car, CarState
from initialization.traffic_init import TrafficAgent, CAR_LENGTH

from learning.feasibility_common import (
    ArenaCommonConfig, ScenarioFamilyConfigError, scenario_id,
    place_relative, place_absolute, perturb, check_no_overlap,
)

# model.traffic_step doesn't export a module-level MAX_BRAKING constant of its own -- step_surr_agents'
# own default (8.0) is the same clamp value learning.env.MAX_BRAKING already mirrors for ego; reusing
# that existing constant here instead of a fresh, independently-invented one.
from learning.env import MAX_BRAKING as SURR_MAX_BRAKING, DT as _DT

FAMILY = "sandwich"
SLOT_NAMES = ("front_1", "front_2", "rear_1", "rear_2", "blocker_1", "blocker_2")

# -- Fixed secondary-actor formation, same convention/constants as feasibility_cutin.py.
FRONT2_EXTRA_GAP_M = 30.0
REAR2_EXTRA_GAP_M = 30.0
SECONDARY_RELATIVE_SPEED_MPS = 0.0

S_STOP_MIN_M = 100.0
S_STOP_MAX_M = 380.0   # same validated event-location domain as the cut-in family

_PCT_PERTURB = 0.05
_SPEED_PERTURB_MPS = 0.5
_GAP_CENTER_PERTURB_M = 1.0   # blocker-gap center, per spec
_EVENT_PERTURB_M = 2.0

THETA_BOUNDS_SANDWICH: dict[str, tuple[float, float]] = {
    "road_mu": (0.4, 1.0),
    "road_kappa_max": (0.0, 0.02),
    "s_stop_m": (S_STOP_MIN_M, S_STOP_MAX_M),
    "front_gap_m": (8.0, 60.0),
    "front_relative_speed_mps": (-12.0, 5.0),
    "rear_gap_m": (8.0, 60.0),
    "rear_relative_speed_mps": (-10.0, 12.0),
    "escape_gap_center_m": (-20.0, 30.0),
    "escape_gap_length_m": (8.0, 40.0),
    "blocker_relative_speed_mps": (-10.0, 10.0),
}
THETA_FIELDS_SANDWICH = tuple(THETA_BOUNDS_SANDWICH)


@dataclass(frozen=True)
class SandwichConfig(ArenaCommonConfig):
    """theta_sandwich (10 fields, see THETA_BOUNDS_SANDWICH) + sandwich-
    specific experiment settings. All theta fields have defaults, so
    SandwichConfig() alone is already a valid, usable scenario.

    s_stop_m:            nominal ego road station that triggers front_1's
                          prescribed emergency stop.
    front_gap_m/front_relative_speed_mps:
                          front_1's initial gap/relative speed to ego.
    rear_gap_m/rear_relative_speed_mps:
                          rear_1's initial gap/relative speed to ego
                          (rear_1 approaches -- ordinary IDM decides whether
                          it brakes in time, never prescribed).
    escape_gap_center_m/escape_gap_length_m/blocker_relative_speed_mps:
                          same escape-gap geometry as the base arena spec:
                          blocker_1/blocker_2 straddle a usable opening of
                          length escape_gap_length_m centred escape_gap_
                          center_m from ego, both at ego's speed +
                          blocker_relative_speed_mps.
    emergency_deceleration_mps2:
                          front_1's prescribed deceleration once triggered
                          -- an experiment setting, not theta. Defaults to
                          -8.0, i.e. exactly model.traffic_step.MAX_BRAKING
                          (the same clamp already shared by every surr IDM
                          car) rather than a new, independent limit.
    """
    road_mu: float = 1.0
    road_kappa_max: float = 0.005
    s_stop_m: float = 240.0
    front_gap_m: float = 30.0
    front_relative_speed_mps: float = 0.0
    rear_gap_m: float = 20.0
    rear_relative_speed_mps: float = 2.0
    escape_gap_center_m: float = 0.0
    escape_gap_length_m: float = 20.0
    blocker_relative_speed_mps: float = 0.0
    emergency_deceleration_mps2: float = -SURR_MAX_BRAKING

    def theta(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in THETA_FIELDS_SANDWICH}

    def scenario_id(self) -> str:
        return scenario_id(FAMILY, self.blocker_side, self.mode, self.theta())


def validate_sandwich(cfg: SandwichConfig, lane_num: int, ego_lane: int) -> None:
    for name, (lo, hi) in THETA_BOUNDS_SANDWICH.items():
        v = getattr(cfg, name)
        if not (lo <= v <= hi):
            raise ScenarioFamilyConfigError(f"{name}={v!r} outside validated bounds [{lo}, {hi}]")
    if cfg.blocker_side not in (-1, 1):
        raise ScenarioFamilyConfigError(f"blocker_side must be -1 or +1, got {cfg.blocker_side!r}")
    if not (-SURR_MAX_BRAKING <= cfg.emergency_deceleration_mps2 < 0):
        raise ScenarioFamilyConfigError(
            f"emergency_deceleration_mps2={cfg.emergency_deceleration_mps2!r} must be in "
            f"[-{SURR_MAX_BRAKING}, 0) -- {SURR_MAX_BRAKING} m/s^2 is the same clamp every surr "
            f"IDM car is already limited to (model.traffic_step.MAX_BRAKING), not a new bound "
            f"invented here.")

    blocker_lane = ego_lane + cfg.blocker_side
    if not (0 <= blocker_lane < lane_num):
        raise ScenarioFamilyConfigError(
            f"blocker_side={cfg.blocker_side} from ego_lane={ego_lane} implies lane {blocker_lane}, "
            f"outside [0, {lane_num}) for this {lane_num}-lane road.")

    max_realized_s_stop = cfg.s_stop_m + (0.0 if cfg.mode == "exact" else _EVENT_PERTURB_M)
    if max_realized_s_stop + 50.0 > cfg.goal_distance_m + 50.0:
        raise ScenarioFamilyConfigError(
            f"s_stop_m={cfg.s_stop_m} leaves too little road before the goal "
            f"(EGO_S0 + goal_distance_m={cfg.goal_distance_m}) to observe the stop and any recovery.")


def _build_realized(cfg: SandwichConfig, ego_s: float, ego_lane: int, ego_v0: float,
                     agents: list[TrafficAgent], s_stop_realized: float,
                     worker_rank: int, episode_index: int, scenario_seed: int | None) -> dict:
    def car_record(agent: TrafficAgent, role: str) -> dict:
        s = agent.car.state
        return {"id": int(agent.car.car_id), "role": role, "s": float(s.s), "lane": int(s.lane),
                "v_x": float(s.v_x)}

    return {
        "family": FAMILY,
        "ego": {"s": float(ego_s), "lane": int(ego_lane), "v_x": float(ego_v0)},
        "agents": [car_record(a, role) for a, role in zip(agents, SLOT_NAMES)],
        "s_stop_nominal_m": float(cfg.s_stop_m),
        "s_stop_realized_m": float(s_stop_realized),
        "sandwich_config": {"blocker_side": cfg.blocker_side, "mode": cfg.mode, **cfg.theta()},
        "scenario_id": cfg.scenario_id(),
        "seed_info": {"base_seed": int(cfg.seed), "worker_rank": int(worker_rank),
                      "episode_index": int(episode_index),
                      "scenario_seed": int(scenario_seed) if scenario_seed is not None else None},
    }


def generate_sandwich_arena(
        cfg: SandwichConfig, road: Road, ego_s: float, ego_lane: int, rng: np.random.Generator,
        *, worker_rank: int = 0, episode_index: int = 0, scenario_seed: int | None = None,
) -> tuple[list[TrafficAgent], float, dict]:
    """Realize a SandwichConfig into exactly 6 TrafficAgents, fixed
    [front_1, front_2, rear_1, rear_2, blocker_1, blocker_2] order (car_ids
    0-5), ego's own initial speed, and a JSON-safe realized-scenario record
    -- same rng/ego_v0-drawn-here contract as
    feasibility_cutin.generate_cutin_arena."""
    validate_sandwich(cfg, lane_num=road.lane_num, ego_lane=ego_lane)

    ego_v0 = perturb(rng, cfg.nominal_ego_speed_mps, _SPEED_PERTURB_MPS, cfg.mode)
    g_f = perturb(rng, cfg.front_gap_m, cfg.front_gap_m * _PCT_PERTURB, cfg.mode)
    dv_f = perturb(rng, cfg.front_relative_speed_mps, _SPEED_PERTURB_MPS, cfg.mode)
    g_r = perturb(rng, cfg.rear_gap_m, cfg.rear_gap_m * _PCT_PERTURB, cfg.mode)
    dv_r = perturb(rng, cfg.rear_relative_speed_mps, _SPEED_PERTURB_MPS, cfg.mode)
    gap_center = perturb(rng, cfg.escape_gap_center_m, _GAP_CENTER_PERTURB_M, cfg.mode)
    gap_length = perturb(rng, cfg.escape_gap_length_m, cfg.escape_gap_length_m * _PCT_PERTURB, cfg.mode)
    dv_b = perturb(rng, cfg.blocker_relative_speed_mps, _SPEED_PERTURB_MPS, cfg.mode)
    s_stop_realized = perturb(rng, cfg.s_stop_m, _EVENT_PERTURB_M, cfg.mode)

    blocker_lane = ego_lane + cfg.blocker_side

    front_1 = place_relative(ego_s, ego_lane, ego_v0, g_f, dv_f, ego_lane, car_id=0)
    front_2 = place_relative(ego_s, ego_lane, ego_v0, g_f + FRONT2_EXTRA_GAP_M, dv_f, ego_lane, car_id=1)
    rear_1 = place_relative(ego_s, ego_lane, ego_v0, -g_r, dv_r, ego_lane, car_id=2)
    rear_2 = place_relative(ego_s, ego_lane, ego_v0, -(g_r + REAR2_EXTRA_GAP_M),
                             SECONDARY_RELATIVE_SPEED_MPS, ego_lane, car_id=3)

    # Symmetric span around ego_s + gap_center -- place_absolute, not place_relative: the escape
    # gap's center can itself be ahead of OR behind ego (escape_gap_center_m may be negative), so
    # there is no single "gap from ego" sign to hand place_relative's asymmetric +/-CAR_LENGTH
    # convention (see that function's docstring); computing both absolute endpoints directly is
    # what keeps blocker_2.s - blocker_1.s - CAR_LENGTH exactly equal to gap_length regardless.
    half_span = (gap_length + CAR_LENGTH) / 2.0
    blocker_v0 = ego_v0 + dv_b
    blocker_1 = place_absolute(ego_s + gap_center - half_span, blocker_v0, blocker_lane, car_id=4)
    blocker_2 = place_absolute(ego_s + gap_center + half_span, blocker_v0, blocker_lane, car_id=5)

    cars = [front_1, front_2, rear_1, rear_2, blocker_1, blocker_2]
    check_no_overlap(cars)

    agents = [TrafficAgent(car=c, v0=c.state.v_x, target_lane=c.state.lane) for c in cars]
    realized = _build_realized(cfg, ego_s, ego_lane, ego_v0, agents, s_stop_realized,
                                worker_rank, episode_index, scenario_seed)
    return agents, ego_v0, realized


class SandwichRuntime:
    """Implements learning.feasibility_common.ArenaRuntime for the sandwich
    family -- see CutinRuntime's own docstring for the general contract."""

    STOPPER_CAR_ID = 0   # front_1 -- see SLOT_NAMES

    def __init__(self, cfg: SandwichConfig):
        self.cfg = cfg
        self.goal_distance_m = cfg.goal_distance_m
        self.max_episode_seconds = cfg.max_episode_seconds
        self.min_progress_m = cfg.min_progress_m
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self.triggered = False
        self.trigger_time_s: float | None = None
        self.trigger_ego_s: float | None = None
        self.front_speed_at_trigger: float | None = None
        self.stopped = False
        self.stopping_time_s: float | None = None
        self.stopping_distance_m: float | None = None
        self._s_at_trigger: float | None = None
        self._realized: dict | None = None

    def reset(self, road: Road, ego_s: float, ego_lane: int, rng: np.random.Generator,
              *, worker_rank: int, episode_index: int,
              scenario_seed: int | None) -> tuple[list[TrafficAgent], float, dict]:
        self._reset_episode_state()
        agents, ego_v0, realized = generate_sandwich_arena(
            self.cfg, road, ego_s, ego_lane, rng,
            worker_rank=worker_rank, episode_index=episode_index, scenario_seed=scenario_seed)
        self._realized = realized
        self._s_stop_realized = realized["s_stop_realized_m"]
        return agents, ego_v0, realized

    def pre_surr_step(self, t: float, ego_state: CarState, agents: list[TrafficAgent]) -> dict[int, float] | None:
        stopper = next(a for a in agents if a.car.car_id == self.STOPPER_CAR_ID)

        if not self.triggered:
            if ego_state.s >= self._s_stop_realized:
                self.triggered = True
                self.trigger_time_s = t
                self.trigger_ego_s = ego_state.s
                self.front_speed_at_trigger = stopper.car.state.v_x
                self._s_at_trigger = stopper.car.state.s

        if not self.triggered or stopper.crashed:
            return None   # not yet triggered, or already resolved by a plastic collision -- leave IDM alone

        if self.stopped:
            # Held at exactly 0 forever after -- see below for why this never drifts negative.
            return {self.STOPPER_CAR_ID: 0.0}

        v_x = stopper.car.state.v_x
        # Clamp the commanded decel so this step brings v_x to exactly 0, never past it: a plain
        # constant emergency_deceleration_mps2 would overshoot into reverse on whichever single step
        # crosses zero (v_x's own ODE here is dv_x/dt = accel exactly -- no steering/yaw coupling
        # for a straight-driving car with delta=0 throughout -- so RK4 integrates a constant accel
        # exactly, meaning this clamp lands on precisely 0.0, not just approximately).
        accel = max(self.cfg.emergency_deceleration_mps2, -v_x / _DT)
        if v_x + accel * _DT <= 1e-9:
            self.stopped = True
            self.stopping_time_s = t
            self.stopping_distance_m = stopper.car.state.s - self._s_at_trigger
        return {self.STOPPER_CAR_ID: accel}

    def extra_info(self) -> dict:
        return {
            "family": FAMILY,
            "event_triggered": self.triggered,
            "event_trigger_time_s": self.trigger_time_s,
            "event_trigger_ego_s": self.trigger_ego_s,
            "front_speed_at_trigger": self.front_speed_at_trigger,
            "front_stopped": self.stopped,
            "front_stopping_time_s": self.stopping_time_s,
            "front_stopping_distance_m": self.stopping_distance_m,
        }
