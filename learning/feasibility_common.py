"""
Shared plumbing for the feasibility-surrogate scenario families (cutin,
sandwich) -- see learning.feasibility_cutin / learning.feasibility_sandwich
for the families themselves, and docs/feasibility_pipeline.md for the
overall pipeline this supports.

Kept deliberately small and dependency-light, mirroring learning.scenario's
own "compact parameters in, TrafficAgents out" separation: this module only
owns what genuinely doesn't belong to one family -- common experiment-setting
fields, road-segment stratification (read off Road, never recomputed), and
the scenario-ID hasher both families use identically.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from model.car.car import Car, CarState
from model.car.config import VehicleParameters
from model.road.road import Road
from initialization.traffic_init import TrafficAgent, CAR_LENGTH

ScenarioFamily = Literal["cutin", "sandwich"]
BlockerSide = Literal[-1, 1]
RealizationMode = Literal["exact", "robust"]

SCHEMA_VERSION = 1   # bump whenever a family's theta schema or scenario-ID payload shape changes
MODERATE_BEHAVIOUR = 2   # every actor in both feasibility families is "moderate" -- see each family's module docstring


class ScenarioFamilyConfigError(ValueError):
    """An invalid CutinConfig/SandwichConfig, or the ego_s/ego_lane/road it's
    being realized against -- mirrors learning.scenario.ScenarioConfigError's
    role for the compact 15-car scenario path. Raised, never silently
    corrected."""


@dataclass(frozen=True)
class ArenaCommonConfig:
    """Fields every scenario family needs verbatim -- not part of either
    family's own theta (see each module's THETA_BOUNDS_*), but required to
    fully specify one realizable episode.

    blocker_side:       -1 (left) or +1 (right) relative to ego's lane --
                         treated as two separate experimental strata, never
                         as a continuous axis (see module docstrings).
    seed:                base scenario seed -- combined with worker_rank/
                         episode_index via learning.scenario.derive_seed,
                         exactly like ScenarioConfig.seed.
    mode:                "exact" reconstructs the nominal theta with every
                         perturbation forced to 0 (including the event-
                         station epsilon) -- "can PPO solve this precise
                         scenario?". "robust" applies the small seeded
                         nuisance perturbations each family documents --
                         "can PPO solve a local distribution around it?".
    goal_distance_m/max_episode_seconds/min_progress_m:
                         same fields/semantics as ScenarioConfig's (see
                         learning.scenario) -- EgoTrafficEnv reads these
                         identically regardless of which config type is
                         active (see EgoTrafficEnv.reset()).
    """
    blocker_side: BlockerSide = 1
    seed: int = 0
    mode: RealizationMode = "robust"
    goal_distance_m: float = 400.0
    max_episode_seconds: float = 35.0
    min_progress_m: float = 100.0
    nominal_ego_speed_mps: float = 20.0   # experiment setting, not theta -- every actor's speed in
                                           # both families is expressed *relative* to ego's own (see
                                           # each family's THETA_BOUNDS_*), so ego's own absolute
                                           # speed has to come from somewhere; matches learning.env's
                                           # own NOMINAL_SPEED. Small-perturbed in "robust" mode like
                                           # every other realized quantity (see perturb()).


class RoadSegment:
    """The five road segments a road built from Road.kappa_calc's own
    profile always has, in order -- see Road.length_calc/kappa_calc. Names
    only; boundaries are read off a real Road instance (road_segment_bounds
    below), never hard-coded, so they can never drift out of sync with the
    actual curvature profile."""
    PRE_CURVE_STRAIGHT = "pre_curve_straight"
    CLOTHOID_ENTRY = "clothoid_entry"
    CONSTANT_CURVE = "constant_curve"
    CLOTHOID_EXIT = "clothoid_exit"
    POST_CURVE_STRAIGHT = "post_curve_straight"


def road_segment_bounds(road: Road) -> list[tuple[str, float, float]]:
    """[(segment_name, s_lo, s_hi), ...] in order, read directly off this
    Road instance's own L_enter/L_clothoid/L_curve (set in Road.length_calc)
    -- the exact boundaries Road.kappa_calc itself uses to build kappa(s),
    not a re-derivation. Used for road-segment-stratified sampling of event
    locations (see each family's initial-design sampler)."""
    L_enter, L_clothoid, L_curve = road.L_enter, road.L_clothoid, road.L_curve
    return [
        (RoadSegment.PRE_CURVE_STRAIGHT, 0.0, L_enter),
        (RoadSegment.CLOTHOID_ENTRY, L_enter, L_enter + L_clothoid),
        (RoadSegment.CONSTANT_CURVE, L_enter + L_clothoid, L_enter + L_clothoid + L_curve),
        (RoadSegment.CLOTHOID_EXIT, L_enter + L_clothoid + L_curve, L_enter + 2 * L_clothoid + L_curve),
        (RoadSegment.POST_CURVE_STRAIGHT, L_enter + 2 * L_clothoid + L_curve, road.s_max),
    ]


def road_segment_at(road: Road, s: float) -> str:
    """Which RoadSegment a given arclength s falls in, per road_segment_bounds."""
    for name, lo, hi in road_segment_bounds(road):
        if lo <= s < hi or (name == RoadSegment.POST_CURVE_STRAIGHT and s >= lo):
            return name
    raise ScenarioFamilyConfigError(f"s={s!r} is outside the road (0, {road.s_max}]")


def scenario_id(
        family: ScenarioFamily,
        blocker_side: BlockerSide,
        mode: RealizationMode,
        theta: dict[str, float],
        *,
        schema_version: int = SCHEMA_VERSION,
) -> str:
    """A stable, collision-resistant scenario ID from a canonical
    serialization of everything that defines a nominal scenario (theta only
    -- NOT the realization seed, which varies within one theta's rollouts).

    Deliberately does NOT use fixed-precision float formatting (e.g.
    f"{x:.6g}") -- that would silently collide two meaningfully different
    theta values that happen to round the same way. json.dumps's default
    float formatting is Python's float.__repr__, the shortest decimal string
    that round-trips back to the exact same float -- exact and deterministic
    on a given platform, with no precision loss or quantization decision
    made here.
    """
    payload = {
        "schema_version": schema_version,
        "family": family,
        "blocker_side": blocker_side,
        "mode": mode,
        "theta": {k: theta[k] for k in sorted(theta)},
    }
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def place_relative(ego_s: float, ego_lane: int, ego_v0: float, gap: float, relative_speed: float,
                    lane: int, car_id: int) -> Car:
    """One IDM-following car placed `gap` [m] ahead of ego (bumper-to-bumper), same convention as
    learning.scenario._critical_positions -- gap may be negative for a car placed behind ego.
    Shared by every feasibility-family generator (front/rear/blocker are all "a car at some gap
    from ego", just with different gaps/lanes/roles)."""
    s = ego_s + gap + CAR_LENGTH if gap >= 0 else ego_s + gap - CAR_LENGTH
    v0 = ego_v0 + relative_speed
    state = CarState(s=s, e_y=0.0, e_psi=0.0, v_x=v0, lane=lane)
    return Car(car_id=car_id, state=state, vehicle_params=VehicleParameters(), behaviour=MODERATE_BEHAVIOUR)


def place_absolute(s: float, v_x: float, lane: int, car_id: int) -> Car:
    """One car at an already-computed absolute position/speed -- unlike place_relative (which adds
    a car-length asymmetrically depending on the sign of a *gap from one shared anchor*, correct for
    "one car, one gap from ego"), this is what a symmetric span computed from BOTH ends (e.g. an
    escape gap's two bounding blockers, straddling a center that can itself be ahead of or behind
    ego) needs -- see each family's blocker-pair placement for why place_relative alone double-counts
    CAR_LENGTH in that case."""
    state = CarState(s=s, e_y=0.0, e_psi=0.0, v_x=v_x, lane=lane)
    return Car(car_id=car_id, state=state, vehicle_params=VehicleParameters(), behaviour=MODERATE_BEHAVIOUR)


def perturb(rng: np.random.Generator, nominal: float, spread: float, mode: RealizationMode) -> float:
    """Small seeded nuisance perturbation -- 0 in "exact" mode (or when spread is 0), else
    Uniform(nominal - spread, nominal + spread) drawn from `rng` (never global RNG state)."""
    if mode == "exact" or spread == 0.0:
        return nominal
    return float(nominal + rng.uniform(-spread, spread))


def check_no_overlap(cars: list[Car]) -> None:
    """Raises ScenarioFamilyConfigError on the first same-lane overlapping pair found -- same
    AABB (same lane + |ds| < CAR_LENGTH) test learning.scenario.validate_scenario/model.collision
    use, shared so both families reject invalid initial geometry identically."""
    for i in range(len(cars)):
        for j in range(i + 1, len(cars)):
            a, b = cars[i].state, cars[j].state
            if a.lane == b.lane and abs(a.s - b.s) < CAR_LENGTH:
                raise ScenarioFamilyConfigError(
                    f"cars {cars[i].car_id} and {cars[j].car_id} overlap at spawn: same lane {a.lane}, "
                    f"|ds|={abs(a.s - b.s):.2f}m < CAR_LENGTH={CAR_LENGTH}m.")


class ArenaRuntime(Protocol):
    """Structural contract EgoTrafficEnv's arena-mode reset()/step() rely on
    -- CutinRuntime and SandwichRuntime (see their own modules) each
    implement this without inheriting from it (duck typing, matching this
    codebase's existing preference for plain dataclasses/functions over
    class hierarchies -- this Protocol exists purely for readability/typing,
    EgoTrafficEnv never isinstance-checks it).

    goal_distance_m / max_episode_seconds / min_progress_m: read once per
    reset() -- see ArenaCommonConfig's own fields, which every family's
    config embeds.
    """
    goal_distance_m: float
    max_episode_seconds: float
    min_progress_m: float

    def reset(
            self, road: Road, ego_s: float, ego_lane: int, rng,
            *, worker_rank: int, episode_index: int, scenario_seed: int | None,
    ) -> tuple[list[TrafficAgent], float, dict]:
        """Realize this family's 6 agents (fixed semantic order -- see the
        family's own module docstring) + ego's own initial speed (drawn
        from this runtime's own config -- see ArenaCommonConfig.
        nominal_ego_speed_mps -- exactly like learning.scenario.
        generate_scenario draws its own ego_v0, never passed in by the
        caller) + a JSON-safe realized-scenario record -- same return shape
        as generate_scenario. Also resets this runtime's own latched-event
        state for the new episode."""
        ...

    def pre_surr_step(self, t: float, ego_state: CarState, agents: list[TrafficAgent]) -> dict[int, float] | None:
        """Called once per EgoTrafficEnv.step(), after ego has been
        integrated for this step but before model.traffic_step.
        step_surr_agents runs. May mutate `agents` in place (e.g. a cut-in's
        prescribed lane-change assignment) and/or return a {car_id: accel}
        override dict for step_surr_agents' own accel_override parameter
        (e.g. a sandwich's prescribed emergency stop). Returns None if
        nothing needs overriding this step."""
        ...

    def extra_info(self) -> dict:
        """Family-specific diagnostics for this step's info dict (event
        trigger station/time, phase classification, etc.) -- merged in
        alongside EgoTrafficEnv's own generic keys, under an
        "arena"-namespaced sub-dict so it can never collide with an existing
        key."""
        ...
