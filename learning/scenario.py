"""
Compact, reproducible scenario parameterization for EgoTrafficEnv.

Kept deliberately independent of learning.env / Gymnasium: this module's
job is "compact parameters in, 15 fully-specified TrafficAgents (+ ego's
initial speed + a JSON-safe realized-scenario record) out". The Gym
environment (learning.env.EgoTrafficEnv) is the only thing that knows how
to *step* the result; PPO training/evaluation live above that. See
model.traffic_step / model.collision / controllers.idm / controllers.mobil
for the actual vehicle dynamics and collision/rollover logic -- none of
that is touched or reimplemented here.

theta = (theta_global, theta_critical, seed):
  theta_global:   ScenarioConfig's road_mu/road_kappa_max/background_*/
                  aggressive_fraction/lane_change_tendency/traffic_spread.
                  There is no separate ego_initial_speed: ego's speed is
                  drawn from the *same* distribution as the road's traffic
                  flow, Normal(background_mean_speed, background_speed_std)
                  -- ego is the road speed, not an independently-tunable
                  quantity -- see generate_scenario.
  theta_critical: ScenarioConfig.front/rear/blocker (CriticalActorConfig),
                  each specified *relative* to ego's (now stochastic) speed.
  seed:           ScenarioConfig.seed -- combined with a caller-supplied
                  worker_rank/episode_index via derive_seed() for the
                  actual per-episode background realization (see below).
"""

import math
import dataclasses
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from model.road.road import Road
from model.car.car import Car, CarState
from model.car.config import VehicleParameters
from controllers.idm import IDM_PRESETS
from initialization.traffic_init import TrafficAgent, steady_state_speed, CAR_LENGTH
from controllers.mobil import MobilParams

Behaviour = Literal["conservative", "moderate", "aggressive"]
_BEHAVIOUR_CODE: dict[Behaviour, int] = {"conservative": 1, "moderate": 2, "aggressive": 3}
_BEHAVIOUR_NAME: dict[int, Behaviour] = {v: k for k, v in _BEHAVIOUR_CODE.items()}

N_BACKGROUND = 12
N_CRITICAL = 3
N_SURR = N_BACKGROUND + N_CRITICAL   # 15 -- kept exactly, per the spec

_BACKGROUND_SPEED_MIN = 0.0    # [m/s] clip bounds for the background Normal draw --
_BACKGROUND_SPEED_MAX = 40.0   # "physically meaningful range" isn't pinned down further by the spec


class ScenarioConfigError(ValueError):
    """A ScenarioConfig (or the ego_s/ego_lane/road it's being realized
    against) is invalid -- e.g. a non-existent blocker lane, a non-
    positive gap, a bad behavior string, or a critical actor overlapping
    ego/another critical actor at spawn. Raised, never silently
    corrected -- see validate_scenario."""


@dataclass(frozen=True)
class CriticalActorConfig:
    """One of the three explicit critical actors (front/rear/blocker).
    Which fields are required depends on `role` -- see validate_scenario.

    role:           "front", "rear", or "blocker".
    behavior:       selects IDM_PRESETS/FAR_NEAR_PRESETS via _BEHAVIOUR_CODE.
    relative_speed: [m/s] v_actor - v_ego.
    gap:            [m] front/rear only -- bumper-to-bumper distance from
                    ego (front: ahead; rear: behind). Must be > 0.
    lane_offset:    blocker only -- -1 (adjacent left) or +1 (adjacent
                    right) relative to ego's initial lane.
    relative_s:     [m] blocker only -- signed longitudinal offset from
                    ego (positive = ahead).
    """
    role: Literal["front", "rear", "blocker"]
    behavior: Behaviour = "moderate"
    relative_speed: float = 0.0
    gap: float | None = None
    lane_offset: int | None = None
    relative_s: float | None = None


@dataclass(frozen=True)
class ScenarioConfig:
    """theta_global + theta_critical + seed for one scenario family (see
    module docstring). All fields have defaults so ScenarioConfig() alone
    is already a valid, usable scenario."""

    # -- global --
    road_mu: float = 1.0
    road_kappa_max: float = 0.005
    background_mean_speed: float = 20.0    # [m/s]
    background_speed_std: float = 2.5      # [m/s]
    aggressive_fraction: float = 0.2       # P(background car is aggressive); rest split evenly
                                            # conservative/moderate -- see generate_scenario docstring
    lane_change_tendency: float = 1.0      # multiplier: effective MOBIL threshold = base / tendency
                                            # (tendency > 1 => lower threshold => more lane changes)
    traffic_spread: float = 60.0           # [m] background cars placed within +/- this of ego_s -- 40
                                            # (initialization.traffic_init.generate_traffic's own default)
                                            # was tried first but empirically fails to fit 12 background
                                            # cars ~8% of the time when i.i.d. lane assignment happens to
                                            # crowd 7+ of them into one lane alongside ego's exclusion
                                            # zone; 60 had zero failures across 1000 test seeds
    seed: int = 0
    # No ego_initial_speed field: ego's speed is drawn from the same
    # Normal(background_mean_speed, background_speed_std) distribution as
    # the background traffic flow, not set independently -- see
    # generate_scenario. front/rear/blocker relative_speed is relative to
    # that drawn value.

    # -- fixed-scenario success goal (see EgoTrafficEnv's success/outcome
    # definition in learning.env) -- deliberately decoupled from the
    # modeled road's full length: the 500m road (ROAD_KWARGS) was never
    # calibrated against max_episode_seconds -- even a full nominal-speed
    # (20 m/s) cruise for the whole default 20s covers only 400m, short of
    # the 450m EGO_S0=50 would need to reach s_max=500 -- so treating "hit
    # s_max" as *this* scenario's goal silently demanded a ~25 m/s average
    # from the moment ego spawns, before ever accounting for front/rear/
    # blocker or the road's curve. These three fields let a scenario define
    # success as "cleared its own critical conflict and kept driving",
    # rather than "crossed a possibly-unreachable finish line".
    goal_distance_m: float = 150.0   # [m] past ego's spawn position (EGO_S0) that counts as this
                                      # scenario's goal being reached -- ASSUMPTION, not derived from a
                                      # real spec: chosen to clear where front/rear/blocker spawn (within
                                      # a few tens of meters of ego_s by construction, see
                                      # _critical_positions/traffic_spread) plus a settling margin, while
                                      # staying reachable well inside max_episode_seconds at realistic,
                                      # sub-nominal speed. Override per scenario -- e.g. set to
                                      # ROAD_KWARGS["s_max"] - EGO_S0 (learning.env) to require full-road
                                      # completion as before.
    max_episode_seconds: float = 20.0   # [s] wall-clock episode budget for this scenario -- overrides
                                         # learning.env.EPISODE_SECONDS when this ScenarioConfig is used.
    min_progress_m: float = 50.0   # [m] a timeout (reaching max_episode_seconds unharmed without
                                    # reaching goal_distance_m) only earns EgoTrafficEnv's SURVIVAL_REWARD
                                    # if net forward progress was at least this much -- otherwise idling
                                    # or crawling for the full episode earns nothing, so "stop and never
                                    # risk it" can't out-earn a genuine (if unsuccessful) attempt. Purely a
                                    # training-reward knob -- never affects `success` itself (see
                                    # EgoTrafficEnv.step's outcome/success separation). ASSUMPTION, not
                                    # derived from a spec.

    # -- critical actors --
    front: CriticalActorConfig = field(
        default_factory=lambda: CriticalActorConfig(role="front", gap=30.0))
    rear: CriticalActorConfig = field(
        default_factory=lambda: CriticalActorConfig(role="rear", gap=20.0))
    blocker: CriticalActorConfig = field(
        default_factory=lambda: CriticalActorConfig(role="blocker", lane_offset=1, relative_s=0.0))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioConfig":
        d = dict(d)
        front = CriticalActorConfig(**d.pop("front"))
        rear = CriticalActorConfig(**d.pop("rear"))
        blocker = CriticalActorConfig(**d.pop("blocker"))
        return cls(front=front, rear=rear, blocker=blocker, **d)


def derive_seed(base_seed: int, worker_rank: int, episode_index: int) -> int:
    """Deterministic, well-decorrelated per-(worker, episode) seed derived
    from a base scenario seed -- numpy's own recommended mechanism for
    exactly this (SeedSequence), rather than ad hoc arithmetic like
    base + rank*1000 + episode, which risks collisions/correlation across
    workers. This is the only place a new RNG stream originates from;
    everything downstream uses the resulting np.random.Generator, never
    Python's `random` or numpy's global RNG state."""
    return int(np.random.SeedSequence([base_seed, worker_rank, episode_index]).generate_state(1)[0])


def _critical_positions(cfg: ScenarioConfig, ego_s: float, ego_lane: int) -> dict[str, tuple[float, int]]:
    """role -> (s, lane) -- independent of ego's speed (unlike v_x, a
    critical actor's position never depended on ego_initial_speed even
    before it was removed), so this alone is enough for validate_scenario's
    overlap check, before ego's speed has even been drawn."""
    front, rear, blocker = cfg.front, cfg.rear, cfg.blocker
    return {
        "front": (ego_s + front.gap + CAR_LENGTH, ego_lane),
        "rear": (ego_s - rear.gap - CAR_LENGTH, ego_lane),
        "blocker": (ego_s + blocker.relative_s, ego_lane + blocker.lane_offset),
    }


def _place_critical(
        cfg: ScenarioConfig, ego_s: float, ego_lane: int, ego_v0: float,
) -> dict[str, tuple[float, int, float, int]]:
    """role -> (s, lane, v_x, behaviour_code). ego_v0 is the *drawn* speed
    (see generate_scenario -- ego has no fixed speed of its own anymore,
    it's Normal(background_mean_speed, background_speed_std) like the
    road's traffic flow), so unlike positions, v_x can't be computed from
    cfg alone."""
    positions = _critical_positions(cfg, ego_s, ego_lane)
    front, rear, blocker = cfg.front, cfg.rear, cfg.blocker
    return {
        "front": (*positions["front"], ego_v0 + front.relative_speed, _BEHAVIOUR_CODE[front.behavior]),
        "rear": (*positions["rear"], ego_v0 + rear.relative_speed, _BEHAVIOUR_CODE[rear.behavior]),
        "blocker": (*positions["blocker"], ego_v0 + blocker.relative_speed, _BEHAVIOUR_CODE[blocker.behavior]),
    }


def validate_scenario(cfg: ScenarioConfig, lane_num: int, ego_lane: int, ego_s: float) -> None:
    """Raises ScenarioConfigError on the first problem found. Never
    corrects anything -- e.g. an out-of-range blocker lane is an error,
    not silently clamped to a valid one."""
    if cfg.front.gap is None or cfg.front.gap <= 0:
        raise ScenarioConfigError(f"front.gap must be > 0, got {cfg.front.gap!r}")
    if cfg.rear.gap is None or cfg.rear.gap <= 0:
        raise ScenarioConfigError(f"rear.gap must be > 0, got {cfg.rear.gap!r}")
    if cfg.blocker.lane_offset not in (-1, 1):
        raise ScenarioConfigError(f"blocker.lane_offset must be -1 or +1, got {cfg.blocker.lane_offset!r}")
    if cfg.blocker.relative_s is None:
        raise ScenarioConfigError("blocker.relative_s is required")

    blocker_lane = ego_lane + cfg.blocker.lane_offset
    if not (0 <= blocker_lane < lane_num):
        raise ScenarioConfigError(
            f"blocker.lane_offset={cfg.blocker.lane_offset} from ego_lane={ego_lane} implies lane "
            f"{blocker_lane}, outside [0, {lane_num}) for this {lane_num}-lane road -- the requested "
            f"adjacent lane doesn't exist; pick a valid offset rather than expecting it to be moved.")

    for role_name, actor in (("front", cfg.front), ("rear", cfg.rear), ("blocker", cfg.blocker)):
        if actor.role != role_name:
            raise ScenarioConfigError(f"ScenarioConfig.{role_name}.role must be {role_name!r}, got {actor.role!r}")
        if actor.behavior not in _BEHAVIOUR_CODE:
            raise ScenarioConfigError(
                f"{role_name}.behavior must be one of {tuple(_BEHAVIOUR_CODE)}, got {actor.behavior!r}")
        if not math.isfinite(actor.relative_speed):
            raise ScenarioConfigError(f"{role_name}.relative_speed must be finite, got {actor.relative_speed!r}")

    # No ego_initial_speed / implied-speed check here anymore: ego's speed
    # is drawn stochastically at generation time (Normal(background_mean_
    # speed, background_speed_std), see generate_scenario), not a fixed
    # config value -- so "front/rear/blocker's implied speed >= 0" can
    # only be checked there, against the actual drawn value, not here
    # against cfg alone.

    if not (0.0 <= cfg.aggressive_fraction <= 1.0):
        raise ScenarioConfigError(f"aggressive_fraction must be in [0, 1], got {cfg.aggressive_fraction!r}")
    if cfg.background_speed_std < 0:
        raise ScenarioConfigError(f"background_speed_std must be >= 0, got {cfg.background_speed_std!r}")
    if cfg.traffic_spread <= 0:
        raise ScenarioConfigError(f"traffic_spread must be > 0, got {cfg.traffic_spread!r}")
    if cfg.lane_change_tendency <= 0:
        raise ScenarioConfigError(f"lane_change_tendency must be > 0, got {cfg.lane_change_tendency!r}")
    if cfg.road_mu <= 0:
        raise ScenarioConfigError(f"road_mu must be > 0, got {cfg.road_mu!r}")
    if cfg.road_kappa_max < 0:
        raise ScenarioConfigError(f"road_kappa_max must be >= 0, got {cfg.road_kappa_max!r}")
    if cfg.goal_distance_m <= 0:
        raise ScenarioConfigError(f"goal_distance_m must be > 0, got {cfg.goal_distance_m!r}")
    if cfg.max_episode_seconds <= 0:
        raise ScenarioConfigError(f"max_episode_seconds must be > 0, got {cfg.max_episode_seconds!r}")
    if cfg.min_progress_m < 0:
        raise ScenarioConfigError(f"min_progress_m must be >= 0, got {cfg.min_progress_m!r}")

    # No initial overlap: ego vs. each critical actor, and critical actors
    # vs. each other -- same AABB test model.collision.bodies_overlap uses
    # (same lane + |ds| < CAR_LENGTH; cross-lane never overlaps here since
    # lane spacing >> CAR_WIDTH, matching that module's own reasoning).
    positions = _critical_positions(cfg, ego_s, ego_lane)
    bodies = {"ego": (ego_s, ego_lane), **positions}
    names = list(bodies)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            s1, lane1 = bodies[n1]
            s2, lane2 = bodies[n2]
            if lane1 == lane2 and abs(s1 - s2) < CAR_LENGTH:
                raise ScenarioConfigError(
                    f"{n1!r} and {n2!r} overlap at spawn: same lane {lane1}, |ds|={abs(s1 - s2):.2f}m "
                    f"< CAR_LENGTH={CAR_LENGTH}m -- adjust gaps/relative_s so critical actors and ego "
                    f"don't start inside each other.")


def _sample_background_behaviour(aggressive_fraction: float, rng: np.random.Generator) -> int:
    """Bernoulli(aggressive_fraction) -> aggressive; otherwise split the
    remaining probability evenly conservative/moderate. The spec only
    pins down the aggressive probability -- this 50/50 split of the rest
    is a documented assumption, not derived from anything."""
    if rng.uniform() < aggressive_fraction:
        return 3
    return 1 if rng.uniform() < 0.5 else 2


def _place_background(
        cfg: ScenarioConfig, road: Road, ego_s: float,
        critical: dict[str, tuple[float, int, float, int]],
        rng: np.random.Generator, *,
        n: int, min_gap: float, ego_gap_multiplier: float, max_place_attempts: int,
        car_id_offset: int,
) -> tuple[list[Car], list[float]]:
    """Places n background cars, avoiding ego (ego_gap_multiplier * min_gap
    berth, same idea as initialization.traffic_init.generate_traffic),
    the 3 critical actors, and each other (min_gap, same-lane). car_ids
    are assigned car_id_offset..car_id_offset+n-1 explicitly (see
    generate_scenario) rather than left to Car's own auto-incrementing
    global counter, so that two calls with the same rng stream produce
    byte-identical TrafficAgents -- including car_id -- not just
    identical physics; the global counter's value depends on how many
    Car objects have been created anywhere in the process, which isn't
    reproducible across resets/instances."""
    lanes = tuple(range(road.lane_num))
    d_range = (-cfg.traffic_spread, cfg.traffic_spread)
    ego_min_gap = min_gap * ego_gap_multiplier

    s_taken_by_lane: dict[int, list[float]] = {lane: [] for lane in lanes}
    for _role, (s, lane, _v, _b) in critical.items():
        s_taken_by_lane[lane].append(s)

    cars: list[Car] = []
    v0s: list[float] = []
    for i in range(n):
        lane = int(rng.choice(lanes))
        placed = False
        for _ in range(max_place_attempts):
            d = float(rng.uniform(*d_range))
            s_candidate = ego_s + d
            if abs(d) < ego_min_gap:
                continue
            if all(abs(s_candidate - s) >= min_gap for s in s_taken_by_lane[lane]):
                placed = True
                break
        if not placed:
            # Unlike initialization.traffic_init.generate_traffic (which
            # silently falls through to the last-tried, possibly-
            # overlapping candidate -- an accepted rare edge case there),
            # the no-overlap invariant is a hard requirement here, so a
            # background car that can't find a slot is a clear error, not
            # a silent violation.
            raise ScenarioConfigError(
                f"couldn't place background car {i + 1}/{n} in lane {lane} within {max_place_attempts} "
                f"attempts (traffic_spread={cfg.traffic_spread}, min_gap={min_gap}, already "
                f"{len(s_taken_by_lane[lane])} cars/critical actors in that lane) -- increase "
                f"traffic_spread, or reduce background car count/min_gap, so there's enough room.")
        s_taken_by_lane[lane].append(s_candidate)

        behaviour = _sample_background_behaviour(cfg.aggressive_fraction, rng)
        v0 = float(np.clip(rng.normal(cfg.background_mean_speed, cfg.background_speed_std),
                            _BACKGROUND_SPEED_MIN, _BACKGROUND_SPEED_MAX))

        state = CarState(s=s_candidate, e_y=0.0, e_psi=0.0, v_x=v0, lane=lane)
        car = Car(car_id=car_id_offset + i, state=state, vehicle_params=VehicleParameters(), behaviour=behaviour)
        cars.append(car)
        v0s.append(v0)

    return cars, v0s


def _steady_state_background(cars: list[Car], v0s: list[float]) -> None:
    """Same front-to-back IDM steady-state pass generate_traffic runs,
    applied to the background cars only (critical actors keep their
    explicitly requested speed -- see generate_scenario). Known
    simplification: a background car's steady-state leader is only ever
    another background car in the same lane, never a critical actor or
    ego even if one is physically closer ahead -- "retain the existing
    IDM steady-state initialization when appropriate" was judged not to
    extend to modeling ego (RL-controlled, no IDM concept) or a critical
    actor (explicit fixed speed, not a following car) as a leader."""
    lanes = sorted(set(c.state.lane for c in cars))
    for lane in lanes:
        idx_in_lane = [i for i, c in enumerate(cars) if c.state.lane == lane]
        idx_in_lane.sort(key=lambda i: cars[i].state.s, reverse=True)   # front-most first

        leader_s, leader_v = None, None
        for i in idx_in_lane:
            car, v0 = cars[i], v0s[i]
            idm_params = IDM_PRESETS[car.behaviour]
            if leader_s is None:
                gap, v_leader = float("inf"), 0.0
            else:
                gap = leader_s - car.state.s - CAR_LENGTH
                v_leader = leader_v
            v_ss = steady_state_speed(v0, gap, v_leader, idm_params)
            car.state.v_x = v_ss
            leader_s, leader_v = car.state.s, v_ss


def _build_realized(
        cfg: ScenarioConfig, road: Road, ego_s: float, ego_lane: int, ego_v0: float,
        agents: list[TrafficAgent], roles: list[str],
        worker_rank: int, episode_index: int, scenario_seed: int | None,
) -> dict:
    """A plain-dict, JSON-safe record of every realized initial condition
    -- explicit float()/int() casts throughout since numpy scalar types
    (e.g. np.float64) aren't JSON serializable by the stdlib json module."""
    mobil = MobilParams(threshold=MobilParams().threshold / cfg.lane_change_tendency)

    def car_record(agent: TrafficAgent, role: str) -> dict:
        car = agent.car
        s = car.state
        idm = IDM_PRESETS[car.behaviour]
        return {
            "id": int(car.car_id),
            "role": role,
            "s": float(s.s),
            "e_y": float(s.e_y),
            "lane": int(s.lane),
            "v_x": float(s.v_x),
            "v0": float(agent.v0),
            "behaviour": _BEHAVIOUR_NAME[car.behaviour],
            "idm_params": {k: float(v) for k, v in idm._asdict().items()},
        }

    return {
        "ego": {
            "s": float(ego_s), "e_y": 0.0, "e_psi": 0.0,
            "v_x": float(ego_v0), "lane": int(ego_lane),
        },
        "agents": [car_record(a, r) for a, r in zip(agents, roles)],
        "mobil_params": {
            "politeness": float(mobil.politeness),
            "threshold": float(mobil.threshold),
            "b_safe": float(mobil.b_safe),
        },
        "road": {
            "mu": float(road.mu),
            "kappa_max": float(road.kappa_max),
            "s_max": float(road.s_max),
            "L_clothoid": float(road.L_clothoid),
            "lane_num": int(road.lane_num),
        },
        "scenario_config": cfg.to_dict(),
        "seed_info": {
            "base_seed": int(cfg.seed),
            "worker_rank": int(worker_rank),
            "episode_index": int(episode_index),
            "scenario_seed": int(scenario_seed) if scenario_seed is not None else None,
        },
    }


def generate_scenario(
        cfg: ScenarioConfig,
        road: Road,
        ego_s: float,
        ego_lane: int,
        rng: np.random.Generator,
        *,
        worker_rank: int = 0,
        episode_index: int = 0,
        scenario_seed: int | None = None,
        min_gap: float = 6.0,
        ego_gap_multiplier: float = 1.5,
        max_place_attempts: int = 2000,   # cheap per attempt; keeps the now-hard failure rare (see _place_background)
) -> tuple[list[TrafficAgent], float, dict]:
    """Realize a ScenarioConfig into exactly N_SURR=15 TrafficAgents (3
    critical, at indices 0/1/2 in role order front/rear/blocker, then 12
    background), ego's initial speed, and a JSON-safe realized-scenario
    record (see _build_realized / get_realized_scenario in learning.env).

    `rng` is the *only* source of randomness used here (for ego's own
    speed and every background lane/position/behaviour/speed draw) --
    always a np.random.Generator the caller derived via derive_seed(),
    never numpy's global RNG state or Python's `random`. worker_rank/
    episode_index/scenario_seed are only used for the realized-scenario's
    own seed-provenance record; they don't affect generation directly
    (that's entirely `rng`'s job).

    Returns (agents, ego_v0, realized).
    """
    validate_scenario(cfg, lane_num=road.lane_num, ego_lane=ego_lane, ego_s=ego_s)

    # Ego is the road speed, not an independently-tunable quantity: drawn
    # from the same distribution as background traffic (see module
    # docstring), first -- before any critical actor or background car --
    # so front/rear/blocker.relative_speed have something to be relative to.
    ego_v0 = float(np.clip(rng.normal(cfg.background_mean_speed, cfg.background_speed_std),
                            _BACKGROUND_SPEED_MIN, _BACKGROUND_SPEED_MAX))

    for role_name, actor in (("front", cfg.front), ("rear", cfg.rear), ("blocker", cfg.blocker)):
        v = ego_v0 + actor.relative_speed
        if not math.isfinite(v) or v < 0:
            raise ScenarioConfigError(
                f"{role_name}'s implied speed (drawn ego_v0={ego_v0:.2f} + relative_speed="
                f"{actor.relative_speed!r} = {v:.2f}) must be finite and >= 0. Since ego_v0 is now "
                f"stochastic (Normal(background_mean_speed={cfg.background_mean_speed!r}, "
                f"background_speed_std={cfg.background_speed_std!r})), this depends on the draw -- keep "
                f"relative_speed within a few background_speed_std of 0 to avoid an occasional failure.")

    critical = _place_critical(cfg, ego_s, ego_lane, ego_v0)
    bg_cars, bg_v0s = _place_background(
        cfg, road, ego_s, critical, rng,
        n=N_BACKGROUND, min_gap=min_gap, ego_gap_multiplier=ego_gap_multiplier,
        max_place_attempts=max_place_attempts, car_id_offset=N_CRITICAL,
    )
    _steady_state_background(bg_cars, bg_v0s)

    # car_ids 0/1/2 for front/rear/blocker (fixed role order), 3.. for
    # background (see _place_background's own docstring on why these are
    # assigned explicitly rather than left to Car's global counter).
    agents: list[TrafficAgent] = []
    roles: list[str] = []
    for car_id, role in enumerate(("front", "rear", "blocker")):
        s, lane, v_x, beh = critical[role]
        state = CarState(s=s, e_y=0.0, e_psi=0.0, v_x=v_x, lane=lane)
        car = Car(car_id=car_id, state=state, vehicle_params=VehicleParameters(), behaviour=beh)
        agents.append(TrafficAgent(car=car, v0=v_x, target_lane=lane))
        roles.append(role)
    for car, v0 in zip(bg_cars, bg_v0s):
        agents.append(TrafficAgent(car=car, v0=v0, target_lane=car.state.lane))
        roles.append("background")

    realized = _build_realized(cfg, road, ego_s, ego_lane, ego_v0, agents, roles,
                                worker_rank, episode_index, scenario_seed)
    return agents, ego_v0, realized
