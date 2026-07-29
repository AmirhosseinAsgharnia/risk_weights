"""
The simulation engine: scenario definition + validation, one deterministic
timestep, and structured diagnostics/results. No matplotlib import anywhere in
this module -- rendering (tests/scene_test.py) only ever reads the returned
history, never influences it.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from model.road.road import RoadScenario
from model.car.config import VehicleParameters
from model.car.model import EgoModel, SimulationDivergedError
from model.car_init import Car, resolve_params, frenet_to_global, init_ego, init_surr
from model.IDM_MOBIL import IDMParams, MOBILParams, LaneChangeParams, gap_to_lead, steering_cmd
from model.lane_change import (
    LaneChangeTolerances, lane_change_reference, try_complete_lane_change,
)
from model.traffic import (
    ActorSnapshot, LaneCandidate, build_snapshot, find_neighbors, longitudinal_accel,
    propose_lane_changes, resolve_conflicts, apply_accepted_proposals,
)
from model.collision import (
    CollisionEvent, LossOfControlThresholds, check_collisions, check_road_departure,
    check_lane_index_valid, check_singularity_risk, loss_of_control,
    broad_phase_overlap, narrow_phase_collision,
)


def kappa_at(road: RoadScenario, s: float) -> float:
    idx = min(max(int(round(s / road.ds)), 0), len(road.s) - 1)
    return road.kappa[idx]


def mu_at(road: RoadScenario, s: float) -> float:
    idx = min(max(int(round(s / road.ds)), 0), len(road.s) - 1)
    return road.mu[idx]


# The ego controller sees its own snapshot AND the full-scene snapshot (so it
# can look up leaders/followers, e.g. for IDM) -- broadened from "own
# snapshot only" specifically so car-following controllers are possible
# without changing this interface again later.
EgoControllerType = Callable[
    [ActorSnapshot, "tuple[ActorSnapshot, ...]", RoadScenario, float], tuple[float, float]
]


class CruiseLaneHoldController:
    """
    Ego controller placeholder: holds a constant target speed via a
    proportional law and stays in its current lane, using the exact same
    reference-tracking steering_cmd every other actor uses. It owns the
    vehicle params it needs (drag compensation, wheelbase feedforward) so
    Simulation.step() never contains ego-specific logic; swap in a smarter
    controller (e.g. MPC) by passing a different callable as
    Scenario.ego_controller -- the engine's interface doesn't change.

    Does NOT react to traffic -- see IDMLaneHoldController for that.
    """

    def __init__(self, target_speed: float, vehicle_params: VehicleParameters | None = None,
                 lc_params: LaneChangeParams | None = None, k_cruise: float = 0.5,
                 dt: float = 0.1):
        self.target_speed = target_speed
        self.p = vehicle_params or VehicleParameters()
        self.lc_params = lc_params or LaneChangeParams()
        self.k_cruise = k_cruise
        self.dt = dt
        self._integral = 0.0
        self._prev_delta = None

    def __call__(self, snapshot: ActorSnapshot, full_snapshot: "tuple[ActorSnapshot, ...]",
                 road: RoadScenario, t: float) -> tuple[float, float]:
        vx = snapshot.state[3]
        a_net = self.k_cruise * (self.target_speed - vx)
        a_x = a_net + (self.p.C_d / self.p.m) * vx**2

        kappa_s = kappa_at(road, snapshot.state[0])
        e_y_ref = road.d_c[snapshot.current_lane]  # this placeholder never changes lanes
        delta, self._integral = steering_cmd(
            snapshot.state, kappa_s, e_y_ref, 0.0, self.p.L, self.lc_params,
            integral=self._integral, prev_delta=self._prev_delta, dt=self.dt,
        )
        self._prev_delta = delta
        return float(a_x), float(delta)


class IDMLaneHoldController:
    """
    Ego controller using IDM for longitudinal accel -- reacts to whatever's
    ahead in its own lane(s), unlike CruiseLaneHoldController -- plus the same
    lane-hold steering as everyone else. No MOBIL: it never initiates a lane
    change on its own. A temporary stand-in requested in place of the cruise
    placeholder; swap back to CruiseLaneHoldController (or something smarter)
    via Scenario.ego_controller whenever needed -- the engine doesn't care.
    """

    def __init__(self, vehicle_params: VehicleParameters | None = None,
                 lc_params: LaneChangeParams | None = None, dt: float = 0.1):
        self.p = vehicle_params or VehicleParameters()
        self.lc_params = lc_params or LaneChangeParams()
        self.dt = dt
        self._integral = 0.0
        self._prev_delta = None

    def __call__(self, snapshot: ActorSnapshot, full_snapshot: "tuple[ActorSnapshot, ...]",
                 road: RoadScenario, t: float) -> tuple[float, float]:
        a_net, _leader_id = longitudinal_accel(full_snapshot, snapshot.id)
        vx = snapshot.state[3]
        a_x = a_net + (self.p.C_d / self.p.m) * vx**2

        kappa_s = kappa_at(road, snapshot.state[0])
        e_y_ref = road.d_c[snapshot.current_lane]  # lane-hold only, no MOBIL for the ego
        delta, self._integral = steering_cmd(
            snapshot.state, kappa_s, e_y_ref, 0.0, self.p.L, self.lc_params,
            integral=self._integral, prev_delta=self._prev_delta, dt=self.dt,
        )
        self._prev_delta = delta
        return float(a_x), float(delta)


@dataclass
class Scenario:
    road: RoadScenario
    ego: Car
    traffic: list[Car]
    ego_controller: EgoControllerType

    vehicle_params: VehicleParameters = field(default_factory=VehicleParameters)
    idm_defaults: IDMParams = field(default_factory=IDMParams)
    mobil_defaults: MOBILParams = field(default_factory=MOBILParams)
    lc_defaults: LaneChangeParams = field(default_factory=LaneChangeParams)
    lane_change_tolerances: LaneChangeTolerances = field(default_factory=LaneChangeTolerances)
    loc_thresholds: LossOfControlThresholds = field(default_factory=LossOfControlThresholds)

    v_min: float = 1.0
    dt: float = 0.1
    lane_change_cooldown: float = 3.0
    min_gap_conflict: float = 4.0
    collision_margin: float = 0.5
    singularity_threshold: float = 0.8

    stop_on_collision: bool = True
    stop_on_numerical_failure: bool = True
    stop_on_road_departure: bool = False

    rng: np.random.Generator = field(default_factory=np.random.default_rng)


class ScenarioValidationError(ValueError):
    pass


def validate_scenario(scenario: Scenario) -> None:
    """Raises ScenarioValidationError with a clear message; never lets a bad
    scenario fail later through obscure dynamics."""
    road = scenario.road
    all_cars = [scenario.ego] + list(scenario.traffic)

    if scenario.dt <= 0:
        raise ScenarioValidationError(f"dt must be > 0, got {scenario.dt}")
    if scenario.lane_change_cooldown < 0:
        raise ScenarioValidationError("lane_change_cooldown must be >= 0")
    if scenario.min_gap_conflict < 0:
        raise ScenarioValidationError("min_gap_conflict must be >= 0")
    if scenario.lc_defaults.delta_max <= 0:
        raise ScenarioValidationError("LaneChangeParams.delta_max must be > 0")

    seen_ids = set()
    for car in all_cars:
        if car.id in seen_ids:
            raise ScenarioValidationError(f"duplicate actor id {car.id}")
        seen_ids.add(car.id)

        if not check_lane_index_valid(car.current_lane, road.lane_num):
            raise ScenarioValidationError(
                f"car {car.id}: current_lane {car.current_lane} out of range [0, {road.lane_num - 1}]")
        if not check_lane_index_valid(car.target_lane, road.lane_num):
            raise ScenarioValidationError(
                f"car {car.id}: target_lane {car.target_lane} out of range [0, {road.lane_num - 1}]")

        if not np.all(np.isfinite(car.state)):
            raise ScenarioValidationError(f"car {car.id}: initial state has non-finite entries: {car.state}")
        if car.state[3] < 0:
            raise ScenarioValidationError(f"car {car.id}: initial vx={car.state[3]} must be >= 0")
        if car.v0 is not None and car.v0 <= 0:
            raise ScenarioValidationError(f"car {car.id}: v0={car.v0} must be > 0")
        if car.lane_change_duration <= 0:
            raise ScenarioValidationError(f"car {car.id}: lane_change_duration must be > 0")

    # No two cars may already overlap (broad + narrow phase, reused from model.collision).
    snapshot = build_snapshot(all_cars, road, scenario.idm_defaults, scenario.mobil_defaults)
    for i in range(len(snapshot)):
        for j in range(i + 1, len(snapshot)):
            a, b = snapshot[i], snapshot[j]
            if broad_phase_overlap(a, b, margin=0.0) and narrow_phase_collision(a, b, road):
                raise ScenarioValidationError(f"cars {a.id} and {b.id} overlap at t=0")

    # Road/patch/curvature/friction bounds are validated by RoadScenario's own
    # constructor already (kept as-is, per the preserved API).


# ---------- diagnostics ----------

@dataclass
class ActorLog:
    time: float
    actor_id: int
    state: np.ndarray
    pose: tuple[float, float, float]
    current_lane: int
    target_lane: int
    occupied_lanes: tuple[int, ...]
    lane_change_active: bool
    leader_id: Optional[int]
    follower_id: Optional[int]
    idm_accel: Optional[float]
    mobil_candidates: tuple[LaneCandidate, ...]
    selected_proposal: Optional[str]
    rejection_reason: Optional[str]
    steering_cmd: float
    steering_ref_e_y: float
    slip_front: float
    slip_rear: float
    Fyf_unsat: float
    Fyr_unsat: float
    Fyf_sat: float
    Fyr_sat: float
    mu: float
    saturation_ratio: float
    collision: bool = False
    road_departure: bool = False


@dataclass
class StepRecord:
    time: float
    actors: list[ActorLog]
    events: list[str]
    collisions: list[CollisionEvent]
    loss_of_control_ids: tuple[int, ...]
    min_gap: float
    min_ttc: float


@dataclass
class ScenarioResult:
    completed: bool
    collision: bool
    road_departure: bool
    numerical_failure: bool
    loss_of_control: bool
    min_gap: float
    min_time_to_collision: float
    max_abs_sideslip: float
    max_yaw_rate: float
    max_tire_saturation: float
    lane_changes: int
    terminal_time: float
    events: list[str]
    history: list[StepRecord]


def _pairwise_metrics(snapshot: tuple[ActorSnapshot, ...]) -> tuple[float, float]:
    """(min bumper-to-bumper gap, min time-to-collision) among lane-sharing pairs."""
    min_gap = np.inf
    min_ttc = np.inf
    n = len(snapshot)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = snapshot[i], snapshot[j]
            if not (set(a.occupied) & set(b.occupied)):
                continue
            rear, lead = (a, b) if a.state[0] <= b.state[0] else (b, a)
            gap = gap_to_lead(rear.state[0], lead.state[0], rear.length, lead.length)
            min_gap = min(min_gap, gap)
            closing = rear.state[3] - lead.state[3]
            if closing > 0 and np.isfinite(gap):
                min_ttc = min(min_ttc, max(gap, 0.0) / closing)
    return min_gap, min_ttc


class Simulation:
    def __init__(self, scenario: Scenario):
        validate_scenario(scenario)
        self.scenario = scenario
        self.t = 0.0
        self.step_count = 0

        all_cars = [scenario.ego] + list(scenario.traffic)
        self.cars_by_id: dict[int, Car] = {c.id: c for c in all_cars}
        self.models: dict[int, EgoModel] = {
            c.id: EgoModel(resolve_params(c.vehicle_params, scenario.vehicle_params), v_min=scenario.v_min)
            for c in all_cars
        }
        self._sat_streak: dict[int, int] = {c.id: 0 for c in all_cars}
        self._lane_change_count = 0

    def step(self) -> StepRecord:
        scenario = self.scenario
        road = scenario.road
        dt = scenario.dt
        t = self.t

        all_cars = [scenario.ego] + list(scenario.traffic)
        snapshot = build_snapshot(all_cars, road, scenario.idm_defaults, scenario.mobil_defaults)
        snap_by_id = {a.id: a for a in snapshot}

        # --- MOBIL: propose (traffic only) + resolve + apply, all from the frozen snapshot ---
        # NOTE: neighbour search needs to see EVERY actor (the full snapshot,
        # including the ego) to score safety correctly; only *who gets to
        # propose* is restricted to traffic via eligible_ids. Passing a
        # snapshot that already excluded the ego made MOBIL blind to it and
        # caused a real collision during development -- see traffic.py's docstring.
        traffic_ids = {c.id for c in scenario.traffic}
        proposals, candidates_by_actor = propose_lane_changes(
            snapshot, road, t, scenario.lane_change_cooldown, scenario.rng,
            eligible_ids=traffic_ids,
        )
        accepted, rejected = resolve_conflicts(proposals, snapshot, min_gap=scenario.min_gap_conflict)
        apply_accepted_proposals(self.cars_by_id, accepted, t)
        self._lane_change_count += len(accepted)

        events: list[str] = []
        accepted_by_actor = {p.actor_id: p for p in accepted}
        rejection_reason_by_actor: dict[int, str] = {p.actor_id: reason for p, reason in rejected}
        for p, reason in rejected:
            events.append(f"t={t:.1f} car {p.actor_id}: lane change {p.from_lane}->{p.to_lane} rejected ({reason})")
        for p in accepted:
            events.append(
                f"t={t:.1f} car {p.actor_id}: lane change {p.from_lane}->{p.to_lane} "
                f"started (incentive={p.incentive:.3f})"
            )

        # --- per-actor: longitudinal accel, steering, integrate (still off the frozen snapshot) ---
        actor_logs: list[ActorLog] = []
        for car in all_cars:
            snap = snap_by_id[car.id]
            model = self.models[car.id]
            kappa_s = kappa_at(road, car.state[0])
            mu_s = mu_at(road, car.state[0])

            is_ego = car.id == scenario.ego.id
            leader_id = follower_id = None
            idm_a = None

            if is_ego:
                a_x, delta = scenario.ego_controller(snap, snapshot, road, t)
            else:
                idm_a, leader_id = longitudinal_accel(snapshot, car.id)
                a_net = float(np.clip(idm_a, snap.idm_params.a_min, snap.idm_params.a_max))
                a_x = model.drag_compensated_ax(a_net, car.state[3])
                follower_id = find_neighbors(snapshot, car.id, car.current_lane).follower_id

                lc_p = resolve_params(car.lane_change_params, scenario.lc_defaults)
                e_y_ref, e_y_ref_rate = lane_change_reference(car, road, t)
                # integrate=False during an active manoeuvre: see steering_cmd's
                # docstring -- integral action against an already-planned
                # quintic transient is pure lag, not help.
                delta, car.lane_error_integral = steering_cmd(
                    car.state, kappa_s, e_y_ref, e_y_ref_rate, model.p.L, lc_p,
                    integral=car.lane_error_integral, integrate=not car.lane_change_active,
                    prev_delta=car.last_delta, dt=dt,
                )

            u = (a_x, delta)
            tire = model.tire_diagnostics(car.state, u, kappa_s, mu_s)
            self._sat_streak[car.id] = self._sat_streak[car.id] + 1 if tire.saturation_ratio > 1.0 else 0

            new_state = model.step(car.state, u, kappa_s, mu_s, dt)
            new_state[EgoModel.IDX_VX] = max(new_state[EgoModel.IDX_VX], 0.0)
            car.state = new_state
            car.last_delta = delta

            try_complete_lane_change(car, road, t + dt, scenario.lane_change_tolerances)

            if check_singularity_risk(kappa_s, snap.state[1], scenario.singularity_threshold):
                events.append(f"t={t:.1f} car {car.id}: near Frenet singularity (1-kappa*e_y small)")

            candidates = candidates_by_actor.get(car.id, ())
            if car.id in accepted_by_actor:
                selected = f"lane {accepted_by_actor[car.id].to_lane}"
            elif candidates:
                selected = "stay"
            else:
                selected = None

            pose = frenet_to_global(road, snap.state[0], snap.state[1], snap.state[2])
            actor_logs.append(ActorLog(
                time=t, actor_id=car.id, state=snap.state, pose=pose,
                current_lane=car.current_lane, target_lane=car.target_lane,
                occupied_lanes=snap.occupied, lane_change_active=car.lane_change_active,
                leader_id=leader_id, follower_id=follower_id, idm_accel=idm_a,
                mobil_candidates=candidates, selected_proposal=selected,
                rejection_reason=rejection_reason_by_actor.get(car.id),
                steering_cmd=delta,
                steering_ref_e_y=road.d_c[car.target_lane if car.lane_change_active else car.current_lane],
                slip_front=tire.alpha_f, slip_rear=tire.alpha_r,
                Fyf_unsat=tire.Fyf_unsat, Fyr_unsat=tire.Fyr_unsat,
                Fyf_sat=tire.Fyf, Fyr_sat=tire.Fyr, mu=mu_s,
                saturation_ratio=tire.saturation_ratio,
            ))

        # --- post-step checks: collisions, road departure, loss of control, gap/TTC ---
        post_snapshot = build_snapshot(all_cars, road, scenario.idm_defaults, scenario.mobil_defaults)
        post_by_id = {a.id: a for a in post_snapshot}

        collisions = check_collisions(post_snapshot, road, t + dt, margin=scenario.collision_margin)
        collided_ids = {i for c in collisions for i in c.actor_ids}
        for c in collisions:
            events.append(
                f"t={c.time:.1f} COLLISION {c.actor_ids} ({c.collision_type}, rel_speed={c.relative_speed:.1f})"
            )

        loc_ids = tuple(
            a.id for a in post_snapshot
            if loss_of_control(a, self._sat_streak[a.id], dt, scenario.loc_thresholds)
        )

        for log in actor_logs:
            log.collision = log.actor_id in collided_ids
            log.road_departure = check_road_departure(post_by_id[log.actor_id], road)

        min_gap, min_ttc = _pairwise_metrics(post_snapshot)

        self.t += dt
        self.step_count += 1
        return StepRecord(
            time=self.t, actors=actor_logs, events=events, collisions=collisions,
            loss_of_control_ids=loc_ids, min_gap=min_gap, min_ttc=min_ttc,
        )

    def run(self, n_steps: int) -> ScenarioResult:
        history: list[StepRecord] = []
        events: list[str] = []
        collision = road_departure = numerical_failure = loss_of_control_flag = False
        min_gap = min_ttc = np.inf
        max_sideslip = max_yaw_rate = max_saturation = 0.0
        terminal_time = self.t
        completed = True

        for _ in range(n_steps):
            try:
                record = self.step()
            except SimulationDivergedError as exc:
                numerical_failure = True
                events.append(f"t={self.t:.2f} NUMERICAL FAILURE: {exc}")
                completed = False
                terminal_time = self.t
                break

            history.append(record)
            events.extend(record.events)
            min_gap = min(min_gap, record.min_gap)
            min_ttc = min(min_ttc, record.min_ttc)
            loss_of_control_flag = loss_of_control_flag or bool(record.loss_of_control_ids)
            collision = collision or bool(record.collisions)

            for log in record.actors:
                v_min = self.scenario.loc_thresholds.v_min
                max_sideslip = max(max_sideslip, abs(log.state[4]) / max(log.state[3], v_min))
                max_yaw_rate = max(max_yaw_rate, abs(log.state[5]))
                max_saturation = max(max_saturation, log.saturation_ratio)
                road_departure = road_departure or log.road_departure

            terminal_time = record.time

            if collision and self.scenario.stop_on_collision:
                completed = False
                break
            if road_departure and self.scenario.stop_on_road_departure:
                completed = False
                break

        return ScenarioResult(
            completed=completed, collision=collision, road_departure=road_departure,
            numerical_failure=numerical_failure, loss_of_control=loss_of_control_flag,
            min_gap=min_gap, min_time_to_collision=min_ttc,
            max_abs_sideslip=max_sideslip, max_yaw_rate=max_yaw_rate, max_tire_saturation=max_saturation,
            lane_changes=self._lane_change_count, terminal_time=terminal_time,
            events=events, history=history,
        )


def build_demo_scenario(theta_road, ego_lane: int, ego_speed: float, surr_cars,
                         dt: float = 0.1, lane_change_cooldown: float = 3.0,
                         rng: np.random.Generator | None = None,
                         ego_controller: EgoControllerType | None = None) -> Scenario:
    """Convenience constructor matching the scenario tests/scene_test.py has used
    throughout: a RoadScenario from Theta_road, an ego, and a list of (N, dS, dV) traffic cars.

    ego_controller: defaults to IDMLaneHoldController (reacts to traffic ahead,
    still no MOBIL) -- a temporary stand-in for CruiseLaneHoldController.
    Pass CruiseLaneHoldController(...) explicitly (or anything else matching
    EgoControllerType) to go back to the non-reactive placeholder or swap in
    something smarter; nothing else about the engine needs to change.
    """
    kappa_max, L_s, mu_patch, patch_location = theta_road
    road = RoadScenario(kappa_max=kappa_max, L_s=L_s, mu_patch=mu_patch, patch_location=patch_location)

    ego = init_ego(road, lane=ego_lane, speed=ego_speed)
    traffic = init_surr(road, ego_speed=ego_speed, cars=surr_cars)

    vehicle_params = VehicleParameters()
    lc_defaults = LaneChangeParams()

    if ego_controller is None:
        ego.idm_params = IDMParams(v0=ego_speed)  # ego's own IDM desired speed
        ego_controller = IDMLaneHoldController(vehicle_params=vehicle_params, lc_params=lc_defaults, dt=dt)

    return Scenario(
        road=road, ego=ego, traffic=traffic, ego_controller=ego_controller,
        vehicle_params=vehicle_params, lc_defaults=lc_defaults,
        dt=dt, lane_change_cooldown=lane_change_cooldown,
        rng=rng or np.random.default_rng(0),
    )
