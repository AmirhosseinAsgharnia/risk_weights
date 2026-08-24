"""
Traffic generation for the legacy (no-ScenarioConfig) path: fixed traffic
structure + random continuous realization.

    Fixed (same across every seed):
        vehicle count (16: 1 ego + 15 surrounding), which slot is ego,
        each slot's lane, each slot's front-to-back order within its lane,
        each slot's behaviour class (conservative/moderate/aggressive).

    Random (varies with the RNG):
        the bumper-to-bumper headway between consecutive slots in a lane,
        and each slot's desired speed within its own (fixed) behaviour
        class's range.

    Derived (computed, not sampled):
        every slot's longitudinal position (built by chaining headways,
        then translated so the fixed ego slot lands exactly at ego_s), and
        every slot's actual initial v_x (the IDM steady-state speed for
        the gap/leader it ended up with, resolved front-to-back per lane
        -- ego included, so ego both affects and is affected by the
        traffic it's embedded in).

Changing the seed therefore changes headways and desired speeds, but never
vehicle count, ego identity, lane assignments, per-lane ordering, or
behaviour assignments -- see TRAFFIC_TEMPLATE below.

This module intentionally does not know about ScenarioConfig/critical
actors (front/rear/blocker) -- that is learning.scenario's job, a
separate, stricter generator for RL training reproducibility. This file
stays the simple, structurally-stable generator behind
EgoTrafficEnv's legacy path and any other direct caller.
"""

import numpy as np
from dataclasses import dataclass, field

from model.car.car import Car, CarState
from model.car.config import VehicleParameters
from controllers.idm import idm_accel, IDM_PRESETS

CAR_LENGTH = 4.0   # [m] used for bumper-to-bumper gap, matching the body
                    # rectangle tests/dynamics_test.py draws cars with.
CAR_WIDTH  = 2.0    # [m] shared with tests/traffic_test.py's plotted body
                    # width and model.collision's overlap test.

SPEED_RANGES: dict[int, tuple[float, float]] = {   # [m/s] desired-speed range per behaviour
    1: (15.0, 20.0),   # conservative
    2: (17.5, 22.5),   # moderate
    3: (20.0, 25.0),   # aggressive
}


@dataclass
class TrafficAgent:
    """A traffic car plus the scenario-level bookkeeping Car/CarState don't
    own themselves (v0 is a per-car IDM preference, not a behaviour-fixed
    constant -- see IDM_PRESETS; target_lane/lane_change_t0/prev_delta are
    MOBIL/far-near runtime state, not general vehicle-dynamics state;
    crashed/crash_t/crash_v are set by model.collision once this car has
    been in a plastic collision -- see that module for what CRASHED means
    and why v0 keeps its pre-crash value rather than being cleared)."""
    car: Car
    v0: float                              # [m/s] this car's own IDM desired speed
    target_lane: int                       # lane it's heading for; == car.state.lane when not changing
    lane_change_t0: float | None = None    # sim time the active lane change started; None if not changing
    prev_delta: float = 0.0                # for clip_steering_rate
    last_lane_change_t: float | None = None   # sim time the last lane change committed; None if never
    crashed: bool = False                  # once True, permanent: skip IDM/MOBIL, passive obstacle
    crash_t: float | None = None           # [s] sim time this car crashed; None if not crashed
    crash_v: float | None = None           # [m/s] momentum-conserved speed at the moment of its crash


def steady_state_speed(v0: float, gap: float, v_leader: float, idm_params,
                        tol: float = 1e-4, max_iter: int = 60) -> float:
    """
    The speed v in [0, v0] at which idm_accel(v, gap, v-v_leader, v0, ...)
    == 0 -- i.e. the speed this car settles to, unaccelerating, given a
    leader at a fixed gap moving at v_leader (gap=inf, any v_leader ==>
    free-road, returns v0). idm_accel(v, ...) is monotonically decreasing
    in v (both the free-road and interaction terms shrink as v grows), so
    bisection applies directly. Degenerate case: if even v=0 already wants
    to decelerate (gap tighter than the standstill jam distance s0), there
    is no equilibrium above a standstill -- return 0 rather than search.
    """
    def f(v):
        return idm_accel(v, gap, v - v_leader, v0, idm_params.a_max, idm_params.b,
                          idm_params.s0, idm_params.T, idm_params.delta)

    lo, hi = 0.0, v0
    if f(lo) <= 0.0:
        return lo
    if f(hi) >= 0.0:
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ── Fixed traffic template ──────────────────────────────────────────────────

@dataclass(frozen=True)
class SlotSpec:
    """One fixed slot in the traffic template. lane/order/behaviour/is_ego
    are structural -- identical for every seed. Only this slot's headway to
    its lane-neighbour ahead, and its exact desired speed within its
    behaviour class, are randomized per call (see generate_traffic)."""
    slot_id: int      # 0..15, stable identity across seeds -- also used as Car.car_id
    lane: int         # 0, 1, or 2
    order: int        # position within its lane, ascending with s (0 = rearmost in that lane)
    behaviour: int    # 1 = conservative, 2 = moderate, 3 = aggressive (controllers.idm.IDM_PRESETS)
    is_ego: bool = False


LANE_NUM_TEMPLATE = 3   # this template spans exactly 3 lanes -- matches every current
                         # caller's road (Road(..., lane_num=3)); not a free parameter.
N_SLOTS = 16             # 1 ego + 15 surrounding, fixed regardless of seed
N_SURR = N_SLOTS - 1     # 15 -- number of TrafficAgents generate_traffic returns

# Ego sits in the middle lane (lane 1), at order=2 of that lane's 6 slots --
# 2 slots behind it, 3 ahead -- away from both the front and rear boundary
# of its lane, per the "middle lane, not at the edge" requirement. Ego's own
# behaviour=2 (moderate) here is used ONLY to pick IDM parameters for
# resolving its temporary steady-state speed during generation (see
# generate_traffic) -- it is not a persistent driving style: ego has no
# IDM/MOBIL controller once the scenario starts (it's RL- or user-driven).
# Behaviour classes are hand-assigned for a reasonable mixture across lanes
# (5 conservative / 5 moderate / 5 aggressive among the 15 surrounding
# slots) -- not drawn, so the mixture itself never changes across seeds.
TRAFFIC_TEMPLATE: tuple[SlotSpec, ...] = (
    # lane 0 -- 5 slots, rear to front
    SlotSpec(0,  lane=0, order=0, behaviour=1),
    SlotSpec(1,  lane=0, order=1, behaviour=2),
    SlotSpec(2,  lane=0, order=2, behaviour=3),
    SlotSpec(3,  lane=0, order=3, behaviour=1),
    SlotSpec(4,  lane=0, order=4, behaviour=2),
    # lane 1 -- 6 slots (5 surrounding + ego), rear to front
    SlotSpec(5,  lane=1, order=0, behaviour=3),
    SlotSpec(6,  lane=1, order=1, behaviour=1),
    SlotSpec(7,  lane=1, order=2, behaviour=2, is_ego=True),
    SlotSpec(8,  lane=1, order=3, behaviour=2),
    SlotSpec(9,  lane=1, order=4, behaviour=3),
    SlotSpec(10, lane=1, order=5, behaviour=2),
    # lane 2 -- 5 slots, rear to front
    SlotSpec(11, lane=2, order=0, behaviour=1),
    SlotSpec(12, lane=2, order=1, behaviour=3),
    SlotSpec(13, lane=2, order=2, behaviour=2),
    SlotSpec(14, lane=2, order=3, behaviour=1),
    SlotSpec(15, lane=2, order=4, behaviour=3),
)


def _validate_template(template: tuple[SlotSpec, ...]) -> None:
    """Self-check on the hard-coded template above -- an internal
    invariant (this can only fail if the template itself is edited
    incorrectly), so plain asserts rather than user-facing exceptions."""
    ids = [s.slot_id for s in template]
    assert len(template) == N_SLOTS, f"expected {N_SLOTS} slots, got {len(template)}"
    assert len(set(ids)) == N_SLOTS, "slot_ids must be unique"
    assert sorted(ids) == list(range(N_SLOTS)), "slot_ids must be exactly 0..N_SLOTS-1"
    assert sum(s.is_ego for s in template) == 1, "exactly one slot must be ego"
    for s in template:
        assert 0 <= s.lane < LANE_NUM_TEMPLATE, f"slot {s.slot_id}: lane {s.lane} out of range"
        assert s.behaviour in (1, 2, 3), f"slot {s.slot_id}: invalid behaviour {s.behaviour}"
    for lane in range(LANE_NUM_TEMPLATE):
        orders = [s.order for s in template if s.lane == lane]
        assert len(orders) == len(set(orders)), f"lane {lane}: duplicate order values"
        assert sorted(orders) == list(range(len(orders))), \
            f"lane {lane}: order values must be a contiguous 0..n-1 range, got {sorted(orders)}"


_validate_template(TRAFFIC_TEMPLATE)

TEMPLATE_EGO_SLOT: SlotSpec = next(s for s in TRAFFIC_TEMPLATE if s.is_ego)
TEMPLATE_EGO_LANE: int = TEMPLATE_EGO_SLOT.lane   # convenience for callers that need ego's fixed lane


def generate_traffic(
        ego_s: float = 50.0,
        speed_ranges: dict[int, tuple[float, float]] = SPEED_RANGES,
        mean_headway: float = 20.0,
        headway_std: float = 8.0,
        min_headway: float = 6.0,
        max_headway: float = 45.0,
        rng: np.random.Generator | None = None,
) -> tuple[list[TrafficAgent], float]:
    """
    Realize TRAFFIC_TEMPLATE's fixed 16-slot structure (1 ego + 15
    surrounding, fixed lanes/order/behaviour -- see the template above)
    into concrete positions and speeds. Returns (agents, ego_v0): the 15
    non-ego TrafficAgents, and the ego slot's resolved initial speed.

    Algorithm:
      1. For each lane, chain slots front-to-back: the rearmost slot
         anchors at a small, deterministic per-lane stagger (a fraction of
         mean_headway -- NOT randomized, so lanes don't line up in
         artificial side-by-side rows), then each next slot's position is
         s_{i+1} = s_i + CAR_LENGTH + g_i, with g_i ~ Normal(mean_headway,
         headway_std) clipped to [min_headway, max_headway] -- this
         guarantees every bumper-to-bumper gap lands in
         [min_headway, max_headway] by construction; no retry loop, and no
         placement can ever fail or overlap.
      2. Each slot's desired speed v0 ~ U(speed_ranges[slot.behaviour]) --
         its own (fixed) behaviour class's range, so a seed can change
         *how fast* a conservative car wants to go, never turn it into a
         moderate or aggressive one.
      3. The whole scene (still in arbitrary local coordinates) is
         translated by a single constant, ego_s - <ego slot's local s>, so
         the fixed ego slot ends up exactly at ego_s. This is pure
         arithmetic -- it does not change any gap or consume any RNG.
      4. The full 16-slot scene (ego included) is resolved to IDM steady
         state, front-to-back per lane: the front-most slot in each lane
         keeps its sampled v0; every follower (ego included, using its
         template behaviour's IDM params) gets steady_state_speed(...)
         given the gap to, and resolved speed of, the slot ahead of it.
         No RNG is consumed in this step. This makes ego both affected by
         (a slower leader ahead of it) and affecting (a follower behind
         it) the surrounding traffic, exactly like any other slot.
      5. ego_v0 is the ego slot's resolved speed from step 4; the ego slot
         itself is excluded from the returned `agents` list.

    RNG consumption is fixed in order and count regardless of the seed:
    13 headway draws (5+6+5 slots per lane, 4+5+4 gaps), then 16 desired-
    speed draws (one per slot, in slot_id order) -- so which random values
    get consumed never depends on any earlier random outcome.
    """
    if headway_std < 0:
        raise ValueError(f"headway_std must be >= 0, got {headway_std!r}")
    if not (0 < min_headway <= max_headway):
        raise ValueError(
            f"min_headway must be > 0 and <= max_headway, got min_headway={min_headway!r}, "
            f"max_headway={max_headway!r}")
    if not (min_headway <= mean_headway <= max_headway):
        raise ValueError(
            f"mean_headway ({mean_headway!r}) must be within [min_headway, max_headway] = "
            f"[{min_headway!r}, {max_headway!r}]")

    rng = rng or np.random.default_rng()

    # -- 1. Random headways -> local (untranslated) longitudinal positions,
    # lane by lane, front-to-back, with a fixed (non-random) per-lane stagger.
    local_s: dict[int, float] = {}
    for lane in range(LANE_NUM_TEMPLATE):
        slots_in_lane = sorted((s for s in TRAFFIC_TEMPLATE if s.lane == lane), key=lambda s: s.order)
        s_prev = lane * (mean_headway / LANE_NUM_TEMPLATE)   # deterministic stagger, no RNG
        local_s[slots_in_lane[0].slot_id] = s_prev
        for slot in slots_in_lane[1:]:
            gap = float(np.clip(rng.normal(mean_headway, headway_std), min_headway, max_headway))
            s_prev = s_prev + CAR_LENGTH + gap
            local_s[slot.slot_id] = s_prev

    # -- 2. Random desired speeds, one per slot, fixed slot_id order.
    v0_by_slot: dict[int, float] = {
        slot.slot_id: float(rng.uniform(*speed_ranges[slot.behaviour]))
        for slot in sorted(TRAFFIC_TEMPLATE, key=lambda s: s.slot_id)
    }

    # -- 3. Translate the whole scene so the fixed ego slot lands at ego_s.
    shift = ego_s - local_s[TEMPLATE_EGO_SLOT.slot_id]

    cars_by_id: dict[int, Car] = {}
    for slot in TRAFFIC_TEMPLATE:
        state = CarState(s=local_s[slot.slot_id] + shift, e_y=0.0, e_psi=0.0,
                          v_x=v0_by_slot[slot.slot_id], lane=slot.lane)
        cars_by_id[slot.slot_id] = Car(car_id=slot.slot_id, state=state,
                                        vehicle_params=VehicleParameters(), behaviour=slot.behaviour)

    # -- 4. Physical resolution: IDM steady state, front-to-back per lane,
    # ego included. No RNG consumed here.
    for lane in range(LANE_NUM_TEMPLATE):
        slots_in_lane = sorted((s for s in TRAFFIC_TEMPLATE if s.lane == lane),
                                key=lambda s: s.order, reverse=True)   # front-most first
        leader_s, leader_v = None, None
        for slot in slots_in_lane:
            car = cars_by_id[slot.slot_id]
            v0 = v0_by_slot[slot.slot_id]
            idm_params = IDM_PRESETS[slot.behaviour]
            if leader_s is None:
                gap, v_leader = float("inf"), 0.0
            else:
                gap = leader_s - car.state.s - CAR_LENGTH
                v_leader = leader_v
            v_ss = steady_state_speed(v0, gap, v_leader, idm_params)
            car.state.v_x = v_ss
            leader_s, leader_v = car.state.s, v_ss

    # -- 5. Extract ego's resolved speed; exclude the ego slot from `agents`.
    ego_v0 = cars_by_id[TEMPLATE_EGO_SLOT.slot_id].state.v_x

    agents = [
        TrafficAgent(car=cars_by_id[slot.slot_id], v0=v0_by_slot[slot.slot_id], target_lane=slot.lane)
        for slot in sorted(TRAFFIC_TEMPLATE, key=lambda s: s.slot_id)
        if not slot.is_ego
    ]
    return agents, ego_v0
