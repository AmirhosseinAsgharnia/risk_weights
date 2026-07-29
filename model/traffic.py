"""
Multi-agent traffic logic: per-step immutable snapshots, a single neighbour-search
implementation, best-incentive MOBIL lane selection, and deterministic simultaneous
conflict resolution.

Everything here reads from a frozen ActorSnapshot tuple built at the start of a
step -- nothing in this module ever looks at a live, possibly-already-updated
Car. model.simulation is the only place snapshots are built and proposals applied.
"""

import numpy as np
from dataclasses import dataclass, replace
from model.road.road import RoadScenario
from model.car_init import Car, resolve_params
from model.lane_change import occupied_lanes
from model.IDM_MOBIL import IDMParams, MOBILParams, idm_accel, gap_to_lead, mobil_incentive

# Wide-but-finite floor used ONLY when scoring MOBIL candidates (accel_for's
# clip=False path), as opposed to IDMParams.a_min which bounds real actuation.
# A near-zero gap can otherwise send the raw IDM formula to +-1e9-scale
# "accelerations" (physically meaningless) once bodies are nearly/actually
# overlapping; -6 (the actuation floor) is too tight to tell "bad" from "very
# bad" apart (see the docstring on idm_accel's clip parameter), so this picks
# a middle ground: wide enough to preserve real discrimination between
# candidates, bounded enough to keep incentives numerically sane.
SCORE_A_MIN = -50.0  # [m/s^2]


# ---------- snapshot ----------

@dataclass(frozen=True)
class ActorSnapshot:
    id: int
    current_lane: int
    target_lane: int
    lane_change_active: bool
    last_lane_change_time: float
    occupied: tuple[int, ...]
    state: np.ndarray          # copied -- never a view into the live Car
    length: float
    width: float
    v0: float | None
    idm_params: IDMParams
    mobil_params: MOBILParams


def build_snapshot(cars: list[Car], road: RoadScenario,
                    idm_default: IDMParams, mobil_default: MOBILParams) -> tuple[ActorSnapshot, ...]:
    """One immutable snapshot per car, taken before ANY control or lane decision
    this step. Every decision function below takes a snapshot, never a Car list."""
    snaps = []
    for c in cars:
        idm_params = resolve_params(c.idm_params, idm_default)
        if c.v0 is not None:
            # A car's own desired speed (set once at init from its dV, see
            # car_init.init_surr) always wins over whatever v0 the resolved
            # IDMParams happens to carry -- otherwise every car silently falls
            # back to the scenario default v0 and durable per-car speed
            # differences (the whole point of Car.v0) are lost.
            idm_params = replace(idm_params, v0=c.v0)

        snaps.append(ActorSnapshot(
            id=c.id,
            current_lane=c.current_lane,
            target_lane=c.target_lane,
            lane_change_active=c.lane_change_active,
            last_lane_change_time=c.last_lane_change_time,
            occupied=occupied_lanes(road, c.state[1], c.width),
            state=c.state.copy(),
            length=c.length,
            width=c.width,
            v0=c.v0,
            idm_params=idm_params,
            mobil_params=resolve_params(c.mobil_params, mobil_default),
        ))
    return tuple(snaps)


def _by_id(snapshot: tuple[ActorSnapshot, ...], actor_id: int | None) -> ActorSnapshot | None:
    if actor_id is None:
        return None
    for a in snapshot:
        if a.id == actor_id:
            return a
    return None


# ---------- neighbour search (the single implementation) ----------

@dataclass(frozen=True)
class NeighborResult:
    leader_id: int | None
    follower_id: int | None
    leader_gap: float      # bumper-to-bumper; +inf if none; may be negative if bodies overlap
    follower_gap: float


def find_neighbors(snapshot: tuple[ActorSnapshot, ...], actor_id: int, lane: int) -> NeighborResult:
    """
    Nearest leader/follower to `actor_id` among actors whose occupied lanes
    include `lane`, excluding actor_id itself. A car mid-lane-change occupies
    two lanes (see model.lane_change.occupied_lanes) and is therefore visible
    as a candidate leader/follower in BOTH -- no special-casing needed here.

    Gaps may be negative (overlapping bodies); this function reports the raw
    signed value for diagnostics/collision use. idm_accel clamps internally, so
    IDM itself never divides by a non-positive gap.
    """
    me = _by_id(snapshot, actor_id)
    leader = follower = None
    leader_ds = follower_ds = np.inf

    for a in snapshot:
        if a.id == actor_id or lane not in a.occupied:
            continue
        ds = a.state[0] - me.state[0]
        if ds > 0 and ds < leader_ds:
            leader_ds, leader = ds, a
        elif ds <= 0 and -ds < follower_ds:
            follower_ds, follower = -ds, a

    leader_gap = gap_to_lead(me.state[0], leader.state[0], me.length, leader.length) if leader else np.inf
    follower_gap = gap_to_lead(follower.state[0], me.state[0], follower.length, me.length) if follower else np.inf
    return NeighborResult(
        leader_id=leader.id if leader else None,
        follower_id=follower.id if follower else None,
        leader_gap=leader_gap,
        follower_gap=follower_gap,
    )


def accel_for(snapshot: tuple[ActorSnapshot, ...], actor_id: int | None, leader_id: int | None,
              clip: bool = True) -> float:
    """
    IDM accel of actor_id following leader_id (or free-flow if leader_id is None).

    clip=True (real actuation, via longitudinal_accel): bounded to the actor's
    own [IDMParams.a_min, a_max] -- what the vehicle can actually do.
    clip=False (MOBIL scoring, via evaluate_lane_candidates): bounded instead
    to [SCORE_A_MIN, a_max], much wider than actuation limits so genuinely
    different candidates stay distinguishable, but still finite -- a literal
    near-zero gap would otherwise send the raw IDM formula to a +-1e9-scale,
    physically meaningless "acceleration".
    """
    me = _by_id(snapshot, actor_id)
    if me is None:
        return 0.0
    lead = _by_id(snapshot, leader_id)
    if lead is None:
        gap, dv = np.inf, 0.0
    else:
        gap = gap_to_lead(me.state[0], lead.state[0], me.length, lead.length)
        dv = me.state[3] - lead.state[3]

    a = idm_accel(me.state[3], gap, dv, me.idm_params, clip=False)
    lo = me.idm_params.a_min if clip else SCORE_A_MIN
    return float(np.clip(a, lo, me.idm_params.a_max))


def longitudinal_accel(snapshot: tuple[ActorSnapshot, ...], actor_id: int) -> tuple[float, int | None]:
    """
    IDM accel for actor_id considering EVERY lane its body currently occupies
    (not just current_lane) -- the most conservative (smallest) of the
    per-lane accelerations, with the leader that produced it. This is what a
    car mid-lane-change actually uses each step; a car in a single lane just
    gets that lane's leader. Used for real actuation, so clipped (the default).
    """
    me = _by_id(snapshot, actor_id)
    # occupied_lanes() can be empty for a car that has drifted entirely off
    # every lane band (road departure); fall back to current_lane so this
    # always returns a usable accel rather than None.
    lanes = me.occupied or (me.current_lane,)

    best_accel, best_leader = None, None
    for lane in lanes:
        nb = find_neighbors(snapshot, actor_id, lane)
        a = accel_for(snapshot, actor_id, nb.leader_id)
        if best_accel is None or a < best_accel:
            best_accel, best_leader = a, nb.leader_id
    return best_accel, best_leader


# ---------- MOBIL candidate scoring (best-incentive, no left/right bias) ----------

@dataclass(frozen=True)
class LaneCandidate:
    lane: int
    incentive: float
    safe: bool
    valid: bool               # safe and incentive > a_thr (current lane is always valid, incentive 0)
    a_ego_before: float
    a_ego_after: float
    a_old_follower_before: float
    a_old_follower_after: float
    a_new_follower_before: float
    a_new_follower_after: float


def evaluate_lane_candidates(snapshot: tuple[ActorSnapshot, ...], actor_id: int,
                              road: RoadScenario,
                              rng: np.random.Generator | None = None) -> list[LaneCandidate]:
    """
    Every reachable candidate lane (current +- 1, in range), plus the current
    lane itself as the zero-benefit default. No left-first/right-first bias --
    both directions are always scored; select_best_candidate picks the winner.

    If `rng` is given and an actor's mobil_params.noise_std > 0, a random
    "whim" is added to that actor's changing-lane incentives (never to the
    zero-benefit "stay" candidate, and never relaxing the safety check) -- the
    only place any randomness enters the traffic model, and only ever through
    an explicitly supplied generator (see Scenario.rng), so runs stay
    reproducible under a fixed seed.
    """
    me = _by_id(snapshot, actor_id)
    lane_old = me.current_lane

    nb_old = find_neighbors(snapshot, actor_id, lane_old)
    a_ego_before = accel_for(snapshot, actor_id, nb_old.leader_id, clip=False)
    a_old_follower_before = accel_for(snapshot, nb_old.follower_id, actor_id, clip=False)

    candidates = [LaneCandidate(
        lane=lane_old, incentive=0.0, safe=True, valid=True,
        a_ego_before=a_ego_before, a_ego_after=a_ego_before,
        a_old_follower_before=a_old_follower_before, a_old_follower_after=a_old_follower_before,
        a_new_follower_before=0.0, a_new_follower_after=0.0,
    )]

    for lane_new in (lane_old - 1, lane_old + 1):
        if not (0 <= lane_new < road.lane_num):
            continue

        nb_new = find_neighbors(snapshot, actor_id, lane_new)
        a_ego_after = accel_for(snapshot, actor_id, nb_new.leader_id, clip=False)
        a_old_follower_after = accel_for(snapshot, nb_old.follower_id, nb_old.leader_id, clip=False)
        a_new_follower_before = accel_for(snapshot, nb_new.follower_id, nb_new.leader_id, clip=False)
        a_new_follower_after = accel_for(snapshot, nb_new.follower_id, actor_id, clip=False)

        incentive = mobil_incentive(
            a_ego_after, a_ego_before,
            a_new_follower_after, a_new_follower_before,
            a_old_follower_after, a_old_follower_before,
            me.mobil_params,
        )
        if rng is not None and me.mobil_params.noise_std > 0:
            incentive = incentive + rng.normal(0.0, me.mobil_params.noise_std)

        safe = (nb_new.follower_id is None) or (a_new_follower_after >= -me.mobil_params.b_safe)
        valid = safe and incentive > me.mobil_params.a_thr

        candidates.append(LaneCandidate(
            lane=lane_new, incentive=incentive, safe=safe, valid=valid,
            a_ego_before=a_ego_before, a_ego_after=a_ego_after,
            a_old_follower_before=a_old_follower_before, a_old_follower_after=a_old_follower_after,
            a_new_follower_before=a_new_follower_before, a_new_follower_after=a_new_follower_after,
        ))

    return candidates


def select_best_candidate(candidates: list[LaneCandidate]) -> LaneCandidate:
    """Argmax incentive among valid candidates. The current lane is always
    valid with incentive 0, so this never raises, and staying wins whenever no
    change clears both the safety and threshold bars."""
    valid = [c for c in candidates if c.valid]
    return max(valid, key=lambda c: c.incentive)


# ---------- proposals ----------

@dataclass(frozen=True)
class LaneProposal:
    actor_id: int
    from_lane: int
    to_lane: int
    incentive: float
    safety_margin: float               # a_new_follower_after - (-b_safe); larger = safer
    candidates: tuple[LaneCandidate, ...]  # every evaluated candidate, for diagnostics


def propose_lane_changes(
    snapshot: tuple[ActorSnapshot, ...], road: RoadScenario, t: float, cooldown: float,
    rng: np.random.Generator | None = None,
    eligible_ids: set[int] | None = None,
) -> tuple[list[LaneProposal], dict[int, tuple[LaneCandidate, ...]]]:
    """
    One proposal per eligible actor that (a) isn't mid-manoeuvre, (b) is past
    its post-completion cooldown, and (c) has a best candidate lane != its
    current lane. Everyone else implicitly "stays" (no proposal emitted).

    `snapshot` must contain EVERY actor in the scene, including non-eligible
    ones (e.g. the ego) -- neighbour search needs to see them to score safety
    correctly, even though only `eligible_ids` (default: everyone) gets to
    propose a change. Passing a snapshot that already excludes the ego would
    make MOBIL blind to it and is exactly the kind of bug this parameter
    split prevents.

    Also returns every evaluated actor's full candidate list keyed by actor id
    (including actors whose best candidate was "stay"), for diagnostics -- an
    actor skipped entirely (mid-manoeuvre, in cooldown, or ineligible) has no entry.
    """
    proposals = []
    candidates_by_actor: dict[int, tuple[LaneCandidate, ...]] = {}
    for a in snapshot:
        if eligible_ids is not None and a.id not in eligible_ids:
            continue
        if a.lane_change_active:
            continue
        if t - a.last_lane_change_time < cooldown:
            continue

        candidates = evaluate_lane_candidates(snapshot, a.id, road, rng)
        candidates_by_actor[a.id] = tuple(candidates)
        best = select_best_candidate(candidates)
        if best.lane == a.current_lane:
            continue

        safety_margin = best.a_new_follower_after + a.mobil_params.b_safe
        proposals.append(LaneProposal(
            actor_id=a.id, from_lane=a.current_lane, to_lane=best.lane,
            incentive=best.incentive, safety_margin=safety_margin,
            candidates=tuple(candidates),
        ))
    return proposals, candidates_by_actor


# ---------- deterministic simultaneous conflict resolution ----------

def resolve_conflicts(proposals: list[LaneProposal], snapshot: tuple[ActorSnapshot, ...],
                       min_gap: float = 4.0) -> tuple[list[LaneProposal], list[tuple[LaneProposal, str]]]:
    """
    Priority order (fixed, independent of input list order):
        1. higher MOBIL incentive
        2. larger safety margin
        3. lower actor id (final, arbitrary-but-stable tie-break)

    Proposals are accepted greedily in that order; a proposal conflicts with an
    already-accepted one (and is rejected) if:
      - both target the same lane and their current longitudinal gap is below
        min_gap                                     ("insufficient shared gap")
      - they're a direct swap (p.from == a.to and p.to == a.from) within
        min_gap + width                                    ("swap through each other")
      - their (from_lane, to_lane) sets intersect at all and they're within
        min_gap longitudinally                            ("trajectory overlap")

    Returns (accepted, rejected_with_reason) -- rejections are reported, not
    silently dropped, so diagnostics can record why.
    """
    ordered = sorted(proposals, key=lambda p: (-p.incentive, -p.safety_margin, p.actor_id))
    accepted: list[LaneProposal] = []
    rejected: list[tuple[LaneProposal, str]] = []

    for p in ordered:
        p_snap = _by_id(snapshot, p.actor_id)
        reason = None

        for a in accepted:
            a_snap = _by_id(snapshot, a.actor_id)
            long_gap = abs(p_snap.state[0] - a_snap.state[0]) - (p_snap.length + a_snap.length) / 2.0

            if p.to_lane == a.to_lane and long_gap < min_gap:
                reason = "insufficient gap in shared target lane"
            elif (p.from_lane == a.to_lane and p.to_lane == a.from_lane
                  and long_gap < min_gap + max(p_snap.width, a_snap.width)):
                reason = "swap through each other"
            elif ({p.from_lane, p.to_lane} & {a.from_lane, a.to_lane}) and long_gap < min_gap:
                reason = "trajectory overlap"

            if reason is not None:
                break

        if reason is None:
            accepted.append(p)
        else:
            rejected.append((p, reason))

    return accepted, rejected


def apply_accepted_proposals(cars_by_id: dict[int, Car], accepted: list[LaneProposal], t: float) -> None:
    """Start the lane-change state machine for every accepted proposal."""
    from model.lane_change import start_lane_change
    for p in accepted:
        start_lane_change(cars_by_id[p.actor_id], p.to_lane, t)
