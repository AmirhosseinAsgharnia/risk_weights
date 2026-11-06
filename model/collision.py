"""
Surr-surr collision detection and perfectly-plastic post-crash response.

Scope: surrounding ("surr") traffic only. The ego vehicle is never run
through the plastic-crash model here -- an ego/surr overlap is a
different event entirely (episode termination, not a wreck) and is left
to the caller; see `ego_overlaps_any` below.

Modeling choices (deliberate simplifications -- this is a stopping
obstacle model, not a crash-mechanics model):
  - restitution e = 0 (perfectly plastic): colliding cars share one
    momentum-conserved post-impact longitudinal speed.
  - longitudinal momentum only -- no rotation, no crumple geometry, no
    energy partition.
  - lateral dynamics collapse at the instant of impact: e_y freezes
    wherever it was, e_psi -> 0. The lateral merge dynamics *during* the
    impact itself are not resolved.
  - two crashed bodies that later overlap (e.g. a pileup) stay distinct
    and co-located -- never fused into one rigid body. Hitting an
    already-crashed car only affects the *new* arrival; the standing
    wreck's own state is left untouched.
  - overlap is tested as an axis-aligned box in the road's own (s, e_y)
    frame (CAR_LENGTH x CAR_WIDTH, heading ignored) -- consistent with
    how every other module here reasons about position.
"""

import math

from initialization.traffic_init import CAR_LENGTH, CAR_WIDTH
from model.car.config import G

_BODY_RADIUS = math.hypot(CAR_LENGTH, CAR_WIDTH) / 2.0   # centre to corner: no overlap beyond twice this


def lane_backbone_e_y(agent, road) -> float:
    """This car's lateral offset in the road's common backbone frame
    (same convention as the frenet_to_global call in the sim loop) --
    the shared coordinate that makes cars in different lanes comparable."""
    lane_obj = road.lanes[agent.car.state.lane]
    return -lane_obj.offset + agent.car.state.e_y  # type: ignore


def bodies_overlap(agent_a, agent_b, road) -> bool:
    """Axis-aligned CAR_LENGTH x CAR_WIDTH box overlap in (s, e_y)."""
    ds  = abs(agent_a.car.state.s - agent_b.car.state.s)
    dey = abs(lane_backbone_e_y(agent_a, road) - lane_backbone_e_y(agent_b, road))
    return ds < CAR_LENGTH and dey < CAR_WIDTH


def body_half_extent_e_y(e_psi: float) -> float:
    """[m] Half the lateral (e_y) extent of a CAR_LENGTH x CAR_WIDTH body at heading error e_psi."""
    return 0.5 * (CAR_LENGTH * abs(math.sin(e_psi)) + CAR_WIDTH * abs(math.cos(e_psi)))


def _body_corners(s: float, e_y: float, e_psi: float) -> list[tuple[float, float]]:
    """Corners, in order, of the CAR_LENGTH x CAR_WIDTH body in the (s, e_y) frame; the body's long
    axis is (cos(e_psi), sin(e_psi)) there, which is d(s, e_y)/dt at v_y = 0."""
    c, si, hl, hw = math.cos(e_psi), math.sin(e_psi), CAR_LENGTH / 2.0, CAR_WIDTH / 2.0
    return [(s + a * hl * c - b * hw * si, e_y + a * hl * si + b * hw * c)
            for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def _projections_overlap(ca: list, cb: list) -> bool:
    """Separating-axis test on the two rectangles' four edge normals. Strict, like bodies_overlap:
    touching is not overlapping."""
    for (x0, y0), (x1, y1) in (ca[0], ca[1]), (ca[1], ca[2]), (cb[0], cb[1]), (cb[1], cb[2]):
        nx, ny = y1 - y0, x0 - x1
        pa = [nx * x + ny * y for x, y in ca]
        pb = [nx * x + ny * y for x, y in cb]
        if max(pa) <= min(pb) or max(pb) <= min(pa):
            return False
    return True


def _point_segment_distance(p, a, b) -> float:
    abx, aby = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / (abx * abx + aby * aby)))
    return math.hypot(p[0] - a[0] - t * abx, p[1] - a[1] - t * aby)


def body_gap(s_a: float, e_y_a: float, e_psi_a: float, s_b: float, e_y_b: float, e_psi_b: float) -> float:
    """[m] Distance between two oriented bodies in the (s, e_y) frame, exactly 0 when they overlap."""
    ca, cb = _body_corners(s_a, e_y_a, e_psi_a), _body_corners(s_b, e_y_b, e_psi_b)
    if _projections_overlap(ca, cb):
        return 0.0
    return min(min(_point_segment_distance(p, poly[i], poly[(i + 1) % 4]) for p in other for i in range(4))
               for other, poly in ((ca, cb), (cb, ca)))


def _plastic_speed(agent_a, agent_b) -> float:
    """Momentum-conserved common post-impact v_x; falls back to the mean
    of the two speeds if the combined mass is degenerate (guards div/0,
    never NaN)."""
    m_a = agent_a.car.vehicle_params.m
    m_b = agent_b.car.vehicle_params.m
    total = m_a + m_b
    if total <= 0.0:
        return 0.5 * (agent_a.car.state.v_x + agent_b.car.state.v_x)
    return (m_a * agent_a.car.state.v_x + m_b * agent_b.car.state.v_x) / total


def _crash(agent, v_c: float, t: float) -> None:
    """Transition one not-yet-crashed agent into the CRASHED state:
    common post-impact speed (clamped >= 0), e_y frozen where it is
    (simply left untouched), e_psi/v_y/r/delta collapsed to 0."""
    agent.crashed = True
    agent.crash_t = t
    agent.crash_v = max(v_c, 0.0)
    state = agent.car.state
    state.v_x   = agent.crash_v
    state.e_psi = 0.0
    state.v_y   = 0.0
    state.r     = 0.0
    state.delta = 0.0


def resolve_surr_collisions(agents, road, t: float) -> None:
    """One O(n^2) sweep over surr-surr pairs -- n is small here, a
    spatial-hash speedup would be over-engineering. Call once per step,
    after every not-yet-crashed agent has moved for this step.

    A pair already both CRASHED is skipped (a car crashes once). A pair
    with exactly one CRASHED side still collides: the moving side
    coalesces into the wreck (its own momentum-conserved v_c against the
    wreck's *current* v_x) and joins it; the standing wreck is left
    untouched -- this is what turns a stopped wreck into a growing
    pileup instead of a single-crash special case. A pair with neither
    side CRASHED becomes a fresh mutual crash sharing one v_c.
    """
    n = len(agents)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = agents[i], agents[j]
            if a.crashed and b.crashed:
                continue
            if not bodies_overlap(a, b, road):
                continue
            v_c = _plastic_speed(a, b)
            if not a.crashed:
                _crash(a, v_c, t)
            if not b.crashed:
                _crash(b, v_c, t)


def bleed_crashed(agent, road, dt: float, k: float = 1.0, g: float = G) -> None:
    """Advance one CRASHED car for this step: decelerate v_x toward 0 at
    a = k * road.mu * g (low mu => longer slide => the wreck lands
    deeper into whatever's behind it -- intentional coupling), clamp at
    0, and integrate s with the trapezoidal average speed. Once v_x
    reaches 0 no special-case is needed -- s simply stops advancing."""
    state = agent.car.state
    v_new = max(0.0, state.v_x - k * road.mu * g * dt)
    state.s += 0.5 * (state.v_x + v_new) * dt
    state.v_x = v_new


def ego_overlaps_any(ego_s: float, ego_e_y_backbone: float, agents, road, ego_e_psi: float = 0.0) -> bool:
    """EGO RULE hook: True if an ego body (CAR_LENGTH x CAR_WIDTH, centred
    at (ego_s, ego_e_y_backbone) in the same backbone frame as
    `lane_backbone_e_y`, turned by ego_e_psi) overlaps ANY surr body, each
    turned by its own e_psi, crashed or not. Heading matters: a body at
    30 deg is 3.7 m wide across the road, not 2 m. Callers
    must treat a True return as episode termination -- never route it
    through resolve_surr_collisions/_crash; ego is never converted into a
    wreck.

    Not yet wired into tests/traffic_test.py: that script has no ego
    vehicle instantiated (see its own "no ego car yet" notes) -- only a
    reference point ego_s traffic is generated around. Call this once an
    ego Car/CarState exists, each step, before/instead of resolving surr-
    surr collisions for that same step.
    """
    ego_corners = _body_corners(ego_s, ego_e_y_backbone, ego_e_psi)
    for agent in agents:
        s, e_y = agent.car.state.s, lane_backbone_e_y(agent, road)
        if math.hypot(ego_s - s, ego_e_y_backbone - e_y) >= 2.0 * _BODY_RADIUS:
            continue
        if _projections_overlap(ego_corners, _body_corners(s, e_y, agent.car.state.e_psi)):
            return True
    return False
