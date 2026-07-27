"""
Explicit safety-event detection: vehicle-vehicle collisions (broad + narrow
phase), road/lane-boundary departure, Frenet singularity proximity, and a
loss-of-control detector. None of this is inferred from IDM gaps -- IDM only
ever sees a gap number; whether that gap represents an actual overlapping body
is decided here, independently, from vehicle geometry.
"""

import numpy as np
from dataclasses import dataclass
from model.road import RoadScenario
from model.car_init import frenet_to_global
from model.traffic import ActorSnapshot


# ---------- oriented-rectangle collision (SAT) ----------

def rectangle_corners(x, y, psi, length, width):
    """Four global-frame corners of a vehicle's body, centred at (x, y)."""
    hl, hw = length / 2.0, width / 2.0
    local = ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
    c, s = np.cos(psi), np.sin(psi)
    return [(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local]


def _edge_axes(corners):
    axes = []
    n = len(corners)
    for i in range(n):
        (x1, y1), (x2, y2) = corners[i], corners[(i + 1) % n]
        axes.append((-(y2 - y1), x2 - x1))  # outward normal of this edge
    return axes


def _project(corners, axis):
    dots = [px * axis[0] + py * axis[1] for px, py in corners]
    return min(dots), max(dots)


def sat_overlap(corners_a, corners_b) -> bool:
    """Separating Axis Theorem for two convex quads: they overlap iff no
    candidate axis (each rectangle's two unique edge normals) separates them."""
    for axis in _edge_axes(corners_a) + _edge_axes(corners_b):
        norm = np.hypot(axis[0], axis[1])
        if norm < 1e-9:
            continue
        amin, amax = _project(corners_a, axis)
        bmin, bmax = _project(corners_b, axis)
        if amax < bmin or bmax < amin:
            return False
    return True


# ---------- broad phase ----------

def broad_phase_overlap(a: ActorSnapshot, b: ActorSnapshot, margin: float = 0.5) -> bool:
    """Cheap Frenet-frame filter using longitudinal and lateral extents; a
    False here guarantees no collision, so the (more expensive) narrow phase
    only ever runs on plausible pairs."""
    long_gap = abs(a.state[0] - b.state[0]) - (a.length + b.length) / 2.0 - margin
    if long_gap > 0:
        return False
    lat_close = abs(a.state[1] - b.state[1]) - (a.width + b.width) / 2.0 - margin <= 0
    lanes_share = bool(set(a.occupied) & set(b.occupied))
    return bool(lat_close or lanes_share)


def narrow_phase_collision(a: ActorSnapshot, b: ActorSnapshot, road: RoadScenario) -> bool:
    ax, ay, apsi = frenet_to_global(road, a.state[0], a.state[1], a.state[2])
    bx, by, bpsi = frenet_to_global(road, b.state[0], b.state[1], b.state[2])
    corners_a = rectangle_corners(ax, ay, apsi, a.length, a.width)
    corners_b = rectangle_corners(bx, by, bpsi, b.length, b.width)
    return sat_overlap(corners_a, corners_b)


def classify_collision_type(a: ActorSnapshot, b: ActorSnapshot) -> str:
    """Heuristic classification from lane/manoeuvre state -- NOT a contact-force
    simulation, just a label for logging/analysis."""
    if a.lane_change_active or b.lane_change_active:
        return "sideswipe_lane_change"
    if a.current_lane == b.current_lane:
        return "rear_end"
    if set(a.occupied) & set(b.occupied):
        return "sideswipe"
    return "other"


@dataclass(frozen=True)
class CollisionEvent:
    collision: bool
    actor_ids: tuple[int, int]
    time: float
    collision_type: str
    relative_speed: float


def check_collisions(snapshot: tuple[ActorSnapshot, ...], road: RoadScenario,
                      t: float, margin: float = 0.5) -> list[CollisionEvent]:
    events = []
    n = len(snapshot)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = snapshot[i], snapshot[j]
            if not broad_phase_overlap(a, b, margin):
                continue
            if narrow_phase_collision(a, b, road):
                events.append(CollisionEvent(
                    collision=True,
                    actor_ids=(a.id, b.id),
                    time=t,
                    collision_type=classify_collision_type(a, b),
                    relative_speed=abs(a.state[3] - b.state[3]),
                ))
    return events


# ---------- road / lane boundary + singularity checks ----------

def check_lane_index_valid(lane: int, lane_num: int) -> bool:
    return 0 <= lane < lane_num


def check_road_departure(actor: ActorSnapshot, road: RoadScenario) -> bool:
    """True if the vehicle's BODY (not just its centre) is entirely outside the
    outermost lane boundaries."""
    lo = actor.state[1] - actor.width / 2.0
    hi = actor.state[1] + actor.width / 2.0
    outer_right = road.d_c[0] + road.l_w / 2.0    # lane 0 is rightmost (road.py convention)
    outer_left = road.d_c[-1] - road.l_w / 2.0    # lane N-1 is leftmost
    return bool(hi < outer_left or lo > outer_right)


def check_singularity_risk(kappa: float, e_y: float, threshold: float = 0.8) -> bool:
    """True when the Frenet projection 1 - kappa*e_y is getting close to 0."""
    return bool(abs(1.0 - kappa * e_y) < threshold)


# ---------- loss of control ----------

@dataclass(frozen=True)
class LossOfControlThresholds:
    sideslip: float = 0.3              # [rad] approx |vy/vx| threshold (small-angle atan(vy/vx))
    yaw_rate: float = 2.0              # [rad/s]
    saturation_duration: float = 1.0   # [s] sustained tire saturation before flagging
    v_min: float = 1.0                 # [m/s] low-speed guard for the sideslip ratio


def loss_of_control(actor: ActorSnapshot, saturation_streak_steps: int, dt: float,
                     thresholds: LossOfControlThresholds | None = None) -> bool:
    """Kept independent of EgoModel/car_model -- reads only state + a streak
    counter the caller (model.simulation) maintains."""
    thresholds = thresholds or LossOfControlThresholds()
    vx_safe = max(actor.state[3], thresholds.v_min)
    beta = abs(actor.state[4]) / vx_safe
    if beta > thresholds.sideslip:
        return True
    if abs(actor.state[5]) > thresholds.yaw_rate:
        return True
    if saturation_streak_steps * dt > thresholds.saturation_duration:
        return True
    return False
