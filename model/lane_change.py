"""
Lane occupancy geometry and the lane-change state machine + trajectory.

MOBIL (model.traffic) decides *whether* and *where* to change lanes. This module
owns the continuous *how*: a smooth quintic reference for e_y, and the
current_lane/target_lane state machine that governs when a manoeuvre starts,
what it's tracking mid-flight, and when it's done.
"""

import numpy as np
from dataclasses import dataclass
from model.road.road import RoadScenario


@dataclass(frozen=True)
class LaneChangeTolerances:
    e_y: float = 0.15     # [m]   lateral-offset tolerance to call a change "complete"
    e_psi: float = 0.03   # [rad] heading-error tolerance
    vy: float = 0.3       # [m/s] lateral-velocity tolerance


# ---------- lane occupancy (lateral body extent vs. lane bands) ----------

def occupied_lanes(road: RoadScenario, e_y: float, width: float) -> tuple[int, ...]:
    """
    Which lanes a vehicle's body overlaps, from its lateral extent
    [e_y - width/2, e_y + width/2] against each lane's [d_c[n] +- l_w/2] band.
    A car straddling a lane boundary (e.g. mid-lane-change) naturally returns two
    lane indices -- callers never need to special-case "in transition".
    """
    lo, hi = e_y - width / 2.0, e_y + width / 2.0
    lanes = []
    for n in range(road.lane_num):
        band_lo = road.d_c[n] - road.l_w / 2.0
        band_hi = road.d_c[n] + road.l_w / 2.0
        if lo < band_hi and hi > band_lo:
            lanes.append(n)
    return tuple(lanes)


# ---------- quintic lane-change reference ----------

def _q(tau):
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def _q_dot(tau):
    """d(_q)/d(tau)."""
    return 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4


def lane_change_reference(car, road: RoadScenario, t: float) -> tuple[float, float]:
    """
    (e_y_ref, e_y_ref_rate) to track right now.

    Not actively changing lanes: hold the current lane's centreline, zero rate.
    Actively changing: a quintic blend from the lane-change's start offset
    (car.e_y_start, captured once in start_lane_change -- never recomputed here)
    to the target lane's centreline, zero rate/acceleration at both tau=0 and
    tau=1 by construction of the polynomial.
    """
    if not car.lane_change_active:
        return road.d_c[car.current_lane], 0.0

    duration = car.lane_change_duration
    tau = min(max((t - car.lane_change_start_time) / duration, 0.0), 1.0)

    e_y_target = road.d_c[car.target_lane]
    span = e_y_target - car.e_y_start
    e_y_ref = car.e_y_start + span * _q(tau)
    e_y_ref_rate = span * _q_dot(tau) / duration
    return e_y_ref, e_y_ref_rate


# ---------- lane-change state machine ----------

def start_lane_change(car, target_lane: int, t: float) -> None:
    """Begin a manoeuvre toward target_lane. e_y_start is captured HERE, once."""
    car.target_lane = target_lane
    car.lane_change_active = True
    car.lane_change_start_time = t
    car.e_y_start = car.state[1]
    # fresh manoeuvre, fresh steering-PID integrator -- don't carry over
    # whatever trim the old lane-holding accumulated.
    car.lane_error_integral = 0.0


def try_complete_lane_change(car, road: RoadScenario, t: float,
                              tol: LaneChangeTolerances | None = None) -> bool:
    """
    If the manoeuvre's quintic duration has elapsed AND the vehicle has actually
    converged (within `tol` on e_y, e_psi, vy), promote current_lane to
    target_lane, end the manoeuvre, and start the cooldown (last_lane_change_time
    is set here, on completion -- not on initiation).
    """
    if not car.lane_change_active:
        return False

    tol = tol or LaneChangeTolerances()
    e_y, e_psi, vy = car.state[1], car.state[2], car.state[4]
    e_y_target = road.d_c[car.target_lane]

    tau_done = (t - car.lane_change_start_time) >= car.lane_change_duration
    converged = (
        abs(e_y - e_y_target) < tol.e_y
        and abs(e_psi) < tol.e_psi
        and abs(vy) < tol.vy
    )

    if tau_done and converged:
        car.current_lane = car.target_lane
        car.lane_change_active = False
        car.lane_change_start_time = None
        car.e_y_start = None
        car.last_lane_change_time = t
        return True
    return False
