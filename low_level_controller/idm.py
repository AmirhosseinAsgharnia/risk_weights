"""
Intelligent Driver Model (IDM) — longitudinal speed/car-following controller.

For surrounding traffic only, not the ego vehicle (ego will be driven by an
MPC, designed later).
"""

import math
from typing import NamedTuple


def idm_accel(v: float, gap: float, dv: float,
              v0: float, a_max: float, b: float,
              s0: float = 2.0, T: float = 1.5, delta: float = 4.0,
              gap_floor: float = 1e-3) -> float:
    """
    IDM longitudinal acceleration:
        a  = a_max * (1 - (v / v0)^delta - (s_star / gap)^2)
        s_star = s0 + max(0, v*T + v*dv / (2 * sqrt(a_max * b)))

    With no leader (gap = inf), the interaction term (s_star/gap)^2 -> 0 and
    this reduces to pure free-road acceleration toward v0 — pass
    math.inf as gap for that case rather than special-casing it.

    Inputs:
      v:         [m/s] this car's current speed.
      gap:       [m] bumper-to-bumper gap to the leader ahead; math.inf for
                 free-road driving with no leader.
      dv:        [m/s] closing speed v - v_leader; positive means
                 approaching the leader. Irrelevant when gap is inf.
      v0:        [m/s] desired (free-road) speed.
      a_max:     [m/s^2] maximum acceleration.
      b:         [m/s^2] comfortable deceleration (used only inside s_star).
      s0:        [m] minimum (jam) gap at a standstill.
      T:         [s] desired time headway to the leader.
      delta:     [-] free-road acceleration exponent (4 is the IDM default).
      gap_floor: [m] floor on gap, avoids a division blow-up if the car is
                 (numerically) touching its leader.

    Returns:
      a: [m/s^2] commanded longitudinal acceleration (unclamped — this is
         the raw IDM value, not saturated against the vehicle's true
         actuation limits).
    """
    gap = max(gap, gap_floor)
    s_star = s0 + max(0.0, v * T + (v * dv) / (2.0 * math.sqrt(a_max * b)))
    return a_max * (1.0 - (v / v0) ** delta - (s_star / gap) ** 2)


class IDMBehaviorParams(NamedTuple):
    """IDM tuning for one driving style. Deliberately excludes v0 (desired
    speed) -- unlike the other fields here, that's drawn per-car, not
    fixed by behaviour (see initialization.traffic_init)."""
    T: float       # [s] desired time headway
    s0: float      # [m] minimum (jam) gap
    a_max: float   # [m/s^2] max acceleration
    b: float       # [m/s^2] comfortable deceleration
    delta: float = 4.0   # [-] free-road acceleration exponent


IDM_PRESETS: dict[int, IDMBehaviorParams] = {
    1: IDMBehaviorParams(T=2.0, s0=3.0, a_max=0.8, b=1.3),   # conservative
    2: IDMBehaviorParams(T=1.4, s0=2.0, a_max=1.5, b=2.0),   # moderate
    3: IDMBehaviorParams(T=0.8, s0=1.0, a_max=2.5, b=3.0),   # aggressive
}
