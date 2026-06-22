"""
Scenario classes for environment design.

Each scenario is a class with:
  - M: integer class identifier
  - to_vector() / from_vector(): fixed-length numpy encoding
  - ego_init_state(): initial [e_y, psi, e_psi, psi_dot, v_x, v_y]
  - active_vehicles(): list of valid surrounding vehicle specs

Current classes
---------------
  M=0  SandwichScenario  —  highway with N surrounding vehicles

Environment vector layout for SandwichScenario (length 49):
  [0]     M          = 0
  [1]     n_lanes    in {1, 2, 3}
  [2]     ego_lane   in {0, …, n_lanes-1}  (0 = rightmost)
  [3]     lane_width (m)
  [4]     k0         constant curvature (rad/m)
  [5]     k1         linear curvature coefficient (rad/m²)
  [6]     e_y_ego    initial within-lane SAE J670 lateral offset (m)
  [7]     e_psi_ego  initial heading error (rad)
  [8]     v_x_ego    initial longitudinal speed (m/s)
  [9:49]  10 × (x_rel, v_x, lane, valid)
             x_rel  : m ahead (+) or behind (−) of ego
             v_x    : speed (m/s, must be > 0 when valid=1)
             lane   : 0 = rightmost
             valid  : 1 = vehicle exists, 0 = empty slot

Lane convention:  0 = rightmost,  n_lanes-1 = leftmost.
e_y convention:   SAE J670 — zero at lane centre, positive rightward.
Curvature:        κ(s) = k0 + k1·s  (positive = left turn).
Frenet reference: left road boundary.
"""

import numpy as np
from typing import List, Optional

N_VEH_MAX = 10


class SandwichScenario:
    M       = 0
    VEC_LEN = 6 + 3 + 4 * N_VEH_MAX   # 49

    def __init__(
        self,
        n_lanes    : int   = 3,
        ego_lane   : int   = 1,
        lane_width : float = 3.7,
        k0         : float = 0.0,
        k1         : float = 0.0,
        ego_ey     : float = 0.0,
        ego_epsi   : float = 0.0,
        v0         : float = 25.0,
        vehicles   : Optional[List[dict]] = None,
        dt         : float = 0.05,
        t_end      : float = 8.0,
    ):
        assert 1 <= n_lanes <= 3,          "n_lanes must be 1, 2, or 3"
        assert 0 <= ego_lane < n_lanes,    "ego_lane out of range"
        assert lane_width > 0
        assert v0 > 0

        self.n_lanes    = n_lanes
        self.ego_lane   = ego_lane
        self.lane_width = lane_width
        self.k0         = k0
        self.k1         = k1
        self.ego_ey     = ego_ey
        self.ego_epsi   = ego_epsi
        self.v0         = v0
        self.dt         = dt
        self.t_end      = t_end

        base = vehicles if vehicles is not None else []
        self.vehicles: List[dict] = []
        for i in range(N_VEH_MAX):
            if i < len(base):
                v = base[i]
                self.vehicles.append({
                    'x_rel': float(v['x_rel']),
                    'v_x'  : float(v['v_x']),
                    'lane' : int(v['lane']),
                    'valid': int(v.get('valid', 1)),
                })
            else:
                self.vehicles.append(
                    {'x_rel': 0.0, 'v_x': 0.0, 'lane': 0, 'valid': 0}
                )

    # ── geometry helpers ──────────────────────────────────────────────────────

    def lane_center_abs(self, lane: int) -> float:
        """Rightward distance from left road boundary to centre of lane."""
        return (self.n_lanes - lane - 0.5) * self.lane_width

    def road_width(self) -> float:
        return self.n_lanes * self.lane_width

    # ── initial conditions ────────────────────────────────────────────────────

    def ego_init_state(self) -> np.ndarray:
        """Initial ego state [e_y, psi, e_psi, psi_dot, v_x, v_y]."""
        return np.array([self.ego_ey, 0.0, self.ego_epsi, 0.0, self.v0, 0.0])

    def active_vehicles(self) -> List[dict]:
        """Return only valid (existing) surrounding vehicle specs."""
        return [v for v in self.vehicles if v['valid'] == 1]

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_vector(self) -> np.ndarray:
        """Encode as flat numpy vector of length VEC_LEN=49."""
        vec = np.zeros(self.VEC_LEN)
        vec[0] = self.M
        vec[1] = self.n_lanes
        vec[2] = self.ego_lane
        vec[3] = self.lane_width
        vec[4] = self.k0
        vec[5] = self.k1
        vec[6] = self.ego_ey
        vec[7] = self.ego_epsi
        vec[8] = self.v0
        for i, v in enumerate(self.vehicles):
            b = 9 + 4 * i
            vec[b], vec[b+1] = v['x_rel'], v['v_x']
            vec[b+2], vec[b+3] = v['lane'], v['valid']
        return vec

    @classmethod
    def from_vector(cls, vec, dt: float = 0.05, t_end: float = 8.0):
        vec = np.asarray(vec, float)
        assert len(vec) == cls.VEC_LEN, f"Expected {cls.VEC_LEN} elements, got {len(vec)}"
        assert int(round(vec[0])) == cls.M, f"M mismatch: expected {cls.M}"

        vehicles = []
        for i in range(N_VEH_MAX):
            b = 9 + 4 * i
            vehicles.append({
                'x_rel': float(vec[b]),
                'v_x'  : float(vec[b+1]),
                'lane' : int(round(vec[b+2])),
                'valid': int(round(vec[b+3])),
            })

        return cls(
            n_lanes    = int(round(vec[1])),
            ego_lane   = int(round(vec[2])),
            lane_width = float(vec[3]),
            k0         = float(vec[4]),
            k1         = float(vec[5]),
            ego_ey     = float(vec[6]),
            ego_epsi   = float(vec[7]),
            v0         = float(vec[8]),
            vehicles   = vehicles,
            dt         = dt,
            t_end      = t_end,
        )

    def __repr__(self):
        n_active = sum(1 for v in self.vehicles if v['valid'])
        return (
            f"SandwichScenario(n_lanes={self.n_lanes}, ego_lane={self.ego_lane}, "
            f"lw={self.lane_width}, k0={self.k0:.4f}, k1={self.k1:.5f}, "
            f"v0={self.v0}, vehicles={n_active}/{N_VEH_MAX})"
        )
