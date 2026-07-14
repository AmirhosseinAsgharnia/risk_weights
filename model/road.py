import numpy as np
from scipy.integrate import cumulative_trapezoid
from dataclasses import dataclass

@dataclass
class Lane:
    index: int
    x: float
    y: float
    l_w: float

class RoadScenario(Lane):
    "Clothoid Roal Class"

    def __init__(self, S_max = 500 , kappa_max = 0.01, L_s = 60 , lane_num = 3):
        
        self.s_max = S_max
        self.kappa_max = kappa_max
        
        self.L_s = L_s
        self.L_c = 80 # constant
        self.L_e = (S_max - self.L_s * 2 - self.L_c) / 2
        self.L_1 = (S_max - self.L_s * 2 - self.L_c) / 2

        self.l_w = 3.8 # [m] lane width 

        self.lane_num = lane_num

        self.ds = 0.1
        self.s = np.arange(0 , S_max , self.ds)
        self.kappa = np.zeros_like(self.s)
        self.sigma = self.kappa_max / self.L_s

        self.centreline_builder()

    def _lane_centre(self):
        """
        Lane centrelines offset from the road reference centreline.

        Convention: d > 0 is LEFT of the direction of travel.
        Lanes are ordered rightmost (index 0) -> leftmost (index N-1).
        self.x, self.y, self.heading are arrays sampled along s.
        """
        N = int(self.lane_num)
        if N < 1:
            raise ValueError(f"lane_num must be >= 1, got {N}")

        # lateral offset of each lane centre from the reference line
        self.d_c = (np.arange(N, dtype=np.float64) - (N - 1) / 2.0) * self.l_w

        # left-hand unit normal to the tangent (cos psi, sin psi)
        n_x = -np.sin(self.heading)
        n_y = np.cos(self.heading)

        # (N, S) arrays: outer product of offsets with the normal field
        self.x_lanes = self.x[None, :] + self.d_c[:, None] * n_x[None, :]
        self.y_lanes = self.y[None, :] + self.d_c[:, None] * n_y[None, :]



    def kappa_builder(self):

        "Building the curvature"

        mask_1 = (self.s >= 0) & (self.s < self.L_1)
        self.kappa[mask_1] = 0

        mask_2 = (self.s >= self.L_1) & (self.s < self.L_1 + self.L_s)
        self.kappa[mask_2] = self.sigma * (self.s[mask_2] - self.L_1)

        mask_3 = (self.s >= self.L_1 + self.L_s) & (self.s < self.L_1 + self.L_s + self.L_c)
        self.kappa[mask_3] = self.kappa_max

        mask_4 = (self.s >= self.L_1 + self.L_s + self.L_c) & (self.s < self.L_1 + self.L_s + self.L_c + self.L_s)
        self.kappa[mask_4] = self.kappa_max - self.sigma * (self.s[mask_4] - self.L_1 - self.L_s - self.L_c)

        mask_5 = (self.s >= self.L_1 + self.L_s + self.L_c + self.L_s) & (self.s <= self.s_max)
        self.kappa[mask_5] = 0
    
    def heading_builder(self):

        "Building the heading"

        self.heading = np.zeros_like(self.s)

        self.heading = cumulative_trapezoid(self.kappa , self.s , self.ds , initial = 0.0)

    def cartesean_builder(self):

        self.x = cumulative_trapezoid(np.cos(self.heading) , self.s , self.ds , initial = 0.0)
        self.y = cumulative_trapezoid(np.sin(self.heading) , self.s , self.ds , initial = 0.0)

    def centreline_builder(self):

        self.kappa_builder()
        self.heading_builder()
        self.cartesean_builder()

