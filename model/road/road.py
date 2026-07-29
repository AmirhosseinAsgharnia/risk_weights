import numpy as np
from scipy.integrate import cumulative_trapezoid
from dataclasses import dataclass

class Road:
    "Clothoid Roal Class"

    L_patch = 80.0 # [m] friction patch length, fixed

    def __init__(self,
                 s_max     = 500,
                 kappa_max = 0.02,
                 L_clothoid= 50,
                 lane_num  = 3,
                 mu_road   = 1.0,
                 mu_patch  = 0.3,
                 patch_location = 225.0):
        
        self.s_max     = s_max
        self.kappa_max = kappa_max

        self.L_clothoid = L_clothoid
        self.L_curve = 80 # constant
        self.L_e = (s_max - self.L_clothoid * 2 - self.L_curve) / 2
        self.L_1 = (s_max - self.L_clothoid * 2 - self.L_curve) / 2

        self.l_w = 4.0 # [m] lane width
        self.lane_num = lane_num

        self.ds    = 0.1
        self.s     = np.arange(0 , s_max , self.ds)
        self.mu    = mu_road * np.ones_like(self.s)
        self.sigma = self.kappa_max / self.L_clothoid

    @property
    def kappa(self):

        "Building the curvature"

        kappa = np.zeros_like(self.s)

        # kappa = 0
        mask_1 = (self.s >= 0) & (self.s < self.L_1)
        kappa[mask_1] = 0

        # kappa is evolving to kappa_max
        mask_2 = (self.s >= self.L_1) & (self.s < self.L_1 + self.L_clothoid)
        kappa[mask_2] = self.sigma * (self.s[mask_2] - self.L_1)

        # kappa is equal to kappa_max
        mask_3 = (self.s >= self.L_1 + self.L_clothoid) & (self.s < self.L_1 + self.L_clothoid + self.L_curve)
        kappa[mask_3] = self.kappa_max

        # kappa is decreasing to zero 
        mask_4 = (self.s >= self.L_1 + self.L_clothoid + self.L_curve) & (self.s < self.L_1 + self.L_clothoid + self.L_curve + self.L_clothoid)
        kappa[mask_4] = self.kappa_max - self.sigma * (self.s[mask_4] - self.L_1 - self.L_clothoid - self.L_curve)

        # kappa = 0
        mask_5 = (self.s >= self.L_1 + self.L_clothoid + self.L_curve + self.L_clothoid) & (self.s <= self.s_max)
        kappa[mask_5] = 0

        return kappa

    @property
    def heading(self):

        "Building the heading"

        heading = np.zeros_like(self.s)

        heading = cumulative_trapezoid(self.kappa , self.s , self.ds , initial = 0.0)

        return heading

    @property
    def x_road(self):
        return cumulative_trapezoid(np.cos(self.heading) , self.s , self.ds , initial = 0.0)

    @property
    def y_road(self):
        return cumulative_trapezoid(np.sin(self.heading) , self.s , self.ds , initial = 0.0)