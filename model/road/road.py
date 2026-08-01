import numpy as np
from scipy.integrate import cumulative_trapezoid
from dataclasses import dataclass

@dataclass
class Lane:
    
    lane_id : int
    width   : float      | None = None
    offset  : float      | None = None
    x       : np.ndarray | None = None
    y       : np.ndarray | None = None
    s       : np.ndarray | None = None
    heading : np.ndarray | None = None
    kappa   : np.ndarray | None = None
    mu      : np.ndarray | None = None

class Road:
    "Clothoid Roal Class"

    L_patch = 80.0 # [m] friction patch length, fixed

    def __init__(self,
                 s_max     = 500,
                 kappa_max = 0.02,
                 L_clothoid= 50,
                 mu_road   = 1.0,
                 mu_patch  = 0.3,
                 patch_location = 225.0,
                 lane_num  = 3):
        
        self.s_max      = s_max
        self.kappa_max  = kappa_max
        self.L_clothoid = L_clothoid
        self.mu         = mu_road
        self.mu_patch   = mu_patch
        self.patch_location = patch_location
        self.lane_num   = lane_num

        self.l_w = 4.0 # [m] lane width
        
        self.ds    = 0.1
        self.s     = np.arange(0 , s_max , self.ds)

        self.lanes = [Lane(lane_id = i) for i in range(self.lane_num)]

        self.calculate()

    def calculate(self):

        self.length_calc()

        self.kappa_calc()

        self.heading = self.heading_calc(self.kappa , self.s)

        self.x, self.y = self.cartesean_calc(self.heading ,self.s , 0.0 , 0.0)

        self.lane_calc()

    def length_calc(self):

        self.L_curve = 80 # constant
        self.L_enter = (self.s_max - self.L_clothoid * 2 - self.L_curve) / 2
        self.sigma   = self.kappa_max / self.L_clothoid

    def kappa_calc(self):

        "Building the curvature"

        self.kappa = np.zeros_like(self.s)

        # kappa = 0
        mask_1 = (self.s >= 0) & (self.s < self.L_enter)
        self.kappa[mask_1] = 0

        # kappa is evolving to kappa_max
        mask_2 = (self.s >= self.L_enter) & (self.s < self.L_enter + self.L_clothoid)
        self.kappa[mask_2] = self.sigma * (self.s[mask_2] - self.L_enter)

        # kappa is equal to kappa_max
        mask_3 = (self.s >= self.L_enter + self.L_clothoid) & (self.s < self.L_enter + self.L_clothoid + self.L_curve)
        self.kappa[mask_3] = self.kappa_max

        # kappa is decreasing to zero 
        mask_4 = (self.s >= self.L_enter + self.L_clothoid + self.L_curve) & (self.s < self.L_enter + self.L_clothoid + self.L_curve + self.L_clothoid)
        self.kappa[mask_4] = self.kappa_max - self.sigma * (self.s[mask_4] - self.L_enter - self.L_clothoid - self.L_curve)

        # kappa = 0
        mask_5 = (self.s >= self.L_enter + self.L_clothoid + self.L_curve + self.L_clothoid) & (self.s <= self.s_max)
        self.kappa[mask_5] = 0
    
    def heading_calc(self , kappa , s):

        "Building the heading"

        return cumulative_trapezoid(kappa , s , axis = -1 , initial = 0.0)

    def cartesean_calc(self, heading , s , x_init, y_init):

        "Build Carteasian from Frenet-Serret"

        x = x_init + cumulative_trapezoid(np.cos(heading) , s , axis = -1 , initial = 0.0)
        y = y_init + cumulative_trapezoid(np.sin(heading) , s , axis = -1 , initial = 0.0)

        return x , y

    def lane_calc(self):

        L_w = self.l_w * self.lane_num

        if self.lane_num != 1:
            offsets = np.linspace(- (L_w - 0.5 * L_w) / 2 , (L_w - 0.5 * L_w) / 2 , self.lane_num)
        else:
            offsets = np.array([0])

        for l in range(self.lane_num):

            self.lanes[l].width = self.l_w

            self.lanes[l].offset= offsets[l]

            self.lanes[l].kappa = self.kappa / (1 - offsets[l] * self.kappa)

            self.lanes[l].s     = cumulative_trapezoid( 1 - offsets[l] * self.kappa , self.s , initial = 0.0)

            self.lanes[l].heading = self.heading

            x_init = offsets[l] * np.sin(self.lanes[l].heading[0]) # type: ignore
            y_init = offsets[l] * np.cos(self.lanes[l].heading[0]) # type: ignore

            self.lanes[l].x , self.lanes[l].y= self.cartesean_calc( self.lanes[l].heading , self.lanes[l].s, x_init, y_init)