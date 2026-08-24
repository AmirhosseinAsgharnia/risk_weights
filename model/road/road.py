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

class Road:
    "Clothoid Roal Class"

    def __init__(self,
                 s_max     = 500,
                 kappa_max = 0.02,
                 L_clothoid= 50,
                 mu        = 1.0,
                 lane_num  = 3):

        self.s_max      = s_max
        self.kappa_max  = kappa_max
        self.L_clothoid = L_clothoid
        self.mu         = mu
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

    def index_at(self, s: float) -> int:
        """Nearest sample index into self.s -- and into every per-index
        array aligned with it (self.kappa, self.heading, each Lane's own
        .kappa/.heading, which are built on this same index, not on the
        lane's own re-parameterized .s) -- for an arbitrary arclength s."""
        idx = int(round(s / self.ds))
        return min(max(idx, 0), len(self.s) - 1)

    def frenet_to_global(self, s: float, e_y: float, e_psi: float = 0.0) -> tuple[float, float, float]:
        """Global (x, y, heading) of a point at arclength s, offset e_y from
        the road's own backbone (offset-0) curve.

        Sign conventions -- these are SAE J670 (+right), matching CarState/
        CarDynamics, NOT Lane.offset's own "+left" convention (offsets used
        by lane_calc's x_init/y_init only coincide with this at heading=0,
        where it seeds the arc-length-correct integration that actually
        traces out each lane; that formula isn't a general perpendicular
        for heading != 0). Callers combining a Lane.offset with an e_y here
        must negate the offset first.

          e_y:  the perpendicular unit vector for a SAE "+right" offset, at
                heading psi, is (sin(psi), -cos(psi)) -- verified
                perpendicular to the tangent (cos(psi), sin(psi)) for every
                psi (dot product 0), unlike (sin(psi), cos(psi)) which is
                only perpendicular at psi = 0.
          e_psi: CarDynamics' own e_psi state accumulates as
                psi_vehicle - psi_path, positive = clockwise (SAE) --
                opposite of psi's counter-clockwise-positive (cos, sin)
                convention here, hence the minus sign below.
        """
        idx = self.index_at(s)
        psi = self.heading[idx]
        x = self.x[idx] + e_y * np.sin(psi)
        y = self.y[idx] - e_y * np.cos(psi)
        return x, y, psi - e_psi

    def offset_curve(self, offset: float) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) of the curve running everywhere `offset` [m] from this
        road's own backbone (offset-0) curve -- same (sin(heading),
        cos(heading)) offset-direction convention lane_calc's own
        x_init/y_init use (see frenet_to_global's docstring for how this
        differs from that method's SAE "+right" convention).

        Correctly accounts for the arc-length stretch/compression a
        laterally-offset curve undergoes through a curved section (the
        `1 - offset * kappa` factor, same as lane_calc): integrating
        cos/sin(heading) against this offset-scaled arc length, rather
        than against the backbone's own arc length, is what keeps a fixed
        offset from the backbone truly fixed all the way through and
        after a curve. A naive per-point (sin, cos) shift added directly
        onto an already-built lane centreline does NOT compose this way
        -- e.g. extending lane 0's centreline outward by a further
        half-width to get the road's outer edge visibly (and permanently)
        narrows through a curve and never recovers; call this method with
        the edge's own total offset instead of doing that.
        """
        lane_s = cumulative_trapezoid(1 - offset * self.kappa, self.s, initial=0.0)
        x_init = offset * np.sin(self.heading[0])
        y_init = offset * np.cos(self.heading[0])
        return self.cartesean_calc(self.heading, lane_s, x_init, y_init)

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

            self.lanes[l].x , self.lanes[l].y = self.offset_curve(offsets[l])