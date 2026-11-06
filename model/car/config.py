from dataclasses import dataclass

G = 9.81  # [m/s^2] gravitational acceleration

@dataclass(frozen=True)
class VehicleParameters:
    # SUV: high CG, comparatively narrow track -> rollover is reachable
    # at the same speed regime as tire slip (SSF = W_c / (2*h_CG) ~ mu_max).
    
    m: float = 2000.0  # [kg] vehicle mass

    l_f: float  = 1.2  # [m] front axle to CG
    l_r: float  = 1.6  # [m] rear axle to CG
    I_zz: float = 2875  # [kg.m^2] yaw moment of inertia
    C_f: float  = 19000  # [N/rad] front axle cornering stiffness
    C_r: float  = 33000  # [N/rad] rear axle cornering stiffness
    h_CG: float = 0.80  # [m] CG height
    W_c: float  = 1.55  # [m] track width (used as the rollover moment arm)
    C_d: float  = 2.35  # [kg/m] aerodynamic drag coefficient (not used by CarDynamics)

    delta_max: float = 0.5     # [rad] front-wheel steering lock, ~29 deg

    # Longitudinal force allocation: front axle's share of the request, in [0, 1]. None = proportional to
    # the axle's normal load, the ideal split: both axles reach mu F_z together, so the straight-line
    # limit is exactly mu g. brake None stands for ABS/EBD, drive None for an idealized AWD. These are
    # assumptions (no drivetrain spec exists); a fixed share gives a lower traction limit.
    brake_front_share: float | None = None
    drive_front_share: float | None = None
    v_stop: float = 0.5        # [m/s] brake force fades to 0 below this: brakes stop the car, never reverse it
    v_tire_min: float = 1.0    # [m/s] slip-angle denominator floor; below it the tire damps wheel side-slip.
                               # Stiffest mode is ~45/v_tire_min [1/s] for the defaults: lowering this
                               # below ~0.8 m/s takes RK4 at dt = 0.05 s out of its stability region.

    def __post_init__(self):
        for share in (self.brake_front_share, self.drive_front_share):
            assert share is None or 0.0 <= share <= 1.0, share
        assert self.v_stop > 0 and self.v_tire_min > 0

    @property
    def L(self) -> float:
        return self.l_f + self.l_r

    @property
    def Fz_nom(self) -> float:
        """[N] nominal per-axle static load, used to normalize tire stiffness."""
        return self.m * G / 2.0

    @property
    def SSF(self) -> float:
        """[-] Static Stability Factor: W_c / (2*h_CG). Rollover threshold
        lateral accel (in g) -- exceeding SSF*g is the rollover onset."""
        return self.W_c / (2.0 * self.h_CG)