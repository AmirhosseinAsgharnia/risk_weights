from dataclasses import dataclass

G = 9.81  # [m/s^2] gravitational acceleration


@dataclass(frozen=True)
class VehicleParameters:
    # SUV: high CG, comparatively narrow track -> rollover is reachable
    # at the same speed regime as tire slip (SSF = W_c / (2*h_CG) ~ mu_max).
    m: float = 2000.0  # [kg] vehicle mass

    l_f: float = 1.2  # [m] front axle to CG
    l_r: float = 1.6  # [m] rear axle to CG
    I_zz: float = 2875  # [kg.m^2] yaw moment of inertia
    C_f: float = 19000  # [N/rad] front axle cornering stiffness
    C_r: float = 33000  # [N/rad] rear axle cornering stiffness
    h_CG: float = 0.80  # [m] CG height
    W_c: float = 1.55  # [m] track width (used as the rollover moment arm)
    C_d: float = 2.35  # [kg/m] aerodynamic drag coefficient

    mu_max: float = 1.0  # [-] peak tire-road friction available (dry asphalt ceiling)
    k_f: float = 0.02  # [s/m] friction-speed decay: mu_eff(v) = mu_max / (1 + k_f*v)
    U_mu: float = 0.05  # [-] std-dev of the friction estimate available to the observer

    @property
    def L(self) -> float:
        return self.l_f + self.l_r

    @property
    def Fz_nom(self) -> float:
        """[N] nominal per-axle static load, used to normalize tire stiffness."""
        return self.m * G / 2.0