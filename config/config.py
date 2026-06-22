import numpy as np

# Vehicle parameters
l_r    = 1.6      # m
l_f    = 1.2      # m
L      = 2.8      # m
m      = 1575     # kg
I_zz   = 2875     # kg·m²
h_cg   = 0.55     # m
g      = 9.81     # m/s²
C_f    = 19000    # N/rad
C_r    = 33000    # N/rad
mu     = 1.0
Fz_nom = 7724     # N
W_c    = 1.84     # m
L_veh  = 4.5      # m
C_d    = 2.35     # kg/m

# Road parameters
n_lanes    = 3
lane_width = 3.7   # m
ego_lane   = 2     # start in middle lane (1-indexed)
road_radius = np.inf            # m; np.inf gives a straight road
road_curvature = 0.0

# Simulation parameters
dt        = 0.05   # s
t_end     = 8.0    # s
v0        = 25.0   # m/s

# Action bounds
delta_min = -0.5   # rad
delta_max =  0.5   # rad
a_min     = -8.0   # m/s²
a_max     =  3.0   # m/s²

# Risk weights — collision avoidance dominates lane-departure penalty so the
# MPC prefers a lane change (P_ld≈1, cost=0.10) over staying in a sandwich
# (P_c2c→1, cost=0.70).
w_slip = 0.1
w_roll = 0.1
w_c2c  = 0.70
w_ld   = 0.1

# Uncertainty parameters
U_hcg           = 0.15
U_mu            = 0.1
k_f             = 0.01
alpha_max_base  = 8.6 * np.pi / 180
sensor_accuracy = 0.95

# MPC / prediction parameters
T_horizon   = 4.0   # s
dt_pred     = 0.1   # s
lambda_disc = 0.1

# Reproducibility
RANDOM_SEED = 42
