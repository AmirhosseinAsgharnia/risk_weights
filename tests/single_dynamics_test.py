import numpy as np
from solvers.rk4 import rk4
from model.car.config import VehicleParameters
from model.car.model import CarDynamics
from model.road.road import Road
from matplotlib.animation import Animation

# Initialize the road:
road = Road(s_max = 500, kappa_max = 0.005 , L_clothoid = 60 , mu_road = 1.0 , mu_patch = 0.5 , patch_location = 225 , lane_num = 3)

# Ego vehicle initilialization
