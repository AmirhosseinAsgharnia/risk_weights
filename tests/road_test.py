import numpy as np
from model.road.road import Road
import matplotlib.pyplot as plt

road = Road(s_max = 500, kappa_max = 0.002 , L_clothoid = 50 , lane_num = 3 , mu_road = 1.0 , mu_patch = 0.3 , patch_location = 225)

fig, axe = plt.subplots(1 , 1, figsize = (5 , 5))
axe.plot(road.x_road , road.y_road , linestyle = 'solid')
plt.tight_layout
plt.show()