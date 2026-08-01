import numpy as np
from model.road.road import Road
import matplotlib.pyplot as plt

lane_num = 2

road = Road(s_max = 500, kappa_max = 0.001 , L_clothoid = 100 , lane_num = lane_num , mu_road = 1.0 , mu_patch = 0.3 , patch_location = 225)

fig, axe = plt.subplots(1 , 2, figsize = (5 , 5))
axe[0].plot(road.y , road.x , linestyle = 'solid', color = "red")

for l in range(lane_num):
    axe[0].plot(road.lanes[l].y , road.lanes[l].x , linestyle = 'dashed', color = "blue") # type: ignore

axe[1].plot(road.lanes[l].s , road.lanes[l].heading , linestyle = 'dashed', color = "blue") # type: ignore
axe[0].set_aspect('equal')
plt.tight_layout()
plt.show()