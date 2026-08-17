import numpy as np
import matplotlib.pyplot as plt
from model.road.road import Road

lane_num = 3

road = Road(s_max = 500, kappa_max = 0.001 , L_clothoid = 100 , lane_num = lane_num , mu_road = 1.0 , mu_patch = 0.3 , patch_location = 225)

fig, axe = plt.subplots(1 , 2, figsize = (10 , 5))

# ── Asphalt: filled band between the outermost lanes' outer edges ──
first, last = road.lanes[0], road.lanes[-1]
edge_low_x  = first.x - (first.width / 2) * np.sin(first.heading)  # type: ignore
edge_low_y  = first.y - (first.width / 2) * np.cos(first.heading)  # type: ignore
edge_high_x = last.x  + (last.width  / 2) * np.sin(last.heading)   # type: ignore
edge_high_y = last.y  + (last.width  / 2) * np.cos(last.heading)   # type: ignore

poly_x = np.concatenate([edge_low_x, edge_high_x[::-1]])
poly_y = np.concatenate([edge_low_y, edge_high_y[::-1]])

# Plotted (y, x) throughout, matching the centreline below -- this Road's
# clothoid runs mostly along x, so swapping axes displays it tall/narrow
# instead of long/flat.
axe[0].set_facecolor("#e3efe0")
axe[0].fill(poly_y, poly_x, color = "white", zorder = 1)
axe[0].plot(edge_low_y,  edge_low_x,  linestyle = "solid", color = "dimgray", linewidth = 1.2, zorder = 2)
axe[0].plot(edge_high_y, edge_high_x, linestyle = "solid", color = "dimgray", linewidth = 1.2, zorder = 2)

# ── Lane dividers (between adjacent lanes) ──
for l in range(lane_num - 1):
    div_x = (road.lanes[l].x + road.lanes[l + 1].x) / 2  # type: ignore
    div_y = (road.lanes[l].y + road.lanes[l + 1].y) / 2  # type: ignore
    axe[0].plot(div_y, div_x, linestyle = "dashed", color = "dimgray", linewidth = 1.5, zorder = 3)

# ── Centrelines (each lane's own centreline) ──
for lane in road.lanes:
    axe[0].plot(lane.y, lane.x, linestyle = "dotted", color = "goldenrod", linewidth = 2.0, zorder = 4)  # type: ignore

axe[0].set_aspect('equal')

axe[1].plot(road.lanes[-1].s , road.lanes[-1].heading , linestyle = 'dashed', color = "blue") # type: ignore
plt.tight_layout()
plt.show()
