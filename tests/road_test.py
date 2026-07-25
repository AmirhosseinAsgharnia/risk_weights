import numpy as np
from model.road import RoadScenario
import matplotlib.pyplot as plt

r = RoadScenario()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(r.x_lanes[0,:] , r.y_lanes[0,:] , linestyle="--")
ax[0].plot(r.x_lanes[1,:] , r.y_lanes[1,:] , linestyle="--")
ax[0].plot(r.x_lanes[2,:] , r.y_lanes[2,:] , linestyle="--")
ax[0].set_aspect("equal")
# ax[1].plot()
plt.show()

# assert np.abs(np.diff(r.kappa)).max() < 2 * r.sigma * r.ds, "curvature jumps"
# assert np.isclose(r.kappa.max(), r.kappa_max), "arc never reached"
# assert np.isclose(r.kappa[-1], 0.0), "road doesn't end straight"