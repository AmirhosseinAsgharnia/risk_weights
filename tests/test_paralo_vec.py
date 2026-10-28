import numpy as np
import pytest

from initialization.traffic_init import CAR_LENGTH, CAR_WIDTH
from model.car.config import VehicleParameters
from risks.c2c import c2c, C2CParams, EgoStep, ObjectStep, ObjectTrajectory
from risks.departure import departure, DepartureParams, LaneCenterline, TrajectoryStep as DepStep
from risks.slippage import slippage, SlippageParams, TrajectoryStep as SlipStep
from learning.paralo_vec import p_c2c_pairs, p_departure_vec, p_slip_vec

VP = VehicleParameters()
RNG = np.random.default_rng(0)


@pytest.mark.parametrize("sigma", [0.3, 1e-5])
def test_c2c_matches_module_on_single_state(sigma):
    ex, ey = RNG.uniform(-10, 10, 300), RNG.uniform(-5, 5, 300)
    ox, oy = RNG.uniform(-10, 10, 300), RNG.uniform(-5, 5, 300)
    var = 2 * sigma ** 2
    vec = p_c2c_pairs(ex, ey, ox, oy, CAR_LENGTH, CAR_WIDTH, var, var)
    params = C2CParams(ego_length_m=CAR_LENGTH, ego_width_m=CAR_WIDTH, default_sigma_m=sigma)
    ref = [c2c([EgoStep(x=a, y=b, v=20.0, heading=0.0, var_x=sigma ** 2, var_y=sigma ** 2)],
               [ObjectTrajectory(track_id=1, length_m=CAR_LENGTH, width_m=CAR_WIDTH,
                                 steps=[ObjectStep(x=c, y=d, v=15.0, heading=0.0, sigma_x=sigma, sigma_y=sigma)])],
               VP, params)[0] for a, b, c, d in zip(ex, ey, ox, oy)]
    np.testing.assert_allclose(vec, ref, atol=1e-12)


def test_slippage_matches_module():
    v, d, a = RNG.uniform(0, 30, 400), RNG.uniform(-0.3, 0.3, 400), RNG.uniform(-8, 4, 400)
    d[:5] = 0.0
    for mu in (0.4, 1.0):
        p = SlippageParams(mu_max=mu, mu_uncertainty=0.05)
        ref = [slippage([SlipStep(v=x, delta=y, a=z)], VP, p)[0] for x, y, z in zip(v, d, a)]
        np.testing.assert_allclose(p_slip_vec(v, d, a, VP, p), ref, atol=1e-12)


@pytest.mark.parametrize("width", [4.0, 12.0])
def test_departure_matches_module_against_exact_straight_centreline(width):
    D, c = RNG.uniform(-8, 8, 300), 4.0
    p = DepartureParams(lane_width_m=width, vehicle_width_m=CAR_WIDTH, default_sigma_lateral_m=0.3)
    lane = LaneCenterline((0.0, 0.0, c), np.zeros((3, 3)))
    ref = [departure([DepStep(x=0.0, y=c + x, v=20.0)], lane, VP, p)[0] for x in D]
    np.testing.assert_allclose(p_departure_vec(D, p), ref, atol=1e-12)
