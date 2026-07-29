import numpy as np
from model.road import RoadScenario
from model.car_init import frenet_to_global
from model.car.model import EgoModel


def test_frenet_to_global_on_straight_road_matches_direct_geometry():
    road = RoadScenario(kappa_max=0.0)
    x, y, psi = frenet_to_global(road, s=50.0, e_y=0.0)
    assert psi == 0.0
    assert np.isclose(x, 50.0, atol=0.2)
    assert np.isclose(y, 0.0, atol=1e-9)


def test_frenet_to_global_lateral_offset_is_right_positive_at_zero_heading():
    road = RoadScenario(kappa_max=0.0)
    # at heading 0, right-positive e_y must map to NEGATIVE y (road.py's own convention)
    x_right, y_right, _ = frenet_to_global(road, s=50.0, e_y=4.0)
    x_left, y_left, _ = frenet_to_global(road, s=50.0, e_y=-4.0)
    assert y_right < 0
    assert y_left > 0
    assert np.isclose(y_right, -y_left)


def test_frenet_to_global_extrapolates_past_tabulated_road_length():
    road = RoadScenario(kappa_max=0.0, S_max=100.0, patch_location=25.0)
    x1, y1, psi1 = frenet_to_global(road, s=150.0, e_y=0.0)
    x2, y2, psi2 = frenet_to_global(road, s=160.0, e_y=0.0)
    # still moving forward at the same heading, not frozen at the boundary sample
    assert psi1 == psi2 == 0.0
    assert np.isclose(x2 - x1, 10.0, atol=1e-6)


def test_egomodel_e_y_dynamics_are_right_positive():
    """
    de_y/dt = -(vx*sin(e_psi) + vy*cos(e_psi)): a POSITIVE heading error (nose
    turned toward local +y, i.e. LEFT) must DECREASE e_y, since e_y is
    right-positive. This is the sign that was fixed after the reversed-heading
    animation bug; regressing it silently reverses every lane change.
    """
    model = EgoModel()
    x = np.array([0.0, 0.0, 0.05, 20.0, 0.0, 0.0])  # small positive e_psi
    dxdt = model.state_space(x, u=(0.0, 0.0), kappa=0.0, mu=1.0)
    assert dxdt[EgoModel.IDX_EY] < 0

    x_neg = np.array([0.0, 0.0, -0.05, 20.0, 0.0, 0.0])
    dxdt_neg = model.state_space(x_neg, u=(0.0, 0.0), kappa=0.0, mu=1.0)
    assert dxdt_neg[EgoModel.IDX_EY] > 0
