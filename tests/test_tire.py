import numpy as np
import pytest
from model.car.model import EgoModel, SimulationDivergedError, assert_finite


def test_tire_forces_are_saturated_to_friction_limit():
    model = EgoModel()
    # a large slip demand: big lateral velocity + yaw rate, modest speed, low mu
    x = np.array([0.0, 0.0, 0.0, 15.0, 8.0, 1.5])
    diag = model.tire_diagnostics(x, u=(0.0, 0.3), kappa=0.0, mu=0.3)

    Fzf, Fzr = diag.Fzf, diag.Fzr
    mu = 0.3
    assert abs(diag.Fyf) <= mu * Fzf + 1e-6
    assert abs(diag.Fyr) <= mu * Fzr + 1e-6
    assert diag.saturation_ratio > 1.0  # the unsaturated demand exceeded grip
    assert abs(diag.Fyf_unsat) >= abs(diag.Fyf)


def test_tire_forces_unsaturated_when_demand_is_modest():
    model = EgoModel()
    x = np.array([0.0, 0.0, 0.0, 20.0, 0.1, 0.02])
    diag = model.tire_diagnostics(x, u=(0.0, 0.01), kappa=0.0, mu=1.0)
    assert diag.saturation_ratio < 1.0
    assert np.isclose(diag.Fyf, diag.Fyf_unsat)
    assert np.isclose(diag.Fyr, diag.Fyr_unsat)


def test_low_speed_does_not_produce_nan_or_explode():
    model = EgoModel(v_min=1.0)
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # starting from a dead stop
    for _ in range(200):
        x = model.step(x, u=(1.0, 0.05), kappa=0.0, mu=1.0, dt=0.05)
        assert np.all(np.isfinite(x))


def test_assert_finite_raises_on_nan():
    with pytest.raises(SimulationDivergedError):
        assert_finite(np.array([1.0, np.nan, 3.0]), "test state")


def test_assert_finite_passes_on_finite_state():
    assert_finite(np.array([1.0, 2.0, 3.0]), "test state")  # must not raise


def test_step_raises_diverged_error_instead_of_returning_nan():
    model = EgoModel()
    # a deliberately pathological input (huge steering at near-zero mass-normalised
    # stiffness isn't realistic to trigger from delta alone with saturation in
    # place, so drive it via an already-corrupted incoming state instead).
    x_bad = np.array([0.0, 0.0, 0.0, np.inf, 0.0, 0.0])
    with pytest.raises(SimulationDivergedError):
        model.step(x_bad, u=(0.0, 0.0), kappa=0.0, mu=1.0, dt=0.1)
