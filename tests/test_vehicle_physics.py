"""Rollover / slippage physics regression (docs/physics_audit.md §A2, spec §6, §27).
Steady states come from the real CarDynamics (RK4), never from a re-model."""

import math

import numpy as np
import pytest

from model.car.car import Car, CarState
from model.car.config import VehicleParameters, G
from learning.closed_loop import ScenarioSpec, Stratum, build_controller
from learning.mpc import MPCWeights
from learning.paralo_mpc import ParaloMPCConfig
from learning.paralo_vec import p_rollover_vec, p_slip_vec
from learning.physical_eval import run_physical_episode
from learning.vehicle_physics import (
    FrictionInfoConfig, delta_equivalent, friction_demand, p_roll_legacy, paralo_steer_input, realized_curvature,
    rollover_accel_threshold, rollover_legacy_v1, rollover_physical_v2, slip_friction_circle,
    slip_lateral_saturation, steady_state_yaw_rate, tire_state, understeer_gradient,
)
from risks.slippage import SlippageParams

VP = VehicleParameters()
AY_ROLL = rollover_accel_threshold(VP)


def steady(v, delta, mu=1.0, accel=0.0, T=8.0, dt=0.01):
    car = Car(state=CarState(v_x=v), vehicle_params=VP)
    for _ in range(int(T / dt)):
        car.step(0.0, delta, 0.0, mu, dt)
    st = car.state
    return st, tire_state(st.as_array(), accel, delta, 0.0, mu, VP)


def test_reference_threshold_is_the_quasi_static_ssf():
    assert AY_ROLL == pytest.approx(VP.W_c / (2 * VP.h_CG) * G) and AY_ROLL == pytest.approx(9.503, abs=1e-3)


def test_straight_line_has_no_risk():
    st, ts = steady(25.0, 0.0)
    assert abs(ts.a_y) < 1e-9 and not rollover_physical_v2(ts, VP) and not rollover_legacy_v1(st.v_x, 0.0, VP)
    assert not slip_lateral_saturation(ts) and not slip_friction_circle(ts, VP)


def test_gentle_curve_at_highway_speed_is_a_legacy_false_positive_only():
    # the default curve (kappa 0.005) at 20 m/s: the SUV needs delta ~ L kappa + K a_y ~ 0.082
    st, ts = steady(20.0, 0.0816)
    assert ts.a_y == pytest.approx(1.96, abs=0.05) and st.r / st.v_x == pytest.approx(0.005, rel=0.06)   # v_x bleeds ~3% (no drive force)
    assert rollover_legacy_v1(st.v_x, 0.0816, VP)                       # the documented false positive
    assert not rollover_physical_v2(ts, VP) and abs(ts.a_y) < 0.25 * AY_ROLL
    d_eq = paralo_steer_input("kinematic_equivalent", 0.0816, st.v_x, st.r, VP)
    assert p_rollover_vec(st.v_x, d_eq, VP, 0.1) < 1e-6                  # corrected PARALO input
    assert p_slip_vec(st.v_x, d_eq, 0.0, VP, SlippageParams(mu_max=1.0)) < 1e-3


def test_lateral_accel_increases_with_curvature_and_speed_and_rollover_tracks_it():
    ay_by_delta = [steady(20.0, d)[1].a_y for d in (0.02, 0.06, 0.12, 0.2)]
    ay_by_speed = [steady(v, 0.06)[1].a_y for v in (10.0, 15.0, 20.0, 25.0)]
    assert np.all(np.diff(ay_by_delta) > 0) and np.all(np.diff(ay_by_speed) > 0)
    for v, d in [(20.0, 0.02), (20.0, 0.2), (25.0, 0.06)]:
        st, ts = steady(v, d)
        assert rollover_physical_v2(ts, VP) == (abs(ts.a_y) >= AY_ROLL)


def test_high_lateral_accel_can_reach_tipping_only_when_mu_exceeds_ssf():
    # |a_y| <= mu g in this tire model; tipping needs mu >= SSF = 0.969
    for mu in (0.5, 0.8, 0.95):
        ays = []
        for delta in np.linspace(0.05, 0.6, 12):
            car = Car(state=CarState(v_x=25.0), vehicle_params=VP)
            for _ in range(150):
                car.step(0.0, delta, 0.0, mu, 0.01)
                ays.append(abs(tire_state(car.state.as_array(), 0.0, delta, 0.0, mu, VP).a_y))
        assert max(ays) <= mu * G + 1e-6 and max(ays) < AY_ROLL
    ts = tire_state(np.array([0.0, 0.0, 0.0, 25.0, -6.0, 0.6]), 0.0, 0.6, 0.0, 1.0, VP)   # both axles saturated, mu = 1
    assert abs(ts.a_y) == pytest.approx(G, rel=1e-6) and rollover_physical_v2(ts, VP)


def test_low_friction_saturates_tires_without_rollover():
    st, ts = steady(20.0, 0.2, mu=0.4)
    assert abs(ts.a_y) <= 0.4 * G + 1e-6 and not rollover_physical_v2(ts, VP)
    assert slip_lateral_saturation(ts) or slip_friction_circle(ts, VP)


def test_understeering_trajectory_realized_curvature_matches_steady_state_model():
    K = understeer_gradient(VP)
    assert K == pytest.approx(0.0342, abs=5e-4)
    for v, d in [(10.0, 0.03), (20.0, 0.0816), (25.0, 0.06)]:
        st, ts = steady(v, d)
        assert st.r == pytest.approx(float(steady_state_yaw_rate(v, d, VP)), rel=0.03)      # MPC prediction model
        assert st.r < 0.5 * float(steady_state_yaw_rate(v, d, VP, K=0.0))                    # kinematic over-predicts
        assert rollover_legacy_v1(st.v_x, d, VP) or v < 15
        assert not rollover_physical_v2(ts, VP)


def test_combined_braking_and_cornering_uses_the_friction_circle():
    ts_dry = tire_state(np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0]), -8.0, 0.0, 0.0, 1.0, VP)
    ts_wet = tire_state(np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0]), -8.0, 0.0, 0.0, 0.4, VP)
    assert friction_demand(ts_dry, VP) == pytest.approx(8.0 / G, rel=1e-6) and not slip_friction_circle(ts_dry, VP)
    assert slip_friction_circle(ts_wet, VP) and not slip_lateral_saturation(ts_wet)   # braking alone, no lateral
    st, ts = steady(20.0, 0.1, mu=0.6)
    assert not slip_friction_circle(ts, VP)
    ts_b = tire_state(st.as_array(), -6.0, 0.1, 0.0, 0.6, VP)
    assert friction_demand(ts_b, VP) > friction_demand(ts, VP) and slip_friction_circle(ts_b, VP)


def test_low_speed_and_zero_curvature_are_numerically_stable():
    assert realized_curvature(0.0, 0.3) == 0.0 and realized_curvature(0.5, 0.3) == 0.0
    assert realized_curvature(-10.0, 0.1) == pytest.approx(0.01)
    assert float(delta_equivalent(0.0, VP)) == 0.0
    d = paralo_steer_input("kinematic_equivalent", np.zeros(3), np.array([0.0, 1e-9, 20.0]), np.zeros(3), VP)
    assert np.all(np.isfinite(d)) and np.all(d == 0.0)
    ts = tire_state(np.zeros(6), 0.0, 0.0, 0.0, 1.0, VP)
    assert np.isfinite(ts.a_y) and np.isfinite(friction_demand(ts, VP)) and p_roll_legacy(0.0, 0.3, VP) == 0.0


def test_recorded_closed_loop_false_positive_is_not_a_physical_rollover():
    # original MPC on a PPO-demonstrated cut-in (kappa 0.0138, mu 0.76): legacy_v1 ended this episode as a
    # ROLLOVER at peak |a_y| ~ 2.1 m/s^2 (found by the audit's diagnostic sweep, seed 1000)
    theta = {"cutter_gap_m": 12.262487559512964, "cutter_relative_speed_mps": -4.597664490144757,
             "escape_gap_length_m": 25.54464626482166, "front_gap_m": 97.06700568730841,
             "front_relative_speed_mps": -2.21605146123245, "road_kappa_max": 0.013831898453487264,
             "road_mu": 0.7595297981760167, "s_cutin_m": 177.7614775003682}
    spec = ScenarioSpec.make(Stratum("cutin", 1), theta)
    w = MPCWeights().to_vector()
    legacy = run_physical_episode(build_controller({"name": "mpc"}, w), spec, 1000, "legacy_v1")
    assert legacy["events"]["rollover"] and legacy["peak_abs_lat_accel"] < 0.3 * AY_ROLL
    phys = run_physical_episode(build_controller({"name": "mpc"}, w), spec, 1000, "physical_v2")
    assert not phys["events"]["rollover"] and not phys["events"]["rollover_physical_indicator"]
    assert phys["events"]["rollover_legacy_indicator"] and phys["steps"] > legacy["steps"]


def test_friction_information_modes():
    assert FrictionInfoConfig().estimate(0.55, 3) == 0.55
    est = FrictionInfoConfig(mode="estimated", bias=0.1)
    assert est.estimate(0.55, 3) == pytest.approx(0.65) and est.estimate(1.15, 3) == 1.2
    noisy = FrictionInfoConfig(mode="estimated", noise_std=0.05)
    assert noisy.estimate(0.6, 7) == noisy.estimate(0.6, 7) != noisy.estimate(0.6, 8)   # per-episode, seeded
    assert FrictionInfoConfig(mode="conservative", margin=0.1).estimate(0.6, 0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        FrictionInfoConfig(mode="oracle", bias=0.1)
    with pytest.raises(AssertionError):
        ParaloMPCConfig(friction_mode="guess")
