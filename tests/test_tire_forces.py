"""Physical validation of the combined tire model, independent of PPO and of any controller
(docs/description.md §4.1). The letters are the validation plan's: A straight braking, B pure cornering,
C combined braking, D combined drive, E low speed, F load transfer."""

import dataclasses
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest

from initialization.traffic_init import CAR_LENGTH, CAR_WIDTH
from learning.closed_loop import ScenarioSpec, Stratum, build_controller, FAMILIES
from learning.env import ROAD_EDGE_MARGIN_M
from learning.env import ACCEL_MAX, ACCEL_MIN, EgoTrafficEnv
from learning.mpc import MPCWeights
from learning.physical_eval import run_physical_episode
from learning.vehicle_physics import (
    friction_demand, slip_friction_circle, slip_lateral_saturation, stability_loss, tire_state, tire_utilization,
)
from model.car.car import Car, CarState
from model.car.config import VehicleParameters, G
from model.car.model import CarDynamics, tire_forces
from model.collision import body_gap, body_half_extent_e_y, ego_overlaps_any

VP = VehicleParameters()
DT = 0.05
TOL = 1e-6   # [N]


class RecordingDynamics(CarDynamics):
    """CarDynamics that logs the forces of every derivative evaluation, i.e. of every RK4 stage."""
    stages: list = []

    def step(self):
        super().step()
        type(self).stages.append((self.tires, self.mu))


def make_car(v=20.0, vp=VP, model=CarDynamics):
    return Car(state=CarState(v_x=v), vehicle_params=vp, model=model)


def drive(v0, accel, delta, mu, T, kappa=0.0, vp=VP, dt=DT, model=CarDynamics):
    car = make_car(v0, vp, model)
    states = []
    for _ in range(int(round(T / dt))):
        car.step(accel, delta, kappa, mu, dt)
        states.append(car.state)
    return states


def assert_circle(t, mu):
    assert math.hypot(t.F_xf, t.F_yf) <= mu * t.F_zf + TOL and math.hypot(t.F_xr, t.F_yr) <= mu * t.F_zr + TOL


# ── A: straight-line braking ────────────────────────────────────────────────
@pytest.mark.parametrize("mu", [0.4, 0.6, 0.8, 1.0])
def test_a_braking_is_limited_by_adhesion_whatever_is_commanded(mu):
    limit = mu * G
    for cmd in (8.0, 20.0, 100.0):
        states = drive(25.0, -cmd, 0.0, mu, 12.0)
        v = np.array([25.0] + [s.v_x for s in states])
        moving = v[1:] > 2.0
        decel = -np.diff(v)[moving] / DT
        expected = min(cmd, limit)
        assert decel == pytest.approx(np.full_like(decel, expected), rel=1e-3)
        assert states[-1].s >= 25.0 ** 2 / (2 * mu * G)                    # never shorter than the adhesion limit
        assert states[-1].s == pytest.approx(25.0 ** 2 / (2 * expected), rel=0.03)
        assert states[-1].v_x < 1e-3 and min(s.v_x for s in states) >= -1e-9   # stopped: no reversal, no limit cycle
        assert abs(states[-1].v_y) < 1e-9 and abs(states[-1].r) < 1e-9


def test_a_stopping_distance_is_ordered_by_friction():
    dist = [drive(25.0, -8.0, 0.0, mu, 12.0)[-1].s for mu in (0.4, 0.6, 0.8)]
    assert dist[0] > dist[1] > dist[2]


def test_a_fixed_brake_bias_cannot_reach_the_load_proportional_limit():
    rear_biased = dataclasses.replace(VP, brake_front_share=0.3)
    t_ideal = tire_forces(20.0, 0.0, 0.0, -7.0, 0.0, 0.8, VP)
    t_fixed = tire_forces(20.0, 0.0, 0.0, -7.0, 0.0, 0.8, rear_biased)
    assert t_ideal.a_x == pytest.approx(-7.0) and abs(t_fixed.a_x) < 6.5   # the rear locks up first; no redistribution


# ── B: pure cornering ───────────────────────────────────────────────────────
def hold_speed(state, v_ref):
    return 10.0 * (v_ref - state.v_x)


def corner(v, kappa, mu, T=10.0):
    """Follows a constant-curvature road with a simple feedforward + feedback steering law."""
    car = make_car(v)
    trace = []
    for _ in range(int(T / 0.01)):
        st = car.state
        delta = VP.L * kappa + 0.0342 * v * v * kappa - 0.1 * st.e_y - 0.8 * st.e_psi
        car.step(hold_speed(st, v), delta, kappa, mu, 0.01)
        trace.append((car.state, tire_state(car.state.as_array(), 0.0, delta, kappa, mu, VP)))
    return trace


def test_b_cornering_within_the_limit_tracks_the_curve():
    trace = corner(15.0, 0.02, 1.0)
    st, ts = trace[-1]
    assert abs(st.e_y) < 1.0 and abs(st.r - 15.0 * 0.02) < 0.015           # both: this test's low-gain controllers
    assert ts.a_y == pytest.approx(15.0 ** 2 * 0.02, rel=0.05) and not slip_lateral_saturation(ts)


def test_b_cornering_beyond_the_limit_saturates_and_runs_wide_without_blowing_up():
    trace = corner(15.0, 0.03, 0.4)   # needs 6.75 m/s^2, mu g is 3.9
    assert max(abs(ts.a_y) for _, ts in trace) <= 0.4 * G + 1e-6
    assert any(slip_lateral_saturation(ts) for _, ts in trace)
    assert abs(trace[-1][0].e_y) > 2.0                                    # physically leaves the curve
    assert all(np.all(np.isfinite(st.as_array())) for st, _ in trace)


# ── C: combined braking and cornering (the main regression) ─────────────────
GRID = list(itertools.product((3.0, 8.0, 15.0, 25.0), (-0.5, -0.2, -0.05, 0.05, 0.2, 0.5), (0.4, 0.6, 0.8, 1.0)))


def test_c_friction_circle_holds_at_every_rk4_stage_under_braking():
    RecordingDynamics.stages = []
    for (v, delta, mu), accel in itertools.product(GRID, (-1.0, -4.0, -8.0)):
        drive(v, accel, delta, mu, 1.5, model=RecordingDynamics)
    assert len(RecordingDynamics.stages) == 4 * 30 * len(GRID) * 3         # four stages, all recorded
    for t, mu in RecordingDynamics.stages:
        assert_circle(t, mu)


def test_c_braking_takes_lateral_capacity_away():
    state = (20.0, -4.0, 0.8)                                             # both axles asking for more than mu F_z
    forces = [tire_forces(*state, a, 0.5, 0.5, VP) for a in (0.0, -2.0, -4.0, -6.0, -8.0)]
    assert abs(forces[0].F_yf_req) > 0.5 * forces[0].F_zf and abs(forces[0].F_yr_req) > 0.5 * forces[0].F_zr
    assert np.all(np.diff([t.F_yf_max + t.F_yr_max for t in forces]) < 0)
    assert np.all(np.diff([abs(t.F_yf) + abs(t.F_yr) for t in forces]) < 0)
    for t in forces:                                                      # saturated: F_y sits on what is left
        assert abs(t.F_yf) == pytest.approx(t.F_yf_max) and abs(t.F_yr) == pytest.approx(t.F_yr_max)


def test_c_lateral_capacity_matches_the_circle_formula():
    t = tire_forces(20.0, -0.5, 0.25, -5.0, 0.15, 0.8, VP)
    assert t.F_yf_max == pytest.approx(math.sqrt((0.8 * t.F_zf) ** 2 - t.F_xf ** 2))
    assert t.F_yr_max == pytest.approx(math.sqrt((0.8 * t.F_zr) ** 2 - t.F_xr ** 2))


@pytest.mark.parametrize("mu", [0.4, 0.8])
def test_c_total_realized_acceleration_never_exceeds_mu_g(mu):
    states = drive(20.0, -8.0, 0.15, mu, 2.0)
    for st in states:
        t = tire_forces(st.v_x, st.v_y, st.r, -8.0, 0.15, mu, VP)
        assert math.hypot(t.a_x, t.a_y) <= mu * G + 1e-6


def test_c_hard_braking_in_a_turn_gives_up_cornering_compared_with_rolling():
    rolling = drive(20.0, 0.0, 0.1, 0.8, 1.5)[-1]
    braking = drive(20.0, -8.0, 0.1, 0.8, 1.5)[-1]
    assert abs(braking.e_psi) < 0.5 * abs(rolling.e_psi)                  # understeers instead of out-turning it


@pytest.mark.parametrize("v,mu,delta", list(itertools.product((12.0, 20.0, 30.0), (0.3, 0.5, 0.8, 1.0, 1.2),
                                                               (0.05, 0.15, 0.3, 0.5))))
def test_c_limit_braking_never_turns_the_car_against_its_steering(v, mu, delta):
    # If F_x took the whole circle first, F_y would be 0 on both axles and the brake force along the
    # steered wheel would yaw the car the wrong way, by up to 284 deg (647 of 1944 sweep cases).
    assert min(s.e_psi for s in drive(v, -8.0, delta, mu, 8.0)) > math.radians(-1.0)


# ── D: combined acceleration and cornering ─────────────────────────────────
def test_d_friction_circle_holds_at_every_rk4_stage_under_drive():
    RecordingDynamics.stages = []
    for vp in (VP, dataclasses.replace(VP, drive_front_share=0.0), dataclasses.replace(VP, drive_front_share=1.0)):
        car_model = RecordingDynamics
        for (v, delta, mu), accel in itertools.product(GRID, (1.0, 4.0)):
            drive(v, accel, delta, mu, 1.0, vp=vp, model=car_model)
    for t, mu in RecordingDynamics.stages:
        assert_circle(t, mu)


def test_d_drive_is_traction_limited_and_depends_on_the_allocation():
    mu = 0.3
    awd = tire_forces(10.0, 0.0, 0.0, 4.0, 0.0, mu, VP)
    assert awd.a_x == pytest.approx(mu * G)
    rwd = tire_forces(10.0, 0.0, 0.0, 4.0, 0.0, mu, dataclasses.replace(VP, drive_front_share=0.0))
    fwd = tire_forces(10.0, 0.0, 0.0, 4.0, 0.0, mu, dataclasses.replace(VP, drive_front_share=1.0))
    assert rwd.F_xf == 0.0 and rwd.F_xr == pytest.approx(mu * rwd.F_zr)
    assert fwd.F_xr == 0.0 and fwd.F_xf == pytest.approx(mu * fwd.F_zf)
    assert rwd.a_x < awd.a_x and fwd.a_x < awd.a_x                         # one driven axle has less to push on
    unlimited = tire_forces(10.0, 0.0, 0.0, 2.0, 0.0, 1.0, VP)
    assert unlimited.a_x == pytest.approx(2.0)


def test_d_drive_takes_lateral_capacity_away():
    gentle, hard = (tire_forces(15.0, -0.3, 0.2, a, 0.1, 0.4, VP) for a in (0.5, 3.5))
    assert hard.F_yf_max + hard.F_yr_max < gentle.F_yf_max + gentle.F_yr_max


# ── E: low speed and standstill ─────────────────────────────────────────────
def test_e_parked_car_stays_parked():
    for delta, accel in ((0.0, 0.0), (0.3, 0.0), (-0.5, 0.0), (0.3, -8.0)):
        st = drive(0.0, accel, delta, 1.0, 3.0)[-1]
        assert np.all(st.as_array() == 0.0)


def test_e_braking_at_rest_never_reverses():
    assert min(s.v_x for s in drive(0.0, -8.0, 0.0, 1.0, 3.0)) >= 0.0


@pytest.mark.parametrize("mu", [0.4, 1.0])
def test_e_brakes_to_a_complete_stop_and_stays_there(mu):
    states = drive(10.0, -8.0, 0.0, mu, 8.0)
    assert min(s.v_x for s in states) >= -1e-9 and states[-1].v_x < 1e-3
    assert states[-1].s == pytest.approx(states[-2].s, abs=1e-3)


def test_e_acceleration_from_rest_is_finite_and_at_the_request():
    states = drive(0.0, 2.0, 0.0, 1.0, 2.0)
    assert states[-1].v_x == pytest.approx(4.0, rel=1e-6)


def test_e_steering_while_creeping_is_kinematic():
    for v, delta in ((0.2, 0.3), (0.5, -0.4), (0.9, 0.5)):
        st = drive(v, 0.0, delta, 1.0, 2.0, dt=DT)[-1]
        assert np.all(np.isfinite(st.as_array()))
        assert abs(st.r) <= 1.2 * abs(st.v_x * math.tan(delta)) / VP.L + 1e-3   # no spin-up beyond kinematic


@pytest.mark.parametrize("v_y,r", [(1.0, 0.0), (0.0, 1.0), (-0.5, -1.0), (2.0, 1.5)])
def test_e_residual_slide_at_standstill_dies_out(v_y, r):
    car = Car(state=CarState(v_x=0.0, v_y=v_y, r=r), vehicle_params=VP)
    for _ in range(100):
        car.step(0.0, 0.0, 0.0, 1.0, DT)
        assert np.all(np.isfinite(car.state.as_array()))
    assert abs(car.state.v_y) < 1e-3 and abs(car.state.r) < 1e-3
    assert math.hypot(car.state.v_x, car.state.v_y) <= math.hypot(v_y, r * VP.L)   # side-slip energy is only ever lost


@pytest.mark.parametrize("mu", [0.2, 0.4, 1.0])
def test_e_aggressive_steering_stays_bounded_and_finite(mu):
    rng = np.random.default_rng(0)
    car = make_car(25.0)
    delta = 0.0
    for _ in range(2000):
        delta = float(np.clip(delta + rng.uniform(-0.05, 0.05), -0.5, 0.5))
        car.step(float(rng.uniform(-8, 4)), delta, 0.0, mu, DT)
        st = car.state
        assert np.all(np.isfinite(st.as_array())) and abs(st.r) < 6.0 and math.hypot(st.v_x, st.v_y) < 150.0


# ── F: load transfer ────────────────────────────────────────────────────────
def test_f_static_loads():
    t = tire_forces(20.0, 0.0, 0.0, 0.0, 0.0, 1.0, VP)
    assert t.F_zf == pytest.approx(VP.m * G * VP.l_r / VP.L) and t.F_zr == pytest.approx(VP.m * G * VP.l_f / VP.L)


def test_f_braking_loads_the_front_and_drive_loads_the_rear():
    static, brake, drive_ = (tire_forces(20.0, 0.0, 0.0, a, 0.0, 1.0, VP) for a in (0.0, -6.0, 3.0))
    assert brake.F_zf > static.F_zf > drive_.F_zf and brake.F_zr < static.F_zr < drive_.F_zr


def test_f_transfer_follows_the_realized_not_the_requested_acceleration():
    realized = tire_forces(20.0, 0.0, 0.0, -8.0, 0.0, 0.4, VP)
    assert realized.a_x == pytest.approx(-0.4 * G)
    assert realized.F_zf == pytest.approx((VP.m * G * VP.l_r - VP.m * VP.h_CG * realized.a_x) / VP.L, rel=1e-9)


def test_f_conservation_positivity_and_fixed_point_on_the_whole_grid():
    worst_residual = 0.0
    for (v, delta, mu), a, v_y, r in itertools.product(GRID, (-100.0, -8.0, -2.0, 0.0, 4.0), (-3.0, 0.0, 3.0), (-1.0, 1.0)):
        t = tire_forces(v, v_y, r, a, delta, mu, VP)
        assert t.F_zf + t.F_zr == pytest.approx(VP.m * G) and t.F_zf > 0.0 and t.F_zr > 0.0
        assert_circle(t, mu)
        worst_residual = max(worst_residual, abs((VP.m * G * VP.l_r - t.F_zf * VP.L) / (VP.m * VP.h_CG) - t.a_x))
    assert worst_residual < 0.02   # [m/s^2] gap between a_x and the a_x used for F_z: the fixed point converged


def test_f_normal_loads_never_go_negative():
    tall = dataclasses.replace(VP, h_CG=3.0)
    for a in (-100.0, 100.0):
        t = tire_forces(20.0, 0.0, 0.0, a, 0.0, 1.2, tall)
        assert min(t.F_zf, t.F_zr) >= 0.0 and t.F_zf + t.F_zr == pytest.approx(tall.m * G)
        assert_circle(t, 1.2)


def test_surr_steering_stays_within_the_lock_while_stopped():
    cfg = FAMILIES["sandwich"][0](seed=0, mode="exact", s_stop_m=110.0)
    env = EgoTrafficEnv(arena=FAMILIES["sandwich"][1](cfg), mode="fixed")
    env.reset()
    stopped_steps = 0
    for _ in range(250):
        _, _, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        stopper = next(a for a in env.agents if a.car.car_id == 0)
        assert abs(stopper.car.state.delta) <= VP.delta_max
        stopped_steps += info["arena"]["front_stopped"]
        if term or trunc:
            break
    assert stopped_steps > 50


# ── Conventions preserved ──────────────────────────────────────────────────
@pytest.mark.parametrize("v_x,v_y,r,delta", [(20.0, 0.5, 0.2, 0.1), (5.0, -0.3, 0.4, -0.2), (30.0, 0.0, 0.0, 0.05)])
def test_slip_angles_are_the_original_formula_above_the_low_speed_floor(v_x, v_y, r, delta):
    t = tire_forces(v_x, v_y, r, 0.0, delta, 1.0, VP)
    assert t.alpha_f == pytest.approx(delta - math.atan2(v_y + VP.l_f * r, v_x), abs=1e-12)
    assert t.alpha_r == pytest.approx(-math.atan2(v_y - VP.l_r * r, v_x), abs=1e-12)


def test_steering_right_still_yaws_right_on_straight_rolling():
    st = drive(20.0, 0.0, 0.05, 1.0, 1.0)[-1]
    assert st.r > 0 and st.e_y > 0 and st.e_psi > 0


def test_front_forces_are_rotated_by_the_steering_angle():
    delta = 0.3
    d = CarDynamics(np.array([0, 0, 0, 20.0, 0.0, 0.0]), -3.0, delta, 0.0, 1.0)
    d.step()
    t = d.tires
    assert t.a_x == pytest.approx((t.F_xf * math.cos(delta) - t.F_yf * math.sin(delta) + t.F_xr) / VP.m)
    assert t.a_y == pytest.approx((t.F_xf * math.sin(delta) + t.F_yf * math.cos(delta) + t.F_yr) / VP.m)
    F_yb = t.F_xf * math.sin(delta) + t.F_yf * math.cos(delta)
    assert d.d_states[CarDynamics.IDX_r] == pytest.approx((VP.l_f * F_yb - VP.l_r * t.F_yr) / VP.I_zz)


def test_cornering_costs_speed_through_the_steered_tire():
    t = tire_forces(20.0, 0.0, 0.0, 0.0, 0.2, 1.0, VP)
    assert t.F_yf > 0 and t.a_x < 0


# ── Event definitions ──────────────────────────────────────────────────────
def test_events_separate_demand_saturation_and_stability():
    st = np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0])
    braking = tire_state(st, -8.0, 0.0, 0.0, 0.4, VP)
    assert friction_demand(braking, VP) == pytest.approx(8.0 / (0.4 * G))         # excess demand ...
    assert slip_friction_circle(braking, VP) and tire_utilization(braking) <= 1.0  # ... never realized
    assert not slip_lateral_saturation(braking) and not stability_loss(braking)    # and no skid either
    assert braking.a_x == pytest.approx(-0.4 * G) and braking.a_cmd == -8.0
    at_limit = tire_state(np.array([0.0, 0.0, 0.0, 20.0, -0.6, 0.0]), 0.0, 0.3, 0.0, 0.4, VP)
    assert slip_lateral_saturation(at_limit) and not stability_loss(at_limit)      # saturated, sideslip 1.7 deg
    sliding = tire_state(np.array([0.0, 0.0, 0.0, 20.0, 6.0, 0.0]), 0.0, 0.0, 0.0, 0.4, VP)
    assert stability_loss(sliding)
    parked = tire_state(np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.0]), 0.0, 0.0, 0.0, 0.4, VP)
    assert not stability_loss(parked)                                              # undefined at walking pace


# ── Footprint geometry ─────────────────────────────────────────────────────
def _agent(s, e_y, e_psi=0.0):
    state = SimpleNamespace(s=s, e_y=e_y, e_psi=e_psi, lane=0)
    return SimpleNamespace(car=SimpleNamespace(state=state))


ROAD = SimpleNamespace(lanes=[SimpleNamespace(offset=0.0)])


def test_collision_box_matches_the_old_rule_at_zero_heading():
    for ds, dey in itertools.product(np.linspace(0, 6, 13), np.linspace(0, 4, 9)):
        assert ego_overlaps_any(0.0, 0.0, [_agent(ds, dey)], ROAD) == bool(ds < CAR_LENGTH and dey < CAR_WIDTH)


def test_collision_box_turns_with_heading():
    beside = _agent(0.0, 2.1)                                             # just clear when both are straight
    assert not ego_overlaps_any(0.0, 0.0, [beside], ROAD)
    assert ego_overlaps_any(0.0, 0.0, [beside], ROAD, ego_e_psi=0.5)
    assert ego_overlaps_any(0.0, 0.0, [_agent(0.0, 2.1, 0.5)], ROAD)      # a surr body turning in counts too
    assert not ego_overlaps_any(0.0, 0.0, [_agent(4.5, 1.9)], ROAD, ego_e_psi=math.pi / 2)   # a turned body is shorter


def test_body_gap_is_zero_exactly_on_overlap_and_the_box_distance_when_straight():
    assert body_gap(0, 0, 0, 7.0, 6.0, 0) == pytest.approx(math.hypot(3.0, 4.0))
    assert body_gap(0, 0, 0, 7.0, 0.0, 0) == pytest.approx(3.0)
    rng = np.random.default_rng(1)
    for _ in range(400):
        ds, dey, pa, pb = rng.uniform(-8, 8), rng.uniform(-5, 5), rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        gap = body_gap(0, 0, pa, ds, dey, pb)
        assert (gap == 0.0) == ego_overlaps_any(0.0, 0.0, [_agent(ds, dey, pb)], ROAD, ego_e_psi=pa)


def test_off_road_allows_half_a_metre_past_the_edge_for_the_turned_body():
    cfg = SPEC.config(seed=1000)
    env = EgoTrafficEnv(arena=FAMILIES["cutin"][1](cfg), mode="fixed")
    env.reset()
    limit = env._road_half_width + ROAD_EDGE_MARGIN_M
    lane_offset = env.road.lanes[env.ego_car.state.lane].offset
    def off(e_y_backbone, e_psi):
        env.ego_car.state = dataclasses.replace(env.ego_car.state, e_y=e_y_backbone + lane_offset, e_psi=e_psi)
        return env._ego_off_road()
    assert ROAD_EDGE_MARGIN_M == 0.5
    assert not off(limit - CAR_WIDTH / 2 - 0.01, 0.0) and off(limit - CAR_WIDTH / 2 + 0.01, 0.0)
    extent = body_half_extent_e_y(0.5)
    assert extent > CAR_WIDTH / 2
    assert not off(limit - extent - 0.01, 0.5) and off(limit - extent + 0.01, 0.5)   # turning uses up the margin
    assert not off(-(limit - CAR_WIDTH / 2 - 0.01), 0.0) and off(-(limit - CAR_WIDTH / 2 + 0.01), 0.0)
    assert not off(env.road.lanes[-1].offset, 0.0) and not off(env.road.lanes[0].offset, 0.0)   # the outer lanes stay legal


def test_off_road_margin_matches_the_logged_edge_clearance():
    cfg = SPEC.config(seed=1000)
    env = EgoTrafficEnv(arena=FAMILIES["cutin"][1](cfg), mode="fixed")
    env.reset()
    lane_offset = env.road.lanes[env.ego_car.state.lane].offset
    e_y = env._road_half_width + ROAD_EDGE_MARGIN_M - CAR_WIDTH / 2 + 0.01
    env.ego_car.state = dataclasses.replace(env.ego_car.state, e_y=e_y + lane_offset, e_psi=0.0)
    clearance = env._road_half_width - abs(env._ego_backbone_e_y()) - body_half_extent_e_y(0.0)
    assert clearance < -ROAD_EDGE_MARGIN_M and env._ego_off_road()


# ── Environment smoke test: an interface check, no evidence about PPO or feasibility ──────
THETA = {"cutin": {"road_mu": 0.5, "road_kappa_max": 0.01, "s_cutin_m": 160.0, "cutter_gap_m": 20.0,
                   "cutter_relative_speed_mps": -2.0, "front_gap_m": 40.0, "front_relative_speed_mps": 0.0,
                   "escape_gap_length_m": 20.0}}
SPEC = ScenarioSpec.make(Stratum("cutin", 1), THETA["cutin"])


def _random_rollout(env, seed, n_steps=200):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs) and np.all(np.isfinite(obs))
    info = {}
    for _ in range(n_steps):
        action = rng.uniform(-1, 1, 2).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs) and np.all(np.isfinite(obs)) and math.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        st = env.ego_car.state
        assert np.all(np.isfinite(st.as_array())) and st.v_x > -0.5
        assert ACCEL_MIN <= info["accel_requested"] <= ACCEL_MAX and ACCEL_MIN <= info["accel_actuator"] <= ACCEL_MAX
        assert abs(info["accel_realized"]) <= env.road.mu * G + 1e-6
        assert info["max_tire_utilization"] <= 1.0 + 1e-9
        if terminated or truncated:
            break
    return info


@pytest.mark.parametrize("family", ["cutin", "sandwich"])
def test_env_random_rollouts_stay_finite_and_keep_the_interface(family):
    cfg_cls = FAMILIES[family]
    env = EgoTrafficEnv(arena=cfg_cls[1](cfg_cls[0](seed=3, road_mu=0.5)), mode="distribution",
                        rollover_event="physical_v2")
    assert env.action_space.shape == (2,) and env.observation_space.shape == (37,)
    starts = set()
    for episode in range(4):
        info = _random_rollout(env, episode)
        starts.add(round(float(env.agents[0].car.state.s), 3))
        assert {"tire", "stability_loss", "max_friction_demand", "lack_of_progress"} <= set(info)
    assert len(starts) > 1                                                 # randomization still runs


def test_env_truncates_and_reports_lack_of_progress():
    env = EgoTrafficEnv(arena=FAMILIES["cutin"][1](SPEC.config(seed=1000)), mode="fixed", rollover_event="physical_v2")
    env.reset()
    full_brake = np.array([-1.0, 0.0], dtype=np.float32)
    done, info = False, {}
    while not done:
        _, _, term, trunc, info = env.step(full_brake)
        done = term or trunc
    if info["timeout"]:
        assert info["lack_of_progress"] and not info["success"]
    assert env.ego_car.state.v_x >= -1e-9                                  # holding the brake never reverses


def test_env_requested_and_realized_acceleration_differ_on_low_friction():
    env = EgoTrafficEnv(arena=FAMILIES["cutin"][1](SPEC.config(seed=1000)), mode="fixed", rollover_event="physical_v2")
    env.reset()
    for _ in range(10):
        _, _, _, _, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert info["accel_requested"] == ACCEL_MIN and info["accel_actuator"] == ACCEL_MIN
    assert info["accel_realized"] == pytest.approx(-0.5 * G, rel=1e-3)
    assert info["tire"]["a_cmd"] == ACCEL_MIN and info["max_friction_demand"] > 1.0


def test_existing_controller_closes_the_loop_and_the_record_is_reproducible():
    w = MPCWeights().to_vector()
    a = run_physical_episode(build_controller({"name": "mpc"}, w), SPEC, 1000)
    b = run_physical_episode(build_controller({"name": "mpc"}, w), SPEC, 1000)
    assert a["peak_tire_utilization"] <= 1.0 + 1e-9
    assert a["peak_abs_Fx"] <= a["peak_abs_Fx_req"] + TOL and a["peak_abs_accel"] <= 0.5 * G + 1e-6
    for key in ("progress_m", "steps", "peak_abs_accel", "peak_friction_demand", "min_clearance_m"):
        assert a[key] == b[key]
