import numpy as np
import pytest
from scipy.optimize import brentq

from model.road import RoadScenario
from model.car.config import VehicleParameters
from model.car.model import EgoModel, propagate_ctrv_trajectory
from model.risk import (
    RiskParams, EKFPropagator, numerical_jacobian,
    propagate_ego_trajectory,
    frenet_to_global_with_cov, evaluate_risk_profile,
    rollover_risk, slip_risk, lane_departure_risk, road_departure_risk, car_to_car_risk,
)

STRAIGHT_ROAD = RoadScenario(kappa_max=0.0, lane_num=3)
CURVED_ROAD = RoadScenario(kappa_max=0.02, lane_num=3)  # kappa=0.02 within s in [210, 290)
VP = VehicleParameters()
P = RiskParams()


def _ego_state(s=100.0, e_y=0.0, e_psi=0.0, vx=20.0, vy=0.0, r=0.0):
    return np.array([s, e_y, e_psi, vx, vy, r], dtype=np.float64)


# ---------- range: every p in [0,1] over randomized trajectories ----------

def test_all_probabilities_in_unit_range_over_randomized_trajectories():
    rng = np.random.default_rng(0)
    for _ in range(30):
        state = _ego_state(
            s=rng.uniform(50, 400), e_y=rng.uniform(-5, 5), e_psi=rng.uniform(-0.3, 0.3),
            vx=rng.uniform(5, 30), vy=rng.uniform(-5, 5), r=rng.uniform(-1, 1),
        )
        cov = P.R_ego * rng.uniform(0.1, 5.0)
        u = (rng.uniform(-3, 3), rng.uniform(-0.3, 0.3))

        p_r, _, _ = rollover_risk(state, cov, VP, CURVED_ROAD, u, P)
        p_s, _, _ = slip_risk(state, cov, VP, CURVED_ROAD, u, P)
        p_l, _, _ = lane_departure_risk(state, cov, CURVED_ROAD, 2.0, 1, P)
        p_rd, _, _ = road_departure_risk(state, cov, CURVED_ROAD, 2.0, P)

        for val in (p_r, p_s, p_l, p_rd):
            assert 0.0 <= val <= 1.0
            assert np.isfinite(val)
            assert isinstance(val, float)

    # c2c across randomized neighbour placements
    ego_state = _ego_state(s=100.0)
    for _ in range(30):
        n_mean = np.array([
            rng.uniform(80, 220), rng.uniform(-10, 10), rng.uniform(-1, 1),
            rng.uniform(5, 30), rng.uniform(-0.5, 0.5),
        ])
        agg, per = car_to_car_risk(ego_state, P.R_ego, STRAIGHT_ROAD, (4.0, 2.0),
                                    [n_mean], [P.R_ctrv], [(4.0, 2.0)], P)
        assert 0.0 <= agg <= 1.0 and np.isfinite(agg)
        for p_i, g_i, sigma_i in per:
            assert 0.0 <= p_i <= 1.0
            assert np.isfinite(g_i) and np.isfinite(sigma_i)


# ---------- boundary: p = 0.5 at each risk's threshold ----------

def test_lane_departure_boundary_is_half_when_body_edge_on_the_line():
    # lane1 edges at [-l_w/2, l_w/2] = [-2, 2]; width=2 car -> right edge on
    # the boundary when e_y = 2 - 1 = 1
    state = _ego_state(e_y=1.0)
    prob, g, _ = lane_departure_risk(state, P.R_ego, STRAIGHT_ROAD, 2.0, 1, P)
    assert prob == pytest.approx(0.5, abs=1e-9)
    assert g == pytest.approx(0.0, abs=1e-9)


def test_road_departure_boundary_is_half_when_body_edge_on_the_line():
    # 3-lane road: outer edges at +-6.0; width=2 car -> right edge on the
    # outer boundary when e_y = 6 - 1 = 5
    state = _ego_state(e_y=5.0)
    prob, g, _ = road_departure_risk(state, P.R_ego, STRAIGHT_ROAD, 2.0, P)
    assert prob == pytest.approx(0.5, abs=1e-9)
    assert g == pytest.approx(0.0, abs=1e-9)


def test_rollover_boundary_is_half_when_ay_equals_ssf_times_g():
    def g_of_vy(vy):
        state = _ego_state(vx=20.0, vy=vy, r=0.3)
        _, g, _ = rollover_risk(state, P.R_ego, VP, STRAIGHT_ROAD, (0.0, 0.05), P)
        return g

    vy_root = brentq(g_of_vy, 0.0, 15.0, xtol=1e-10)
    state = _ego_state(vx=20.0, vy=vy_root, r=0.3)
    prob, g, _ = rollover_risk(state, P.R_ego, VP, STRAIGHT_ROAD, (0.0, 0.05), P)
    assert g == pytest.approx(0.0, abs=1e-6)
    assert prob == pytest.approx(0.5, abs=1e-6)


def test_slip_boundary_is_half_when_rho_equals_one():
    def g_of_vy(vy):
        state = _ego_state(vx=20.0, vy=vy, r=0.3)
        _, g, _ = slip_risk(state, P.R_ego, VP, STRAIGHT_ROAD, (0.0, 0.05), P)
        return g

    # bracket a sign change
    lo, hi = 0.0, 15.0
    assert g_of_vy(lo) < 0 < g_of_vy(hi)
    vy_root = brentq(g_of_vy, lo, hi, xtol=1e-10)
    state = _ego_state(vx=20.0, vy=vy_root, r=0.3)
    prob, g, _ = slip_risk(state, P.R_ego, VP, STRAIGHT_ROAD, (0.0, 0.05), P)
    assert g == pytest.approx(0.0, abs=1e-6)
    assert prob == pytest.approx(0.5, abs=1e-6)


def test_c2c_boundary_is_half_when_gap_equals_d_safe():
    ego_state = _ego_state(s=100.0, e_y=0.0)
    ego_x, ego_y, ego_psi = 100.0, 0.0, 0.0  # straight road, s=100 -> x=100, y=0 approx
    # place neighbour directly ahead so gap = d_safe exactly:
    # bumper-to-bumper gap = center_distance - (len_ego+len_n)/2
    center_distance = P.d_safe + (4.0 + 4.0) / 2.0
    n_mean = np.array([ego_x + center_distance, ego_y, 0.0, 20.0, 0.0])
    prob_agg, per = car_to_car_risk(ego_state, P.R_ego, STRAIGHT_ROAD, (4.0, 2.0),
                                     [n_mean], [P.R_ctrv], [(4.0, 2.0)], P)
    prob_i, g_i, _ = per[0]
    assert g_i == pytest.approx(P.d_safe, abs=1e-6)
    assert prob_i == pytest.approx(0.5, abs=1e-6)


# ---------- monotonicity: in gap, and in uncertainty ----------

def test_c2c_probability_increases_as_gap_closes():
    ego_state = _ego_state(s=100.0, e_y=0.0)
    distances = [30.0, 20.0, 10.0, 5.0, 2.0]
    probs = []
    for d in distances:
        n_mean = np.array([100.0 + d, 0.0, 0.0, 20.0, 0.0])
        agg, _ = car_to_car_risk(ego_state, P.R_ego, STRAIGHT_ROAD, (4.0, 2.0),
                                  [n_mean], [P.R_ctrv], [(4.0, 2.0)], P)
        probs.append(agg)
    assert all(b >= a - 1e-12 for a, b in zip(probs, probs[1:])), probs


def test_rollover_probability_moves_toward_half_as_uncertainty_grows_at_fixed_safe_gap():
    # vy=0, r=0 exactly would sit at a genuine stationary point of |a_y|
    # (a_y is odd in vy, so |a_y| has zero gradient there by construction --
    # not a bug, just the documented EKF/delta-method limitation the module
    # itself flags: first-order sensitivity vanishes at a symmetric
    # zero-crossing). Use a realistic near-straight state instead, safely on
    # the low side, where the gradient is genuinely nonzero.
    state = _ego_state(vx=20.0, vy=1.0, r=0.1)
    scales = [0.1, 1.0, 10.0, 100.0, 1000.0]
    probs = []
    for scale in scales:
        prob, g, _ = rollover_risk(state, P.R_ego * scale, VP, STRAIGHT_ROAD, (0.0, 0.0), P)
        assert g < 0  # confirm we're on the safe side throughout
        probs.append(prob)
    # monotonically increasing toward 0.5 as sigma grows from a safe (p<0.5) state
    assert all(b >= a - 1e-12 for a, b in zip(probs, probs[1:])), probs
    assert probs[0] < 0.5
    assert probs[-1] > probs[0]


def test_lane_departure_probability_moves_toward_half_as_uncertainty_grows():
    state = _ego_state(e_y=0.0)  # dead-centre in the lane -- very safe
    scales = [0.1, 1.0, 10.0, 100.0, 1000.0]
    probs = [lane_departure_risk(state, P.R_ego * s, STRAIGHT_ROAD, 2.0, 1, P)[0] for s in scales]
    assert all(b >= a - 1e-12 for a, b in zip(probs, probs[1:])), probs
    assert probs[0] < 0.1
    assert probs[-1] > probs[0]


# ---------- frame/covariance: Jacobian actually applied on a curved step ----------

def test_ego_global_covariance_uses_the_frenet_jacobian_on_a_curve():
    # pick s inside the constant-curvature arc so kappa != 0
    s_curve = 250.0
    assert CURVED_ROAD.kappa[int(round(s_curve / CURVED_ROAD.ds))] != 0.0

    state = _ego_state(s=s_curve, e_y=0.5, e_psi=0.1)
    cov = np.diag([0.1, 0.2, 0.01, 0.3, 0.1, 0.02]).astype(np.float64)

    _, cov_global = frenet_to_global_with_cov(state, cov, CURVED_ROAD)

    # the WRONG answer: skip J and just reuse the (s,e_y,e_psi) block of the
    # raw Frenet covariance as if it were already (x,y,psi) -- if the
    # implementation actually used J, cov_global must differ from this.
    wrong_reuse = cov[:3, :3]
    assert not np.allclose(cov_global, wrong_reuse)

    # sanity: still a valid (symmetric, PSD-ish) 3x3 covariance
    assert cov_global.shape == (3, 3)
    assert np.allclose(cov_global, cov_global.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(cov_global) > -1e-8)


def test_covariance_grows_monotonically_over_the_forecast_horizon():
    propagator = EKFPropagator(Q=P.Q_ego)
    x0 = _ego_state(s=100.0, vx=20.0)
    controls = [(0.0, 0.01)] * 10
    traj = propagate_ego_trajectory(x0, P.R_ego, controls, CURVED_ROAD, VP, propagator, dt=0.1)
    traces = [np.trace(cov) for _, cov in traj]
    assert all(b > a for a, b in zip(traces, traces[1:])), traces


# ---------- smoothness: finite dp/d(input), nonzero near thresholds ----------

def test_lane_departure_is_smooth_near_its_threshold():
    e_y0 = 1.0  # the boundary found above
    h = 1e-4

    def p_at(e_y):
        state = _ego_state(e_y=e_y)
        return lane_departure_risk(state, P.R_ego, STRAIGHT_ROAD, 2.0, 1, P)[0]

    dp = (p_at(e_y0 + h) - p_at(e_y0 - h)) / (2 * h)
    assert np.isfinite(dp)
    assert abs(dp) > 1e-6


def test_c2c_is_smooth_near_its_threshold():
    ego_state = _ego_state(s=100.0, e_y=0.0)
    center_distance0 = P.d_safe + 4.0
    h = 1e-3

    def p_at(center_distance):
        n_mean = np.array([100.0 + center_distance, 0.0, 0.0, 20.0, 0.0])
        agg, _ = car_to_car_risk(ego_state, P.R_ego, STRAIGHT_ROAD, (4.0, 2.0),
                                  [n_mean], [P.R_ctrv], [(4.0, 2.0)], P)
        return agg

    dp = (p_at(center_distance0 + h) - p_at(center_distance0 - h)) / (2 * h)
    assert np.isfinite(dp)
    assert abs(dp) > 1e-6


def test_rollover_is_smooth_near_its_threshold():
    def g_of_vy(vy):
        state = _ego_state(vx=20.0, vy=vy, r=0.3)
        return rollover_risk(state, P.R_ego, VP, STRAIGHT_ROAD, (0.0, 0.05), P)[1]

    vy0 = brentq(g_of_vy, 0.0, 15.0, xtol=1e-10)
    h = 1e-4

    def p_at(vy):
        state = _ego_state(vx=20.0, vy=vy, r=0.3)
        return rollover_risk(state, P.R_ego, VP, STRAIGHT_ROAD, (0.0, 0.05), P)[0]

    dp = (p_at(vy0 + h) - p_at(vy0 - h)) / (2 * h)
    assert np.isfinite(dp)
    assert abs(dp) > 1e-6


# ---------- dead-feature guard: rollover is reachable, not structurally dead ----------

def test_rollover_risk_is_reachable_past_half_on_a_narrow_tall_vehicle():
    """A narrow/tall vehicle (h_CG=0.90, matching the SSF docstring's own
    example) has SSF well below mu -- rollover becomes reachable before slip
    saturates the tires. Confirms p_roll can actually cross 0.5 given real
    tire/road numbers, not just in the abstract Phi(g/sigma) formula."""
    narrow_tall_vp = VehicleParameters(h_CG=0.90)
    assert narrow_tall_vp.SSF < narrow_tall_vp.mu_max  # rollover reachable before slip, by construction

    def g_of_vy(vy):
        state = _ego_state(vx=25.0, vy=vy, r=0.6)
        return rollover_risk(state, P.R_ego, narrow_tall_vp, STRAIGHT_ROAD, (0.0, 0.15), P)[1]

    # search for a state where the rollover margin actually goes positive
    vys = np.linspace(0.0, 20.0, 50)
    gs = [g_of_vy(vy) for vy in vys]
    assert max(gs) > 0.0, "rollover risk never crosses its threshold -- structurally dead"

    vy_bad = vys[int(np.argmax(gs))]
    state = _ego_state(vx=25.0, vy=vy_bad, r=0.6)
    prob, g, _ = rollover_risk(state, P.R_ego, narrow_tall_vp, STRAIGHT_ROAD, (0.0, 0.15), P)
    assert prob > 0.5


# ---------- purity / hard constraints ----------

def test_risk_functions_do_not_mutate_inputs():
    state = _ego_state()
    state_copy = state.copy()
    cov = P.R_ego.copy()
    cov_copy = cov.copy()

    rollover_risk(state, cov, VP, STRAIGHT_ROAD, (0.0, 0.0), P)
    slip_risk(state, cov, VP, STRAIGHT_ROAD, (0.0, 0.0), P)
    lane_departure_risk(state, cov, STRAIGHT_ROAD, 2.0, 1, P)
    road_departure_risk(state, cov, STRAIGHT_ROAD, 2.0, P)

    assert np.array_equal(state, state_copy)
    assert np.array_equal(cov, cov_copy)


def test_evaluate_risk_profile_instantaneous_case_is_length_one():
    state = _ego_state()
    ego_traj = [(state, P.R_ego)]
    controls = [(0.0, 0.0)]
    n_mean = np.array([150.0, 0.0, 0.0, 20.0, 0.0])
    profile = evaluate_risk_profile(
        ego_traj, controls, [[(n_mean, P.R_ctrv)]], [(4.0, 2.0)],
        STRAIGHT_ROAD, VP, (4.0, 2.0), 1, P,
    )
    for arr in (profile.p_slip, profile.p_roll, profile.p_lane, profile.p_road, profile.p_c2c):
        assert arr.shape == (1,)
        assert 0.0 <= arr[0] <= 1.0


def test_numerical_jacobian_matches_known_linear_map():
    A = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -1.0]])

    def f(x):
        return A @ x

    x0 = np.array([1.0, 2.0, 3.0])
    J = numerical_jacobian(f, x0)
    assert np.allclose(J, A, atol=1e-6)
