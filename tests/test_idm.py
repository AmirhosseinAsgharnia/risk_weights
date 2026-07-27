import numpy as np
from model.IDM_MOBIL import IDMParams, idm_accel


def test_free_road_accelerates_toward_a_max_when_far_below_v0():
    p = IDMParams(v0=30.0, a_max=1.5)
    a = idm_accel(v=20.0, gap=np.inf, dv=0.0, p=p)
    assert 0.0 < a < p.a_max


def test_at_desired_speed_with_huge_gap_accel_is_near_zero():
    p = IDMParams(v0=30.0)
    a = idm_accel(v=30.0, gap=1e6, dv=0.0, p=p)
    assert abs(a) < 1e-6


def test_speed_above_desired_speed_decelerates():
    p = IDMParams(v0=20.0)
    a = idm_accel(v=25.0, gap=1e6, dv=0.0, p=p)
    assert a < 0.0


def test_steady_following_at_the_equilibrium_gap_is_near_zero():
    """
    a = a_max * (1 - (v/v0)**delta - (s*/gap)**2) is zero (equilibrium) when
    gap = s* / sqrt(1 - (v/v0)**delta) -- NOT at gap == s* itself (that's the
    jam/minimum distance term, not the equilibrium spacing; at gap == s* with
    dv == 0 the formula still commands a full braking response unless v == v0,
    and v == v0 makes the equilibrium gap infinite). v must be strictly below
    v0 for a finite equilibrium gap to exist.
    """
    p = IDMParams(v0=30.0, T=1.5, s0=2.0, a_max=1.5, b=2.0, delta=4.0)
    v = 20.0
    s_star = p.s0 + v * p.T
    free_flow_term = 1.0 - (v / p.v0) ** p.delta
    gap_eq = s_star / np.sqrt(free_flow_term)

    a = idm_accel(v=v, gap=gap_eq, dv=0.0, p=p)
    assert abs(a) < 1e-6


def test_approaching_slower_leader_decelerates():
    p = IDMParams(v0=30.0)
    a = idm_accel(v=25.0, gap=30.0, dv=10.0, p=p)  # closing fast
    assert a < 0.0


def test_dangerously_small_gap_hits_hard_braking_and_is_clipped():
    p = IDMParams(v0=30.0, a_min=-6.0)
    a = idm_accel(v=25.0, gap=0.5, dv=15.0, p=p)
    assert a == p.a_min


def test_no_leader_is_free_flow():
    p = IDMParams(v0=25.0)
    a_inf_gap = idm_accel(v=20.0, gap=np.inf, dv=0.0, p=p)
    a_huge_gap = idm_accel(v=20.0, gap=1e9, dv=0.0, p=p)
    assert np.isclose(a_inf_gap, a_huge_gap, atol=1e-6)


def test_zero_speed_gives_max_accel_regardless_of_leader():
    p = IDMParams(v0=25.0, a_max=1.5)
    a = idm_accel(v=0.0, gap=1e6, dv=0.0, p=p)  # gap large enough that its term is negligible
    assert np.isclose(a, p.a_max, atol=1e-6)


def test_negative_gap_does_not_raise_and_commands_max_braking():
    p = IDMParams(a_min=-6.0)
    a = idm_accel(v=20.0, gap=-3.0, dv=5.0, p=p)
    assert a == p.a_min


def test_clip_false_can_exceed_actuation_bounds_but_stays_finite():
    p = IDMParams(a_min=-6.0, a_max=1.5)
    a_clipped = idm_accel(v=25.0, gap=0.1, dv=20.0, p=p, clip=True)
    a_unclipped = idm_accel(v=25.0, gap=0.1, dv=20.0, p=p, clip=False)
    assert a_clipped == p.a_min
    assert a_unclipped < a_clipped  # unclipped is more negative -- distinguishable from other "bad" cases
    assert np.isfinite(a_unclipped)
