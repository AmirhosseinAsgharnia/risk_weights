"""
MOBIL (Minimize Overall Braking Induced by Lane change) — lane-change
decision controller, used alongside IDM (car-following) and far-near
(steering) for surrounding traffic.

For surrounding traffic only, not the ego vehicle (ego will be driven by an
MPC, designed later).
"""

from typing import NamedTuple


class MobilParams(NamedTuple):
    """
    politeness: [-] p -- weight on how much the change disturbs the old and
                new followers, relative to ego's own gain. p=0 is pure
                egoism (ignore neighbours); higher p makes ego defer more.
    threshold:  [m/s^2] a_thr -- minimum net (politeness-weighted) advantage
                required before it's worth changing lanes at all, i.e. a
                deadband against constant lane-hopping for a marginal gain.
    b_safe:     [m/s^2] hard safety cap -- ego may never force the new
                follower to brake harder than this.
    """
    politeness: float = 0.2
    threshold: float = 0.2
    b_safe: float = 4.0


def mobil_decision(
        a_ego_before: float, a_ego_after: float,
        a_old_follower_before: float, a_old_follower_after: float,
        a_new_follower_before: float, a_new_follower_after: float,
        p: MobilParams | None = None,
) -> tuple[bool, float]:
    """
    MOBIL lane-change decision for one candidate lane, given the IDM
    acceleration ego and its old/new-lane followers would have before vs.
    after the change (compute these with idm_accel against the relevant
    leader/follower gaps in each lane -- this function only implements the
    MOBIL inequalities themselves, not the traffic-snapshot lookup that
    finds those neighbours).

    Safety: the new follower must not be forced to brake harder than
    b_safe if ego merges in front of it.
        a_new_follower_after >= -b_safe

    Incentive: ego's own gain, plus a politeness-weighted (dis)benefit to
    the old and new followers, must exceed threshold.
        (a_ego_after - a_ego_before)
        + p * [(a_new_follower_after - a_new_follower_before)
               + (a_old_follower_after - a_old_follower_before)]
        > a_thr

    Returns (should_change, incentive). incentive is the left-hand side of
    the inequality above (-inf if the safety check fails), so callers
    comparing several candidate lanes can pick the best rather than just
    the first that clears the threshold.
    """
    p = p or MobilParams()

    if a_new_follower_after < -p.b_safe:
        return False, float("-inf")

    incentive = (
        (a_ego_after - a_ego_before)
        + p.politeness * (
            (a_new_follower_after - a_new_follower_before)
            + (a_old_follower_after - a_old_follower_before)
        )
    )
    return incentive > p.threshold, incentive
