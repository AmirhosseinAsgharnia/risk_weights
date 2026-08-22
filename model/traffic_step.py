"""
One-step advance for a list of surrounding ("surr") TrafficAgents: MOBIL
lane-change decisions, IDM car-following, far-near steering (with
curvature-preview feedforward), and the surr-surr plastic-collision sweep.

Extracted out of tests/traffic_test.py so its per-step physics has exactly
one implementation, shared between that visualization script and
learning/env.py's training environment -- the two must never be able to
silently diverge.

Ego is out of scope here entirely: step_surr_agents only advances `agents`
(the surr TrafficAgent list). A caller that also has an ego vehicle must
step it separately and handle ego/surr overlap itself via
model.collision.ego_overlaps_any -- never route an ego overlap through
resolve_surr_collisions (see that function's own docstring).
"""

from controllers.idm import idm_accel, IDM_PRESETS
from controllers.far_near import (
    far_near_steering, far_near_lookahead_offset, clip_steering_rate,
    far_near_curvature_feedforward, FAR_NEAR_PRESETS,
)
from controllers.mobil import mobil_decision, MobilParams
from initialization.traffic_init import CAR_LENGTH
from model.collision import resolve_surr_collisions, bleed_crashed


def find_leader(agents, lane, s, exclude_id):
    ahead = [a for a in agents if a.car.state.lane == lane and a.car.car_id != exclude_id and a.car.state.s > s]
    return min(ahead, key = lambda a: a.car.state.s) if ahead else None


def find_follower(agents, lane, s, exclude_id):
    behind = [a for a in agents if a.car.state.lane == lane and a.car.car_id != exclude_id and a.car.state.s < s]
    return max(behind, key = lambda a: a.car.state.s) if behind else None


def _idm_accel_of(follower_state, v0, idm_params, leader_agent):
    """leader_agent may be any object with .car.state.s/.v_x (a TrafficAgent),
    or None for free-road."""
    if leader_agent is None:
        gap, dv = float("inf"), 0.0
    else:
        gap = leader_agent.car.state.s - follower_state.s - CAR_LENGTH
        dv = follower_state.v_x - leader_agent.car.state.v_x
    return idm_accel(follower_state.v_x, gap, dv, v0, idm_params.a_max, idm_params.b,
                      idm_params.s0, idm_params.T, idm_params.delta)


def evaluate_mobil(agents, agent, candidate_lane, mobil_params: MobilParams):
    """MOBIL incentive/safety for `agent` moving from its current lane to
    candidate_lane, given the current traffic snapshot."""
    car = agent.car
    agent_lane = car.state.lane
    agent_idm = IDM_PRESETS[car.behaviour]

    old_leader   = find_leader(agents, agent_lane, car.state.s, car.car_id)
    old_follower = find_follower(agents, agent_lane, car.state.s, car.car_id)
    new_leader   = find_leader(agents, candidate_lane, car.state.s, car.car_id)
    new_follower = find_follower(agents, candidate_lane, car.state.s, car.car_id)

    a_agent_before = _idm_accel_of(car.state, agent.v0, agent_idm, old_leader)
    a_agent_after  = _idm_accel_of(car.state, agent.v0, agent_idm, new_leader)

    if old_follower is None:
        a_old_follower_before = a_old_follower_after = 0.0
    else:
        of_idm = IDM_PRESETS[old_follower.car.behaviour]
        a_old_follower_before = _idm_accel_of(old_follower.car.state, old_follower.v0, of_idm, agent)
        a_old_follower_after  = _idm_accel_of(old_follower.car.state, old_follower.v0, of_idm, old_leader)

    if new_follower is None:
        a_new_follower_before = a_new_follower_after = 0.0
    else:
        nf_idm = IDM_PRESETS[new_follower.car.behaviour]
        a_new_follower_before = _idm_accel_of(new_follower.car.state, new_follower.v0, nf_idm, new_leader)
        a_new_follower_after  = _idm_accel_of(new_follower.car.state, new_follower.v0, nf_idm, agent)

    return mobil_decision(a_agent_before, a_agent_after,
                           a_old_follower_before, a_old_follower_after,
                           a_new_follower_before, a_new_follower_after,
                           mobil_params)


def step_surr_agents(
        agents, road, t: float, dt: float, *,
        mobil_params: MobilParams,
        lane_num: int,
        max_braking: float = 8.0,
        lane_change_cooldown: float = 1.0,
        crash_bleed_k: float = 1.0,
) -> None:
    """Advance every agent in `agents` by one step of dt, in place.

    Step numbering matches what used to be inline in tests/traffic_test.py:
    1. MOBIL lane-change decision (skipped mid-change or on cooldown).
    2. IDM car-following accel, clamped to max_braking.
    3. Far-near steering + curvature-preview feedforward, rate-limited.
    4. Road curvature/friction lookup, then Car.step integrates dynamics.
    5. Commit the lane change once its duration has elapsed.
    6. Global (x, y, heading) pose, stored on car.state for any caller
       (plotting, observation construction) that needs it.
    A CRASHED agent skips 1-5 entirely (see model.collision.bleed_crashed).
    Step 7, once for the whole list: surr-surr collision resolution.
    """
    for agent in agents:
        car = agent.car

        if agent.crashed:
            bleed_crashed(agent, road, dt, k = crash_bleed_k)
        else:
            # 1. MOBIL: only consider a new lane change once any active one has
            # committed, and not within lane_change_cooldown of the last one.
            on_cooldown = (agent.last_lane_change_t is not None
                           and (t - agent.last_lane_change_t) < lane_change_cooldown)
            if agent.lane_change_t0 is None and not on_cooldown:
                best_lane, best_incentive = None, mobil_params.threshold
                for candidate in (car.state.lane - 1, car.state.lane + 1):
                    if not (0 <= candidate < lane_num):
                        continue
                    should_change, incentive = evaluate_mobil(agents, agent, candidate, mobil_params)
                    if should_change and incentive > best_incentive:
                        best_lane, best_incentive = candidate, incentive
                if best_lane is not None:
                    agent.target_lane = best_lane
                    agent.lane_change_t0 = t

            # 2. IDM: follow the target lane's leader (a car commits to the
            # new lane's traffic stream as soon as a change starts, not just
            # once it completes). find_leader doesn't filter by crashed
            # status, so a stopped wreck is automatically a valid leader.
            idm_p = IDM_PRESETS[car.behaviour]
            leader = find_leader(agents, agent.target_lane, car.state.s, car.car_id)
            accel = _idm_accel_of(car.state, agent.v0, idm_p, leader)
            # idm_accel is deliberately unclamped (see its own docstring) -- a
            # near-zero gap sends (s_star/gap)^2, and so accel, toward -inf.
            # Physical actuation limit, not part of the IDM formula itself.
            accel = max(accel, -max_braking)

            # 3. Far-near steering, target ramped from the old lane's centreline
            # to the new one over this car's lane_change_duration.
            fn_p = FAR_NEAR_PRESETS[car.behaviour]
            if agent.lane_change_t0 is not None:
                old_lane_obj = road.lanes[car.state.lane]
                new_lane_obj = road.lanes[agent.target_lane]
                full_shift = old_lane_obj.offset - new_lane_obj.offset   # type: ignore
                progress = min(1.0, (t - agent.lane_change_t0) / fn_p.lane_change_duration)
                e_y_ref = car.state.e_y - progress * full_shift
            else:
                e_y_ref = car.state.e_y

            e_y_near = far_near_lookahead_offset(e_y_ref, car.state.e_psi, fn_p.L_n)
            d_far    = car.state.v_x * fn_p.T_f
            e_y_far  = far_near_lookahead_offset(e_y_ref, car.state.e_psi, d_far)
            delta_cmd = far_near_steering(e_y_near, e_y_far, car.state.v_x, fn_p.k_n, fn_p.k_f)

            # Curvature-preview feedforward: look up the road's curvature at
            # the far lookahead point (not the car's current s) so delta
            # starts ramping toward what the upcoming curve needs before the
            # car geometrically reaches it -- pure e_y/e_psi feedback always
            # reacts after the fact (see far_near_curvature_feedforward).
            idx_preview = road.index_at(car.state.s + d_far)
            kappa_preview = -road.lanes[car.state.lane].kappa[idx_preview]  # type: ignore
            delta_cmd += far_near_curvature_feedforward(kappa_preview, car.vehicle_params.L)

            delta = clip_steering_rate(delta_cmd, agent.prev_delta, fn_p.steer_rate_limit, dt)
            agent.prev_delta = delta

            # 4. Road curvature/friction at this car's current position, using
            # its nominal (not target) lane, matching the leader lookups above.
            idx = road.index_at(car.state.s)
            kappa = -road.lanes[car.state.lane].kappa[idx]  # type: ignore
            mu = road.mu

            car.step(accel, delta, kappa, mu, dt)

            # 5. Commit the lane change once its duration has elapsed.
            if agent.lane_change_t0 is not None and (t - agent.lane_change_t0) >= fn_p.lane_change_duration:
                old_lane_obj = road.lanes[car.state.lane]
                new_lane_obj = road.lanes[agent.target_lane]
                car.state.e_y = car.state.e_y + (new_lane_obj.offset - old_lane_obj.offset)  # type: ignore
                car.state.lane = agent.target_lane
                agent.lane_change_t0 = None
                agent.last_lane_change_t = t

        # 6. Global pose (common to both branches).
        lane_obj = road.lanes[car.state.lane]
        backbone_e_y = -lane_obj.offset + car.state.e_y  # type: ignore
        x, y, heading = road.frenet_to_global(car.state.s, backbone_e_y, car.state.e_psi)
        car.state.x, car.state.y, car.state.heading = x, y, heading

    # 7. Surr-surr collision detection + plastic response, once per step
    # after everyone (crashed or not) has moved -- see model.collision.
    resolve_surr_collisions(agents, road, t)
