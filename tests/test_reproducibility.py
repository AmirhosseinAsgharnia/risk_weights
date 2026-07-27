import itertools
import numpy as np
import model.car_init
from model.simulation import build_demo_scenario, Simulation

THETA_ROAD = [0.01, 50, 0.3, 225.0]
SURR_CARS = [
    (0, -40.0, 1.0), (0, -5.0, 5.0), (0, 35.0, -5.0),
    (1, -25.0, 1.0), (1, 50.0, -2.0),
    (2, -45.0, 3.0), (2, 10.0, -4.0),
]


def _run(seed):
    # Actor ids come from a module-level counter, so two constructions of "the
    # same" scenario in one process get different (but each internally
    # consistent) id sequences -- reset it so this test isolates trajectory
    # determinism from that unrelated global-counter state.
    model.car_init._id_counter = itertools.count()
    scenario = build_demo_scenario(
        THETA_ROAD, ego_lane=1, ego_speed=20.0, surr_cars=SURR_CARS,
        rng=np.random.default_rng(seed),
    )
    # exercise the only stochastic path (MOBIL noise) so reproducibility is
    # actually meaningful, not trivially true because nothing random happened
    scenario.mobil_defaults = type(scenario.mobil_defaults)(
        politeness=scenario.mobil_defaults.politeness,
        a_thr=scenario.mobil_defaults.a_thr,
        b_safe=scenario.mobil_defaults.b_safe,
        noise_std=0.5,
    )
    return Simulation(scenario).run(n_steps=100)


def test_same_seed_produces_identical_trajectories():
    result_a = _run(seed=42)
    result_b = _run(seed=42)

    assert len(result_a.history) == len(result_b.history)
    for rec_a, rec_b in zip(result_a.history, result_b.history):
        for log_a, log_b in zip(rec_a.actors, rec_b.actors):
            assert np.array_equal(log_a.state, log_b.state)
    assert result_a.events == result_b.events
    assert result_a.lane_changes == result_b.lane_changes


def test_different_seeds_can_produce_different_lane_change_timing():
    result_a = _run(seed=1)
    result_b = _run(seed=2)
    # not a strict requirement that they differ, but with noise_std=0.5 over
    # 7 cars and 100 steps it would be suspicious if they were identical
    same = all(
        np.array_equal(la.state, lb.state)
        for ra, rb in zip(result_a.history, result_b.history)
        for la, lb in zip(ra.actors, rb.actors)
    )
    assert not same
