"""
Tests for the design-of-experiments sampler (learning.feasibility_sampling)
-- bounds/reproducibility/stratification, not any real PPO training.

Run with: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_feasibility_sampling.py -v
"""

import pytest

from learning.feasibility_cutin import THETA_BOUNDS_CUTIN
from learning.feasibility_sandwich import THETA_BOUNDS_SANDWICH
from learning.feasibility_sampling import sample_cutin, sample_sandwich


def test_sampled_theta_within_bounds():
    configs, rejected = sample_cutin(n=12, blocker_side=1, seed=0)
    assert len(configs) == 12
    for cfg in configs:
        for name, (lo, hi) in THETA_BOUNDS_CUTIN.items():
            assert lo <= getattr(cfg, name) <= hi, f"{name}={getattr(cfg, name)} outside [{lo}, {hi}]"


def test_reproducible_given_same_seed():
    a, _ = sample_cutin(n=10, blocker_side=1, seed=42)
    b, _ = sample_cutin(n=10, blocker_side=1, seed=42)
    assert [c.scenario_id() for c in a] == [c.scenario_id() for c in b]
    assert [c.theta() for c in a] == [c.theta() for c in b]


def test_different_seed_gives_different_design():
    a, _ = sample_cutin(n=10, blocker_side=1, seed=1)
    b, _ = sample_cutin(n=10, blocker_side=1, seed=2)
    assert [c.theta() for c in a] != [c.theta() for c in b]


def test_blocker_side_stratified_not_combined():
    """Both sides drawn from the SAME master seed must still produce
    different designs (independent per-side sampling, per the spec's own
    "treat as separate strata" requirement) -- and every config must
    actually carry the requested side."""
    pos, _ = sample_cutin(n=8, blocker_side=1, seed=0)
    neg, _ = sample_cutin(n=8, blocker_side=-1, seed=0)
    assert all(c.blocker_side == 1 for c in pos)
    assert all(c.blocker_side == -1 for c in neg)
    assert [c.theta() for c in pos] != [c.theta() for c in neg]


def test_every_sampled_theta_has_a_unique_seed():
    configs, _ = sample_cutin(n=20, blocker_side=1, seed=7)
    seeds = [c.seed for c in configs]
    assert len(set(seeds)) == len(seeds)


def test_sample_space_is_filling_not_clustered():
    """A crude space-filling sanity check: across enough LHS samples, the
    observed range of one dimension should cover most of its bound span,
    not cluster in a narrow sub-band."""
    configs, _ = sample_cutin(n=20, blocker_side=1, seed=0)
    lo, hi = THETA_BOUNDS_CUTIN["escape_gap_length_m"]
    values = [c.escape_gap_length_m for c in configs]
    span = hi - lo
    assert (max(values) - min(values)) > 0.6 * span


def test_sobol_method_also_works():
    configs, rejected = sample_cutin(n=8, blocker_side=1, seed=0, method="sobol")
    assert len(configs) == 8


def test_sandwich_family_sampling():
    configs, rejected = sample_sandwich(n=6, blocker_side=-1, seed=3)
    assert len(configs) == 6
    for cfg in configs:
        for name, (lo, hi) in THETA_BOUNDS_SANDWICH.items():
            assert lo <= getattr(cfg, name) <= hi


def test_common_config_kwargs_are_applied():
    configs, _ = sample_cutin(n=3, blocker_side=1, seed=0, goal_distance_m=250.0, max_episode_seconds=20.0)
    for cfg in configs:
        assert cfg.goal_distance_m == 250.0
        assert cfg.max_episode_seconds == 20.0
