"""
Design-of-experiments sampling for the feasibility pipeline -- generates a
seeded, space-filling initial batch of nominal theta values over a scenario
family's validated bounds, rejecting (and re-drawing) any point the
family's own validator considers physically invalid, so the training/
evaluation stage (learning.feasibility_train_one) never even sees an
unrealizable scenario. See learning.feasibility_pipeline (Runner B) for how
this batch feeds into training.

Sampling itself only ever draws from a single seeded np.random.Generator
per call (no untracked global randomness) via scipy.stats.qmc, which is
deterministic given that seed -- the same (n, blocker_side, seed) always
reproduces the same batch of theta values.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from learning.feasibility_common import BlockerSide, ScenarioFamilyConfigError
from learning.feasibility_cutin import CutinConfig, THETA_BOUNDS_CUTIN, validate_cutin
from learning.feasibility_sandwich import SandwichConfig, THETA_BOUNDS_SANDWICH, validate_sandwich
from learning.scenario import derive_seed

_LANE_NUM = 3   # matches learning.env.LANE_NUM -- validate_cutin/validate_sandwich only need
                # lane_num/ego_lane to check blocker_side's implied lane exists; the standard
                # 3-lane, middle-lane-ego road (see learning.env.reset()'s arena branch) is the
                # only configuration this pipeline trains against.
_EGO_LANE = _LANE_NUM // 2

_FAMILY_SPECS = {
    "cutin": (CutinConfig, THETA_BOUNDS_CUTIN, validate_cutin),
    "sandwich": (SandwichConfig, THETA_BOUNDS_SANDWICH, validate_sandwich),
}


@dataclass(frozen=True)
class RejectedSample:
    """One rejected (and then re-drawn) candidate -- kept for the pipeline's
    own logging, never silently dropped (see module docstring)."""
    theta: dict
    reason: str


def _unit_cube_to_theta(cube_row: np.ndarray, bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    """cube_row in [0, 1]^d (one qmc.LatinHypercube/qmc.Sobol sample) -> theta dict, linearly
    mapping each dimension from [0, 1] to that field's own (lo, hi) -- bounds.items() order is
    the same order the sampler's dimensionality was constructed with (see sample_*)."""
    return {name: float(lo + cube_row[i] * (hi - lo)) for i, (name, (lo, hi)) in enumerate(bounds.items())}


def _sample_family(
        family: str, n: int, blocker_side: BlockerSide, seed: int, *,
        method: str = "lhs", oversample_factor: int = 4, max_rounds: int = 20,
        extra_config_kwargs: dict | None = None,
) -> tuple[list, list[RejectedSample]]:
    """n validated configs for one family + blocker_side, plus every
    rejected candidate along the way (with its reason). Draws
    n * oversample_factor candidates per round and keeps the first n that
    validate; if fewer than n survive (bounds alone never trigger a
    rejection here -- see each family's own bounds vs. goal_distance_m
    defaults -- so this is normally a no-op safety net, not the common
    case), draws another round with a fresh-but-deterministic seed rather
    than looping forever. Raises if max_rounds is exhausted, so a
    misconfigured bound (e.g. an unreasonably small goal_distance_m) fails
    loudly instead of hanging."""
    cfg_cls, bounds, validate = _FAMILY_SPECS[family]
    extra_config_kwargs = extra_config_kwargs or {}
    dim = len(bounds)

    sampler_cls = {"lhs": qmc.LatinHypercube, "sobol": qmc.Sobol}[method]
    # derive_seed feeds into numpy's SeedSequence, which requires non-negative entropy -- blocker_side
    # is +/-1, so it can't be passed there directly; 0/1 still disambiguates the two sides completely.
    side_code = 0 if blocker_side == 1 else 1
    rejected: list[RejectedSample] = []
    accepted: list = []
    # Two independent designs, not a mirrored one: the LHS/Sobol sampler's OWN seed must also vary
    # by side (not just each accepted theta's downstream scenario_seed below), or both sides would
    # draw the identical grid of theta VALUES and differ only in which side/seed label is attached --
    # that covers the same slice of theta-space twice instead of stratifying real coverage across it.
    round_seed = derive_seed(seed, side_code, 0)

    for _round in range(max_rounds):
        remaining = n - len(accepted)
        if remaining <= 0:
            break
        sampler = sampler_cls(d=dim, seed=round_seed)
        # Sobol's own balance properties degrade for a non-power-of-2 sample count -- .random()
        # (rather than .random_base2()) still returns a valid, deterministic QMC sequence for an
        # arbitrary count, just without that specific guarantee; acceptable here since points are
        # independently rejection-checked anyway, and LHS (this module's default) has no such
        # restriction at all.
        cube = sampler.random(n=remaining * oversample_factor)
        for cube_row in cube:
            if len(accepted) >= n:
                break
            theta = _unit_cube_to_theta(cube_row, bounds)
            # Each accepted theta gets its OWN scenario seed -- derive_seed(seed, blocker_side, i),
            # deterministic from (master seed, side, position in the batch) -- rather than every
            # theta in the batch sharing the literal `seed` value, which would otherwise perfectly
            # correlate their realization-noise streams (same percentile draw for e.g. front_gap_m's
            # perturbation on every theta's first episode) for no reason.
            theta_seed = derive_seed(seed, side_code, len(accepted) + len(rejected))
            try:
                cfg = cfg_cls(blocker_side=blocker_side, seed=theta_seed, mode="robust",
                               **extra_config_kwargs, **theta)
                validate(cfg, lane_num=_LANE_NUM, ego_lane=_EGO_LANE)
            except ScenarioFamilyConfigError as exc:
                rejected.append(RejectedSample(theta=theta, reason=str(exc)))
                continue
            accepted.append(cfg)
        round_seed += 1   # a fresh, still-deterministic stream for any refill round
    else:
        raise ScenarioFamilyConfigError(
            f"only {len(accepted)}/{n} samples validated after {max_rounds} rounds "
            f"({len(rejected)} rejected total) -- check the configured bounds/goal_distance_m "
            f"for {family} aren't mutually inconsistent (see the rejection reasons collected so far).")

    return accepted, rejected


def sample_cutin(
        n: int, blocker_side: BlockerSide, seed: int, *, method: str = "lhs",
        **common_config_kwargs,
) -> tuple[list[CutinConfig], list[RejectedSample]]:
    """n validated CutinConfig theta samples for one blocker_side, via a
    seeded space-filling design (Latin Hypercube by default, or Sobol) over
    THETA_BOUNDS_CUTIN. common_config_kwargs are passed straight to every
    CutinConfig (e.g. goal_distance_m=, max_episode_seconds=) -- only
    blocker_side/seed/mode are set by this function itself. Returns
    (accepted_configs, rejected_samples) -- see RejectedSample."""
    return _sample_family("cutin", n, blocker_side, seed, method=method,
                           extra_config_kwargs=common_config_kwargs)


def sample_sandwich(
        n: int, blocker_side: BlockerSide, seed: int, *, method: str = "lhs",
        **common_config_kwargs,
) -> tuple[list[SandwichConfig], list[RejectedSample]]:
    """Sandwich-family counterpart of sample_cutin -- same contract, over
    THETA_BOUNDS_SANDWICH."""
    return _sample_family("sandwich", n, blocker_side, seed, method=method,
                           extra_config_kwargs=common_config_kwargs)
