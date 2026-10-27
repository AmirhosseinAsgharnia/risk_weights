"""
Result interpreter: render feasible vs. non-feasible regions of theta-space
as 2D heatmaps, one per pair of theta fields, for one (scenario_family,
blocker_side)'s accumulated dataset.

Two panels per pair (or one, if --surrogate is omitted):

  Empirical panel (always shown, needs only scenarios.jsonl -- no fitted
  surrogate required): every sampled theta, projected onto this pair of
  dimensions, colored by its own empirical success_rate; a linear
  interpolation (scipy.interpolate.griddata) fills the area BETWEEN sampled
  points for a continuous-looking heatmap, but only inside the convex hull
  of the actual samples -- outside it (or wherever too few nearby points
  exist to interpolate) is left blank (shown as the axes' own gray
  background) rather than fabricating a value. This is a MARGINAL
  projection: a point's color reflects its true 8- (or 10-) dimensional
  theta, including whatever the OTHER dimensions happened to be for that
  sample -- it is not a controlled slice.

  Surrogate panel (only when --surrogate is given): p_hat from the fitted
  SurrogateArtifact, evaluated on a dense regular grid over this pair's
  bounds with every OTHER theta dimension held fixed at the dataset's own
  median value (see _median_theta) -- an actual controlled 2D slice through
  theta-space, and (unlike griddata) defined everywhere in-bounds, not just
  inside the convex hull of what's been sampled so far.

Usage:
    python -m learning.feasibility_interpret --scenario-family cutin --blocker-side 1 \\
        --out-dir artifacts/feasibility/heatmaps/cutin_side1

    python -m learning.feasibility_interpret --scenario-family cutin --blocker-side 1 \\
        --surrogate artifacts/feasibility/surrogates/cutin_side1.joblib \\
        --dims front_gap_m,cutter_relative_speed_mps,s_cutin_m \\
        --out-dir artifacts/feasibility/heatmaps/cutin_side1
"""

import argparse
import itertools
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # this tool only ever saves PNGs -- never assume a display is available
                         # (it typically runs directly against the dataset on the headless training
                         # machine, same reasoning as learning.animate_rollout's --save path).
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import QhullError

from learning.feasibility_dataset import ARTIFACTS_ROOT, scenarios_path, read_jsonl
from learning.feasibility_cutin import THETA_BOUNDS_CUTIN
from learning.feasibility_sandwich import THETA_BOUNDS_SANDWICH
from learning.feasibility_surrogate import SurrogateArtifact, normalize_theta

_THETA_BOUNDS = {"cutin": THETA_BOUNDS_CUTIN, "sandwich": THETA_BOUNDS_SANDWICH}
_CMAP = "RdYlGn"   # red = infeasible (success_rate/p_hat -> 0), green = feasible (-> 1)


def _median_theta(scenario_records: list[dict], bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    """The dataset's own median theta -- used as the "other dimensions held
    fixed here" reference point for the surrogate panel's 2D slice, since
    it's guaranteed to be a physically representative point actually near
    the explored region (unlike, say, each field's dataclass default,
    which may sit anywhere relative to what was actually sampled)."""
    return {name: float(np.median([r["theta"][name] for r in scenario_records])) for name in bounds}


def _empirical_grid(scenario_records: list[dict], dim_a: str, dim_b: str,
                     bounds: dict[str, tuple[float, float]], resolution: int):
    points = np.array([[r["theta"][dim_a], r["theta"][dim_b]] for r in scenario_records])
    values = np.array([r["success_rate"] for r in scenario_records])
    lo_a, hi_a = bounds[dim_a]
    lo_b, hi_b = bounds[dim_b]
    grid_a, grid_b = np.meshgrid(np.linspace(lo_a, hi_a, resolution), np.linspace(lo_b, hi_b, resolution))

    grid_p = None
    for method in ("linear", "nearest"):
        try:
            grid_p = griddata(points, values, (grid_a, grid_b), method=method)
            break
        except QhullError:
            continue   # too few / degenerate (e.g. collinear) points for this method -- try the next
    # grid_p stays None (an all-blank fill, scatter points still shown) if even "nearest" fails --
    # legitimate with a literal handful of scenarios, not an error worth crashing over.

    return grid_a, grid_b, grid_p, points, values


def _surrogate_grid(artifact: SurrogateArtifact, dim_a: str, dim_b: str,
                     bounds: dict[str, tuple[float, float]], reference_theta: dict[str, float],
                     resolution: int):
    lo_a, hi_a = bounds[dim_a]
    lo_b, hi_b = bounds[dim_b]
    grid_a, grid_b = np.meshgrid(np.linspace(lo_a, hi_a, resolution), np.linspace(lo_b, hi_b, resolution))

    # normalize_theta expects a full theta dict (every bounds field) -- start from the reference point
    # and override just the two varying dims per grid cell, matching artifact.theta_bounds' own field
    # order internally (see normalize_theta) so this lines up with how the model was actually trained.
    rows = []
    for a, b in zip(grid_a.ravel(), grid_b.ravel()):
        theta = dict(reference_theta)
        theta[dim_a] = float(a)
        theta[dim_b] = float(b)
        rows.append(normalize_theta(theta, artifact.theta_bounds))
    X = np.stack(rows)
    p = artifact.model.predict_proba(X)[:, 1]
    return grid_a, grid_b, p.reshape(grid_a.shape)


def plot_pair(
        family: str, blocker_side: int, dim_a: str, dim_b: str, scenario_records: list[dict],
        bounds: dict[str, tuple[float, float]], out_dir: Path, *,
        surrogate: SurrogateArtifact | None = None, resolution: int = 60,
) -> Path:
    n_panels = 2 if surrogate is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.5), squeeze=False)
    axes = axes[0]

    ax = axes[0]
    ga, gb, gp, points, values = _empirical_grid(scenario_records, dim_a, dim_b, bounds, resolution)
    ax.set_facecolor("#dddddd")   # visibly distinct from any real (red-to-green) value, incl. blank/NaN cells
    if gp is not None:
        im = ax.pcolormesh(ga, gb, gp, cmap=_CMAP, vmin=0, vmax=1, shading="auto")
        fig.colorbar(im, ax=ax, label="success rate")
    ax.scatter(points[:, 0], points[:, 1], c=values, cmap=_CMAP, vmin=0, vmax=1,
               s=45, edgecolors="black", linewidths=0.7, zorder=3)
    ax.set_xlabel(dim_a)
    ax.set_ylabel(dim_b)
    ax.set_xlim(bounds[dim_a])
    ax.set_ylim(bounds[dim_b])
    ax.set_title(f"Empirical (n={len(scenario_records)} scenarios, marginal projection)")

    if surrogate is not None:
        ax2 = axes[1]
        reference_theta = _median_theta(scenario_records, bounds)
        ga2, gb2, gp2 = _surrogate_grid(surrogate, dim_a, dim_b, bounds, reference_theta, resolution)
        im2 = ax2.pcolormesh(ga2, gb2, gp2, cmap=_CMAP, vmin=0, vmax=1, shading="auto")
        fig.colorbar(im2, ax=ax2, label="p_hat")
        ax2.scatter(points[:, 0], points[:, 1], c=values, cmap=_CMAP, vmin=0, vmax=1,
                    s=30, edgecolors="black", linewidths=0.5, alpha=0.55, zorder=3)
        ax2.set_xlabel(dim_a)
        ax2.set_ylabel(dim_b)
        ax2.set_xlim(bounds[dim_a])
        ax2.set_ylim(bounds[dim_b])
        other_dims = ", ".join(f"{k}={v:.3g}" for k, v in reference_theta.items() if k not in (dim_a, dim_b))
        preliminary_tag = " [PRELIMINARY]" if surrogate.preliminary else ""
        ax2.set_title(f"Surrogate p_hat{preliminary_tag} (other dims @ dataset median)\n{other_dims}",
                       fontsize=7.5)

    fig.suptitle(f"{family} / blocker_side={blocker_side}: {dim_a} vs {dim_b}")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"heatmap_{dim_a}_vs_{dim_b}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario-family", choices=["cutin", "sandwich"], required=True)
    parser.add_argument("--blocker-side", type=int, choices=[-1, 1], required=True)
    parser.add_argument("--root", type=str, default=str(ARTIFACTS_ROOT))
    parser.add_argument("--surrogate", type=str, default=None,
                         help="path to a fitted SurrogateArtifact .joblib (learning.feasibility_pipeline "
                              "--mode fit-only's --surrogate-out) -- adds a smooth, model-based panel "
                              "next to the empirical one. Omit to only plot raw scenario data (no fitted "
                              "surrogate needed at all).")
    parser.add_argument("--dims", type=str, default=None,
                         help="comma-separated theta field names to restrict which PAIRS get plotted "
                              "(every 2-combination of these). Default: every pair among ALL of this "
                              "family's theta fields.")
    parser.add_argument("--resolution", type=int, default=60,
                         help="grid resolution per axis for both the interpolated empirical fill and "
                              "the surrogate panel.")
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    bounds = _THETA_BOUNDS[args.scenario_family]
    root = Path(args.root)
    scenario_records = read_jsonl(scenarios_path(args.scenario_family, args.blocker_side, root))
    if not scenario_records:
        raise SystemExit(f"no scenarios found for family={args.scenario_family} "
                          f"blocker_side={args.blocker_side} under {root} -- run `--mode run` first.")

    dims = args.dims.split(",") if args.dims else list(bounds)
    for d in dims:
        if d not in bounds:
            raise SystemExit(f"unknown theta field {d!r} for family={args.scenario_family} -- "
                              f"valid fields: {list(bounds)}")
    if len(dims) < 2:
        raise SystemExit(f"need at least 2 theta fields to form a pair, got --dims={dims!r}")

    surrogate = None
    if args.surrogate:
        surrogate = SurrogateArtifact.load(args.surrogate)
        if surrogate.family != args.scenario_family or surrogate.blocker_side != args.blocker_side:
            raise SystemExit(f"{args.surrogate} was fit for family={surrogate.family} "
                              f"blocker_side={surrogate.blocker_side}, not "
                              f"{args.scenario_family}/{args.blocker_side} -- refusing to mix them.")
        if surrogate.preliminary:
            warnings.warn(f"{args.surrogate} is PRELIMINARY (fit on only {surrogate.n_scenarios} "
                           f"scenarios) -- its heatmap panel is a rough pipeline check, not a validated "
                           f"feasibility map.")

    pairs = list(itertools.combinations(dims, 2))
    out_dir = Path(args.out_dir)
    print(f"{len(scenario_records)} scenarios loaded -- plotting {len(pairs)} pairs to {out_dir}/")
    for dim_a, dim_b in pairs:
        path = plot_pair(args.scenario_family, args.blocker_side, dim_a, dim_b, scenario_records, bounds,
                          out_dir, surrogate=surrogate, resolution=args.resolution)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
