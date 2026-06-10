"""
Visualizes one preprocessed scenario from data/processd/train/.

Usage:
    python script/visualizer.py                     # first scenario found
    python script/visualizer.py path/to/scene.pt   # specific file
"""

import glob
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# RoadLine type constants (from map.proto)
# (color, is_solid, is_double, is_passing_double)
#   is_passing_double: one solid + one broken line (TYPE_PASSING_DOUBLE_YELLOW)
_LINE_PROPS = {
    0: ("white",   False, False, False),  # TYPE_UNKNOWN
    1: ("white",   False, False, False),  # TYPE_BROKEN_SINGLE_WHITE
    2: ("white",   True,  False, False),  # TYPE_SOLID_SINGLE_WHITE
    3: ("white",   True,  True,  False),  # TYPE_SOLID_DOUBLE_WHITE
    4: ("#dddd00", False, False, False),  # TYPE_BROKEN_SINGLE_YELLOW
    5: ("#dddd00", False, True,  False),  # TYPE_BROKEN_DOUBLE_YELLOW
    6: ("#dddd00", True,  False, False),  # TYPE_SOLID_SINGLE_YELLOW
    7: ("#dddd00", True,  True,  False),  # TYPE_SOLID_DOUBLE_YELLOW
    8: ("#dddd00", True,  True,  True),   # TYPE_PASSING_DOUBLE_YELLOW
}
_DOUBLE_OFFSET = 0.35  # metres between the two parallel lines

# Agent type constants (from scenario.proto)
TYPE_VEHICLE    = 1
TYPE_PEDESTRIAN = 2
TYPE_CYCLIST    = 3

TRUCK_LENGTH_M = 6.0  # vehicles longer than this are treated as trucks

# Feature column indices (must match FEATURE_NAMES in dataloader.py)
X, Y, LENGTH, VALID = 0, 1, 6, 9


def _offset_polyline(pts: np.ndarray, d: float) -> np.ndarray:
    """Shift a polyline by d metres in the left-normal direction (xy only)."""
    if len(pts) < 2:
        return pts.copy()
    xy = pts[:, :2]
    tangents = np.diff(xy, axis=0)                       # (N-1, 2)
    t = np.vstack([tangents, tangents[-1:]])             # (N, 2)
    t[1:-1] = (tangents[:-1] + tangents[1:]) / 2        # smooth interior points
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    t /= norms
    normals = np.stack([-t[:, 1], t[:, 0]], axis=1)     # rotate 90° CCW
    result = pts.copy()
    result[:, :2] = xy + normals * d
    return result


def _draw_line(ax: plt.Axes, pts: np.ndarray, color: str, is_solid: bool,
               lw: float = 0.8, zorder: int = 2) -> None:
    kw: dict = dict(color=color, linewidth=lw, zorder=zorder)
    if is_solid:
        kw["linestyle"] = "-"
    else:
        kw["linestyle"] = "--"
        kw["dashes"] = (5, 4)
    ax.plot(pts[:, 0], pts[:, 1], **kw)


def _in_intersection(data: dict, t: int) -> bool:
    """Return True if ego at timestep t is within 2.5 m of any interpolating (intersection) lane."""
    agents  = data["agents"]
    sdc_idx = int(data["sdc_track_index"])

    ego = np.array([float(agents[t, X, sdc_idx]),
                    float(agents[t, Y, sdc_idx])])
    THRESHOLD = 2.5

    for lane in data["lanes"]:
        if not lane["interpolating"]:
            continue
        pts = lane["polyline"]
        if hasattr(pts, "numpy"):
            pts = pts.numpy()
        pts_xy = pts[:, :2]
        for j in range(len(pts_xy) - 1):
            a, b = pts_xy[j], pts_xy[j + 1]
            ab = b - a
            ab_sq = float(np.dot(ab, ab))
            if ab_sq < 1e-8:
                d = float(np.linalg.norm(ego - a))
            else:
                s = float(np.clip(np.dot(ego - a, ab) / ab_sq, 0.0, 1.0))
                d = float(np.linalg.norm(ego - (a + s * ab)))
            if d < THRESHOLD:
                return True
    return False


def _plot(data: dict, ax: plt.Axes) -> None:
    ax.set_facecolor("#868686")

    # Road edges
    for edge in data["road_edges"]:
        p = edge["polyline"]
        ax.plot(p[:, 0], p[:, 1], color="#000000", linewidth=1.5, zorder=1)

    # Road lines — type-aware rendering (single/double, solid/broken, white/yellow)
    for rl in data["road_lines"]:
        pts = rl["polyline"].numpy()
        t   = int(rl["type"])
        color, is_solid, is_double, is_passing = _LINE_PROPS.get(t, _LINE_PROPS[0])
        if is_double:
            p1 = _offset_polyline(pts, +_DOUBLE_OFFSET)
            p2 = _offset_polyline(pts, -_DOUBLE_OFFSET)
            if is_passing:
                # TYPE_PASSING_DOUBLE_YELLOW: solid centre-line + broken passing-side line
                _draw_line(ax, p1, color, is_solid=True)
                _draw_line(ax, p2, color, is_solid=False)
            else:
                _draw_line(ax, p1, color, is_solid)
                _draw_line(ax, p2, color, is_solid)
        else:
            _draw_line(ax, pts, color, is_solid)

    # Crosswalks
    for cw in data["crosswalks"]:
        p = cw["polygon"]
        closed = torch.cat([p, p[:1]], dim=0)
        ax.fill(closed[:, 0], closed[:, 1], color="#88bbdd", alpha=0.3, zorder=1)
        ax.plot(closed[:, 0], closed[:, 1], color="#88bbdd", linewidth=0.8, zorder=2)

    # Agents
    agents      = data["agents"]           # (T, F, N)
    agent_types = data["agent_types"]      # (N,)
    sdc_idx     = int(data["sdc_track_index"])
    t_now       = int(data["current_time_index"])
    N           = agents.shape[2]

    for i in range(N):
        valid = agents[:, VALID, i].bool()
        if not valid.any():
            continue

        xs = agents[:, X, i]
        ys = agents[:, Y, i]
        atype  = int(agent_types[i])
        length = float(agents[t_now, LENGTH, i])

        if i == sdc_idx:
            color, zorder, ms = "red", 6, 9
        elif atype == TYPE_PEDESTRIAN:
            color, zorder, ms = "limegreen", 4, 6
        elif atype == TYPE_CYCLIST:
            color, zorder, ms = "yellow", 4, 6
        elif atype == TYPE_VEHICLE and length > TRUCK_LENGTH_M:
            color, zorder, ms = "white", 3, 6
        else:
            color, zorder, ms = "#5599ff", 3, 6

        # History — solid
        hx = xs[:t_now + 1][valid[:t_now + 1]]
        hy = ys[:t_now + 1][valid[:t_now + 1]]
        if len(hx) > 1:
            ax.plot(hx, hy, color=color, linewidth=1.2, alpha=0.9, zorder=zorder)

        # Future — dashed, faded
        fx = xs[t_now:][valid[t_now:]]
        fy = ys[t_now:][valid[t_now:]]
        if len(fx) > 1:
            ax.plot(fx, fy, color=color, linewidth=0.8, linestyle="--",
                    alpha=0.45, zorder=zorder)

        # Current position marker
        if valid[t_now]:
            ax.plot(float(xs[t_now]), float(ys[t_now]), "o",
                    color=color, markersize=ms, zorder=zorder + 1,
                    markeredgecolor="black", markeredgewidth=0.5)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        files = sorted(glob.glob("data/processd/train/*.pt"))
        if not files:
            raise FileNotFoundError("No .pt files in data/processd/train/ — run preprocess.py first")
        path = files[0]

    print(f"Visualizing: {path}")
    data = torch.load(path, weights_only=False)

    agents  = data["agents"]
    sdc_idx = int(data["sdc_track_index"])
    T       = agents.shape[0]
    for t in range(T):
        ego_x  = float(agents[t, X, sdc_idx])
        ego_y  = float(agents[t, Y, sdc_idx])
        in_int = _in_intersection(data, t)
        print(f"t={t:02d}  pos=({ego_x:.2f}, {ego_y:.2f})  in_intersection={in_int}")

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.set_facecolor("#363636")
    _plot(data, ax)

    ax.set_aspect("equal")
    ax.set_title(f"Scenario  {data['scenario_id']}", fontsize=10)
    ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
