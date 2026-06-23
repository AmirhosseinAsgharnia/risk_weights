"""
Top-down visualisation for the parameterisable curved-road environment.

Coordinate system
-----------------
Frenet reference : left road boundary.
y_abs            : rightward distance from left boundary (m).
Lane k centre    : y_abs = (n_lanes - k - 0.5) * lane_width.

World XY conversion:  road_geom.to_world(s, y_abs)
  x = x_ref(s) + y_abs * sin(θ(s))
  y = y_ref(s) − y_abs * cos(θ(s))
where θ(s) = (A/ω)·(1 − cos(ω·s)) is the road heading.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v2 as imageio
from scipy.integrate import cumulative_trapezoid

from dynamics.env_dynamics import theta_road, lane_center_abs
from environment.scenarios import SandwichScenario
from config.config import L_veh, W_c

_VEH_COLORS  = ['red', 'darkorange', 'gold', 'limegreen', 'mediumpurple', 'cyan',
                 'hotpink', 'deepskyblue', 'white', 'lightgreen']
_WINDOW_S    = 80.0    # m shown along road
_WINDOW_LAT  = 4.0     # m padding outside road edges


# ── Road geometry ─────────────────────────────────────────────────────────────

class RoadGeometry:
    """Pre-computes the world-XY path of the left road boundary for fast lookup."""

    def __init__(self, kappa_A: float, kappa_omega: float, s_max: float = 400.0, ds: float = 0.5):
        self.kappa_A, self.kappa_omega = kappa_A, kappa_omega
        s       = np.arange(0.0, s_max + ds, ds)
        theta   = theta_road(s, kappa_A, kappa_omega)
        x_ref   = np.concatenate([[0.0], cumulative_trapezoid(np.cos(theta), s)])
        y_ref   = np.concatenate([[0.0], cumulative_trapezoid(np.sin(theta), s)])
        self._s     = s
        self._x_ref = x_ref
        self._y_ref = y_ref
        self._theta = theta

    def to_world(self, s, y_abs):
        """Frenet (s, y_abs) → world (x, y). Accepts scalars or arrays."""
        s     = np.asarray(s, float)
        y_abs = np.asarray(y_abs, float)
        x_ref = np.interp(s, self._s, self._x_ref)
        y_ref = np.interp(s, self._s, self._y_ref)
        theta = np.interp(s, self._s, self._theta)
        return x_ref + y_abs * np.sin(theta), y_ref - y_abs * np.cos(theta)

    def heading_at(self, s: float) -> float:
        return float(np.interp(s, self._s, self._theta))


# ── Frame drawing ─────────────────────────────────────────────────────────────

def _draw_frame(
    ego_state,        # [e_y, psi, e_psi, psi_dot, v_x, v_y]
    ego_lane: int,
    s_ego: float,
    vehicles: list,   # runtime vehicle dicts {s, y_abs, v_x, lane, valid}
    ego_fut: dict,    # {e_y, y_abs, e_psi, v_x, s}
    sur_fut: list,    # list of {s, y_abs} or None per vehicle
    risks: dict,      # {P_slip, P_roll, P_c2c, P_ld}
    road_geom: RoadGeometry,
    scenario: SandwichScenario,
    step_n: int,
    epoch_n: int = 0,
):
    fig, (ax_road, ax_risk) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={'width_ratios': [3, 1]},
    )

    n_lanes  = scenario.n_lanes
    lw       = scenario.lane_width

    # ── Ego world position ────────────────────────────────────────────────────
    e_y_ego = ego_state[0]
    psi_ego = ego_state[1]
    y_abs_ego = lane_center_abs(ego_lane, n_lanes, lw) + e_y_ego
    x_ego, y_ego = road_geom.to_world(s_ego, y_abs_ego)
    x_ego, y_ego = float(x_ego), float(y_ego)

    # ── Road surface ──────────────────────────────────────────────────────────
    s_lo  = max(s_ego - _WINDOW_S / 2, 0.0)
    s_hi  = s_ego + _WINDOW_S / 2
    s_road = np.linspace(s_lo, s_hi, 400)

    ax_road.set_facecolor('#555555')
    ax_road.set_xlabel('World x (m)')
    ax_road.set_ylabel('World y (m)')
    ax_road.set_aspect('equal')

    for k in range(n_lanes + 1):
        y_bound = k * lw        # left boundary at k=0, right at k=n_lanes
        xb, yb  = road_geom.to_world(s_road, np.full_like(s_road, y_bound))
        ls      = 'solid' if k in (0, n_lanes) else 'dashed'
        ax_road.plot(xb, yb, color='white', linewidth=1.2, linestyle=ls)

    # ── Predicted ego path ────────────────────────────────────────────────────
    x_pred, y_pred = road_geom.to_world(ego_fut['s'], ego_fut['y_abs'])
    ax_road.plot(x_pred, y_pred, 'b--', linewidth=1.2, zorder=3, label='ego pred.')

    # ── Predicted surrounding vehicle positions ───────────────────────────────
    for i, sf in enumerate(sur_fut):
        if sf is None:
            continue
        col = _VEH_COLORS[i % len(_VEH_COLORS)]
        xs, ys = road_geom.to_world(sf['s'], sf['y_abs'])
        ax_road.scatter(xs, ys, c=col, s=4, zorder=3, alpha=0.5)

    # ── Vehicle rectangles ────────────────────────────────────────────────────
    _draw_vehicle(ax_road, x_ego, y_ego, psi_ego, 'deepskyblue', 'ego')

    active_idx = 0
    for i, veh in enumerate(vehicles):
        if not veh['valid']:
            continue
        xv, yv = road_geom.to_world(veh['s'], veh['y_abs'])
        theta_v = road_geom.heading_at(veh['s'])
        col   = _VEH_COLORS[i % len(_VEH_COLORS)]
        _draw_vehicle(ax_road, float(xv), float(yv), theta_v, col,
                      f"veh-{i} L{veh['lane']}")
        active_idx += 1

    # ── View window ───────────────────────────────────────────────────────────
    theta_ego = road_geom.heading_at(s_ego)
    _set_view(ax_road, x_ego, y_ego, theta_ego,
              _WINDOW_S, scenario.road_width(), _WINDOW_LAT)

    ax_road.legend(loc='upper left', fontsize=7, framealpha=0.7)
    ax_road.set_title(
        f'Epoch {epoch_n}  |  Step {step_n}  |  '
        f'lane={ego_lane}  e_y={e_y_ego:+.2f}m  '
        f'ψ={np.degrees(psi_ego):.1f}°  v_x={ego_state[4]:.1f}m/s',
        fontsize=8,
    )

    # ── Risk bar chart ────────────────────────────────────────────────────────
    labels = ['P_slip', 'P_roll', 'P_c2c', 'P_ld']
    values = [risks['P_slip'], risks['P_roll'], risks['P_c2c'], risks['P_ld']]
    colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']

    bars = ax_risk.bar(labels, values, color=colors, edgecolor='black', width=0.6)
    ax_risk.set_ylim(0, 1)
    ax_risk.set_ylabel('Risk probability')
    ax_risk.set_title('Risk components', fontsize=9)
    ax_risk.tick_params(axis='x', labelsize=8)
    for bar, val in zip(bars, values):
        ax_risk.text(
            bar.get_x() + bar.get_width() / 2, val + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7,
        )

    fig.tight_layout()
    fig.canvas.draw()
    w, h  = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    frame = frame.reshape(h, w, 4)
    frame = np.roll(frame, -1, axis=2)
    plt.close(fig)
    return frame


def _draw_vehicle(ax, x, y, psi, color, label):
    rect = patches.Rectangle(
        (x - L_veh / 2, y - W_c / 2),
        L_veh, W_c,
        angle=np.degrees(psi),
        rotation_point='center',
        linewidth=1, edgecolor='black', facecolor=color, zorder=4, label=label,
    )
    ax.add_patch(rect)


def _set_view(ax, cx, cy, road_angle, window_s, road_width, pad):
    half_s   = window_s / 2
    half_lat = (road_width + 2 * pad) / 2
    cos_r, sin_r = np.cos(road_angle), np.sin(road_angle)
    corners = np.array([
        [ half_s,  half_lat], [-half_s,  half_lat],
        [ half_s, -half_lat], [-half_s, -half_lat],
    ]) @ np.array([[cos_r, sin_r], [-sin_r, cos_r]])
    ax.set_xlim(corners[:, 0].min() + cx, corners[:, 0].max() + cx)
    ax.set_ylim(corners[:, 1].min() + cy, corners[:, 1].max() + cy)


# ── Mode-2 frame: full road history view ─────────────────────────────────────

def _draw_frame_history(
    history: list,    # all history dicts accumulated so far (current step included)
    ego_state,
    ego_lane: int,
    s_ego: float,
    vehicles: list,
    ego_fut: dict,
    sur_fut: list,
    risks: dict,
    road_geom: RoadGeometry,
    scenario: SandwichScenario,
    step_n: int,
    epoch_n: int = 0,
):
    """Full-road history view: shows the entire road traveled + ego trail."""
    fig, (ax_road, ax_risk) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={'width_ratios': [3, 1]},
    )

    n_lanes = scenario.n_lanes
    lw      = scenario.lane_width

    s_start = history[0]['s_ego'] if history else s_ego
    s_lo    = max(s_start - 5.0, 0.0)
    s_hi    = s_ego + 20.0

    s_road  = np.linspace(s_lo, s_hi, max(400, int((s_hi - s_lo) * 5)))

    ax_road.set_facecolor('#555555')
    ax_road.set_xlabel('World x (m)')
    ax_road.set_ylabel('World y (m)')
    ax_road.set_aspect('equal')

    # ── Road boundaries ───────────────────────────────────────────────────────
    for k in range(n_lanes + 1):
        y_bound = k * lw
        xb, yb  = road_geom.to_world(s_road, np.full_like(s_road, y_bound))
        ls      = 'solid' if k in (0, n_lanes) else 'dashed'
        ax_road.plot(xb, yb, color='white', linewidth=1.2, linestyle=ls)

    # ── Historical ego trail ──────────────────────────────────────────────────
    if len(history) > 1:
        s_hist    = np.array([h['s_ego'] for h in history])
        ey_hist   = np.array([h['ego_state'][0] for h in history])
        lane_hist = [h['ego_lane'] for h in history]
        yabs_hist = np.array([
            lane_center_abs(lane_hist[i], n_lanes, lw) + ey_hist[i]
            for i in range(len(history))
        ])
        x_trail, y_trail = road_geom.to_world(s_hist, yabs_hist)
        ax_road.plot(x_trail, y_trail, color='cyan', linewidth=1.5,
                     alpha=0.7, zorder=2, label='ego history')

    # ── Predicted future path ─────────────────────────────────────────────────
    x_pred, y_pred = road_geom.to_world(ego_fut['s'], ego_fut['y_abs'])
    ax_road.plot(x_pred, y_pred, 'b--', linewidth=1.2, zorder=3, label='ego pred.')

    # ── Ego current position ──────────────────────────────────────────────────
    e_y_ego   = ego_state[0]
    psi_ego   = ego_state[1]
    y_abs_ego = lane_center_abs(ego_lane, n_lanes, lw) + e_y_ego
    x_ego, y_ego = road_geom.to_world(s_ego, y_abs_ego)
    x_ego, y_ego = float(x_ego), float(y_ego)
    _draw_vehicle(ax_road, x_ego, y_ego, psi_ego, 'deepskyblue', 'ego')

    # ── Surrounding vehicles ──────────────────────────────────────────────────
    for i, veh in enumerate(vehicles):
        if not veh['valid']:
            continue
        xv, yv  = road_geom.to_world(veh['s'], veh['y_abs'])
        theta_v = road_geom.heading_at(veh['s'])
        col     = _VEH_COLORS[i % len(_VEH_COLORS)]
        _draw_vehicle(ax_road, float(xv), float(yv), theta_v, col,
                      f"veh-{i} L{veh['lane']}")

    # ── Auto-fit axis to full road extent ─────────────────────────────────────
    all_y_bounds = [road_geom.to_world(s_road, np.full_like(s_road, k * lw))
                    for k in range(n_lanes + 1)]
    all_x = np.concatenate([b[0] for b in all_y_bounds])
    all_y = np.concatenate([b[1] for b in all_y_bounds])
    pad   = _WINDOW_LAT
    ax_road.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax_road.set_ylim(all_y.min() - pad, all_y.max() + pad)

    ax_road.legend(loc='upper left', fontsize=7, framealpha=0.7)
    ax_road.set_title(
        f'Epoch {epoch_n}  |  Step {step_n}  |  '
        f'lane={ego_lane}  e_y={e_y_ego:+.2f}m  '
        f'ψ={np.degrees(psi_ego):.1f}°  v_x={ego_state[4]:.1f}m/s  s={s_ego:.1f}m',
        fontsize=8,
    )

    # ── Risk bar chart ────────────────────────────────────────────────────────
    labels = ['P_slip', 'P_roll', 'P_c2c', 'P_ld']
    values = [risks['P_slip'], risks['P_roll'], risks['P_c2c'], risks['P_ld']]
    colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']

    bars = ax_risk.bar(labels, values, color=colors, edgecolor='black', width=0.6)
    ax_risk.set_ylim(0, 1)
    ax_risk.set_ylabel('Risk probability')
    ax_risk.set_title('Risk components', fontsize=9)
    ax_risk.tick_params(axis='x', labelsize=8)
    for bar, val in zip(bars, values):
        ax_risk.text(
            bar.get_x() + bar.get_width() / 2, val + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7,
        )

    fig.tight_layout()
    fig.canvas.draw()
    w, h  = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    frame = frame.reshape(h, w, 4)
    frame = np.roll(frame, -1, axis=2)
    plt.close(fig)
    return frame


# ── GIF export ────────────────────────────────────────────────────────────────

def make_gif(frames, path='env_demo.gif', fps=20):
    print(f"Saving {path} ({len(frames)} frames) …")
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"Saved: {path}")
