#!/usr/bin/env python3
"""
Departure Node — predictive lane departure probability along the EKF-predicted
ego trajectory.

Subscriptions:
  /ego/predicted_trajectory    (perception_interfaces/EgoPredictedTrajectory)
  /localization/state          (perception_interfaces/LocalizationState)
  /lane/model                  (perception_interfaces/LaneCoeffs)
  /config/camera/front         (config_interfaces/CameraConfig) — only while either
  sensor/camera/front/image_raw (sensor_msgs/Image)                gated image output
                                (/risk/departure/image/bev or /risk/departure/image/overlay)
                                has subscribers. Both subscribed/unsubscribed together,
                                dynamically — see _update_camera_subscription.

Publications:
  /risk/departure/max_probability  (std_msgs/Float64) — max_t P_LD(t) over the horizon
  /risk/departure/score            (std_msgs/Float64) — cumulative lane-departure risk R,
                                    PARALO §5.2.5 "Cumulative Risk Calculation" eq. (5.51),
                                    no per-channel VF (kinetic-energy severity only — see below):
                                      S(t)  = 0.5·m·v(t)²
                                      S_max = max_t S(t)
                                      R     = S_max · (1 − Π_t (1 − (S(t)/S_max)·P_LD(t)))
  /risk/departure/image/bev        (sensor_msgs/Image) — gated on subscriber count
  /risk/departure/image/overlay    (sensor_msgs/Image) — gated on subscriber count

Coordinate frame: both /ego/predicted_trajectory (forward_m, lateral_m) and
/lane/model (left/right/centre_coeffs) are ego-frame, SAE convention
(+forward, +right) — confirmed from each message's own field comments.
lateral_m and the lane polynomials are combined directly, unnegated.
(loc.e_y, +right per LocalizationState, is separately negated only to seed
ego_lateral_left below — a rendering-only quantity, not used in the
departure probability/risk math.)

Lane-departure probability P_LD(t) — Gaussian clearance model against the
centerline-derived lane boundaries (perception_interfaces/LaneCoeffs
centre_coeffs/centre_cov/centre_valid; NOT the independent left/right
lines — see compute_departure_risk). Lane width and vehicle width come from
this node's own parameters (lane_width_m, vehicle_width_m — LaneCoeffs
carries no width field). D(t) = y(t) − y_C(x(t)) is the ego center's
lateral clearance from the centerline; its variance is first-order
propagated from the centerline coefficient covariance (Sigma_theta) and the
ego trajectory's own per-step position covariance (cov_xy), assumed
independent of each other (no lane/ego-prediction cross-covariance is
published). P_LD(t) = 1 − [Φ((h−mu_D)/sigma_D) − Φ((−h−mu_D)/sigma_D)],
h = (lane_width_m − vehicle_width_m)/2. An invalid/unavailable centerline
forces P_LD(t)=0 and R=0 for that callback (never reused stale).

Per-step lane-departure probability and severity are computed at every
predicted step; neither is published per-point — only their reduction to
max_probability and the cumulative R go out, plus whatever the gated
overlays render.
"""

import math
from typing import NamedTuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy,
)

from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge

from perception_interfaces.msg import EgoPredictedTrajectory, LocalizationState, LaneCoeffs
from config_interfaces.msg import CarConstants, CameraConfig, RiskWeights

_EPS = 1e-9


class _RenderPoint(NamedTuple):
    """Minimal per-step data needed to draw the BEV/camera overlays.

    Nothing here is published (see module docstring) — it's discarded once
    the overlay images (if anyone is subscribed) are drawn.
    """
    x_forward:        float   # ego-frame forward position [m]
    ego_lateral_left: float   # predicted ego lateral position, road-frame, +left [m]
    y_left_boundary:  float   # left boundary lateral position, ego-frame +left [m]
    y_right_boundary: float   # right boundary lateral position, ego-frame +left [m]
    p_lane_departure: float   # OR-fused departure probability [0,1]

# Color map: green(low) → yellow → orange → red(high), BGR
_COLORMAP_BGR: list[tuple[int, int, int]] = [
    (0,   200,   0),   # green    p=0.00
    (0,   220, 220),   # yellow   p=0.33
    (0,   140, 255),   # orange   p=0.66
    (0,     0, 220),   # red      p=1.00
]


# ──────────────────────────────────────────────────────────────────────────────
# Numerical helpers
# ──────────────────────────────────────────────────────────────────────────────

def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def safe_float(x, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def normal_cdf(z: float) -> float:
    """Φ(z) via erfc for numerical stability at large |z|."""
    z = safe_float(z, 0.0)
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def eval_quad(coeffs, x: float) -> float:
    """Evaluate parabolic lane boundary y = a*x^2 + b*x + c."""
    a, b, c = coeffs
    return float(a) * x * x + float(b) * x + float(c)


def eval_slope(coeffs, x: float) -> float:
    """Centerline slope dy/dx = 2*a*x + b for y = a*x^2 + b*x + c."""
    a, b, _ = coeffs
    return 2.0 * float(a) * x + float(b)


def propagate_poly_variance(cov9, x: float, fallback_var: float) -> tuple[float, bool]:
    """
    Propagate KF posterior covariance to boundary lateral variance at distance x.
    sigma^2 = phi @ P @ phi,  phi = [x^2, x, 1].
    Returns (variance, valid_flag).
    """
    try:
        P   = np.array(cov9, dtype=float).reshape(3, 3)
        phi = np.array([x * x, x, 1.0], dtype=float)
        var = float(phi @ P @ phi)
        if not math.isfinite(var) or var < -1e-8:
            return fallback_var, False
        return max(var, 0.0), True
    except Exception:
        return fallback_var, False


def sanitize_cov3(cov9) -> tuple[np.ndarray, bool]:
    """
    Validate/repair the 3x3 centerline coefficient covariance Sigma_theta.

    Applies only minimal numerical repair — symmetrization and clipping of
    tiny negative eigenvalues caused by floating-point error. A non-finite
    or materially non-PSD matrix (a negative eigenvalue beyond that
    tolerance) is reported invalid rather than silently masked.
    Returns (repaired_3x3_matrix, valid_flag).
    """
    try:
        P = np.array(cov9, dtype=float).reshape(3, 3)
    except Exception:
        return np.zeros((3, 3)), False
    if not np.all(np.isfinite(P)):
        return np.zeros((3, 3)), False

    P_sym = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P_sym)
    tol = max(1e-9, 1e-6 * float(np.max(np.abs(eigvals))))
    if float(np.min(eigvals)) < -tol:
        return P_sym, False

    eigvals_clipped = np.clip(eigvals, 0.0, None)
    P_repaired = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return 0.5 * (P_repaired + P_repaired.T), True


def get_cov_xy_step(cov_xy, k: int, fallback_var: float) -> tuple[float, float, float, bool]:
    """
    Extract (sigma2_x, sigma2_y, sigma_xy) for step k from EgoPredictedTrajectory's
    flattened per-step 2x2 [P_ff, P_fl, P_lf, P_ll] covariance array.
    Falls back to an isotropic variance with zero cross-term if the array is
    missing, too short, or has a negative reported variance.
    Returns (sigma2_x, sigma2_y, sigma_xy, valid_flag).
    """
    i = 4 * k
    try:
        if len(cov_xy) < i + 4:
            return fallback_var, fallback_var, 0.0, False
        p_ff = safe_float(cov_xy[i],     fallback_var)
        p_fl = safe_float(cov_xy[i + 1], 0.0)
        p_ll = safe_float(cov_xy[i + 3], fallback_var)
        if p_ff < 0.0 or p_ll < 0.0:
            return fallback_var, fallback_var, 0.0, False
        return p_ff, p_ll, p_fl, True
    except Exception:
        return fallback_var, fallback_var, 0.0, False


def paralo_cumulative_risk(s_list: list[float], p_list: list[float]) -> float:
    """
    PARALO §5.2.5 "Cumulative Risk Calculation" eq. (5.51), discrete form:

        q_k   = clamp((S_k / S_max) * P_k, 0, 1)
        R     = S_max * (1 - exp(sum_k log1p(-q_k)))

    equivalent to S_max * (1 - prod_k (1 - q_k)) but computed in log-space
    via log1p for numerical stability. q_k == 1 is handled explicitly so the
    result becomes S_max without an invalid log(0).

    NOTE: this value depends on the number and spacing of the prediction
    samples in s_list/p_list — no independent dt correction is applied here.
    """
    if not s_list:
        return 0.0
    S_max = max(s_list)
    if S_max <= _EPS:
        return 0.0

    log_survival = 0.0
    for S_k, P_k in zip(s_list, p_list):
        q_k = clamp((S_k / S_max) * P_k, 0.0, 1.0)
        if q_k >= 1.0:
            return S_max
        log_survival += math.log1p(-q_k)
    return S_max * (1.0 - math.exp(log_survival))


# ──────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ──────────────────────────────────────────────────────────────────────────────

def _p_color(p: float) -> tuple[int, int, int]:
    """Interpolate BGR colour in _COLORMAP_BGR for probability p ∈ [0,1]."""
    p = clamp(p, 0.0, 1.0)
    n = len(_COLORMAP_BGR) - 1
    idx_f = p * n
    lo = int(idx_f)
    hi = min(lo + 1, n)
    t = idx_f - lo
    c0, c1 = _COLORMAP_BGR[lo], _COLORMAP_BGR[hi]
    return tuple(int(c0[i] + t * (c1[i] - c0[i])) for i in range(3))


# ──────────────────────────────────────────────────────────────────────────────
# BEV overlay
# ──────────────────────────────────────────────────────────────────────────────

def create_bev_overlay(
        points:       list[_RenderPoint],
        bev_w_m:      float = 20.0,
        bev_fwd_m:    float = 50.0,
        res:          float = 0.1,
        current_speed: float = 0.0,
) -> np.ndarray:
    """
    Render a top-down BEV lane-departure probability map.

    Canvas:
      Origin at bottom-centre (ego position), +forward → up, +left → left.

    Draws:
      - Left and right lane boundary curves
      - Ego trajectory points coloured by p_lane_departure
      - Path segments between consecutive points
      - Text overlay (speed, max probability)
    """
    bev_h_px = int(round(bev_fwd_m / res))
    bev_w_px = int(round(bev_w_m   / res))
    canvas = np.zeros((bev_h_px, bev_w_px, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)

    def metric_to_px(fwd: float, lat_left: float) -> tuple[int, int]:
        u = int(round(bev_w_px / 2.0 - lat_left / res))
        v = int(round(bev_h_px - 1   - fwd       / res))
        return u, v

    # Ego marker at origin
    ego_u, ego_v = bev_w_px // 2, bev_h_px - 1
    cv2.circle(canvas, (ego_u, ego_v), 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(canvas, "EGO", (ego_u - 15, ego_v - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    n = len(points)
    if n == 0:
        return canvas

    # Pre-compute ego pixel positions
    ego_px = [metric_to_px(pt.x_forward, pt.ego_lateral_left) for pt in points]

    # ── Lane boundary curves ───────────────────────────────────────────────────
    for attr, color in [("y_left_boundary", (0, 180, 0)), ("y_right_boundary", (180, 0, 0))]:
        bnd_px = [metric_to_px(pt.x_forward, getattr(pt, attr)) for pt in points]
        for k in range(1, n):
            p1, p2 = bnd_px[k - 1], bnd_px[k]
            if (0 <= p1[0] < bev_w_px and 0 <= p1[1] < bev_h_px and
                    0 <= p2[0] < bev_w_px and 0 <= p2[1] < bev_h_px):
                cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

    # ── Coloured path segments ────────────────────────────────────────────────
    for k in range(1, n):
        p1, p2 = ego_px[k - 1], ego_px[k]
        in_bounds = (0 <= p1[0] < bev_w_px and 0 <= p1[1] < bev_h_px and
                     0 <= p2[0] < bev_w_px and 0 <= p2[1] < bev_h_px)
        if not in_bounds:
            continue
        cv2.line(canvas, p1, p2, _p_color(points[k].p_lane_departure), 3, cv2.LINE_AA)

    # ── Waypoint circles ───────────────────────────────────────────────────────
    for k, (pt, (u, v)) in enumerate(zip(points, ego_px)):
        if not (0 <= u < bev_w_px and 0 <= v < bev_h_px):
            continue
        radius = 4 + int(3 * k / max(n - 1, 1))
        cv2.circle(canvas, (u, v), radius, _p_color(pt.p_lane_departure), -1, cv2.LINE_AA)

    # ── Text overlay ───────────────────────────────────────────────────────────
    max_p  = max(pt.p_lane_departure for pt in points)
    mean_p = sum(pt.p_lane_departure for pt in points) / n

    def _put(text: str, row: int, color=(220, 220, 220)):
        cv2.putText(canvas, text, (6, row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    _put(f"Speed:   {current_speed:5.1f} m/s", 20)
    _put(f"Max  P:  {max_p:.3f}",  42, _p_color(max_p))
    _put(f"Mean P:  {mean_p:.3f}", 62, _p_color(mean_p))

    # Colour-scale legend
    bar_x  = bev_w_px - 14
    bar_h  = bev_h_px // 2
    bar_y0 = bev_h_px // 4
    for row in range(bar_h):
        p_row = 1.0 - row / max(bar_h - 1, 1)
        cv2.line(canvas, (bar_x, bar_y0 + row), (bar_x + 10, bar_y0 + row), _p_color(p_row), 1)
    _put("1.0", bar_y0 - 4,         (180, 180, 180))
    _put("0.0", bar_y0 + bar_h + 2, (180, 180, 180))

    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Camera overlay
# ──────────────────────────────────────────────────────────────────────────────

def _ground_to_image(
        fwd_m:        float,
        lat_left_m:   float,
        cam_K:        np.ndarray,
        cam_Rcw:      np.ndarray,
        cam_Cw:       np.ndarray,
        img_w:        int,
        img_h:        int,
        hood_rows:    int = 0,
) -> tuple[int, int] | None:
    """
    Project an ego-frame ground point to camera image pixels.
    World frame: Xw=forward, Yw=right → lat_left_m maps to -Yw.
    Returns (u, v) or None if behind/outside camera.
    hood_rows: number of pixel rows at the bottom to exclude (car hood).
    """
    Pw = np.array([fwd_m, -lat_left_m, 0.0], dtype=np.float64)
    Pc = cam_Rcw @ (Pw - cam_Cw)
    if Pc[2] < 0.1:
        return None
    uv = cam_K @ Pc
    u  = uv[0] / uv[2]
    v  = uv[1] / uv[2]
    if not (0 <= u < img_w and 0 <= v < img_h - hood_rows):
        return None
    return int(round(u)), int(round(v))


def create_camera_overlay(
        points:        list[_RenderPoint],
        bgr:           np.ndarray,
        cam_K:         np.ndarray,
        cam_Rcw:       np.ndarray,
        cam_Cw:        np.ndarray,
        current_speed: float = 0.0,
        ego_lat_left_0: float = 0.0,
        hood_rows:     int = 0,
) -> np.ndarray:
    """
    Draw predicted lane departure probability onto the camera image.

    Projects ego trajectory points and lane boundaries from the ground plane
    (z=0, ego frame) into pixel coordinates.  Path segments are coloured by
    p_lane_departure (green→yellow→orange→red).  Lane boundaries are drawn
    as solid green (left) and red (right) lines.

    ego_lat_left_0: current lateral offset from lane center (positive-left).
      pt.ego_lateral_left is in the lane frame; subtract this to get ego frame
      before projecting into the camera.
    hood_rows: pixel rows at the bottom to exclude (car hood region).
    """
    overlay = bgr.copy()
    img_h, img_w = overlay.shape[:2]
    n = len(points)
    if n == 0:
        return overlay

    def proj(fwd, lat_ego):
        return _ground_to_image(fwd, lat_ego, cam_K, cam_Rcw, cam_Cw, img_w, img_h,
                                hood_rows=hood_rows)

    # ego_lateral_left is road-frame (offset from lane center); subtract ego_lat_left_0 to
    # recover ego-frame lateral (= lat_k, displacement from current vehicle position).
    # y_left/right_boundary are already ego-frame (lane_node fits from ego-centric BEV).
    ego_px  = [proj(pt.x_forward, pt.ego_lateral_left - ego_lat_left_0) for pt in points]
    left_px = [proj(pt.x_forward, pt.y_left_boundary)                   for pt in points]
    rgt_px  = [proj(pt.x_forward, pt.y_right_boundary)                  for pt in points]

    # ── Lane boundary lines ────────────────────────────────────────────────────
    for px_list, color in [(left_px, (0, 200, 0)), (rgt_px, (0, 0, 200))]:
        for k in range(1, n):
            p1, p2 = px_list[k - 1], px_list[k]
            if p1 is not None and p2 is not None:
                cv2.line(overlay, p1, p2, color, 2, cv2.LINE_AA)

    # ── Ego path segments coloured by probability ──────────────────────────────
    for k in range(1, n):
        p1, p2 = ego_px[k - 1], ego_px[k]
        if p1 is not None and p2 is not None:
            cv2.line(overlay, p1, p2, _p_color(points[k].p_lane_departure), 4, cv2.LINE_AA)

    # ── Waypoint circles ───────────────────────────────────────────────────────
    for k, (pt, pix) in enumerate(zip(points, ego_px)):
        if pix is None:
            continue
        color  = _p_color(pt.p_lane_departure)
        radius = max(4, 8 - k // 5)
        cv2.circle(overlay, pix, radius, color,         -1, cv2.LINE_AA)
        cv2.circle(overlay, pix, radius, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Text overlay ───────────────────────────────────────────────────────────
    max_p = max((pt.p_lane_departure for pt in points), default=0.0)

    def _put(text: str, row: int, color=(220, 220, 220)):
        cv2.putText(overlay, text, (8, row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, (8, row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color,     1, cv2.LINE_AA)

    _put(f"Speed:  {current_speed:5.1f} m/s", 28)
    _put(f"Max P:  {max_p:.3f}", 54, _p_color(max_p))

    return overlay


# ──────────────────────────────────────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────────────────────────────────────

class DepartureRiskNode(Node):

    def __init__(self):
        super().__init__("departure_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("lane_width_m",                  3.8)
        self.declare_parameter("vehicle_width_m",               1.85)
        self.declare_parameter("default_sigma_lateral_m",       0.30)
        self.declare_parameter("min_sigma_y_m",                 0.05)
        self.declare_parameter("max_sigma_y_m",                 3.0)
        self.declare_parameter("ego_prediction_sigma_base_m",   0.05)
        self.declare_parameter("ego_prediction_sigma_rate_mps", 0.10)
        self.declare_parameter("use_heading_uncertainty",       True)
        self.declare_parameter("use_lane_model",                True)
        self.declare_parameter("allow_lane_width_fallback",     True)
        self.declare_parameter("stale_lane_model_timeout_s",    0.50)
        self.declare_parameter("stale_localization_timeout_s",  0.50)
        self.declare_parameter("bev_forward_range_m",           50.0)
        self.declare_parameter("bev_width_m",                   20.0)
        self.declare_parameter("bev_resolution_m_per_pixel",    0.10)
        self.declare_parameter("camera_hood_rows",              140)

        self._load_params()

        # ── QoS profiles ──────────────────────────────────────────────────────
        stream_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        lane_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        config_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )

        # ── State ─────────────────────────────────────────────────────────────
        self._w_departure: float  = 1.0
        self._mass:        float  = 1575.0
        self._latest_loc:       LocalizationState | None = None
        self._latest_loc_time                            = None
        self._latest_lane:      LaneCoeffs | None        = None
        self._latest_lane_time                           = None

        # Camera projection matrices (filled from /config/camera/front)
        # World frame: Xw=forward, Yw=right, Zw=up
        self._cam_K:   np.ndarray | None = None
        self._cam_Rcw: np.ndarray | None = None
        self._cam_Cw:  np.ndarray | None = None
        self._latest_bgr: np.ndarray | None = None

        self._bridge = CvBridge()

        # QoS kept for lazily (de)creating the camera subscriptions — see
        # _update_camera_subscription.
        self._stream_qos = stream_qos
        self._config_qos = config_qos

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(
            EgoPredictedTrajectory, "/ego/predicted_trajectory",
            self.trajectory_callback, stream_qos,
        )
        self.create_subscription(
            LocalizationState, "/localization/state",
            self.localization_callback, stream_qos,
        )
        self.create_subscription(
            LaneCoeffs, "/lane/model",
            self.lane_model_callback, lane_qos,
        )
        self.create_subscription(
            CarConstants, "/config/car",
            self._car_config_callback, config_qos,
        )
        self.create_subscription(
            RiskWeights, "/config/risk_weights",
            self._risk_weights_callback, config_qos,
        )

        # Camera calibration (/config/camera/front) and the raw frame — only
        # subscribed while either gated image output (/risk/departure/image/bev
        # or /risk/departure/image/overlay) has subscribers, to avoid the
        # per-frame cv_bridge decode/copy cost when nobody wants any image.
        # See _update_camera_subscription.
        self._sub_cam_cfg = None
        self._sub_cam_img = None

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_max_p        = self.create_publisher(Float64, "/risk/departure/max_probability", pub_qos)
        self._pub_weighted_risk = self.create_publisher(Float64, "/risk/departure/score",           pub_qos)
        self._pub_bev          = self.create_publisher(Image,   "/risk/departure/image/bev",        stream_qos)
        self._pub_overlay      = self.create_publisher(Image,   "/risk/departure/image/overlay",    stream_qos)

        # Periodic check (subscriber counts aren't event-driven in rclpy) —
        # see _update_camera_subscription.
        self._cam_sub_timer = self.create_timer(1.0, self._update_camera_subscription)

        self.get_logger().info("Departure risk node ready.")

    # ── Dynamic camera-image subscription ──────────────────────────────────────

    def _update_camera_subscription(self) -> None:
        """Subscribe to /config/camera/front and the raw camera frame only
        while /risk/departure/image/bev or /risk/departure/image/overlay has
        subscribers; unsubscribe from both — and drop the cached
        calibration/frame — the moment neither does.
        """
        wants_image = (
            self._pub_bev.get_subscription_count() > 0
            or self._pub_overlay.get_subscription_count() > 0
        )

        if wants_image and self._sub_cam_img is None:
            self._sub_cam_cfg = self.create_subscription(
                CameraConfig,
                "/config/camera/front",
                self._cam_config_callback,
                self._config_qos,
            )
            self._sub_cam_img = self.create_subscription(
                Image,
                "sensor/camera/front/image_raw",
                self._cam_image_callback,
                self._stream_qos,
            )
            self.get_logger().info(
                "Image subscriber detected — subscribing to camera config + raw image."
            )
        elif not wants_image and self._sub_cam_img is not None and self._sub_cam_cfg is not None:
            self.destroy_subscription(self._sub_cam_img)
            self.destroy_subscription(self._sub_cam_cfg)
            self._sub_cam_img = None
            self._sub_cam_cfg = None
            self._latest_bgr  = None
            self._cam_K       = None
            self._cam_Rcw     = None
            self._cam_Cw      = None
            self.get_logger().info(
                "No image subscribers — unsubscribing from camera config + raw image."
            )

    # ── Parameter loading ──────────────────────────────────────────────────────

    def _load_params(self) -> None:
        gp = lambda n: self.get_parameter(n).value
        self._lane_width_m    = float(gp("lane_width_m"))
        self._veh_width_m     = float(gp("vehicle_width_m"))
        self._def_sigma_lat   = float(gp("default_sigma_lateral_m"))
        self._min_sigma_y     = float(gp("min_sigma_y_m"))
        self._max_sigma_y     = float(gp("max_sigma_y_m"))
        self._sigma_base      = float(gp("ego_prediction_sigma_base_m"))
        self._sigma_rate      = float(gp("ego_prediction_sigma_rate_mps"))
        self._use_heading_unc = bool(gp("use_heading_uncertainty"))
        self._use_lane_model  = bool(gp("use_lane_model"))
        self._allow_fallback  = bool(gp("allow_lane_width_fallback"))
        self._stale_lane_s    = float(gp("stale_lane_model_timeout_s"))
        self._stale_loc_s     = float(gp("stale_localization_timeout_s"))
        self._bev_fwd         = float(gp("bev_forward_range_m"))
        self._bev_wid         = float(gp("bev_width_m"))
        self._bev_res         = float(gp("bev_resolution_m_per_pixel"))
        self._hood_rows       = int(gp("camera_hood_rows"))

    # ── Config callbacks ───────────────────────────────────────────────────────

    def _car_config_callback(self, msg: CarConstants) -> None:
        self._mass = float(msg.m)

    def _risk_weights_callback(self, msg: RiskWeights) -> None:
        self._w_departure = float(msg.w_departure)
        self.get_logger().info(f"Risk weights: w_departure={self._w_departure:.3f}")

    def _cam_config_callback(self, msg: CameraConfig) -> None:
        """Build camera projection matrices from /config/camera/front."""
        intr = msg.intrinsic
        self._cam_K = np.array([
            [intr.fx, 0.0,     intr.cx],
            [0.0,     intr.fy, intr.cy],
            [0.0,     0.0,     1.0    ],
        ], dtype=np.float64)

        ext   = msg.extrinsic
        roll  = math.radians(float(ext.roll_deg))
        pitch = math.radians(float(ext.pitch_deg))
        yaw   = math.radians(float(ext.yaw_deg))

        # Level camera: world → optical axes (Xw=fwd, Yw=right, Zw=up → Xc=right, Yc=down, Zc=fwd)
        R0 = np.array([
            [0.0, 1.0,  0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0,  0.0],
        ], dtype=np.float64)

        def _Rx(a):
            c, s = math.cos(a), math.sin(a)
            return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)
        def _Ry(a):
            c, s = math.cos(a), math.sin(a)
            return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)
        def _Rz(a):
            c, s = math.cos(a), math.sin(a)
            return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

        self._cam_Rcw = R0 @ _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)
        # tx=forward, ty=left → Yw=right = -ty
        self._cam_Cw = np.array(
            [float(ext.tx), -float(ext.ty), float(ext.tz)], dtype=np.float64
        )
        self.get_logger().info("Camera config received — projection matrices ready.")

    def _cam_image_callback(self, msg: Image) -> None:
        try:
            self._latest_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warning(f"Camera image decode failed: {e}",
                                      throttle_duration_sec=5.0)

    # ── Data callbacks ─────────────────────────────────────────────────────────

    def localization_callback(self, msg: LocalizationState) -> None:
        self._latest_loc      = msg
        self._latest_loc_time = self.get_clock().now()

    def lane_model_callback(self, msg: LaneCoeffs) -> None:
        self._latest_lane      = msg
        self._latest_lane_time = self.get_clock().now()

    def trajectory_callback(self, msg: EgoPredictedTrajectory) -> None:
        now = self.get_clock().now()

        if self._latest_loc is None or self._latest_loc_time is None:
            self.get_logger().warn(
                "No localization received — skipping departure risk.",
                throttle_duration_sec=2.0,
            )
            return

        loc_age = (now - self._latest_loc_time).nanoseconds * 1e-9
        if loc_age > self._stale_loc_s:
            self.get_logger().warn(
                f"Localization stale ({loc_age:.2f} s) — skipping.",
                throttle_duration_sec=2.0,
            )
            return

        lane_fresh = (
            self._latest_lane is not None
            and self._latest_lane_time is not None
            and (now - self._latest_lane_time).nanoseconds * 1e-9 <= self._stale_lane_s
        )

        points, max_probability, cumulative_risk, ego_lat_left_0 = self.compute_departure_risk(
            msg, self._latest_loc,
            self._latest_lane if lane_fresh else None,
        )
        if not points:
            return

        self._pub_max_p.publish(Float64(data=max_probability))
        self._pub_weighted_risk.publish(Float64(data=cumulative_risk))

        # ── BEV overlay ───────────────────────────────────────────────────────
        if self._pub_bev.get_subscription_count() > 0:
            speed0 = safe_float(msg.speed_mps[0], 0.0) if msg.speed_mps else 0.0
            canvas = create_bev_overlay(
                points,
                bev_w_m=self._bev_wid,
                bev_fwd_m=self._bev_fwd,
                res=self._bev_res,
                current_speed=speed0,
            )
            bev_img_msg = self._bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
            bev_img_msg.header = msg.header
            self._pub_bev.publish(bev_img_msg)

        # ── Camera overlay ────────────────────────────────────────────────────
        if (self._pub_overlay.get_subscription_count() > 0
                and self._latest_bgr is not None
                and self._cam_K is not None):
            speed0 = safe_float(msg.speed_mps[0], 0.0) if msg.speed_mps else 0.0
            cam_canvas = create_camera_overlay(
                points,
                bgr=self._latest_bgr,
                cam_K=self._cam_K,
                cam_Rcw=self._cam_Rcw,
                cam_Cw=self._cam_Cw,
                current_speed=speed0,
                ego_lat_left_0=ego_lat_left_0,
                hood_rows=self._hood_rows,
            )
            cam_img_msg = self._bridge.cv2_to_imgmsg(cam_canvas, encoding="bgr8")
            cam_img_msg.header = msg.header
            self._pub_overlay.publish(cam_img_msg)

    # ── Core computation ───────────────────────────────────────────────────────

    def _zero_departure_result(
        self, traj: EgoPredictedTrajectory, n: int, ego_lat_left_0: float,
    ) -> list[_RenderPoint]:
        """Section 1 / invalid-covariance result: P_LD=0 at every step, no
        lane boundaries to draw (never reuse stale lane data) — the boundary
        fields collapse onto the ego path itself so the overlays render a
        plain (uncoloured) path instead of extrapolating a boundary from
        nothing.
        """
        points: list[_RenderPoint] = []
        for k in range(n):
            x_k = safe_float(traj.forward_m[k], 0.0)
            y_k = safe_float(traj.lateral_m[k], 0.0)
            points.append(_RenderPoint(
                x_forward=x_k,
                ego_lateral_left=ego_lat_left_0 + y_k,
                y_left_boundary=y_k,
                y_right_boundary=y_k,
                p_lane_departure=0.0,
            ))
        return points

    def compute_departure_risk(
        self,
        traj:     EgoPredictedTrajectory,
        loc:      LocalizationState,
        lane_msg,
    ) -> tuple[list[_RenderPoint], float, float, float]:
        """Returns (points, max_probability, cumulative_risk, ego_lat_left_0).
        points is empty when there's nothing to report at all (n == 0, or an
        invalid lane/vehicle-width geometry — see the early returns below);
        callers should treat that as "skip this callback". An invalid/stale
        centerline is different: it still returns one zero-probability point
        per trajectory step so max_probability=0 and cumulative_risk=0 are
        published normally (see module docstring / _zero_departure_result).
        ego_lat_left_0 is the current lateral position from lane center (positive-left,
        rendering-only — see module docstring), needed to convert stored
        lane-frame laterals back to ego frame for camera projection.
        """
        n = min(len(traj.forward_m), len(traj.lateral_m), len(traj.speed_mps))
        if n == 0:
            return [], 0.0, 0.0, 0.0

        if not (len(traj.forward_m) == len(traj.lateral_m) == len(traj.speed_mps)):
            self.get_logger().warn(
                "Trajectory array length mismatch — using minimum valid length.",
                throttle_duration_sec=2.0,
            )

        # ego_lat_left_0 seeds the rendering-only road-frame lateral offset
        # (camera overlay projection); it does not feed the P_LD/R math below.
        ego_lat_left_0 = -safe_float(loc.e_y, 0.0)

        # ── Section 1: centerline validity gate ──────────────────────────────
        # centre_valid is LaneCoeffs' single centerline-validity flag. Despite
        # the .msg comment ("valid only when both lanes valid"), lane_node's
        # actual _compute_centerline is left_valid OR right_valid: true
        # average when both sides agree, a fallback to whichever single side
        # is more certain when both are valid but incompatible, or one side
        # shifted by the lane half-width when only that side is valid — False
        # only when neither side is valid. Whichever way centre_coeffs/
        # centre_cov were derived, they're consumed here as-is as the
        # centerline mean/covariance; only centre_valid=False (no usable lane
        # info at all) or an intentionally-disabled lane model
        # (self._use_lane_model) means zero departure risk, not "treat as
        # highly uncertain lane".
        if not self._use_lane_model or lane_msg is None or not bool(lane_msg.centre_valid):
            zero_points = self._zero_departure_result(traj, n, ego_lat_left_0)
            return zero_points, 0.0, 0.0, ego_lat_left_0

        # ── Section 2: centerline coefficients + covariance ──────────────────
        theta = (
            safe_float(lane_msg.centre_coeffs[0], 0.0),
            safe_float(lane_msg.centre_coeffs[1], 0.0),
            safe_float(lane_msg.centre_coeffs[2], 0.0),
        )
        Sigma_theta, cov_ok = sanitize_cov3(lane_msg.centre_cov)
        if not cov_ok:
            self.get_logger().warn(
                "Centerline coefficient covariance is non-finite or "
                "materially non-PSD — treating centerline as invalid.",
                throttle_duration_sec=2.0,
            )
            zero_points = self._zero_departure_result(traj, n, ego_lat_left_0)
            return zero_points, 0.0, 0.0, ego_lat_left_0

        # ── Section 4: available center clearance h = (W_L - W_v)/2 ──────────
        # lane_width_m/vehicle_width_m are this node's own parameters (kept —
        # LaneCoeffs carries no width field; vehicle width/mass sources are
        # otherwise unchanged from the existing node configuration).
        half_lane = self._lane_width_m / 2.0
        half_veh  = self._veh_width_m  / 2.0
        h = half_lane - half_veh
        if h <= 0.0:
            self.get_logger().warn(
                f"Invalid lane geometry: lane_width_m={self._lane_width_m:.2f} <= "
                f"vehicle_width_m={self._veh_width_m:.2f} — skipping.",
                throttle_duration_sec=2.0,
            )
            return [], 0.0, 0.0, ego_lat_left_0

        fallback_var = self._def_sigma_lat ** 2
        sigma_floor2 = self._min_sigma_y ** 2

        # ── Per-step loop ──────────────────────────────────────────────────────
        # points[] (used only for the gated overlay renders) is discarded
        # after the caller draws them; nothing per-point is published.
        points:    list[_RenderPoint] = []
        p_ld_list: list[float]        = []
        s_list:    list[float]        = []

        for k in range(n):
            x_k = safe_float(traj.forward_m[k], 0.0)
            y_k = safe_float(traj.lateral_m[k], 0.0)
            v_k = safe_float(traj.speed_mps[k], 0.0)

            # ── Section 4/5: relative lateral position D_k = y_k - y_C(x_k) ──
            # Sigma_theta (lane) and Sigma_X,k (ego prediction, from this
            # step's cov_xy) are assumed independent — no lane/ego-prediction
            # cross-covariance is published — so their variances add.
            sigma2_x, sigma2_y, sigma_xy, cov_step_ok = get_cov_xy_step(
                traj.cov_xy, k, fallback_var,
            )
            if not cov_step_ok:
                self.get_logger().warn(
                    "Ego trajectory cov_xy missing/invalid at a predicted "
                    "step — using fallback lateral variance.",
                    throttle_duration_sec=2.0,
                )

            var_theta, _ = propagate_poly_variance(Sigma_theta.flatten(), x_k, fallback_var)
            g_x   = eval_slope(theta, x_k)
            mu_D  = y_k - eval_quad(theta, x_k)
            var_D = var_theta + g_x * g_x * sigma2_x + sigma2_y - 2.0 * g_x * sigma_xy
            sigma_D = math.sqrt(max(var_D, sigma_floor2))

            # ── Section 6: instantaneous lane-departure probability ─────────
            p_inside = normal_cdf((h - mu_D) / sigma_D) - normal_cdf((-h - mu_D) / sigma_D)
            p_ld_k   = clamp(1.0 - p_inside, 0.0, 1.0)

            # ── Section 7: kinetic-energy severity, no additional VF ────────
            S_k = 0.5 * self._mass * v_k * v_k

            # ── Boundaries for the gated overlays only (Section 3) ──────────
            y_ctr_k   = eval_quad(theta, x_k)
            y_left_k  = y_ctr_k - half_lane
            y_right_k = y_ctr_k + half_lane
            ego_lat_left_k = ego_lat_left_0 + y_k

            points.append(_RenderPoint(
                x_forward=x_k,
                ego_lateral_left=ego_lat_left_k,
                y_left_boundary=y_left_k,
                y_right_boundary=y_right_k,
                p_lane_departure=p_ld_k,
            ))
            p_ld_list.append(p_ld_k)
            s_list.append(S_k)

        max_probability = max(p_ld_list) if p_ld_list else 0.0
        cumulative_risk = paralo_cumulative_risk(s_list, p_ld_list)
        return points, max_probability, cumulative_risk, ego_lat_left_0


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = DepartureRiskNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
