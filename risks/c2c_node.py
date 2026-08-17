#!/usr/bin/env python3
"""
C2C Risk Node — predictive car-to-car (C2C) collision probability between
the ego vehicle and all surrounding tracked objects.

Subscriptions:
  /ego/predicted_trajectory             (perception_interfaces/EgoPredictedTrajectory)
  /localization/state                   (perception_interfaces/LocalizationState)
  /surrounding/predicted_trajectories   (perception_interfaces/SurroundingPredictedTrajectories)
  /config/car                           (config_interfaces/CarConstants)  TRANSIENT_LOCAL
  /config/risk_weights                  (config_interfaces/RiskWeights)   TRANSIENT_LOCAL
  /config/camera/front         (config_interfaces/CameraConfig) — only while either
  sensor/camera/front/image_raw (sensor_msgs/Image)                gated image output
                                (/risk/collision/image/bev or /risk/collision/image/overlay)
                                has subscribers. Both subscribed/unsubscribed together,
                                dynamically — see _update_camera_subscription.

Publications:
  /risk/collision/max_probability  (std_msgs/Float64) — max_{j,k} P_C2C,j,k over every
                                    surrounding object and predicted step (diagnostic)
  /risk/collision/score            (std_msgs/Float64) — PARALO cumulative C2C risk R,
                                    weighted by RiskWeights.w_c2c (see below)
  /risk/collision/image/bev        (sensor_msgs/Image) — gated on subscriber count
  /risk/collision/image/overlay    (sensor_msgs/Image) — gated on subscriber count

Coordinate frame — SAE J670 throughout (+x forward, +y right, +z down):
  EgoPredictedTrajectory.lateral_m, SurroundingPredictedTrajectory.lateral_m and
  LocalizationState.e_y are all natively +right per their own message-field
  comments — confirmed by inspection, nothing here is positive-left. No
  boundary conversion is performed because none is needed; loc.e_y is used
  as-is (NOT negated — the previous "ego_lat_left_0 = -loc.e_y" conversion
  was incorrect and has been removed). left_to_right_2x2() is provided and
  unit-tested for any future source that genuinely is positive-left, but is
  not invoked on the current live inputs.

Ego position distribution Sigma_e,k: preferably read directly from
EgoPredictedTrajectory.cov_xy (the EKF-propagated per-step 2x2 forward/
lateral position covariance, already +forward/+right). Falls back to the
previous heuristic uncertainty-growth model (diagonal-only, no cross term)
only when cov_xy is unavailable or fails validation for a given step — see
get_ego_cov_step()/compute_collision_risk().

Surrounding-vehicle position distribution Sigma_j,k: SurroundingPredictedTrajectory
only carries 1-sigma per-axis std (sigma_forward_m, sigma_lateral_m, NOT
variance) — squared into a diagonal covariance, no cross term available.

Relative position R_j,k = X_j,k - X_e,k; assuming the lane/ego-prediction
and surrounding-object estimators are independent (no cross-covariance is
published), Sigma_r,j,k = Sigma_j,k + Sigma_e,k.

C2C probability: independent-axis AABB rectangle-overlap probability (this
repository's risk_pkg does not currently depend on SciPy — it's used
elsewhere in perception_ws/tracking_pkg but is not a risk_pkg dependency —
so no bivariate-normal CDF is used here rather than add a new heavy
dependency for this change; the relative covariance's off-diagonal term is
therefore ignored in the probability calculation). See
rectangle_overlap_probability()/axis_interval_probability() — the latter
also implements the near-deterministic step-function limit so a numerical
variance floor can never fabricate spurious probability.

C2C severity S_j,k = 0.5 * m_e * (v_e,k - v_j,k*cos(psi_j,k - psi_e,k))^2 —
PARALO kinetic-energy-of-closure surrogate, no extra vulnerability factor.
Ego heading psi_e,k is read directly from EgoPredictedTrajectory.heading_rad
when available (it is); psi_e=0 is the documented fallback otherwise.
Surrounding heading psi_j,k from SurroundingPredictedTrajectory.heading_rad;
a same-direction (psi_j=0) fallback is used when it is unavailable.

Cumulative C2C risk: PARALO §5.2.5 "Cumulative Risk Calculation" eq. (5.51)
generalized from a single trajectory to every (object, step) hypothesis:
  S_max = max_{j,k} S_j,k
  q_j,k = clamp((S_j,k / S_max) * P_C2C,j,k, 0, 1)
  R     = S_max * (1 - prod_{j,k} (1 - q_j,k))
computed in log-space via log1p for numerical stability. The product treats
every (object, step) hypothesis as independent even though predicted states
are physically correlated over time and across nearby objects — documented
here per spec, no extra correction is applied. /risk/collision/score
publishes w_c2c * R (RiskWeights.w_c2c is this channel's dedicated weight —
the previous code read a nonexistent "w_collision" field and would have
raised AttributeError the first time RiskWeights was received).

Per-object/per-step diagnostics (per-axis overlap probabilities, relative
means, severities, etc.) are kept as fields on the internal _ObjPoint/
_ObjectResult NamedTuples — nothing per-point is published (same pattern as
the other risk nodes); only max_probability and score go out, plus whatever
the gated overlays render.
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

from perception_interfaces.msg import (
    EgoPredictedTrajectory, LocalizationState, SurroundingPredictedTrajectories,
)
from config_interfaces.msg import CarConstants, CameraConfig, RiskWeights

_EPS = 1e-9

_DEFAULT_OBJ_LENGTH_M = 4.5
_DEFAULT_OBJ_WIDTH_M  = 1.85

# Color map: green(low) → yellow → orange → red(high), BGR
_COLORMAP_BGR: list[tuple[int, int, int]] = [
    (0,   200,   0),   # green    p=0.00
    (0,   220, 220),   # yellow   p=0.33
    (0,   140, 255),   # orange   p=0.66
    (0,     0, 220),   # red      p=1.00
]


class _ObjPoint(NamedTuple):
    """Minimal per-step, per-object data needed for the gated overlay
    renders and the PARALO aggregation.

    Nothing here is published (see module docstring) — it's discarded once
    the overlay images (if anyone is subscribed) are drawn. All positions
    are SAE J670 (+x forward, +y right).
    """
    obj_forward:     float   # object predicted forward position, ego-frame [m]
    obj_y_right:     float   # object predicted lateral position, ego-frame +right [m]
    obj_heading_rad: float   # object predicted heading [rad], SAE (+right/CW)
    p_long:          float   # P_x,j,k — longitudinal overlap probability [0,1]
    p_lat:           float   # P_y,j,k — lateral overlap probability [0,1]
    p_collision:     float   # P_C2C,j,k = p_long * p_lat [0,1]
    mu_rx:           float   # relative longitudinal mean R_x,j,k [m]
    mu_ry:           float   # relative lateral mean R_y,j,k [m]
    severity:        float   # S_j,k [J] — PARALO severity surrogate


class _ObjectResult(NamedTuple):
    """Per-object aggregate over the prediction horizon."""
    track_id:        int
    points:          list[_ObjPoint]
    max_probability: float
    half_length:     float   # object half-length used for its AABB [m]
    half_width:      float   # object half-width used for its AABB [m]


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


def sanitize_cov2(cov4, fallback: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Validate/repair a 2x2 covariance matrix (e.g. one step of
    EgoPredictedTrajectory.cov_xy). Applies only minimal numerical repair —
    symmetrization and clipping of tiny negative eigenvalues caused by
    floating-point error. A non-finite or materially non-PSD matrix (a
    negative eigenvalue beyond that tolerance) is reported invalid rather
    than silently masked; `fallback` is returned unmodified in that case.
    Returns (repaired_2x2_matrix, valid_flag).
    """
    try:
        P = np.array(cov4, dtype=float).reshape(2, 2)
    except Exception:
        return fallback, False
    if not np.all(np.isfinite(P)):
        return fallback, False

    P_sym = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P_sym)
    tol = max(1e-9, 1e-6 * float(np.max(np.abs(eigvals))))
    if float(np.min(eigvals)) < -tol:
        return fallback, False

    eigvals_clipped = np.clip(eigvals, 0.0, None)
    P_repaired = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return 0.5 * (P_repaired + P_repaired.T), True


def left_to_right_2x2(mu_left, sigma_left) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform a positive-left (x-forward, y-left) 2D mean/covariance into
    SAE J670 positive-right using T = diag(1, -1):
        mu_right    = T @ mu_left
        Sigma_right = T @ Sigma_left @ T.T
    This leaves sigma_x^2/sigma_y^2 unchanged but flips the sign of
    sigma_xy. Provided per spec and unit-tested; not invoked on the live
    callback path because every currently-consumed topic is already native
    SAE J670 (+right) — see module docstring.
    """
    T = np.array([[1.0, 0.0], [0.0, -1.0]])
    mu_right = T @ np.asarray(mu_left, dtype=float)
    sigma_right = T @ np.asarray(sigma_left, dtype=float) @ T.T
    return mu_right, sigma_right


def axis_interval_probability(mu: float, var: float, h: float, var_floor: float) -> float:
    """
    P(-h <= X <= h) for X ~ N(mu, var), one axis of the relative-position
    Gaussian. Near-zero variance (<= var_floor) uses the exact deterministic
    step function instead of letting the numerical floor fabricate spurious
    probability (task/spec section 13):
        P = 1 if -h <= mu <= h else 0.
    Ordinary nonzero variance uses the Gaussian interval formula. Clamped
    only against floating-point drift.
    """
    if var <= var_floor:
        return 1.0 if -h <= mu <= h else 0.0
    sigma = math.sqrt(var)
    p_inside = normal_cdf((h - mu) / sigma) - normal_cdf((-h - mu) / sigma)
    return clamp(p_inside, 0.0, 1.0)


def rectangle_overlap_probability(
    mu_rx: float, var_rx: float,
    mu_ry: float, var_ry: float,
    hx: float, hy: float,
    var_floor: float,
) -> tuple[float, float, float]:
    """
    Independent-axis AABB rectangle-overlap probability — the probability
    that the relative-position random variable R=(Rx,Ry) lies inside the
    [-hx,hx] x [-hy,hy] collision rectangle, treating Rx and Ry as
    independent (off-diagonal Sigma_r is ignored — see module docstring).
    Returns (p_long, p_lat, p_collision).
    """
    p_long = axis_interval_probability(mu_rx, var_rx, hx, var_floor)
    p_lat  = axis_interval_probability(mu_ry, var_ry, hy, var_floor)
    return p_long, p_lat, clamp(p_long * p_lat, 0.0, 1.0)


def paralo_severity(m_e: float, v_e: float, v_j: float, psi_e: float, psi_j: float) -> float:
    """
    PARALO C2C severity surrogate:
        S = 0.5 * m_e * (v_e - v_j * cos(psi_j - psi_e))^2
    Always nonnegative (squared). No additional vulnerability factor.
    """
    dv = v_e - v_j * math.cos(psi_j - psi_e)
    return 0.5 * m_e * dv * dv


def paralo_cumulative_risk(s_list: list[float], p_list: list[float]) -> float:
    """
    PARALO cumulative C2C risk across every (object, step) hypothesis:
        S_max = max_{j,k} S_j,k
        q_j,k = clamp((S_j,k / S_max) * P_j,k, 0, 1)
        R     = S_max * (1 - prod_{j,k} (1 - q_j,k))
    computed in log-space via log1p for numerical stability; q=1 is handled
    explicitly so the result becomes S_max without an invalid log(0). This
    product treats adjacent prediction steps and different vehicle
    hypotheses as independent even though the underlying predicted states
    are physically correlated over time — documented per spec, no
    additional dt correction is applied.
    """
    if not s_list:
        return 0.0
    S_max = max(s_list)
    if S_max <= _EPS:
        return 0.0

    log_survival = 0.0
    for S, P in zip(s_list, p_list):
        q = clamp((S / S_max) * P, 0.0, 1.0)
        if q >= 1.0:
            return S_max
        log_survival += math.log1p(-q)
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


def bev_to_pixel(
        fwd_m:     float,
        y_right_m: float,
        bev_w_px:  int,
        bev_h_px:  int,
        res:       float,
) -> tuple[int, int]:
    """
    Map an ego-frame ground point (SAE J670, +right) to BEV canvas pixels.
    Canvas origin is bottom-centre (ego position), +forward → up.
    SAE J670: increasing y (right) must move toward the right-hand side of
    the displayed image, i.e. +y_right_m/res (not the old -lateral/res).
    """
    u = int(round(bev_w_px / 2.0 + y_right_m / res))
    v = int(round(bev_h_px - 1   - fwd_m     / res))
    return u, v


def _ground_to_image(
        fwd_m:      float,
        y_right_m:  float,
        cam_K:      np.ndarray,
        cam_Rcw:    np.ndarray,
        cam_Cw:     np.ndarray,
        img_w:      int,
        img_h:      int,
) -> tuple[int, int] | None:
    """
    Project an ego-frame ground point (SAE J670, +right) to camera image
    pixels. World frame: Xw=forward, Yw=right — y_right_m maps to Yw
    directly (no sign flip, unlike the old +left convention).
    Returns (u, v) or None if behind/outside the image.
    """
    Pw = np.array([fwd_m, y_right_m, 0.0], dtype=np.float64)
    Pc = cam_Rcw @ (Pw - cam_Cw)
    if Pc[2] < 0.1:
        return None
    uv = cam_K @ Pc
    u = uv[0] / uv[2]
    v = uv[1] / uv[2]
    if not (0 <= u < img_w and 0 <= v < img_h):
        return None
    return int(round(u)), int(round(v))


# ──────────────────────────────────────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────────────────────────────────────

class C2CRiskNode(Node):

    def __init__(self):
        super().__init__("c2c_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("ego_length_m",                      4.5)
        self.declare_parameter("ego_width_m",                       1.85)
        self.declare_parameter("min_sigma_m",                       0.05)
        self.declare_parameter("max_sigma_m",                       5.0)
        self.declare_parameter("ego_prediction_sigma_lat_base_m",   0.05)
        self.declare_parameter("ego_prediction_sigma_lat_rate_mps", 0.10)
        self.declare_parameter("ego_prediction_sigma_fwd_base_m",   0.10)
        self.declare_parameter("ego_prediction_sigma_fwd_rate_mps", 0.20)
        self.declare_parameter("use_heading_uncertainty",           True)
        self.declare_parameter("use_surrounding_predictions",       True)
        self.declare_parameter("stale_localization_timeout_s",      0.50)
        self.declare_parameter("stale_surrounding_timeout_s",       0.50)
        self.declare_parameter("bev_forward_range_m",               50.0)
        self.declare_parameter("bev_width_m",                       20.0)
        self.declare_parameter("bev_resolution_m_per_pixel",        0.10)
        # Numerical variance floor (task/spec section 8/13): below this,
        # an axis is treated as deterministic instead of applying the
        # Gaussian interval formula with an artificially inflated sigma.
        self.declare_parameter("variance_floor_m2",                 1e-6)

        self._load_params()

        # ── QoS profiles ──────────────────────────────────────────────────────
        stream_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
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
        self._w_c2c:        float = 1.0
        self._ego_mass_kg:  float = 1575.0

        self._latest_loc:              LocalizationState | None                = None
        self._latest_loc_time                                                  = None
        self._latest_surrounding:      SurroundingPredictedTrajectories | None = None
        self._latest_surrounding_time                                          = None

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
            SurroundingPredictedTrajectories, "/surrounding/predicted_trajectories",
            self.surrounding_callback, stream_qos,
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
        # subscribed while either gated image output (/risk/collision/image/bev
        # or /risk/collision/image/overlay) has subscribers, to avoid the
        # per-frame cv_bridge decode/copy cost when nobody wants any image.
        # See _update_camera_subscription.
        self._sub_cam_cfg = None
        self._sub_cam_img = None

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_max_p   = self.create_publisher(Float64, "/risk/collision/max_probability", pub_qos)
        self._pub_risk    = self.create_publisher(Float64, "/risk/collision/score",           pub_qos)
        self._pub_bev     = self.create_publisher(Image,   "/risk/collision/image/bev",       stream_qos)
        self._pub_overlay = self.create_publisher(Image,   "/risk/collision/image/overlay",   stream_qos)

        # Periodic check (subscriber counts aren't event-driven in rclpy) —
        # see _update_camera_subscription.
        self._cam_sub_timer = self.create_timer(1.0, self._update_camera_subscription)

        self.get_logger().info("C2C risk node ready.")

    # ── Dynamic camera-image subscription ──────────────────────────────────────

    def _update_camera_subscription(self) -> None:
        """Subscribe to /config/camera/front and the raw camera frame only
        while /risk/collision/image/bev or /risk/collision/image/overlay has
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
        self._ego_length_m        = float(gp("ego_length_m"))
        self._ego_width_m         = float(gp("ego_width_m"))
        self._min_sigma_m         = float(gp("min_sigma_m"))
        self._max_sigma_m         = float(gp("max_sigma_m"))
        self._sigma_lat_base      = float(gp("ego_prediction_sigma_lat_base_m"))
        self._sigma_lat_rate      = float(gp("ego_prediction_sigma_lat_rate_mps"))
        self._sigma_fwd_base      = float(gp("ego_prediction_sigma_fwd_base_m"))
        self._sigma_fwd_rate      = float(gp("ego_prediction_sigma_fwd_rate_mps"))
        self._use_heading_unc     = bool(gp("use_heading_uncertainty"))
        self._use_surrounding     = bool(gp("use_surrounding_predictions"))
        self._stale_loc_s         = float(gp("stale_localization_timeout_s"))
        self._stale_surrounding_s = float(gp("stale_surrounding_timeout_s"))
        self._bev_fwd             = float(gp("bev_forward_range_m"))
        self._bev_wid             = float(gp("bev_width_m"))
        self._bev_res             = float(gp("bev_resolution_m_per_pixel"))
        self._var_floor           = float(gp("variance_floor_m2"))

    # ── Config callbacks ───────────────────────────────────────────────────────

    def _car_config_callback(self, msg: CarConstants) -> None:
        self._ego_length_m = float(msg.l)
        self._ego_width_m  = float(msg.w_c)
        self._ego_mass_kg  = float(msg.m)
        self.get_logger().info(
            f"Car config: length={self._ego_length_m:.2f} m, width={self._ego_width_m:.2f} m"
        )

    def _risk_weights_callback(self, msg: RiskWeights) -> None:
        self._w_c2c = float(msg.w_c2c)
        self.get_logger().info(f"Risk weights: w_c2c={self._w_c2c:.3f}")

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
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

        def _Ry(a):
            c, s = math.cos(a), math.sin(a)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

        def _Rz(a):
            c, s = math.cos(a), math.sin(a)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

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

    def surrounding_callback(self, msg: SurroundingPredictedTrajectories) -> None:
        self._latest_surrounding      = msg
        self._latest_surrounding_time = self.get_clock().now()

    def trajectory_callback(self, msg: EgoPredictedTrajectory) -> None:
        now = self.get_clock().now()

        # ── Localization staleness check ───────────────────────────────────────
        if self._latest_loc is None or self._latest_loc_time is None:
            self.get_logger().warn(
                "No localization received — skipping C2C risk.",
                throttle_duration_sec=2.0,
            )
            self._publish_empty()
            return

        loc_age = (now - self._latest_loc_time).nanoseconds * 1e-9
        if loc_age > self._stale_loc_s:
            self.get_logger().warn(
                f"Localization stale ({loc_age:.2f} s) — skipping.",
                throttle_duration_sec=2.0,
            )
            self._publish_empty()
            return

        # ── Surrounding staleness check ────────────────────────────────────────
        if self._use_surrounding:
            if self._latest_surrounding is None or self._latest_surrounding_time is None:
                self.get_logger().warn(
                    "No surrounding trajectories received — skipping C2C risk.",
                    throttle_duration_sec=2.0,
                )
                self._publish_empty()
                return

            surr_age = (now - self._latest_surrounding_time).nanoseconds * 1e-9
            if surr_age > self._stale_surrounding_s:
                self.get_logger().warn(
                    f"Surrounding trajectories stale ({surr_age:.2f} s) — skipping.",
                    throttle_duration_sec=2.0,
                )
                self._publish_empty()
                return

        # ── Zero objects is normal — publish empty without warning ─────────────
        surrounding_msg = self._latest_surrounding
        if surrounding_msg is None or len(surrounding_msg.trajectories) == 0:
            self._publish_empty()
            self.publish_debug(msg, [])
            return

        # ── Compute and publish ────────────────────────────────────────────────
        # max_probability is published exactly once here (previously
        # duplicated); score is now the PARALO cumulative risk, not the old
        # single max-probability-point estimate.
        objects, max_probability, cumulative_risk = self.compute_collision_risk(
            msg, self._latest_loc, surrounding_msg,
        )
        self._pub_max_p.publish(Float64(data=max_probability))
        self._pub_risk.publish(Float64(data=self._w_c2c * cumulative_risk))
        self.publish_debug(msg, objects)

    # ── Core computation ───────────────────────────────────────────────────────

    def get_ego_cov_step(self, cov_xy, k: int) -> tuple[np.ndarray, bool]:
        """
        Extract & sanitize the per-step 2x2 ego position covariance
        [[P_ff,P_fl],[P_fl,P_ll]] from EgoPredictedTrajectory.cov_xy
        (already forward/+right — see module docstring). Returns
        (Sigma_e_k, ok); ok=False when the array is missing/short/invalid
        for this step, signalling the caller to fall back to the heuristic
        growth model (spec section 4).
        """
        i = 4 * k
        if len(cov_xy) < i + 4:
            return np.zeros((2, 2)), False
        return sanitize_cov2(cov_xy[i:i + 4], np.zeros((2, 2)))

    def compute_collision_risk(
        self,
        ego_traj:        EgoPredictedTrajectory,
        loc:             LocalizationState,
        surrounding_msg: SurroundingPredictedTrajectories,
    ) -> tuple[list[_ObjectResult], float, float]:
        """Returns (objects, max_probability, cumulative_risk)."""

        dt = safe_float(ego_traj.dt_s, 0.1)
        if dt < _EPS:
            dt = 0.1

        # Localization state — SAE J670, +right; used unmodified (no
        # negation — see module docstring) only by the fallback ego
        # lateral-uncertainty growth model below.
        sigma2_ey0   = max(safe_float(loc.cov_diag[0], 0.0), 0.0) if len(loc.cov_diag) > 0 else 0.0
        sigma2_epsi0 = max(safe_float(loc.cov_diag[1], 0.0), 0.0) if len(loc.cov_diag) > 1 else 0.0

        n_ego = len(ego_traj.forward_m)
        # Ego sigma_v array availability check (fallback fwd-sigma growth model only)
        has_sigma_v = (
            hasattr(ego_traj, "sigma_v")
            and len(ego_traj.sigma_v) == n_ego
        )
        # Preferred ego position covariance source (spec section 4).
        has_ego_cov_xy  = len(ego_traj.cov_xy) >= 4 * n_ego
        # Ego heading (spec section 14) — psi_e=0 is the documented fallback.
        has_ego_heading = len(ego_traj.heading_rad) >= n_ego

        ego_half_length = self._ego_length_m / 2.0
        ego_half_width  = self._ego_width_m  / 2.0
        obj_dt          = safe_float(surrounding_msg.dt_s, dt)

        objects: list[_ObjectResult] = []
        all_severities:    list[float] = []
        all_probabilities: list[float] = []

        for obj_traj in surrounding_msg.trajectories:

            # Object dimensions — use defaults when zero
            obj_length_m = safe_float(obj_traj.length_m, 0.0)
            obj_width_m  = safe_float(obj_traj.width_m,  0.0)
            if obj_length_m < _EPS:
                obj_length_m = _DEFAULT_OBJ_LENGTH_M
            if obj_width_m < _EPS:
                obj_width_m = _DEFAULT_OBJ_WIDTH_M
            obj_half_length = obj_length_m / 2.0
            obj_half_width  = obj_width_m  / 2.0
            hx = ego_half_length + obj_half_length
            hy = ego_half_width  + obj_half_width

            # dt alignment warning
            if abs(dt - obj_dt) > 0.005:
                self.get_logger().warn(
                    f"Track {obj_traj.track_id}: ego dt={dt:.3f} s vs "
                    f"surrounding dt={obj_dt:.3f} s — using minimum step count.",
                    throttle_duration_sec=5.0,
                )

            n_steps = min(
                n_ego,
                len(ego_traj.lateral_m),
                len(ego_traj.speed_mps),
                len(obj_traj.forward_m),
                len(obj_traj.lateral_m),
                len(obj_traj.speed_mps),
            )
            if n_steps == 0:
                continue

            has_sigma_obj_fwd = len(obj_traj.sigma_forward_m) >= n_steps
            has_sigma_obj_lat = len(obj_traj.sigma_lateral_m) >= n_steps
            has_obj_heading   = len(obj_traj.heading_rad)     >= n_steps

            points: list[_ObjPoint] = []
            p_list: list[float]     = []
            s_list: list[float]     = []
            cum_sigma_fwd = 0.0

            for k in range(n_steps):
                t_k = k * dt

                ego_fwd_k    = safe_float(ego_traj.forward_m[k],  0.0)
                ego_y_right_k = safe_float(ego_traj.lateral_m[k], 0.0)
                ego_spd_k    = max(safe_float(ego_traj.speed_mps[k], 0.0), 0.0)
                ego_heading_k = safe_float(ego_traj.heading_rad[k], 0.0) if has_ego_heading else 0.0

                obj_fwd_k     = safe_float(obj_traj.forward_m[k],  0.0)
                obj_y_right_k = safe_float(obj_traj.lateral_m[k],  0.0)
                obj_spd_k     = max(safe_float(obj_traj.speed_mps[k], 0.0), 0.0)
                # Same-direction fallback (spec section 14) when object heading is unavailable.
                obj_heading_k = safe_float(obj_traj.heading_rad[k], 0.0) if has_obj_heading else 0.0

                # ── Ego position covariance Sigma_e,k ────────────────────────────
                Sigma_e_k, cov_e_ok = (
                    self.get_ego_cov_step(ego_traj.cov_xy, k) if has_ego_cov_xy
                    else (np.zeros((2, 2)), False)
                )
                if not cov_e_ok:
                    # Fallback: previous heuristic uncertainty-growth model
                    # (diagonal only — it never modeled a cross term).
                    if has_sigma_v:
                        sv = safe_float(ego_traj.sigma_v[k], 0.0)
                        cum_sigma_fwd += sv * dt
                        sigma_ego_fwd_k = max(cum_sigma_fwd, self._sigma_fwd_base)
                    else:
                        sigma_ego_fwd_k = self._sigma_fwd_base + self._sigma_fwd_rate * t_k
                    sigma_ego_fwd_k = clamp(sigma_ego_fwd_k, self._min_sigma_m, self._max_sigma_m)

                    sigma2_heading = (abs(ego_fwd_k) ** 2) * sigma2_epsi0 if self._use_heading_unc else 0.0
                    sigma_pred     = self._sigma_lat_base + self._sigma_lat_rate * t_k
                    sigma2_ego_y_k = sigma2_ey0 + sigma2_heading + sigma_pred ** 2

                    Sigma_e_k = np.diag([sigma_ego_fwd_k ** 2, sigma2_ego_y_k])

                # ── Object position covariance Sigma_j,k ─────────────────────────
                # sigma_forward_m/sigma_lateral_m are 1-sigma std, not
                # variance (msg comment) — squared here (spec section 5).
                sigma_obj_fwd_k = safe_float(obj_traj.sigma_forward_m[k], 0.0) if has_sigma_obj_fwd else 0.0
                sigma_obj_lat_k = safe_float(obj_traj.sigma_lateral_m[k], 0.0) if has_sigma_obj_lat else 0.0
                Sigma_j_k = np.diag([sigma_obj_fwd_k ** 2, sigma_obj_lat_k ** 2])

                # ── Relative position distribution (spec section 7) ──────────────
                # Independence assumption: no ego/surrounding-prediction
                # cross-covariance is published, so variances add.
                mu_rx = obj_fwd_k - ego_fwd_k
                mu_ry = obj_y_right_k - ego_y_right_k
                Sigma_r_k = Sigma_e_k + Sigma_j_k

                # ── C2C rectangle-overlap probability (spec sections 9-13) ───────
                p_long, p_lat, p_collision_k = rectangle_overlap_probability(
                    mu_rx, Sigma_r_k[0, 0], mu_ry, Sigma_r_k[1, 1], hx, hy, self._var_floor,
                )

                # ── C2C severity (spec section 15) ────────────────────────────────
                S_k = paralo_severity(
                    self._ego_mass_kg, ego_spd_k, obj_spd_k, ego_heading_k, obj_heading_k,
                )

                points.append(_ObjPoint(
                    obj_forward=obj_fwd_k,
                    obj_y_right=obj_y_right_k,
                    obj_heading_rad=obj_heading_k,
                    p_long=p_long,
                    p_lat=p_lat,
                    p_collision=p_collision_k,
                    mu_rx=mu_rx,
                    mu_ry=mu_ry,
                    severity=S_k,
                ))
                p_list.append(p_collision_k)
                s_list.append(S_k)

            if not p_list:
                continue

            objects.append(_ObjectResult(
                track_id=int(obj_traj.track_id),
                points=points,
                max_probability=max(p_list),
                half_length=obj_half_length,
                half_width=obj_half_width,
            ))
            all_severities.extend(s_list)
            all_probabilities.extend(p_list)

        max_probability = max((o.max_probability for o in objects), default=0.0)
        cumulative_risk = paralo_cumulative_risk(all_severities, all_probabilities)
        return objects, max_probability, cumulative_risk

    # ── Debug publishing ───────────────────────────────────────────────────────

    def publish_debug(self, ego_traj: EgoPredictedTrajectory, objects: list[_ObjectResult]) -> None:
        ego_speed0 = safe_float(ego_traj.speed_mps[0], 0.0) if ego_traj.speed_mps else 0.0
        self._publish_markers(ego_traj, objects, ego_speed0)

    def _publish_markers(
        self,
        ego_traj:   EgoPredictedTrajectory,
        objects:    list[_ObjectResult],
        ego_speed0: float,
    ) -> None:
        if self._pub_bev.get_subscription_count() > 0:
            canvas = self._create_bev_overlay(ego_traj, objects, ego_speed0)
            bev_msg = self._bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
            bev_msg.header = ego_traj.header
            self._pub_bev.publish(bev_msg)

        if (self._pub_overlay.get_subscription_count() > 0
                and self._latest_bgr is not None
                and self._cam_K is not None):
            cam_canvas = self._create_camera_overlay(ego_traj, objects, ego_speed0)
            cam_msg = self._bridge.cv2_to_imgmsg(cam_canvas, encoding="bgr8")
            cam_msg.header = ego_traj.header
            self._pub_overlay.publish(cam_msg)

    # ── Shared geometry helper ────────────────────────────────────────────────

    @staticmethod
    def _oriented_box_corners_m(cx_m, cy_m, half_l, half_w, heading, project):
        """4 corners of a heading-oriented object box, each run through
        `project` (a ground->canvas or ground->image mapping). (cx_m, cy_m)
        and heading are SAE J670 (+x forward, +y right)."""
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        corners_m = [
            (cx_m + half_l * cos_h - half_w * sin_h,
             cy_m + half_l * sin_h + half_w * cos_h),
            (cx_m + half_l * cos_h + half_w * sin_h,
             cy_m + half_l * sin_h - half_w * cos_h),
            (cx_m - half_l * cos_h + half_w * sin_h,
             cy_m - half_l * sin_h - half_w * cos_h),
            (cx_m - half_l * cos_h - half_w * sin_h,
             cy_m - half_l * sin_h + half_w * cos_h),
        ]
        return [project(f, l) for f, l in corners_m]

    # ── BEV overlay ────────────────────────────────────────────────────────────

    def _create_bev_overlay(
        self,
        ego_traj:   EgoPredictedTrajectory,
        objects:    list[_ObjectResult],
        ego_speed0: float,
    ) -> np.ndarray:
        res      = self._bev_res
        bev_h_px = int(round(self._bev_fwd / res))
        bev_w_px = int(round(self._bev_wid / res))

        canvas = np.zeros((bev_h_px, bev_w_px, 3), dtype=np.uint8)
        canvas[:] = (25, 25, 25)

        def m2px(fwd: float, y_right: float) -> tuple[int, int]:
            return bev_to_pixel(fwd, y_right, bev_w_px, bev_h_px, res)

        def in_bnd(pt: tuple[int, int]) -> bool:
            return 0 <= pt[0] < bev_w_px and 0 <= pt[1] < bev_h_px

        # Current ego lateral offset — SAE J670, +right; used unmodified
        # (no negation — see module docstring).
        ego_y_right_0 = 0.0
        if self._latest_loc is not None:
            ego_y_right_0 = safe_float(self._latest_loc.e_y, 0.0)

        n_ego = min(len(ego_traj.forward_m), len(ego_traj.lateral_m))

        # Per-step max p_collision across all objects
        step_max_p: list[float] = [0.0] * n_ego
        for obj in objects:
            for k, pt in enumerate(obj.points):
                if k < n_ego:
                    step_max_p[k] = max(step_max_p[k], pt.p_collision)

        # Ego predicted path
        ego_px = [
            m2px(safe_float(ego_traj.forward_m[k], 0.0),
                 ego_y_right_0 + safe_float(ego_traj.lateral_m[k], 0.0))
            for k in range(n_ego)
        ]
        for k in range(1, n_ego):
            p1, p2 = ego_px[k - 1], ego_px[k]
            if in_bnd(p1) and in_bnd(p2):
                cv2.line(canvas, p1, p2, _p_color(step_max_p[k]), 3, cv2.LINE_AA)
        for k, pix in enumerate(ego_px):
            if in_bnd(pix):
                cv2.circle(canvas, pix, 4, _p_color(step_max_p[k]), -1, cv2.LINE_AA)

        # Ego bounding box at step 0
        ego_hl = self._ego_length_m / 2.0
        ego_hw = self._ego_width_m  / 2.0
        ego_corners = np.array([
            m2px( ego_hl,  ego_hw + ego_y_right_0),
            m2px( ego_hl, -ego_hw + ego_y_right_0),
            m2px(-ego_hl, -ego_hw + ego_y_right_0),
            m2px(-ego_hl,  ego_hw + ego_y_right_0),
        ], dtype=np.int32)
        cv2.drawContours(canvas, [ego_corners], 0, (255, 255, 255), 1, cv2.LINE_AA)

        ego_ctr = m2px(0.0, ego_y_right_0)
        cv2.circle(canvas, ego_ctr, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, "EGO", (ego_ctr[0] - 15, ego_ctr[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

        # Per-object: centroid dots + oriented bounding box at highest-probability step
        for obj in objects:
            if not obj.points:
                continue

            p_list  = [pt.p_collision for pt in obj.points]
            max_idx = p_list.index(max(p_list))
            obj_px  = [m2px(pt.obj_forward, pt.obj_y_right) for pt in obj.points]

            for pt, pix in zip(obj.points, obj_px):
                if in_bnd(pix):
                    cv2.circle(canvas, pix, 4, _p_color(pt.p_collision), -1, cv2.LINE_AA)

            # Oriented bounding box at the highest-probability step
            pt_max = obj.points[max_idx]
            corners_px = np.array(
                self._oriented_box_corners_m(
                    pt_max.obj_forward, pt_max.obj_y_right,
                    obj.half_length, obj.half_width, pt_max.obj_heading_rad, m2px,
                ),
                dtype=np.int32,
            )
            col = _p_color(obj.max_probability)
            cv2.drawContours(canvas, [corners_px], 0, col, 2, cv2.LINE_AA)

            if obj_px and in_bnd(obj_px[0]):
                lx, ly = obj_px[0][0] + 5, obj_px[0][1] - 5
                cv2.putText(canvas, f"ID{obj.track_id}", (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

        # Text overlay
        max_p = max((o.max_probability for o in objects), default=0.0)

        def _put(text: str, row: int, color=(220, 220, 220)):
            cv2.putText(canvas, text, (6, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        _put(f"Speed:   {ego_speed0:5.1f} m/s", 20)
        _put(f"Max P:   {max_p:.3f}", 42, _p_color(max_p))
        _put(f"Objects: {len(objects)}", 62)

        # Colour-scale legend
        bar_x  = bev_w_px - 14
        bar_h  = bev_h_px // 2
        bar_y0 = bev_h_px // 4
        for row in range(bar_h):
            p_row = 1.0 - row / max(bar_h - 1, 1)
            cv2.line(canvas, (bar_x, bar_y0 + row), (bar_x + 10, bar_y0 + row),
                     _p_color(p_row), 1)
        _put("1.0", bar_y0 - 4,         (180, 180, 180))
        _put("0.0", bar_y0 + bar_h + 2, (180, 180, 180))

        return canvas

    # ── Camera overlay ────────────────────────────────────────────────────────

    def _create_camera_overlay(
        self,
        ego_traj:   EgoPredictedTrajectory,
        objects:    list[_ObjectResult],
        ego_speed0: float,
    ) -> np.ndarray:
        """
        Draw predicted C2C risk onto the camera image: ego path coloured by
        the per-step max p_collision across objects, plus each object's
        projected path and oriented bounding box at its highest-probability
        step. Same ground-plane projection convention as departure_node/
        rollover_node's create_camera_overlay (now SAE J670, +right).
        """
        overlay = self._latest_bgr.copy()
        img_h, img_w = overlay.shape[:2]

        def proj(fwd, y_right):
            return _ground_to_image(
                fwd, y_right, self._cam_K, self._cam_Rcw, self._cam_Cw, img_w, img_h,
            )

        ego_y_right_0 = 0.0
        if self._latest_loc is not None:
            ego_y_right_0 = safe_float(self._latest_loc.e_y, 0.0)

        n_ego = min(len(ego_traj.forward_m), len(ego_traj.lateral_m))

        step_max_p: list[float] = [0.0] * n_ego
        for obj in objects:
            for k, pt in enumerate(obj.points):
                if k < n_ego:
                    step_max_p[k] = max(step_max_p[k], pt.p_collision)

        ego_px = [
            proj(safe_float(ego_traj.forward_m[k], 0.0),
                 ego_y_right_0 + safe_float(ego_traj.lateral_m[k], 0.0))
            for k in range(n_ego)
        ]
        for k in range(1, n_ego):
            p1, p2 = ego_px[k - 1], ego_px[k]
            if p1 is not None and p2 is not None:
                cv2.line(overlay, p1, p2, _p_color(step_max_p[k]), 4, cv2.LINE_AA)
        for k, pix in enumerate(ego_px):
            if pix is not None:
                cv2.circle(overlay, pix, 5, _p_color(step_max_p[k]), -1, cv2.LINE_AA)

        for obj in objects:
            if not obj.points:
                continue

            p_list  = [pt.p_collision for pt in obj.points]
            max_idx = p_list.index(max(p_list))
            obj_px  = [proj(pt.obj_forward, pt.obj_y_right) for pt in obj.points]

            for pt, pix in zip(obj.points, obj_px):
                if pix is not None:
                    color = _p_color(pt.p_collision)
                    cv2.circle(overlay, pix, 4, color,             -1, cv2.LINE_AA)
                    cv2.circle(overlay, pix, 4, (255, 255, 255),    1, cv2.LINE_AA)

            pt_max = obj.points[max_idx]
            corners_px = self._oriented_box_corners_m(
                pt_max.obj_forward, pt_max.obj_y_right,
                obj.half_length, obj.half_width, pt_max.obj_heading_rad, proj,
            )
            if all(c is not None for c in corners_px):
                col = _p_color(obj.max_probability)
                pts = np.array(corners_px, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(overlay, [pts], True, col, 2, cv2.LINE_AA)

            if obj_px and obj_px[0] is not None:
                lx, ly = obj_px[0][0] + 5, obj_px[0][1] - 5
                cv2.putText(overlay, f"ID{obj.track_id}", (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _p_color(obj.max_probability),
                            1, cv2.LINE_AA)

        max_p = max((o.max_probability for o in objects), default=0.0)

        def _put(text: str, row: int, color=(220, 220, 220)):
            cv2.putText(overlay, text, (8, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, text, (8, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color,     1, cv2.LINE_AA)

        _put(f"Speed:  {ego_speed0:5.1f} m/s", 28)
        _put(f"Max P:  {max_p:.3f}", 54, _p_color(max_p))

        return overlay

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _publish_empty(self) -> None:
        self._pub_max_p.publish(Float64(data=0.0))
        self._pub_risk.publish(Float64(data=0.0))


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = C2CRiskNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
