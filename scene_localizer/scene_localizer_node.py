#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
import rclpy.time
import yaml
from ament_index_python.packages import get_package_share_directory
from aruco_opencv_msgs.msg import ArucoDetection
from geometry_msgs.msg import Pose, PointStamped, PoseStamped, TransformStamped
from scene_localizer.msg import BallTrajectory
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


@dataclass
class TransformEstimate:
    translation: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass
class TimedMarkerObservation:
    stamp_sec: float
    marker_id: int
    camera_marker: TransformEstimate


@dataclass
class TopCamState:
    marker_layout: Dict[int, TransformEstimate]

    # One independent rolling buffer per configured marker.
    marker_buffers: Dict[int, List[TimedMarkerObservation]] = field(default_factory=dict)

    # Last pose that passed all window and geometry checks.
    last_valid_estimate: Optional[TransformEstimate] = None

    # Last source timestamp involved in a newly accepted estimate.
    last_valid_source_stamp_sec: Optional[float] = None

    # Optional extra smoothing state.
    last_published_estimate: Optional[TransformEstimate] = None


@dataclass
class CalibrationYamlTransform:
    parent_frame: str
    child_frame: str
    estimate: TransformEstimate


def _clean_frame(frame: str) -> str:
    return str(frame).strip().strip("/")


def _normalize_quaternion_xyzw(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / norm


def _quaternion_to_matrix_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion_xyzw(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _matrix_to_quaternion_xyzw(r: np.ndarray) -> np.ndarray:
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
            w = (r[2, 1] - r[1, 2]) / s
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
            w = (r[0, 2] - r[2, 0]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
            w = (r[1, 0] - r[0, 1]) / s
    return _normalize_quaternion_xyzw(np.array([x, y, z, w], dtype=float))


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _estimate_to_matrix(estimate: TransformEstimate) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _quaternion_to_matrix_xyzw(estimate.quaternion_xyzw)
    transform[:3, 3] = estimate.translation
    return transform


def _matrix_to_estimate(transform: np.ndarray) -> TransformEstimate:
    return TransformEstimate(
        translation=np.array(transform[:3, 3], dtype=float),
        quaternion_xyzw=_matrix_to_quaternion_xyzw(np.array(transform[:3, :3], dtype=float)),
    )


def _invert_matrix(transform: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=float)
    r = transform[:3, :3]
    t = transform[:3, 3]
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t
    return out


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3, :3] @ point + transform[:3, 3]


def _is_finite_vector(v: np.ndarray) -> bool:
    return bool(v.shape == (3,) and np.all(np.isfinite(v)))


def _transform_stamped_to_matrix(msg: TransformStamped) -> np.ndarray:
    q = np.array(
        [
            msg.transform.rotation.x,
            msg.transform.rotation.y,
            msg.transform.rotation.z,
            msg.transform.rotation.w,
        ],
        dtype=float,
    )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _quaternion_to_matrix_xyzw(q)
    transform[:3, 3] = np.array(
        [
            msg.transform.translation.x,
            msg.transform.translation.y,
            msg.transform.translation.z,
        ],
        dtype=float,
    )
    return transform


def _quaternion_angular_distance_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    qa = _normalize_quaternion_xyzw(q1)
    qb = _normalize_quaternion_xyzw(q2)
    dot = float(np.clip(np.abs(np.dot(qa, qb)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _average_quaternions_xyzw(quaternions: List[np.ndarray]) -> np.ndarray:
    if not quaternions:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    ref = _normalize_quaternion_xyzw(quaternions[0])
    accum = np.zeros(4, dtype=float)
    for q in quaternions:
        qn = _normalize_quaternion_xyzw(q)
        if float(np.dot(qn, ref)) < 0.0:
            qn = -qn
        accum += qn

    if float(np.linalg.norm(accum)) < 1e-12:
        return ref
    return _normalize_quaternion_xyzw(accum)


def _blend_quaternions_xyzw(prev_q: np.ndarray, curr_q: np.ndarray, alpha: float) -> np.ndarray:
    alpha_clamped = min(1.0, max(0.0, float(alpha)))
    q_prev = _normalize_quaternion_xyzw(prev_q)
    q_curr = _normalize_quaternion_xyzw(curr_q)
    if float(np.dot(q_prev, q_curr)) < 0.0:
        q_curr = -q_curr
    blended = (1.0 - alpha_clamped) * q_prev + alpha_clamped * q_curr
    if float(np.linalg.norm(blended)) < 1e-12:
        return q_curr
    return _normalize_quaternion_xyzw(blended)


class SceneLocalizerNode(Node):
    """
    Top-camera-only scene localizer.

    Input:
      - /aruco_top_cam/aruco_detections
      - marker layout YAML with table-frame marker coordinates
      - optional calibration YAML describing one of:
          * T_robot_base_top_camera
          * T_top_camera_robot_base
          * T_calibration_link_top_camera
          * T_top_camera_calibration_link

    Output:
      - T_top_camera_table as PoseStamped
      - T_robot_base_top_camera as PoseStamped
      - T_robot_base_table as PoseStamped
      - optional TF broadcasts for robot_base -> top_camera and robot_base -> table
      - valid middle-line/trajectory intersection as PoseStamped in robot_base_frame
      - optional TF broadcast table_frame -> ball_3d_frame from PointStamped ball detections
    """

    def __init__(self) -> None:
        super().__init__("scene_localizer")

        self._last_warn_time_ns: Dict[str, int] = {}
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)

        try:
            share_dir = Path(get_package_share_directory("scene_localizer"))
            default_marker_yaml = share_dir / "config" / "table_marker_layout.yaml"
        except Exception:
            default_marker_yaml = Path(__file__).resolve().parents[1] / "config" / "table_marker_layout.yaml"

        self.declare_parameter("marker_layout_yaml", str(default_marker_yaml))
        self.declare_parameter("base_cam_calibration_yaml", "/home/jau/dyros/calibration/scene_localizer/T_base_cam.yaml")

        self.declare_parameter("robot_base_frame", "base")
        self.declare_parameter("calibration_link_frame", "fr3_link6")
        self.declare_parameter("top_camera_frame", "camera_color_optical_frame")
        self.declare_parameter("table_frame", "table_frame")

        self.declare_parameter("top_detections_topic", "/aruco_top_cam/aruco_detections")
        self.declare_parameter("top_camera_table_pose_topic", "/scene_localizer/top_cam/table_pose_camera")
        self.declare_parameter("robot_base_top_camera_pose_topic", "/scene_localizer/top_cam/camera_pose_robot_base")
        self.declare_parameter("robot_base_table_pose_topic", "/scene_localizer/table_pose_robot_base")

        # The ball_trajectory_estimator is responsible for all trajectory
        # validity/intersection checks. This node only transforms a valid
        # trajectory end_point from table_frame into robot_base_frame.
        self.declare_parameter("ball_trajectory_topic", "/scene/ball_trajectory_table")
        self.declare_parameter(
            "middle_line_intersection_pose_robot_base_topic",
            "/scene/middle_line_intersection_pose_robot_base",
        )
        self.declare_parameter("publish_middle_line_intersection_pose_robot_base", True)
        self.declare_parameter("trajectory_timeout_sec", 0.5)
        self.declare_parameter("tcp_frame", "right_fr3_hand_tcp")
        self.declare_parameter("intersection_pose_orientation_mode", "tcp_current")

        # Optional TF for the latest 3D ball point. The input is expected as
        # geometry_msgs/PointStamped in table_frame, e.g. from ball_3d_pose_estimator.
        self.declare_parameter("ball_3d_topic", "/scene_localizer/top_cam/ball_3d_table")
        self.declare_parameter("publish_table_ball_3d_tf", True)
        self.declare_parameter("ball_3d_frame", "ball_3d")
        self.declare_parameter("ball_3d_timeout_sec", 0.5)

        self.declare_parameter("window_sec", 5.0)

        # Minimum observations required for every required marker.
        self.declare_parameter("min_observations_per_marker", 5)

        # Empty means all markers in the layout are required.
        self.declare_parameter("required_marker_ids", [])

        # Optional: require at least this many different markers rather than all.
        # Set to 0 to require all required_marker_ids/layout markers.
        self.declare_parameter("min_required_markers", 3)

        self.declare_parameter("max_observations_per_marker", 100)

        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("estimation_mode", "marker_center_alignment")
        self.declare_parameter("top_estimation_mode", "")
        self.declare_parameter("max_candidate_translation_deviation", 0.05)
        self.declare_parameter("max_window_translation_deviation", 0.08)
        self.declare_parameter("max_translation_jump", 0.12)
        self.declare_parameter("max_rotation_jump_deg", 35.0)
        self.declare_parameter("smoothing_alpha", 0.25)

        # PoseStamped topics are always published when estimates exist.
        # TF publication is useful for downstream nodes, but disable it if another
        # node already publishes the same child frames.
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("publish_robot_base_top_camera_tf", True)
        self.declare_parameter("publish_robot_base_table_tf", True)
        self.declare_parameter("publish_top_camera_table_tf", False)
        self.declare_parameter("top_camera_table_tf_child_frame", "top_cam_table_frame")

        self.declare_parameter("debug_log", False)

        self._robot_base_frame = _clean_frame(self.get_parameter("robot_base_frame").value)
        self._calibration_link_frame = _clean_frame(self.get_parameter("calibration_link_frame").value)
        self._top_camera_frame = _clean_frame(self.get_parameter("top_camera_frame").value)
        self._table_frame = _clean_frame(self.get_parameter("table_frame").value)

        marker_layout_yaml = str(self.get_parameter("marker_layout_yaml").value)
        self._state = TopCamState(
            marker_layout=self._load_top_marker_layout(marker_layout_yaml)
        )
        self._state.marker_buffers = {
            marker_id: []
            for marker_id in self._state.marker_layout
        }

        self._calibration_yaml_path = str(self.get_parameter("base_cam_calibration_yaml").value)
        self._calibration_yaml_transform = self._load_calibration_yaml(self._calibration_yaml_path)
        self._last_robot_base_top_camera: Optional[TransformEstimate] = None
        self._last_robot_base_table: Optional[TransformEstimate] = None
        self._latest_ball_trajectory: Optional[BallTrajectory] = None
        self._latest_ball_trajectory_time_sec: Optional[float] = None
        self._latest_ball_3d_point: Optional[PointStamped] = None
        self._latest_ball_3d_point_time_sec: Optional[float] = None

        valid_modes = {"marker_pose_average", "marker_center_alignment"}
        global_mode = str(self.get_parameter("estimation_mode").value)
        top_mode_param = str(self.get_parameter("top_estimation_mode").value)
        self._top_estimation_mode = top_mode_param if top_mode_param else global_mode
        if self._top_estimation_mode not in valid_modes:
            self.get_logger().warn(
                f"Invalid top estimation mode '{self._top_estimation_mode}'. "
                f"Use one of {sorted(valid_modes)}."
            )

        top_detections_topic = str(self.get_parameter("top_detections_topic").value)
        self.create_subscription(
            ArucoDetection,
            top_detections_topic,
            self._handle_top_detection_message,
            10,
        )

        ball_trajectory_topic = str(self.get_parameter("ball_trajectory_topic").value)
        self.create_subscription(
            BallTrajectory,
            ball_trajectory_topic,
            self._handle_ball_trajectory_message,
            10,
        )

        ball_3d_topic = str(self.get_parameter("ball_3d_topic").value)
        self.create_subscription(
            PointStamped,
            ball_3d_topic,
            self._handle_ball_3d_point_message,
            10,
        )

        self._top_camera_table_pose_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("top_camera_table_pose_topic").value),
            10,
        )
        self._robot_base_top_camera_pose_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("robot_base_top_camera_pose_topic").value),
            10,
        )
        self._robot_base_table_pose_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("robot_base_table_pose_topic").value),
            10,
        )
        self._middle_line_intersection_pose_robot_base_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("middle_line_intersection_pose_robot_base_topic").value),
            10,
        )

        publish_rate_hz = max(0.1, float(self.get_parameter("publish_rate_hz").value))
        self._publish_timer = self.create_timer(1.0 / publish_rate_hz, self._publish_timer_callback)

        self.get_logger().info(f"Loaded marker layout YAML: {marker_layout_yaml}")
        self.get_logger().info(
            f"Top-cam-only localization: sub={top_detections_topic}, "
            f"ball_trajectory_topic={ball_trajectory_topic}, "
            f"intersection_pose_pub={self.get_parameter('middle_line_intersection_pose_robot_base_topic').value}, "
            f"ball_3d_topic={ball_3d_topic}, "
            f"ball_3d_frame={self.get_parameter('ball_3d_frame').value}, "
            f"markers={len(self._state.marker_layout)}, "
            f"top_camera_frame={self._top_camera_frame}, "
            f"robot_base_frame={self._robot_base_frame}, "
            f"calibration_link_frame={self._calibration_link_frame}, "
            f"table_frame={self._table_frame}, "
            f"estimation_mode={self._top_estimation_mode}, "
            f"window_sec={self.get_parameter('window_sec').value}, "
            f"min_observations_per_marker={self.get_parameter('min_observations_per_marker').value}, "
            f"min_required_markers={self.get_parameter('min_required_markers').value}"
        )
        if self._calibration_yaml_transform is None:
            self.get_logger().warn(
                "No valid base_cam_calibration_yaml loaded; robot-base transforms will not be published."
            )
        else:
            cal = self._calibration_yaml_transform
            self.get_logger().info(
                f"Loaded calibration YAML: {self._calibration_yaml_path}; "
                f"transform={cal.parent_frame}->{cal.child_frame}"
            )

    def _log_debug(self, message: str) -> None:
        if bool(self.get_parameter("debug_log").value):
            self.get_logger().info(message)

    def _warn_throttled(self, key: str, message: str, throttle_sec: float = 1.0) -> None:
        now_ns = self.get_clock().now().nanoseconds
        last_ns = self._last_warn_time_ns.get(key, 0)
        if now_ns - last_ns >= int(throttle_sec * 1e9):
            self._last_warn_time_ns[key] = now_ns
            self.get_logger().warn(message)

    def _load_top_marker_layout(self, yaml_path: str) -> Dict[int, TransformEstimate]:
        layout_path = Path(yaml_path)
        if not layout_path.exists():
            self.get_logger().warn(f"Marker layout YAML not found: {yaml_path}")
            return {}

        try:
            with layout_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:
            self.get_logger().warn(f"Failed to read marker layout YAML: {exc}")
            return {}

        marker_origin = str(data.get("marker_translation_origin", "bottom_left"))
        physical_marker_size = float(data.get("physical_marker_size", -1.0))
        self.get_logger().info(f"Using physical marker size: {physical_marker_size}")

        # Preferred current format:
        # streams:
        #   top_cam:
        #     markers: {...}
        # Also accepts a flat legacy top-level `markers: {...}`.
        streams_data = data.get("streams", {})
        if "top_cam" in streams_data:
            markers_data = streams_data.get("top_cam", {}).get("markers", {})
        else:
            markers_data = data.get("markers", {})

        parsed: Dict[int, TransformEstimate] = {}
        for marker_id_raw, marker_cfg in markers_data.items():
            try:
                marker_id = int(marker_id_raw)
                translation = np.array(marker_cfg.get("translation", [0.0, 0.0, 0.0]), dtype=float)
                rpy = marker_cfg.get("rpy", [0.0, 0.0, 0.0])
                if translation.shape != (3,) or len(rpy) != 3:
                    raise ValueError("translation/rpy must be length 3")

                rotation = _rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
                if marker_origin in ("bottom_left", "bottom-left", "corner"):
                    corner_to_center_marker = np.array(
                        [0.5 * physical_marker_size, 0.5 * physical_marker_size, 0.0],
                        dtype=float,
                    )
                    translation = translation + corner_to_center_marker
                elif marker_origin != "center":
                    self._warn_throttled(
                        "unknown_marker_origin",
                        f"Unknown marker_translation_origin '{marker_origin}', assuming center",
                        5.0,
                    )

                parsed[marker_id] = TransformEstimate(
                    translation=translation,
                    quaternion_xyzw=_matrix_to_quaternion_xyzw(rotation),
                )
            except Exception as exc:
                self._warn_throttled(
                    f"parse_marker_top_cam_{marker_id_raw}",
                    f"[top_cam] failed to parse marker {marker_id_raw}: {exc}",
                    5.0,
                )

        marker_items = []
        for marker_id in sorted(parsed.keys()):
            t = parsed[marker_id].translation
            marker_items.append(
                f"{marker_id}:({float(t[0]):.4f},{float(t[1]):.4f},{float(t[2]):.4f})"
            )
        self.get_logger().info(
            "Loaded top_cam layout summary: "
            f"count={len(parsed)} marker_translation_origin={marker_origin} "
            f"physical_marker_size={physical_marker_size:.4f} "
            f"translations=[{', '.join(marker_items)}]"
        )
        return parsed

    def _load_calibration_yaml(self, yaml_path: str) -> Optional[CalibrationYamlTransform]:
        if not yaml_path:
            return None

        path = Path(yaml_path)
        if not path.exists():
            self.get_logger().warn(f"Calibration YAML not found: {yaml_path}")
            return None

        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:
            self.get_logger().warn(f"Failed to read calibration YAML '{yaml_path}': {exc}")
            return None

        try:
            parent_frame = _clean_frame(data.get("parent_frame", ""))
            child_frame = _clean_frame(data.get("child_frame", ""))
            if not parent_frame or not child_frame:
                raise ValueError("YAML must contain parent_frame and child_frame")

            translation = self._parse_translation(data.get("translation", [0.0, 0.0, 0.0]))
            if "quaternion" in data:
                quaternion = self._parse_quaternion(data.get("quaternion"))
            elif "quaternion_xyzw" in data:
                quaternion = self._parse_quaternion(data.get("quaternion_xyzw"))
            elif "rotation_matrix" in data:
                rotation = np.array(data.get("rotation_matrix"), dtype=float)
                if rotation.shape != (3, 3):
                    raise ValueError("rotation_matrix must be 3x3")
                quaternion = _matrix_to_quaternion_xyzw(rotation)
            else:
                raise ValueError("YAML must contain quaternion, quaternion_xyzw, or rotation_matrix")

            return CalibrationYamlTransform(
                parent_frame=parent_frame,
                child_frame=child_frame,
                estimate=TransformEstimate(
                    translation=translation,
                    quaternion_xyzw=_normalize_quaternion_xyzw(quaternion),
                ),
            )
        except Exception as exc:
            self.get_logger().warn(f"Invalid calibration YAML '{yaml_path}': {exc}")
            return None

    @staticmethod
    def _parse_translation(raw: Any) -> np.ndarray:
        if isinstance(raw, dict):
            return np.array([raw["x"], raw["y"], raw["z"]], dtype=float)
        arr = np.array(raw, dtype=float)
        if arr.shape != (3,):
            raise ValueError("translation must be dict {x,y,z} or length-3 list")
        return arr

    @staticmethod
    def _parse_quaternion(raw: Any) -> np.ndarray:
        if isinstance(raw, dict):
            return np.array([raw["x"], raw["y"], raw["z"], raw["w"]], dtype=float)
        arr = np.array(raw, dtype=float)
        if arr.shape != (4,):
            raise ValueError("quaternion must be dict {x,y,z,w} or length-4 list")
        return arr

    def _handle_ball_trajectory_message(self, msg: BallTrajectory) -> None:
        self._latest_ball_trajectory = msg
        self._latest_ball_trajectory_time_sec = self._stamp_to_sec(msg)

    def _handle_ball_3d_point_message(self, msg: PointStamped) -> None:
        self._latest_ball_3d_point = msg
        self._latest_ball_3d_point_time_sec = self._stamp_to_sec(msg)
        self._publish_table_ball_3d_tf_if_available()

    def _handle_top_detection_message(self, msg: ArucoDetection) -> None:
        markers = self._extract_markers_from_msg(msg)
        stamp_sec = self._stamp_to_sec(msg)

        if not markers:
            self._warn_throttled(
                "no_markers_top_cam",
                "[top_cam] no markers in ArUco detection message",
                1.0,
            )
            return

        if not self._state.marker_layout:
            self._warn_throttled(
                "no_layout_top_cam",
                "[top_cam] marker layout empty; skipping detections",
                2.0,
            )
            return

        detection_frame = _clean_frame(getattr(msg.header, "frame_id", ""))
        if detection_frame and detection_frame != self._top_camera_frame:
            self._warn_throttled(
                "top_detection_frame_mismatch",
                f"[top_cam] detection frame_id='{detection_frame}' differs from "
                f"top_camera_frame='{self._top_camera_frame}'",
                2.0,
            )

        added_ids: List[int] = []

        for marker in markers:
            marker_id = self._extract_marker_id(marker)
            if marker_id is None:
                continue

            if marker_id not in self._state.marker_layout:
                continue

            pose = self._extract_pose_from_marker(marker)
            if pose is None:
                continue

            translation = np.array(
                [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ],
                dtype=float,
            )
            quaternion = _normalize_quaternion_xyzw(
                np.array(
                    [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ],
                    dtype=float,
                )
            )

            if not np.all(np.isfinite(translation)):
                continue

            observation = TimedMarkerObservation(
                stamp_sec=stamp_sec,
                marker_id=marker_id,
                camera_marker=TransformEstimate(
                    translation=translation,
                    quaternion_xyzw=quaternion,
                ),
            )

            self._state.marker_buffers.setdefault(marker_id, []).append(observation)
            added_ids.append(marker_id)

        self._prune_marker_buffers()

        self._log_debug(
            f"[top_cam] stored marker observations ids={added_ids}; "
            f"counts={self._marker_buffer_counts_string()}"
        )

    def _required_marker_ids(self) -> List[int]:
        configured = list(self.get_parameter("required_marker_ids").value)

        if configured:
            return [
                int(marker_id)
                for marker_id in configured
                if int(marker_id) in self._state.marker_layout
            ]

        return sorted(self._state.marker_layout.keys())

    def _get_valid_marker_buffers(
        self,
    ) -> Optional[Dict[int, List[TimedMarkerObservation]]]:
        min_observations = max(
            1,
            int(self.get_parameter("min_observations_per_marker").value),
        )

        required_ids = self._required_marker_ids()
        valid: Dict[int, List[TimedMarkerObservation]] = {}

        for marker_id in required_ids:
            observations = self._state.marker_buffers.get(marker_id, [])
            if len(observations) >= min_observations:
                valid[marker_id] = observations

        min_required = int(self.get_parameter("min_required_markers").value)

        if min_required <= 0:
            if len(valid) != len(required_ids):
                missing = [
                    marker_id
                    for marker_id in required_ids
                    if marker_id not in valid
                ]

                self._warn_throttled(
                    "insufficient_per_marker_observations",
                    "[top_cam] window incomplete: "
                    f"need {min_observations} observations per marker; "
                    f"counts=[{self._marker_buffer_counts_string()}], "
                    f"insufficient={missing}",
                    1.0,
                )
                return None
        else:
            if len(valid) < min_required:
                self._warn_throttled(
                    "insufficient_valid_markers",
                    "[top_cam] too few sufficiently populated marker buffers: "
                    f"{len(valid)} < {min_required}; "
                    f"counts=[{self._marker_buffer_counts_string()}]",
                    1.0,
                )
                return None

        return valid

    def _estimate_window_marker_pose_average(
        self,
        buffers: Dict[int, List[TimedMarkerObservation]],
    ) -> Optional[TransformEstimate]:
        candidate_translations: List[np.ndarray] = []
        candidate_quaternions: List[np.ndarray] = []

        for marker_id, observations in buffers.items():
            table_marker = self._state.marker_layout[marker_id]
            t_table_marker = _estimate_to_matrix(table_marker)
            t_marker_table = _invert_matrix(t_table_marker)

            for observation in observations:
                t_camera_marker = _estimate_to_matrix(
                    observation.camera_marker
                )

                t_camera_table = t_camera_marker @ t_marker_table
                candidate = _matrix_to_estimate(t_camera_table)

                candidate_translations.append(candidate.translation)
                candidate_quaternions.append(candidate.quaternion_xyzw)

        if not candidate_translations:
            return None

        max_deviation = max(
            0.0,
            float(
                self.get_parameter(
                    "max_window_translation_deviation"
                ).value
            ),
        )

        keep_idx = self._reject_translation_outliers(
            candidate_translations,
            max_deviation,
        )

        if not keep_idx:
            self._warn_throttled(
                "window_candidate_reject_all",
                "[top_cam] all full-window pose candidates rejected",
                1.0,
            )
            return None

        translation = np.mean(
            np.stack(
                [candidate_translations[i] for i in keep_idx],
                axis=0,
            ),
            axis=0,
        )

        quaternion = _average_quaternions_xyzw(
            [candidate_quaternions[i] for i in keep_idx]
        )

        return TransformEstimate(
            translation=translation,
            quaternion_xyzw=quaternion,
        )

    def _estimate_window_marker_center_alignment(
        self,
        buffers: Dict[int, List[TimedMarkerObservation]],
    ) -> Optional[TransformEstimate]:
        table_points: List[np.ndarray] = []
        camera_points: List[np.ndarray] = []

        for marker_id, observations in buffers.items():
            if not observations:
                continue

            marker_camera_points = np.stack(
                [
                    obs.camera_marker.translation
                    for obs in observations
                ],
                axis=0,
            )

            camera_center_mean = np.mean(
                marker_camera_points,
                axis=0,
            )

            table_points.append(
                self._state.marker_layout[marker_id].translation
            )
            camera_points.append(camera_center_mean)

        if len(camera_points) < 3:
            self._warn_throttled(
                "too_few_markers_window_alignment",
                "[top_cam] full-window center alignment needs at least "
                f"3 sufficiently populated markers; got {len(camera_points)}",
                1.0,
            )
            return None

        p = np.stack(table_points, axis=0)
        q = np.stack(camera_points, axis=0)

        p_mean = np.mean(p, axis=0)
        q_mean = np.mean(q, axis=0)

        p_centered = p - p_mean
        q_centered = q - q_mean

        h = p_centered.T @ q_centered
        u, _, vt = np.linalg.svd(h)

        rotation = vt.T @ u.T

        if float(np.linalg.det(rotation)) < 0.0:
            vt[-1, :] *= -1.0
            rotation = vt.T @ u.T

        translation = q_mean - rotation @ p_mean

        predicted = (rotation @ p.T).T + translation
        residuals = q - predicted
        rms = float(
            np.sqrt(
                np.mean(
                    np.sum(residuals * residuals, axis=1)
                )
            )
        )

        max_deviation = max(
            0.0,
            float(
                self.get_parameter(
                    "max_window_translation_deviation"
                ).value
            ),
        )

        if rms > max_deviation:
            self._warn_throttled(
                "window_alignment_rms_rejected",
                "[top_cam] rejected full-window center alignment: "
                f"rms={rms:.4f}m > {max_deviation:.4f}m",
                1.0,
            )
            return None

        return TransformEstimate(
            translation=translation,
            quaternion_xyzw=_matrix_to_quaternion_xyzw(rotation),
        )

    def _compute_new_top_camera_table_from_window(
        self,
    ) -> Optional[TransformEstimate]:
        buffers = self._get_valid_marker_buffers()
        if buffers is None:
            return None

        if self._top_estimation_mode == "marker_pose_average":
            estimate = self._estimate_window_marker_pose_average(buffers)
        elif self._top_estimation_mode == "marker_center_alignment":
            estimate = self._estimate_window_marker_center_alignment(buffers)
        else:
            self._warn_throttled(
                "invalid_mode_top_cam",
                f"[top_cam] invalid estimation_mode='{self._top_estimation_mode}', skipping estimates",
                2.0,
            )
            return None

        if estimate is None:
            return None

        if self._state.last_valid_estimate is not None:
            max_translation_jump = max(
                0.0,
                float(self.get_parameter("max_translation_jump").value),
            )
            max_rotation_jump_deg = max(
                0.0,
                float(self.get_parameter("max_rotation_jump_deg").value),
            )

            dt = float(
                np.linalg.norm(
                    estimate.translation
                    - self._state.last_valid_estimate.translation
                )
            )
            dq = _quaternion_angular_distance_deg(
                estimate.quaternion_xyzw,
                self._state.last_valid_estimate.quaternion_xyzw,
            )

            if (
                dt > max_translation_jump
                or dq > max_rotation_jump_deg
            ):
                self._warn_throttled(
                    "window_pose_jump_rejected",
                    "[top_cam] rejected complete-window pose jump: "
                    f"dt={dt:.3f}m dq={dq:.1f}deg",
                    1.0,
                )
                return None

        return estimate

    def _publish_timer_callback(self) -> None:
        self._prune_marker_buffers()

        new_estimate = self._compute_new_top_camera_table_from_window()

        if new_estimate is not None:
            smoothing_alpha = min(
                1.0,
                max(
                    0.0,
                    float(self.get_parameter("smoothing_alpha").value),
                ),
            )

            previous = self._state.last_valid_estimate

            if previous is not None:
                translation = (
                    (1.0 - smoothing_alpha) * previous.translation
                    + smoothing_alpha * new_estimate.translation
                )
                quaternion = _blend_quaternions_xyzw(
                    previous.quaternion_xyzw,
                    new_estimate.quaternion_xyzw,
                    smoothing_alpha,
                )
                new_estimate = TransformEstimate(
                    translation=translation,
                    quaternion_xyzw=quaternion,
                )

            self._state.last_valid_estimate = new_estimate

            all_stamps = [
                obs.stamp_sec
                for observations in self._state.marker_buffers.values()
                for obs in observations
            ]
            if all_stamps:
                self._state.last_valid_source_stamp_sec = max(all_stamps)

        top_camera_table = self._state.last_valid_estimate
        if top_camera_table is None:
            self._warn_throttled(
                "no_valid_pose_yet",
                "[top_cam] no complete valid table pose has been obtained yet",
                1.0,
            )
            return

        stamp = self.get_clock().now().to_msg()
        self._publish_pose(
            self._top_camera_table_pose_pub,
            top_camera_table,
            self._top_camera_frame,
            stamp,
        )

        robot_base_top_camera = self._compute_robot_base_top_camera()
        if robot_base_top_camera is None:
            return

        self._last_robot_base_top_camera = robot_base_top_camera

        self._publish_pose(
            self._robot_base_top_camera_pose_pub,
            robot_base_top_camera,
            self._robot_base_frame,
            stamp,
        )

        t_base_camera = _estimate_to_matrix(robot_base_top_camera)
        t_camera_table = _estimate_to_matrix(top_camera_table)

        robot_base_table = _matrix_to_estimate(
            t_base_camera @ t_camera_table
        )
        self._last_robot_base_table = robot_base_table

        self._publish_pose(
            self._robot_base_table_pose_pub,
            robot_base_table,
            self._robot_base_frame,
            stamp,
        )

        self._publish_middle_line_intersection_pose_if_available(
            robot_base_table
        )
        self._publish_table_ball_3d_tf_if_available()

        self._publish_tfs_if_enabled(
            stamp,
            robot_base_top_camera,
            top_camera_table,
            robot_base_table,
        )

    def _compute_robot_base_top_camera(self) -> Optional[TransformEstimate]:
        cal = self._calibration_yaml_transform
        if cal is None:
            return None

        parent = _clean_frame(cal.parent_frame)
        child = _clean_frame(cal.child_frame)
        robot_base = self._robot_base_frame
        top_cam = self._top_camera_frame
        calibration_link = self._calibration_link_frame
        t_parent_child = _estimate_to_matrix(cal.estimate)

        # Case 1: calibration file already contains T_robot_base_top_camera.
        if parent == robot_base and child == top_cam:
            return cal.estimate

        # Case 2: calibration file contains inverse, T_top_camera_robot_base.
        if parent == top_cam and child == robot_base:
            return _matrix_to_estimate(_invert_matrix(t_parent_child))

        # Case 3: calibration file contains T_calibration_link_top_camera.
        if parent == calibration_link and child == top_cam:
            t_base_link = self._lookup_robot_base_to_calibration_link()
            if t_base_link is None:
                return None
            return _matrix_to_estimate(t_base_link @ t_parent_child)

        # Case 4: calibration file contains T_top_camera_calibration_link.
        if parent == top_cam and child == calibration_link:
            t_base_link = self._lookup_robot_base_to_calibration_link()
            if t_base_link is None:
                return None
            t_link_cam = _invert_matrix(t_parent_child)
            return _matrix_to_estimate(t_base_link @ t_link_cam)

        self._warn_throttled(
            "unsupported_calibration_yaml_frames",
            "Unsupported calibration YAML frames: "
            f"{parent}->{child}. Expected one of: "
            f"{robot_base}->{top_cam}, {top_cam}->{robot_base}, "
            f"{calibration_link}->{top_cam}, {top_cam}->{calibration_link}.",
            5.0,
        )
        return None

    def _lookup_robot_base_to_calibration_link(self) -> Optional[np.ndarray]:
        try:
            msg = self._tf_buffer.lookup_transform(
                self._robot_base_frame,
                self._calibration_link_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException as exc:
            self._warn_throttled(
                "lookup_base_to_calibration_link_failed",
                f"Failed to lookup TF {self._robot_base_frame}->{self._calibration_link_frame}: {exc}",
                1.0,
            )
            return None
        return _transform_stamped_to_matrix(msg)

    def _publish_pose(
        self,
        publisher: Any,
        estimate: TransformEstimate,
        frame_id: str,
        stamp: Any,
    ) -> None:
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.position.x = float(estimate.translation[0])
        pose_msg.pose.position.y = float(estimate.translation[1])
        pose_msg.pose.position.z = float(estimate.translation[2])
        pose_msg.pose.orientation.x = float(estimate.quaternion_xyzw[0])
        pose_msg.pose.orientation.y = float(estimate.quaternion_xyzw[1])
        pose_msg.pose.orientation.z = float(estimate.quaternion_xyzw[2])
        pose_msg.pose.orientation.w = float(estimate.quaternion_xyzw[3])
        publisher.publish(pose_msg)

    def _publish_table_ball_3d_tf_if_available(self) -> None:
        if not bool(self.get_parameter("publish_tf").value):
            return
        if not bool(self.get_parameter("publish_table_ball_3d_tf").value):
            return

        ball_msg = self._latest_ball_3d_point
        if ball_msg is None or self._latest_ball_3d_point_time_sec is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        timeout_sec = max(0.0, float(self.get_parameter("ball_3d_timeout_sec").value))
        age_sec = now_sec - self._latest_ball_3d_point_time_sec
        if age_sec > timeout_sec:
            self._warn_throttled(
                "ball_3d_tf_stale",
                f"Not publishing table->ball_3d TF: ball point stale "
                f"age={age_sec:.3f}s timeout={timeout_sec:.3f}s",
                1.0,
            )
            return

        ball_frame = _clean_frame(getattr(getattr(ball_msg, "header", None), "frame_id", ""))
        accepted_frames = {_clean_frame(self._table_frame), "table", "table_frame"}
        if ball_frame and ball_frame not in accepted_frames:
            self._warn_throttled(
                "ball_3d_tf_frame_mismatch",
                f"Not publishing table->ball_3d TF: ball point frame_id='{ball_frame}' "
                f"does not match table_frame='{self._table_frame}'",
                1.0,
            )
            return

        p_ball_table = np.array(
            [
                float(getattr(ball_msg.point, "x", float("nan"))),
                float(getattr(ball_msg.point, "y", float("nan"))),
                float(getattr(ball_msg.point, "z", float("nan"))),
            ],
            dtype=float,
        )
        if not _is_finite_vector(p_ball_table):
            self._warn_throttled(
                "ball_3d_tf_non_finite",
                f"Not publishing table->ball_3d TF: non-finite point={p_ball_table.tolist()}",
                1.0,
            )
            return

        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self._table_frame
        tf_msg.child_frame_id = _clean_frame(self.get_parameter("ball_3d_frame").value)
        tf_msg.transform.translation.x = float(p_ball_table[0])
        tf_msg.transform.translation.y = float(p_ball_table[1])
        tf_msg.transform.translation.z = float(p_ball_table[2])
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = 0.0
        tf_msg.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(tf_msg)

    def _publish_middle_line_intersection_pose_if_available(
        self,
        robot_base_table: TransformEstimate,
    ) -> None:
        if not bool(self.get_parameter("publish_middle_line_intersection_pose_robot_base").value):
            return

        traj = self._latest_ball_trajectory
        if traj is None or self._latest_ball_trajectory_time_sec is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        timeout_sec = max(0.0, float(self.get_parameter("trajectory_timeout_sec").value))
        age_sec = now_sec - self._latest_ball_trajectory_time_sec
        if age_sec > timeout_sec:
            self._warn_throttled(
                "middle_intersection_trajectory_stale",
                f"Not publishing middle-line intersection pose: trajectory stale "
                f"age={age_sec:.3f}s timeout={timeout_sec:.3f}s",
                1.0,
            )
            return

        if not bool(getattr(traj, "valid", False)):
            return

        traj_frame = _clean_frame(getattr(getattr(traj, "header", None), "frame_id", ""))
        accepted_frames = {_clean_frame(self._table_frame), "table", "table_frame"}
        if traj_frame and traj_frame not in accepted_frames:
            self._warn_throttled(
                "middle_intersection_traj_frame_mismatch",
                f"Not publishing middle-line intersection pose: trajectory frame_id='{traj_frame}' "
                f"does not match table_frame='{self._table_frame}'",
                1.0,
            )
            return

        end_point = getattr(traj, "end_point", None)
        if end_point is None:
            self._warn_throttled(
                "middle_intersection_missing_end_point",
                "Not publishing middle-line intersection pose: BallTrajectory.end_point missing",
                1.0,
            )
            return

        p_hit_table = np.array(
            [
                float(getattr(end_point, "x", float("nan"))),
                float(getattr(end_point, "y", float("nan"))),
                float(getattr(end_point, "z", float("nan"))),
            ],
            dtype=float,
        )
        if not _is_finite_vector(p_hit_table):
            self._warn_throttled(
                "middle_intersection_non_finite",
                f"Not publishing middle-line intersection pose: non-finite end_point={p_hit_table.tolist()}",
                1.0,
            )
            return

        T_base_table = _estimate_to_matrix(robot_base_table)
        p_hit_base = _transform_point(T_base_table, p_hit_table)
        if not _is_finite_vector(p_hit_base):
            self._warn_throttled(
                "middle_intersection_base_non_finite",
                f"Not publishing middle-line intersection pose: transformed point non-finite={p_hit_base.tolist()}",
                1.0,
            )
            return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self._robot_base_frame
        pose_msg.pose.position.x = float(p_hit_base[0])
        pose_msg.pose.position.y = float(p_hit_base[1])
        pose_msg.pose.position.z = float(p_hit_base[2])

        q_base_target = self._intersection_pose_quaternion_xyzw()
        pose_msg.pose.orientation.x = float(q_base_target[0])
        pose_msg.pose.orientation.y = float(q_base_target[1])
        pose_msg.pose.orientation.z = float(q_base_target[2])
        pose_msg.pose.orientation.w = float(q_base_target[3])

        self._middle_line_intersection_pose_robot_base_pub.publish(pose_msg)

    def _intersection_pose_quaternion_xyzw(self) -> np.ndarray:
        mode = str(self.get_parameter("intersection_pose_orientation_mode").value).strip().lower()
        if mode in {"identity", "base_identity", "robot_base_identity"}:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        if mode not in {"tcp_current", "current_tcp", "tcp"}:
            self._warn_throttled(
                "invalid_intersection_pose_orientation_mode",
                f"Invalid intersection_pose_orientation_mode='{mode}', using tcp_current",
                2.0,
            )

        tcp_frame = _clean_frame(self.get_parameter("tcp_frame").value)
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._robot_base_frame,
                tcp_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException as exc:
            self._warn_throttled(
                "lookup_base_to_tcp_for_intersection_pose_failed",
                f"Failed to lookup TF {self._robot_base_frame}->{tcp_frame} for intersection pose orientation; "
                f"using identity orientation: {exc}",
                1.0,
            )
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        q = tf_msg.transform.rotation
        return _normalize_quaternion_xyzw(
            np.array([q.x, q.y, q.z, q.w], dtype=float)
        )

    def _publish_tfs_if_enabled(
        self,
        stamp: Any,
        robot_base_top_camera: TransformEstimate,
        top_camera_table: TransformEstimate,
        robot_base_table: TransformEstimate,
    ) -> None:
        if not bool(self.get_parameter("publish_tf").value):
            return

        transforms: List[TransformStamped] = []

        if bool(self.get_parameter("publish_robot_base_top_camera_tf").value):
            transforms.append(
                self._make_tf_msg(
                    robot_base_top_camera,
                    self._robot_base_frame,
                    self._top_camera_frame,
                    stamp,
                )
            )

        if bool(self.get_parameter("publish_robot_base_table_tf").value):
            transforms.append(
                self._make_tf_msg(
                    robot_base_table,
                    self._robot_base_frame,
                    self._table_frame,
                    stamp,
                )
            )

        if bool(self.get_parameter("publish_top_camera_table_tf").value):
            transforms.append(
                self._make_tf_msg(
                    top_camera_table,
                    self._top_camera_frame,
                    _clean_frame(self.get_parameter("top_camera_table_tf_child_frame").value),
                    stamp,
                )
            )

        if transforms:
            self._tf_broadcaster.sendTransform(transforms)

    @staticmethod
    def _make_tf_msg(
        estimate: TransformEstimate,
        parent_frame: str,
        child_frame: str,
        stamp: Any,
    ) -> TransformStamped:
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = parent_frame
        tf_msg.child_frame_id = child_frame
        tf_msg.transform.translation.x = float(estimate.translation[0])
        tf_msg.transform.translation.y = float(estimate.translation[1])
        tf_msg.transform.translation.z = float(estimate.translation[2])
        tf_msg.transform.rotation.x = float(estimate.quaternion_xyzw[0])
        tf_msg.transform.rotation.y = float(estimate.quaternion_xyzw[1])
        tf_msg.transform.rotation.z = float(estimate.quaternion_xyzw[2])
        tf_msg.transform.rotation.w = float(estimate.quaternion_xyzw[3])
        return tf_msg

    def _marker_buffer_counts_string(self) -> str:
        return ", ".join(
            f"{marker_id}:{len(self._state.marker_buffers.get(marker_id, []))}"
            for marker_id in sorted(self._state.marker_layout)
        )

    def _prune_marker_buffers(self) -> None:
        window_sec = max(
            0.1,
            float(self.get_parameter("window_sec").value),
        )
        max_per_marker = max(
            1,
            int(self.get_parameter("max_observations_per_marker").value),
        )

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        cutoff_sec = now_sec - window_sec

        for marker_id in self._state.marker_layout:
            observations = self._state.marker_buffers.setdefault(marker_id, [])

            observations = [
                obs
                for obs in observations
                if obs.stamp_sec >= cutoff_sec
            ]

            if len(observations) > max_per_marker:
                observations = observations[-max_per_marker:]

            self._state.marker_buffers[marker_id] = observations

    @staticmethod
    def _reject_translation_outliers(translations: List[np.ndarray], max_deviation: float) -> List[int]:
        if not translations:
            return []
        arr = np.stack(translations, axis=0)
        median = np.median(arr, axis=0)
        distances = np.linalg.norm(arr - median, axis=1)
        return [int(i) for i, d in enumerate(distances) if float(d) <= max_deviation]

    @staticmethod
    def _extract_markers_from_msg(msg: ArucoDetection) -> List[Any]:
        if hasattr(msg, "markers") and getattr(msg, "markers") is not None:
            return list(getattr(msg, "markers"))
        if hasattr(msg, "detections") and getattr(msg, "detections") is not None:
            return list(getattr(msg, "detections"))
        if (hasattr(msg, "marker_id") or hasattr(msg, "id")) and hasattr(msg, "pose"):
            return [msg]
        return []

    @staticmethod
    def _extract_marker_id(marker: Any) -> Optional[int]:
        for attr in ("marker_id", "id"):
            if hasattr(marker, attr):
                try:
                    return int(getattr(marker, attr))
                except Exception:
                    return None
        return None

    @staticmethod
    def _extract_pose_from_marker(marker: Any) -> Optional[Pose]:
        pose_field = getattr(marker, "pose", None)
        if pose_field is None:
            return None

        if isinstance(pose_field, Pose):
            return pose_field

        if hasattr(pose_field, "pose"):
            nested = pose_field.pose
            if isinstance(nested, Pose):
                return nested
            if hasattr(nested, "pose") and isinstance(nested.pose, Pose):
                return nested.pose

        if (
            hasattr(pose_field, "position")
            and hasattr(pose_field, "orientation")
            and hasattr(pose_field.position, "x")
            and hasattr(pose_field.position, "y")
            and hasattr(pose_field.position, "z")
            and hasattr(pose_field.orientation, "x")
            and hasattr(pose_field.orientation, "y")
            and hasattr(pose_field.orientation, "z")
            and hasattr(pose_field.orientation, "w")
        ):
            pose = Pose()
            pose.position.x = float(pose_field.position.x)
            pose.position.y = float(pose_field.position.y)
            pose.position.z = float(pose_field.position.z)
            pose.orientation.x = float(pose_field.orientation.x)
            pose.orientation.y = float(pose_field.orientation.y)
            pose.orientation.z = float(pose_field.orientation.z)
            pose.orientation.w = float(pose_field.orientation.w)
            return pose

        return None

    def _stamp_to_sec(self, msg: Any) -> float:
        header = getattr(msg, "header", None)
        if header is not None and hasattr(header, "stamp"):
            stamp = header.stamp
            sec = float(getattr(stamp, "sec", 0.0))
            nanosec = float(getattr(stamp, "nanosec", 0.0))
            stamp_sec = sec + nanosec * 1e-9
            if stamp_sec > 0.0:
                return stamp_sec
        return self.get_clock().now().nanoseconds * 1e-9


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = SceneLocalizerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
