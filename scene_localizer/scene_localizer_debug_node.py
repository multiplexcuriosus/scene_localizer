#!/usr/bin/env python3

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
import rclpy.time
from aruco_opencv_msgs.msg import ArucoDetection
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Pose, PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from fr3_husky_msgs.msg import MiddleLine
from scene_localizer.msg import BallTrajectory
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener


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


def _pose_to_transform(pose: Pose) -> np.ndarray:
    """Return T_parent_child from a geometry_msgs/Pose.

    Convention used in this node: T_parent_child maps child-frame coordinates
    into parent-frame coordinates. Therefore:
    - top_table_pose_topic is interpreted as T_top_camera_table.
    - robot_base_table_pose_topic is interpreted as T_robot_base_table.
    """
    t = np.array(
        [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ],
        dtype=float,
    )
    q = np.array(
        [
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ],
        dtype=float,
    )
    T = np.eye(4, dtype=float)
    T[:3, :3] = _quaternion_to_matrix_xyzw(q)
    T[:3, 3] = t
    return T


def _transform_stamped_to_matrix(transform: TransformStamped) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    pose = Pose()
    pose.position.x = t.x
    pose.position.y = t.y
    pose.position.z = t.z
    pose.orientation.x = q.x
    pose.orientation.y = q.y
    pose.orientation.z = q.z
    pose.orientation.w = q.w
    return _pose_to_transform(pose)


def _invert_transform(T_parent_child: np.ndarray) -> np.ndarray:
    T_child_parent = np.eye(4, dtype=float)
    R_parent_child = T_parent_child[:3, :3]
    t_parent_child = T_parent_child[:3, 3]
    T_child_parent[:3, :3] = R_parent_child.T
    T_child_parent[:3, 3] = -(R_parent_child.T @ t_parent_child)
    return T_child_parent


def _transform_point(T_parent_child: np.ndarray, p_child: np.ndarray) -> np.ndarray:
    return T_parent_child[:3, :3] @ p_child + T_parent_child[:3, 3]


def _is_finite_vector(v: np.ndarray) -> bool:
    return bool(v.shape == (3,) and np.all(np.isfinite(v)))


@dataclass
class TopCameraState:
    name: str
    image_topic: str
    camera_info_topic: str
    detections_topic: str
    table_pose_topic: str
    latest_camera_info: Optional[CameraInfo] = None
    latest_detections: Optional[ArucoDetection] = None
    latest_detection_time_sec: Optional[float] = None
    latest_table_pose: Optional[PoseStamped] = None
    latest_table_pose_time_sec: Optional[float] = None


class SceneLocalizerDebugNode(Node):
    def __init__(self) -> None:
        super().__init__("scene_localizer_debug")

        self._bridge = CvBridge()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._last_warn_time_ns: Dict[str, int] = {}
        self._last_debug_time_ns: Dict[str, int] = {}
        self._text_y = 18
        self._perf_last_report_monotonic = time.monotonic()
        self._perf_window_count = 0
        self._perf_window_sums_ms: Dict[str, float] = {}
        self._perf_window_max_ms: Dict[str, float] = {}

        self._latest_image_lock = threading.Lock()
        self._latest_image_msg: Optional[Image] = None
        self._latest_image_stamp_ns: Optional[int] = None
        self._last_rendered_image_stamp_ns: Optional[int] = None

        self._camera_K_native: Optional[np.ndarray] = None
        self._camera_D: Optional[np.ndarray] = None
        self._camera_projection_is_rectified = False
        self._zero_rvec = np.zeros((3, 1), dtype=np.float64)
        self._zero_tvec = np.zeros((3, 1), dtype=np.float64)

        self.declare_parameter("top_image_topic", "/top_cam/camera/color/image_raw")
        self.declare_parameter("top_camera_info_topic", "/top_cam/camera/color/camera_info")
        self.declare_parameter("top_detections_topic", "/aruco_top_cam/aruco_detections")
        self.declare_parameter("top_debug_image_topic", "/scene_localizer_debug/top_cam/debug_image")
        self.declare_parameter("ball_2d_px_topic", "/ball_tracker2/ball_2d_px")

        # Publishes/consumes T_top_camera_table:
        # pose of table_frame expressed in top_camera_frame.
        self.declare_parameter("top_table_pose_topic", "/scene_localizer/top_cam/table_pose_camera")

        # Ball trajectory is expressed in table_frame.
        self.declare_parameter("ball_trajectory_topic", "/scene/ball_trajectory_table")

        # Publishes/consumes T_robot_base_table:
        # pose of table_frame expressed in robot_base_frame.
        self.declare_parameter("robot_base_table_pose_topic", "/scene_localizer/table_pose_robot_base")

        self.declare_parameter("robot_base_frame", "base")
        self.declare_parameter("top_camera_frame", "camera_color_optical_frame")
        self.declare_parameter("table_frame", "table_frame")
        self.declare_parameter("tcp_frame", "right_fr3_hand_tcp")

        # For real-time debug rendering, keep TF lookup non-blocking by default.
        self.declare_parameter("tf_lookup_timeout_sec", 0.0)
        self.declare_parameter("allow_base_table_fallback_from_tf", True)

        self.declare_parameter("physical_marker_size", -1.0)
        self.declare_parameter("axis_length", 0.08)
        self.declare_parameter("detection_timeout_sec", 0.3)
        self.declare_parameter("publish_debug_images", True)
        self.declare_parameter("debug_render_rate_hz", 30.0)
        self.declare_parameter("log_render_timing", False)
        self.declare_parameter("draw_debug_text", False)

        self.declare_parameter("table_shortedge", 0.6)
        self.declare_parameter("table_longedge", 1.2)
        self.declare_parameter("table_edge_z", 0.0)
        self.declare_parameter("table_pose_timeout_sec", 1.0)
        self.declare_parameter("robot_base_table_pose_timeout_sec", 1.0)
        self.declare_parameter("draw_table_rectangle", True)

        self.declare_parameter("trajectory_timeout_sec", 0.5)
        self.declare_parameter("draw_trajectory_debug_text", True)
        self.declare_parameter("trajectory_line_thickness", 3)
        self.declare_parameter("trajectory_start_radius_px", 5)
        self.declare_parameter("trajectory_end_radius_px", 7)

        # Authoritative captured middle-line state published by
        # trajectory_executor (white-eth). Transient-local so a debug node
        # started after capture still receives the latest state immediately.
        self.declare_parameter(
            "middle_line_state_topic",
            "/trajectory_executor/middle_line_state",
        )

        self.declare_parameter("draw_tcp_middle_line", True)
        # Obsolete for the captured middle-line overlay: geometry now comes
        # entirely from the cached MiddleLine message (center/direction/
        # half_length). Retained only for launch-file compatibility; not read
        # by _draw_captured_middle_line_overlay().
        self.declare_parameter("ball_radius", 0.0325)
        self.declare_parameter("tcp_middle_line_length", 0.3)
        self.declare_parameter("tcp_middle_line_half_length", 0.3)
        self.declare_parameter("tcp_middle_line_thickness", 3)
        self.declare_parameter("draw_trajectory_extrapolation", True)
        self.declare_parameter("trajectory_intersection_mode", "tcp_xz_plane")
        self.declare_parameter("allow_backward_trajectory_intersection", False)
        self.declare_parameter("middle_line_intersection_max_distance", 0.02)
        self.declare_parameter("trajectory_intersection_radius_px", 7)
        self.declare_parameter("ball_2d_overlay_timeout_sec", 0.25)

        self._latest_ball_trajectory: Optional[BallTrajectory] = None
        self._latest_ball_trajectory_time_sec: Optional[float] = None
        self._latest_robot_base_table_pose: Optional[PoseStamped] = None
        self._latest_robot_base_table_pose_time_sec: Optional[float] = None
        self._latest_ball_2d_px: Optional[PointStamped] = None
        self._latest_ball_2d_px_time_sec: Optional[float] = None

        # Authoritative captured middle-line state (persistent scene state).
        # No freshness timeout is applied: once a valid state is received it
        # remains displayed until a newer valid state replaces it.
        self._latest_middle_line: Optional[MiddleLine] = None
        self._latest_middle_line_revision: Optional[int] = None

        self._top = TopCameraState(
            name="top_cam",
            image_topic=str(self.get_parameter("top_image_topic").value),
            camera_info_topic=str(self.get_parameter("top_camera_info_topic").value),
            detections_topic=str(self.get_parameter("top_detections_topic").value),
            table_pose_topic=str(self.get_parameter("top_table_pose_topic").value),
        )

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._debug_pub = self.create_publisher(
            Image,
            str(self.get_parameter("top_debug_image_topic").value),
            image_qos,
        )

        self.create_subscription(
            CameraInfo,
            self._top.camera_info_topic,
            self._camera_info_callback,
            10,
        )
        self.create_subscription(
            ArucoDetection,
            self._top.detections_topic,
            self._detections_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            self._top.table_pose_topic,
            self._table_pose_callback,
            10,
        )
        self.create_subscription(
            Image,
            self._top.image_topic,
            self._image_callback,
            image_qos,
        )
        self.create_subscription(
            BallTrajectory,
            str(self.get_parameter("ball_trajectory_topic").value),
            self._ball_trajectory_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("robot_base_table_pose_topic").value),
            self._robot_base_table_pose_callback,
            10,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter("ball_2d_px_topic").value),
            self._ball_2d_px_callback,
            10,
        )

        # The captured middle-line state is published reliable/transient-local
        # by trajectory_executor so a late-joining debug node still receives
        # the most recent sample. Do not reuse image_qos (best-effort/volatile)
        # for this state topic.
        middle_line_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            MiddleLine,
            str(self.get_parameter("middle_line_state_topic").value),
            self._middle_line_state_callback,
            middle_line_qos,
        )

        render_rate_hz = max(0.1, float(self.get_parameter("debug_render_rate_hz").value))
        self._debug_render_timer = self.create_timer(1.0 / render_rate_hz, self._render_latest_debug_image)

        self.get_logger().info("scene_localizer_debug started in top-cam-only mode")
        self.get_logger().info(
            f"[top_cam] image={self._top.image_topic}, "
            f"camera_info={self._top.camera_info_topic}, "
            f"detections={self._top.detections_topic}, "
            f"table_pose={self._top.table_pose_topic}, "
            f"debug_pub={str(self.get_parameter('top_debug_image_topic').value)}"
        )
        self.get_logger().info(
            "Frame params: "
            f"robot_base_frame={str(self.get_parameter('robot_base_frame').value)}, "
            f"top_camera_frame={str(self.get_parameter('top_camera_frame').value)}, "
            f"table_frame={str(self.get_parameter('table_frame').value)}, "
            f"tcp_frame={str(self.get_parameter('tcp_frame').value)}"
        )
        self.get_logger().info(
            "Middle-line overlay source:\n"
            f"  topic={str(self.get_parameter('middle_line_state_topic').value)}\n"
            "  QoS=reliable/transient_local\n"
            "  geometry=center + direction + half_length\n"
            "  live TCP following disabled"
        )

    def _warn_throttled(self, key: str, message: str, throttle_sec: float = 1.0) -> None:
        now_ns = self.get_clock().now().nanoseconds
        last_ns = self._last_warn_time_ns.get(key, 0)
        if now_ns - last_ns >= int(throttle_sec * 1e9):
            self._last_warn_time_ns[key] = now_ns
            self.get_logger().warn(message)

    def _debug_throttled(self, key: str, message: str, throttle_sec: float = 1.0) -> None:
        now_ns = self.get_clock().now().nanoseconds
        last_ns = self._last_debug_time_ns.get(key, 0)
        if now_ns - last_ns >= int(throttle_sec * 1e9):
            self._last_debug_time_ns[key] = now_ns
            self.get_logger().debug(message)

    def _ball_trajectory_callback(self, msg: BallTrajectory) -> None:
        self._latest_ball_trajectory = msg
        self._latest_ball_trajectory_time_sec = self._stamp_to_sec(msg)

    def _robot_base_table_pose_callback(self, msg: PoseStamped) -> None:
        self._latest_robot_base_table_pose = msg
        self._latest_robot_base_table_pose_time_sec = self._stamp_to_sec(msg)

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self._top.latest_camera_info = msg

        if len(msg.k) < 9:
            self._camera_K_native = None
            self._camera_D = None
            self._camera_projection_is_rectified = False
            return

        K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(K)):
            self._camera_K_native = None
            self._camera_D = None
            self._camera_projection_is_rectified = False
            return

        D = np.asarray(msg.d, dtype=np.float64).reshape(-1, 1)
        is_rectified = D.size == 0 or np.all(np.abs(D) < 1e-9)

        self._camera_K_native = K
        self._camera_D = D
        self._camera_projection_is_rectified = bool(is_rectified)

    def _ball_2d_px_callback(self, msg: PointStamped) -> None:
        self._latest_ball_2d_px = msg
        self._latest_ball_2d_px_time_sec = self._stamp_to_sec(msg)

    def _middle_line_state_callback(self, msg: MiddleLine) -> None:
        """Cache the latest valid captured middle-line state.

        No freshness timeout is applied here: the cached state remains the
        authoritative pink-line geometry until a newer valid message with a
        higher (or equal) revision replaces it. An invalid message never
        clears a previously cached valid state.
        """
        if not bool(msg.valid):
            self._debug_throttled(
                "middle_line_invalid_msg",
                "Ignoring middle-line state: valid=False",
                2.0,
            )
            return

        center = np.array(
            [float(msg.center.x), float(msg.center.y), float(msg.center.z)],
            dtype=float,
        )
        if not _is_finite_vector(center):
            self._warn_throttled(
                "middle_line_center_non_finite",
                "Ignoring middle-line state: center contains non-finite values",
                2.0,
            )
            return

        direction = np.array(
            [float(msg.direction.x), float(msg.direction.y), float(msg.direction.z)],
            dtype=float,
        )
        if not _is_finite_vector(direction):
            self._warn_throttled(
                "middle_line_direction_non_finite",
                "Ignoring middle-line state: direction contains non-finite values",
                2.0,
            )
            return

        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-9:
            self._warn_throttled(
                "middle_line_direction_degenerate",
                "Ignoring middle-line state: direction norm is too small",
                2.0,
            )
            return

        half_length = float(msg.half_length)
        if not np.isfinite(half_length) or half_length <= 0.0:
            self._warn_throttled(
                "middle_line_half_length_invalid",
                f"Ignoring middle-line state: half_length={half_length} is not finite/positive",
                2.0,
            )
            return

        incoming_revision = int(msg.revision)

        if (
            self._latest_middle_line_revision is not None
            and incoming_revision < self._latest_middle_line_revision
        ):
            self._warn_throttled(
                "middle_line_old_revision",
                (
                    f"Ignoring older middle-line revision {incoming_revision}; "
                    f"current revision is {self._latest_middle_line_revision}"
                ),
                2.0,
            )
            return

        if (
            self._latest_middle_line_revision is not None
            and incoming_revision == self._latest_middle_line_revision
        ):
            # Same revision re-delivered (e.g. transient-local replay on a
            # fresh subscription match): refresh the cached message without
            # repeatedly logging it.
            self._latest_middle_line = msg
            return

        self._latest_middle_line = msg
        self._latest_middle_line_revision = incoming_revision

        self.get_logger().info(
            f"Updated captured middle line: revision={incoming_revision}, "
            f"frame={msg.header.frame_id}, "
            f"center=({msg.center.x:.4f}, {msg.center.y:.4f}, {msg.center.z:.4f}), "
            f"direction=({msg.direction.x:.4f}, {msg.direction.y:.4f}, {msg.direction.z:.4f}), "
            f"half_length={msg.half_length:.4f}, "
            f"ee={msg.ee_name}"
        )

    def _detections_callback(self, msg: ArucoDetection) -> None:
        self._top.latest_detections = msg
        self._top.latest_detection_time_sec = self._stamp_to_sec(msg)

    def _table_pose_callback(self, msg: PoseStamped) -> None:
        self._top.latest_table_pose = msg
        self._top.latest_table_pose_time_sec = self._stamp_to_sec(msg)

    def _image_callback(self, msg: Image) -> None:
        with self._latest_image_lock:
            self._latest_image_msg = msg
            self._latest_image_stamp_ns = self._msg_stamp_ns(msg)

    def _msg_stamp_ns(self, msg: Any) -> int:
        header = getattr(msg, "header", None)
        if header is not None and hasattr(header, "stamp"):
            stamp = header.stamp
            sec = int(getattr(stamp, "sec", 0))
            nanosec = int(getattr(stamp, "nanosec", 0))
            total_ns = sec * 1000000000 + nanosec
            if total_ns > 0:
                return total_ns
        return int(self.get_clock().now().nanoseconds)

    def _stamp_age_sec(self, msg: Any, now_sec: float) -> float:
        stamp_ns = self._msg_stamp_ns(msg)
        return max(0.0, now_sec - (float(stamp_ns) * 1e-9))

    def _perf_record_step_ms(self, step: str, elapsed_ms: float) -> None:
        self._perf_window_sums_ms[step] = self._perf_window_sums_ms.get(step, 0.0) + elapsed_ms
        prev_max = self._perf_window_max_ms.get(step, 0.0)
        if elapsed_ms > prev_max:
            self._perf_window_max_ms[step] = elapsed_ms

    def _perf_maybe_report(self) -> None:
        now_mono = time.monotonic()
        if (now_mono - self._perf_last_report_monotonic) < 1.0:
            return

        count = self._perf_window_count
        if count <= 0:
            self._perf_last_report_monotonic = now_mono
            return

        # Report a fixed order so timing lines are stable across reports.
        ordered_steps = [
            "convert",
            "prepare_vga",
            "intrinsics",
            "proj_text",
            "transforms",
            "detection",
            "table",
            "ball_2d",
            "trajectory",
            "publish",
            "total",
        ]
        parts: List[str] = []
        for step in ordered_steps:
            if step not in self._perf_window_sums_ms:
                continue
            avg_ms = self._perf_window_sums_ms[step] / float(count)
            max_ms = self._perf_window_max_ms.get(step, 0.0)
            parts.append(f"{step}:avg={avg_ms:.2f}ms,max={max_ms:.2f}ms")

        self.get_logger().info(
            f"debug_render_timers frames={count} over~{now_mono - self._perf_last_report_monotonic:.2f}s | "
            + " | ".join(parts)
        )

        self._perf_last_report_monotonic = now_mono
        self._perf_window_count = 0
        self._perf_window_sums_ms.clear()
        self._perf_window_max_ms.clear()

    def _render_latest_debug_image(self) -> None:
        if not bool(self.get_parameter("publish_debug_images").value):
            return

        if self._debug_pub.get_subscription_count() == 0:
            return

        with self._latest_image_lock:
            image_msg = self._latest_image_msg
            image_stamp_ns = self._latest_image_stamp_ns

        if image_msg is None or image_stamp_ns is None:
            return

        if self._last_rendered_image_stamp_ns == image_stamp_ns:
            return

        t_fn_start = time.perf_counter()

        t0 = time.perf_counter()
        try:
            frame_native = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception as exc:
            self._warn_throttled("image_convert_failed", f"[top_cam] failed to convert image to bgr8: {exc}", 1.0)
            return
        self._perf_record_step_ms("convert", (time.perf_counter() - t0) * 1000.0)

        if frame_native is None or frame_native.ndim != 3:
            return

        source_height, source_width = frame_native.shape[:2]
        t0 = time.perf_counter()
        frame_vga, scale, pad_x, pad_y = self._prepare_vga_frame(frame_native)
        self._perf_record_step_ms("prepare_vga", (time.perf_counter() - t0) * 1000.0)

        K_vga = None
        t0 = time.perf_counter()
        if self._camera_K_native is not None:
            K_vga = self._camera_K_native.copy()
            K_vga[0, 0] *= scale
            K_vga[1, 1] *= scale
            K_vga[0, 2] = self._camera_K_native[0, 2] * scale + pad_x
            K_vga[1, 2] = self._camera_K_native[1, 2] * scale + pad_y
        self._perf_record_step_ms("intrinsics", (time.perf_counter() - t0) * 1000.0)

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self._begin_frame_text()
        t0 = time.perf_counter()
        self._draw_projection_debug_text(
            frame=frame_vga,
            image_msg=image_msg,
            source_width=source_width,
            source_height=source_height,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )
        self._perf_record_step_ms("proj_text", (time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        transforms = self._get_frame_transforms(now_sec)
        self._perf_record_step_ms("transforms", (time.perf_counter() - t0) * 1000.0)

        if K_vga is None:
            self._draw_status_text(frame_vga, "no valid camera_info/K", (0, 140, 255))
        else:
            detection_timeout_sec = max(0.0, float(self.get_parameter("detection_timeout_sec").value))
            detections_stale = (
                self._top.latest_detection_time_sec is None
                or (now_sec - self._top.latest_detection_time_sec) > detection_timeout_sec
            )

            t0 = time.perf_counter()
            if detections_stale:
                self._draw_status_text(frame_vga, "detections stale", (0, 140, 255))
            elif self._top.latest_detections is None:
                self._draw_status_text(frame_vga, "no detections", (0, 140, 255))
            else:
                physical_marker_size = max(1e-6, float(self.get_parameter("physical_marker_size").value))
                axis_length = max(1e-6, float(self.get_parameter("axis_length").value))
                self._draw_detection_overlay(
                    frame=frame_vga,
                    detections_msg=self._top.latest_detections,
                    physical_marker_size=physical_marker_size,
                    axis_length=axis_length,
                    K=K_vga,
                    D=self._camera_D,
                    image_width=640,
                    image_height=480,
                )
            self._perf_record_step_ms("detection", (time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            self._draw_table_rectangle_overlay(
                frame=frame_vga,
                T_cam_table=transforms.get("T_cam_table"),
                K=K_vga,
                D=self._camera_D,
                image_width=640,
                image_height=480,
            )
            self._perf_record_step_ms("table", (time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            self._draw_ball_2d_px_overlay(
                frame=frame_vga,
                image_msg=image_msg,
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
            )
            self._perf_record_step_ms("ball_2d", (time.perf_counter() - t0) * 1000.0)

            if bool(self.get_parameter("draw_tcp_middle_line").value):
                t0 = time.perf_counter()
                self._draw_captured_middle_line_overlay(
                    frame=frame_vga,
                    K=K_vga,
                    D=self._camera_D,
                    image_width=640,
                    image_height=480,
                )
                self._perf_record_step_ms("middle_line", (time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            self._draw_ball_trajectory_overlay(
                frame=frame_vga,
                now_sec=now_sec,
                T_cam_table=transforms.get("T_cam_table"),
                K=K_vga,
                D=self._camera_D,
                image_width=640,
                image_height=480,
            )
            self._perf_record_step_ms("trajectory", (time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        self._publish_debug_image(image_msg, frame_vga)
        self._perf_record_step_ms("publish", (time.perf_counter() - t0) * 1000.0)
        self._last_rendered_image_stamp_ns = image_stamp_ns

        total_ms = (time.perf_counter() - t_fn_start) * 1000.0
        self._perf_record_step_ms("total", total_ms)
        self._perf_window_count += 1
        self._perf_maybe_report()

        if bool(self.get_parameter("log_render_timing").value):
            render_ms = total_ms
            source_age_sec = self._stamp_age_sec(image_msg, now_sec)
            self._debug_throttled(
                "render_timing",
                f"debug render: {render_ms:.2f} ms, source age: {source_age_sec:.3f} s",
                1.0,
            )

    def _prepare_vga_frame(self, frame_native: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        source_h, source_w = frame_native.shape[:2]
        scale = min(640.0 / float(source_w), 480.0 / float(source_h))
        resized_w = max(1, int(round(float(source_w) * scale)))
        resized_h = max(1, int(round(float(source_h) * scale)))

        resized = cv2.resize(frame_native, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        frame_vga = np.zeros((480, 640, 3), dtype=np.uint8)
        pad_x = (640 - resized_w) // 2
        pad_y = (480 - resized_h) // 2
        frame_vga[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized

        return frame_vga, scale, pad_x, pad_y

    def _get_frame_transforms(self, now_sec: float) -> Dict[str, Optional[np.ndarray]]:
        T_cam_table = self._get_camera_from_table_transform(now_sec)
        T_base_table: Optional[np.ndarray] = None

        if T_cam_table is not None:
            T_base_table = self._get_robot_base_from_table_transform(now_sec, T_cam_table)

        return {
            "T_cam_table": T_cam_table,
            "T_base_table": T_base_table,
        }

    def _publish_debug_image(self, src_msg: Image, frame: np.ndarray) -> None:
        debug_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        debug_msg.header = src_msg.header
        self._debug_pub.publish(debug_msg)

    def _draw_ball_2d_px_overlay(
        self,
        frame: np.ndarray,
        image_msg: Image,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> None:
        ball = self._latest_ball_2d_px
        stamp_sec = self._latest_ball_2d_px_time_sec
        if ball is None or stamp_sec is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        timeout_sec = max(0.0, float(self.get_parameter("ball_2d_overlay_timeout_sec").value))
        if (now_sec - stamp_sec) > timeout_sec:
            return

        image_frame_id = str(getattr(image_msg.header, "frame_id", "")).strip()
        ball_frame_id = str(getattr(ball.header, "frame_id", "")).strip()
        if image_frame_id and ball_frame_id and image_frame_id != ball_frame_id:
            return

        u_native = float(ball.point.x)
        v_native = float(ball.point.y)
        radius_native = float(ball.point.z)
        if not np.isfinite(u_native) or not np.isfinite(v_native):
            return
        if not np.isfinite(radius_native):
            radius_native = 0.0

        u_vga = (scale * u_native) + float(pad_x)
        v_vga = (scale * v_native) + float(pad_y)
        radius_vga = max(0.0, scale * radius_native)

        center = (int(round(u_vga)), int(round(v_vga)))
        radius_px = max(0, int(round(radius_vga)))

        cv2.circle(frame, center, 4, (0, 0, 255), -1)
        if radius_px > 0:
            cv2.circle(frame, center, radius_px, (0, 255, 255), 2)

    def _begin_frame_text(self) -> None:
        self._text_y = 18

    def _draw_status_text(
        self,
        frame: np.ndarray,
        text: str,
        color: Tuple[int, int, int],
        scale: float = 0.48,
        thickness: int = 1,
    ) -> None:
        if not bool(self.get_parameter("draw_debug_text").value):
            return
        cv2.putText(
            frame,
            text,
            (12, self._text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        self._text_y += 18

    def _draw_overlay_label(
        self,
        frame: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        color: Tuple[int, int, int],
        scale: float = 0.5,
        thickness: int = 2,
    ) -> None:
        if not bool(self.get_parameter("draw_debug_text").value):
            return
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_projection_debug_text(
        self,
        frame: np.ndarray,
        image_msg: Image,
        source_width: int,
        source_height: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        image_frame_id = str(getattr(image_msg.header, "frame_id", ""))
        configured_camera_frame = str(self.get_parameter("top_camera_frame").value)

        detections_age = self._format_age(now_sec, self._top.latest_detection_time_sec)
        table_pose_age = self._format_age(now_sec, self._top.latest_table_pose_time_sec)
        base_table_age = self._format_age(now_sec, self._latest_robot_base_table_pose_time_sec)

        self._draw_status_text(
            frame,
            f"src={source_width}x{source_height} -> out=640x480 scale={scale:.3f} pad=({pad_x},{pad_y})",
            (255, 255, 255),
        )
        self._draw_status_text(
            frame,
            f"frame={image_frame_id} configured_camera_frame={configured_camera_frame}",
            (255, 255, 255),
        )
        self._draw_status_text(frame, f"table_pose_camera age: {table_pose_age}", (255, 255, 255))
        self._draw_status_text(frame, f"detections age: {detections_age}", (255, 255, 255))
        self._draw_status_text(frame, f"base_table pose age: {base_table_age}", (255, 255, 255))

    def _draw_detection_overlay(
        self,
        frame: np.ndarray,
        detections_msg: ArucoDetection,
        physical_marker_size: float,
        axis_length: float,
        K: np.ndarray,
        D: Optional[np.ndarray],
        image_width: int,
        image_height: int,
    ) -> None:
        markers = self._extract_markers_from_msg(detections_msg)
        if not markers:
            self._draw_status_text(frame, "no detections", (0, 140, 255))
            return

        local_half = 0.5 * physical_marker_size
        square_local = np.array(
            [
                [-local_half, -local_half, 0.0],
                [local_half, -local_half, 0.0],
                [local_half, local_half, 0.0],
                [-local_half, local_half, 0.0],
            ],
            dtype=float,
        )

        origin = np.array([0.0, 0.0, 0.0], dtype=float)
        axis_x = np.array([axis_length, 0.0, 0.0], dtype=float)
        axis_y = np.array([0.0, axis_length, 0.0], dtype=float)
        axis_z = np.array([0.0, 0.0, axis_length], dtype=float)

        for marker in markers:
            marker_id = self._extract_marker_id(marker)
            pose = self._extract_pose_from_marker(marker)
            if pose is None:
                continue

            t_cm = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
            q_cm = np.array(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
                dtype=float,
            )
            r_cm = _quaternion_to_matrix_xyzw(q_cm)

            points_cam = np.vstack(
                [
                    (r_cm @ square_local.T).T + t_cm,
                    (r_cm @ origin) + t_cm,
                    (r_cm @ axis_x) + t_cm,
                    (r_cm @ axis_y) + t_cm,
                    (r_cm @ axis_z) + t_cm,
                    t_cm,
                ]
            )
            projected = self._project_points(points_cam, K, D, image_width, image_height)

            square_pixels = projected[0:4]
            origin_uv = projected[4]
            x_uv = projected[5]
            y_uv = projected[6]
            z_uv = projected[7]
            center_uv = projected[8]

            if all(p is not None for p in square_pixels):
                pts = np.array(square_pixels, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

            if center_uv is not None:
                cv2.circle(frame, center_uv, 4, (255, 255, 0), -1)
                label = f"id:{marker_id}" if marker_id is not None else "id:?"
                self._draw_overlay_label(
                    frame,
                    label,
                    (center_uv[0] + 6, center_uv[1] - 6),
                    (255, 255, 255),
                    scale=0.55,
                )

            if origin_uv is not None and x_uv is not None:
                cv2.line(frame, origin_uv, x_uv, (0, 0, 255), 2)
            if origin_uv is not None and y_uv is not None:
                cv2.line(frame, origin_uv, y_uv, (0, 255, 0), 2)
            if origin_uv is not None and z_uv is not None:
                cv2.line(frame, origin_uv, z_uv, (255, 0, 0), 2)

    def _draw_table_rectangle_overlay(
        self,
        frame: np.ndarray,
        T_cam_table: Optional[np.ndarray],
        K: np.ndarray,
        D: Optional[np.ndarray],
        image_width: int,
        image_height: int,
    ) -> None:
        if not bool(self.get_parameter("draw_table_rectangle").value):
            return

        if T_cam_table is None:
            self._draw_status_text(frame, "no/stale table_pose_camera", (0, 0, 255))
            return

        table_shortedge = float(self.get_parameter("table_shortedge").value)
        table_longedge = float(self.get_parameter("table_longedge").value)
        table_edge_z = float(self.get_parameter("table_edge_z").value)

        corners_table = [
            np.array([0.0, 0.0, table_edge_z], dtype=float),
            np.array([table_shortedge, 0.0, table_edge_z], dtype=float),
            np.array([table_shortedge, table_longedge, table_edge_z], dtype=float),
            np.array([0.0, table_longedge, table_edge_z], dtype=float),
        ]

        corners_cam = np.vstack([_transform_point(T_cam_table, p_table) for p_table in corners_table])
        projected = self._project_points(corners_cam, K, D, image_width, image_height)
        visible_count = sum(1 for p in projected if p is not None)

        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        clipped_segments: List[Tuple[np.ndarray, np.ndarray]] = []
        for i0, i1 in edges:
            clipped = self._clip_camera_edge_to_near_plane(corners_cam[i0], corners_cam[i1])
            if clipped is not None:
                clipped_segments.append(clipped)

        drawn_edge_count = 0
        if clipped_segments:
            edge_points = np.vstack([np.vstack(seg) for seg in clipped_segments])
            edge_uv = self._project_points(edge_points, K, D, image_width, image_height)
            for i in range(0, len(edge_uv), 2):
                p0_uv = edge_uv[i]
                p1_uv = edge_uv[i + 1]
                if p0_uv is None or p1_uv is None:
                    continue
                cv2.line(frame, p0_uv, p1_uv, (0, 0, 255), 3)
                drawn_edge_count += 1

        labels = ["origin", "x", "x+y", "y"]
        for idx, point in enumerate(projected):
            if point is None:
                continue
            cv2.circle(frame, point, 4, (0, 0, 255), -1)
            self._draw_overlay_label(
                frame,
                labels[idx],
                (point[0] + 6, point[1] - 6),
                (0, 0, 255),
                scale=0.45,
                thickness=1,
            )

        if drawn_edge_count == 0 and visible_count == 0:
            self._draw_status_text(frame, "table rectangle not visible/projectable", (0, 0, 255))

    def _draw_ball_trajectory_overlay(
        self,
        frame: np.ndarray,
        now_sec: float,
        T_cam_table: Optional[np.ndarray],
        K: np.ndarray,
        D: Optional[np.ndarray],
        image_width: int,
        image_height: int,
    ) -> None:
        """Project-only trajectory visualization.

        The ball_trajectory_estimator is responsible for deciding whether a
        trajectory is usable. This debug node only draws a fresh, valid
        BallTrajectory message. In the current contract:
            start_point = latest fitted ball point in the z=ball_radius plane
            end_point   = middle-line intersection/hit point

        The captured middle-line (pink) overlay is persistent scene state and
        is drawn independently by _draw_captured_middle_line_overlay(); it is
        intentionally not nested inside this method so it keeps being drawn
        regardless of ball-trajectory validity/staleness.
        """
        self._draw_status_text(frame, f"traj_overlay now={now_sec:.3f}s", (200, 200, 200))

        if T_cam_table is None:
            self._draw_status_text(frame, "T_cam_table: unavailable", (0, 140, 255))
            self._draw_status_text(frame, "no/stale T_camera_table", (0, 140, 255))
            return

        traj = self._latest_ball_trajectory
        if traj is None or self._latest_ball_trajectory_time_sec is None:
            self._draw_status_text(frame, "trajectory cache: empty", (0, 140, 255))
            self._draw_status_text(frame, "no ball trajectory", (0, 140, 255))
            return

        trajectory_timeout_sec = max(0.0, float(self.get_parameter("trajectory_timeout_sec").value))
        age_sec = now_sec - self._latest_ball_trajectory_time_sec
        self._draw_status_text(
            frame,
            f"trajectory age={age_sec:.3f}s timeout={trajectory_timeout_sec:.3f}s",
            (200, 200, 200),
        )
        if age_sec > trajectory_timeout_sec:
            self._draw_status_text(frame, "ball trajectory stale", (0, 140, 255))
            return

        configured_table_frame = str(self.get_parameter("table_frame").value)
        accepted_frames = {"table", "table_frame", configured_table_frame}
        traj_frame = str(getattr(getattr(traj, "header", None), "frame_id", "")).strip()
        if traj_frame and traj_frame not in accepted_frames:
            self._draw_status_text(frame, f"trajectory frame mismatch: {traj_frame}", (0, 140, 255))
            return

        valid = bool(getattr(traj, "valid", False))
        velocity = getattr(traj, "velocity", None)
        vx = float(getattr(velocity, "x", float("nan"))) if velocity is not None else float("nan")
        vy = float(getattr(velocity, "y", float("nan"))) if velocity is not None else float("nan")
        vz = float(getattr(velocity, "z", float("nan"))) if velocity is not None else float("nan")
        velocity_table = np.array([vx, vy, vz], dtype=float)
        speed = float(np.linalg.norm(velocity_table)) if _is_finite_vector(velocity_table) else float("nan")
        num_obs = int(getattr(traj, "num_observations", 0))
        rms = float(getattr(traj, "fit_rms_error", float("nan")))

        if bool(self.get_parameter("draw_trajectory_debug_text").value):
            self._draw_status_text(
                frame,
                f"trajectory: valid={valid} n={num_obs} rms={rms:.3f}",
                (255, 255, 255),
            )
            self._draw_status_text(
                frame,
                f"v=({vx:.2f},{vy:.2f},{vz:.2f}) speed={speed:.2f} m/s",
                (255, 255, 255),
            )

        if not valid:
            self._draw_status_text(frame, "trajectory invalid: not drawing arrow/hit", (0, 165, 255))
            return

        start_point_msg = getattr(traj, "start_point", None)
        end_point_msg = getattr(traj, "end_point", None)
        if start_point_msg is None or end_point_msg is None:
            self._draw_status_text(frame, "trajectory point fields missing", (0, 140, 255))
            return

        p0_table = np.array(
            [
                float(getattr(start_point_msg, "x", float("nan"))),
                float(getattr(start_point_msg, "y", float("nan"))),
                float(getattr(start_point_msg, "z", float("nan"))),
            ],
            dtype=float,
        )
        p1_table = np.array(
            [
                float(getattr(end_point_msg, "x", float("nan"))),
                float(getattr(end_point_msg, "y", float("nan"))),
                float(getattr(end_point_msg, "z", float("nan"))),
            ],
            dtype=float,
        )
        if not _is_finite_vector(p0_table) or not _is_finite_vector(p1_table):
            self._draw_status_text(frame, "trajectory points non-finite", (0, 140, 255))
            return

        line_color = (255, 0, 255)
        start_color = (0, 255, 255)
        hit_color = (255, 255, 0)
        line_thickness = max(1, int(self.get_parameter("trajectory_line_thickness").value))
        start_radius = max(1, int(self.get_parameter("trajectory_start_radius_px").value))
        hit_radius = max(1, int(self.get_parameter("trajectory_intersection_radius_px").value))

        drawn = self._draw_table_segment(
            frame=frame,
            T_cam_table=T_cam_table,
            p0_table=p0_table,
            p1_table=p1_table,
            color=line_color,
            thickness=line_thickness,
            K=K,
            D=D,
            image_width=image_width,
            image_height=image_height,
            arrow=True,
        )
        if not drawn:
            self._draw_status_text(frame, "valid trajectory not projectable", (0, 140, 255))
            return

        p_table = np.vstack([p0_table, p1_table])
        p_uv = self._project_table_points(p_table, T_cam_table, K, D, image_width, image_height)
        p0_uv, p1_uv = p_uv[0], p_uv[1]

        if p0_uv is not None:
            cv2.circle(frame, p0_uv, start_radius, start_color, -1)
            self._draw_overlay_label(
                frame,
                "ball",
                (p0_uv[0] + 7, p0_uv[1] - 7),
                start_color,
            )

        if p1_uv is not None:
            cv2.circle(frame, p1_uv, hit_radius, hit_color, -1)
            self._draw_overlay_label(
                frame,
                "hit",
                (p1_uv[0] + 7, p1_uv[1] - 7),
                hit_color,
            )

    def _get_camera_from_table_transform(self, now_sec: float) -> Optional[np.ndarray]:
        """Return T_top_camera_table from /scene_localizer/top_cam/table_pose_camera."""
        if self._top.latest_table_pose is None or self._top.latest_table_pose_time_sec is None:
            self._warn_throttled(
                "top_table_pose_missing",
                "top_table_pose unavailable: waiting for PoseStamped on top_table_pose_topic",
                1.0,
            )
            return None

        table_pose_timeout_sec = max(0.0, float(self.get_parameter("table_pose_timeout_sec").value))
        pose_age_sec = now_sec - self._top.latest_table_pose_time_sec
        if pose_age_sec > table_pose_timeout_sec:
            self._warn_throttled(
                "top_table_pose_stale",
                f"top_table_pose stale: age={pose_age_sec:.3f}s timeout={table_pose_timeout_sec:.3f}s",
                1.0,
            )
            return None

        header_frame = str(getattr(self._top.latest_table_pose.header, "frame_id", "")).strip("/")
        configured_camera_frame = str(self.get_parameter("top_camera_frame").value).strip("/")
        if header_frame and header_frame != configured_camera_frame:
            self._warn_throttled(
                "top_table_pose_frame_mismatch",
                f"top table pose frame is '{header_frame}', but top_camera_frame is '{configured_camera_frame}'. "
                "Interpreting pose as T_top_camera_table anyway.",
                2.0,
            )

        return _pose_to_transform(self._top.latest_table_pose.pose)

    def _get_robot_base_from_table_transform(
        self,
        now_sec: float,
        T_cam_table: np.ndarray,
    ) -> Optional[np.ndarray]:
        timeout_sec = max(0.0, float(self.get_parameter("robot_base_table_pose_timeout_sec").value))
        if (
            self._latest_robot_base_table_pose is not None
            and self._latest_robot_base_table_pose_time_sec is not None
            and (now_sec - self._latest_robot_base_table_pose_time_sec) <= timeout_sec
        ):
            header_frame = str(getattr(self._latest_robot_base_table_pose.header, "frame_id", "")).strip("/")
            robot_base_frame = str(self.get_parameter("robot_base_frame").value).strip("/")
            if header_frame and header_frame != robot_base_frame:
                self._warn_throttled(
                    "base_table_pose_frame_mismatch",
                    f"robot_base_table_pose frame is '{header_frame}', but robot_base_frame is '{robot_base_frame}'. "
                    "Interpreting pose as T_robot_base_table anyway.",
                    2.0,
                )
            return _pose_to_transform(self._latest_robot_base_table_pose.pose)

        if not bool(self.get_parameter("allow_base_table_fallback_from_tf").value):
            return None

        T_base_cam = self._lookup_robot_base_from_camera_transform()
        if T_base_cam is None:
            return None
        return T_base_cam @ T_cam_table

    def _lookup_robot_base_from_camera_transform(self) -> Optional[np.ndarray]:
        robot_base_frame = str(self.get_parameter("robot_base_frame").value)
        top_camera_frame = str(self.get_parameter("top_camera_frame").value)
        return self._lookup_transform_matrix(robot_base_frame, top_camera_frame, "base_to_camera")

    def _lookup_transform_matrix(
        self,
        target_frame: str,
        source_frame: str,
        warn_key: str,
    ) -> Optional[np.ndarray]:
        timeout_sec = max(0.0, float(self.get_parameter("tf_lookup_timeout_sec").value))
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=timeout_sec),
            )
        except TransformException as exc:
            self._warn_throttled(
                warn_key,
                f"TF lookup failed {target_frame}->{source_frame}: {exc}",
                1.0,
            )
            return None
        return _transform_stamped_to_matrix(transform)

    def _draw_captured_middle_line_overlay(
        self,
        frame: np.ndarray,
        K: np.ndarray,
        D: Optional[np.ndarray],
        image_width: int,
        image_height: int,
    ) -> None:
        """Draw the authoritative captured middle line.

        Geometry comes only from the cached MiddleLine state (published by
        trajectory_executor on middle_line_state_topic) and a direct TF
        lookup from the message's frame to the camera frame. This overlay
        intentionally does not depend on the live TCP pose, T_table_tcp,
        table-pose estimation, or ball-trajectory state, so it does not move
        when the robot moves and remains visible even if those other inputs
        are unavailable or stale.
        """
        pink = (180, 105, 255)

        line = self._latest_middle_line
        if line is None:
            self._draw_status_text(frame, "middle line: waiting for captured state", pink)
            return

        line_frame = str(line.header.frame_id).strip("/")
        if not line_frame:
            self._warn_throttled(
                "middle_line_frame_empty",
                "Captured middle-line message has an empty header.frame_id; cannot draw.",
                2.0,
            )
            self._draw_status_text(frame, "middle line: empty frame_id", pink)
            return

        camera_frame = str(self.get_parameter("top_camera_frame").value).strip("/")

        # _lookup_transform_matrix(target_frame, source_frame, ...) returns
        # T_target_source. We need camera-from-line (T_camera_line), i.e.
        # target=camera_frame, source=line_frame -- not the inverse.
        T_camera_line = self._lookup_transform_matrix(
            camera_frame,
            line_frame,
            "camera_from_middle_line",
        )
        if T_camera_line is None:
            self._draw_status_text(
                frame,
                f"middle line: no TF {camera_frame} <- {line_frame}",
                pink,
            )
            return

        center_line = np.array(
            [float(line.center.x), float(line.center.y), float(line.center.z)],
            dtype=float,
        )
        direction_line = np.array(
            [float(line.direction.x), float(line.direction.y), float(line.direction.z)],
            dtype=float,
        )
        if not _is_finite_vector(center_line) or not _is_finite_vector(direction_line):
            self._draw_status_text(frame, "middle line: cached geometry non-finite", pink)
            return

        direction_norm = float(np.linalg.norm(direction_line))
        if direction_norm <= 1e-9:
            self._draw_status_text(frame, "middle line: cached direction degenerate", pink)
            return
        direction_line = direction_line / direction_norm

        half_length = float(line.half_length)
        if not np.isfinite(half_length) or half_length <= 0.0:
            self._draw_status_text(frame, "middle line: cached half_length invalid", pink)
            return

        # Published Z is used unchanged: not ball_radius, not table Z, not
        # live TCP Z, and not any other configured overlay height.
        p_start_line = center_line - half_length * direction_line
        p_end_line = center_line + half_length * direction_line

        p_start_cam = _transform_point(T_camera_line, p_start_line)
        p_end_cam = _transform_point(T_camera_line, p_end_line)
        center_cam = _transform_point(T_camera_line, center_line)

        clipped = self._clip_camera_edge_to_near_plane(p_start_cam, p_end_cam)
        if clipped is None:
            self._draw_status_text(frame, "middle line: not projectable (behind camera)", pink)
            return
        c_start_cam, c_end_cam = clipped

        proj = self._project_points(
            np.vstack([c_start_cam, c_end_cam, center_cam]),
            K,
            D,
            image_width,
            image_height,
        )
        p_start_uv, p_end_uv, center_uv = proj[0], proj[1], proj[2]

        if p_start_uv is None or p_end_uv is None:
            self._draw_status_text(frame, "middle line: not projectable", pink)
            return

        thickness = max(1, int(self.get_parameter("tcp_middle_line_thickness").value))
        cv2.line(frame, p_start_uv, p_end_uv, pink, thickness)

        if center_uv is not None:
            cv2.circle(frame, center_uv, 4, pink, -1)
            self._draw_overlay_label(
                frame,
                "captured line",
                (center_uv[0] + 7, center_uv[1] - 7),
                pink,
                scale=0.45,
            )

        self._draw_status_text(
            frame,
            f"middle line rev={self._latest_middle_line_revision} "
            f"half={half_length:.3f} total={2.0 * half_length:.3f} frame={line_frame}",
            pink,
        )

    def _compute_trajectory_intersection_table(
        self,
        p0_table: np.ndarray,
        p1_table: np.ndarray,
        velocity_table: np.ndarray,
        T_table_tcp: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return the extrapolated trajectory hit point in table coordinates.

        Modes:
        - tcp_xz_plane: intersect ball line with the TCP X-Z plane, i.e. the
          plane through TCP origin whose normal is the TCP y-axis.
        - middle_line: closest-approach intersection with TCP x-axis. Accepted
          only if the closest distance is below middle_line_intersection_max_distance.
        - middle_line_then_tcp_xz_plane: try middle_line first, otherwise use
          tcp_xz_plane.
        """
        if not _is_finite_vector(p0_table) or not _is_finite_vector(p1_table):
            return None

        if _is_finite_vector(velocity_table) and float(np.linalg.norm(velocity_table)) > 1e-6:
            direction_table = velocity_table.astype(float)
        else:
            direction_table = p1_table - p0_table

        direction_norm = float(np.linalg.norm(direction_table)) 
        if direction_norm < 1e-9:
            return None
        direction_table = direction_table / direction_norm
        current_table = p1_table

        mode = str(self.get_parameter("trajectory_intersection_mode").value).strip().lower()
        allow_backward = bool(self.get_parameter("allow_backward_trajectory_intersection").value)

        if mode in {"middle_line", "middle_line_then_tcp_xz_plane"}:
            hit = self._compute_middle_line_closest_intersection_table(
                current_table=current_table,
                direction_table=direction_table,
                T_table_tcp=T_table_tcp,
                allow_backward=allow_backward,
            )
            if hit is not None or mode == "middle_line":
                return hit

        tcp_origin_table = T_table_tcp[:3, 3].copy()
        tcp_y_axis_table = T_table_tcp[:3, 1].copy()
        y_norm = float(np.linalg.norm(tcp_y_axis_table))
        if y_norm < 1e-12:
            return None
        tcp_y_axis_table /= y_norm

        denom = float(np.dot(tcp_y_axis_table, direction_table))
        if abs(denom) < 1e-9:
            return None

        s = float(np.dot(tcp_y_axis_table, tcp_origin_table - current_table) / denom)
        if s < 0.0 and not allow_backward:
            return None

        hit_table = current_table + s * direction_table
        return hit_table if _is_finite_vector(hit_table) else None

    def _compute_middle_line_closest_intersection_table(
        self,
        current_table: np.ndarray,
        direction_table: np.ndarray,
        T_table_tcp: np.ndarray,
        allow_backward: bool,
    ) -> Optional[np.ndarray]:
        tcp_origin_table = T_table_tcp[:3, 3].copy()
        tcp_x_axis_table = T_table_tcp[:3, 0].copy()
        x_norm = float(np.linalg.norm(tcp_x_axis_table))
        if x_norm < 1e-12:
            return None
        tcp_x_axis_table /= x_norm

        A = np.column_stack((direction_table, -tcp_x_axis_table))
        b = tcp_origin_table - current_table
        try:
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None

        s = float(sol[0])
        u = float(sol[1])
        if s < 0.0 and not allow_backward:
            return None

        closest_on_ball = current_table + s * direction_table
        closest_on_axis = tcp_origin_table + u * tcp_x_axis_table
        distance = float(np.linalg.norm(closest_on_ball - closest_on_axis))
        max_distance = max(0.0, float(self.get_parameter("middle_line_intersection_max_distance").value))
        if distance > max_distance:
            return None

        hit_table = 0.5 * (closest_on_ball + closest_on_axis)
        return hit_table if _is_finite_vector(hit_table) else None

    def _project_table_points(
        self,
        points_table: np.ndarray,
        T_cam_table: np.ndarray,
        K: np.ndarray,
        D: Optional[np.ndarray],
        image_width: int,
        image_height: int,
    ) -> List[Optional[Tuple[int, int]]]:
        if points_table.ndim != 2 or points_table.shape[1] != 3:
            return []
        points_cam = (_transform_point(T_cam_table, p_table) for p_table in points_table)
        points_cam_np = np.vstack(list(points_cam))
        return self._project_points(points_cam_np, K, D, image_width, image_height)

    def _draw_table_segment(
        self,
        frame: np.ndarray,
        T_cam_table: np.ndarray,
        p0_table: np.ndarray,
        p1_table: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int,
        K: np.ndarray,
        D: Optional[np.ndarray],
        image_width: int,
        image_height: int,
        arrow: bool = False,
    ) -> bool:
        if not _is_finite_vector(p0_table) or not _is_finite_vector(p1_table):
            return False
        p0_cam = _transform_point(T_cam_table, p0_table)
        p1_cam = _transform_point(T_cam_table, p1_table)
        clipped = self._clip_camera_edge_to_near_plane(p0_cam, p1_cam)
        if clipped is None:
            return False
        c0_cam, c1_cam = clipped

        proj = self._project_points(np.vstack([c0_cam, c1_cam]), K, D, image_width, image_height)
        p0_uv, p1_uv = proj[0], proj[1]
        if p0_uv is None or p1_uv is None:
            return False

        if arrow:
            cv2.arrowedLine(frame, p0_uv, p1_uv, color, thickness, tipLength=0.12)
        else:
            cv2.line(frame, p0_uv, p1_uv, color, thickness)
        return True

    @staticmethod
    def _extract_markers_from_msg(msg: ArucoDetection) -> List[Any]:
        if hasattr(msg, "markers") and getattr(msg, "markers") is not None:
            return list(getattr(msg, "markers"))
        if hasattr(msg, "detections") and getattr(msg, "detections") is not None:
            return list(getattr(msg, "detections"))
        if hasattr(msg, "marker_id") or hasattr(msg, "id"):
            if hasattr(msg, "pose"):
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

    def _project_points(
        self,
        points_cam: np.ndarray,
        K: np.ndarray,
        D: Optional[np.ndarray],
        image_width: int,
        image_height: int,
    ) -> List[Optional[Tuple[int, int]]]:
        if points_cam.ndim != 2 or points_cam.shape[1] != 3:
            return []

        n_points = points_cam.shape[0]
        result: List[Optional[Tuple[int, int]]] = [None] * n_points

        finite_mask = np.all(np.isfinite(points_cam), axis=1)
        z_mask = points_cam[:, 2] > 1e-9
        valid_mask = finite_mask & z_mask

        if not np.any(valid_mask):
            return result

        valid_idx = np.where(valid_mask)[0]
        valid_points = points_cam[valid_mask]

        if self._camera_projection_is_rectified:
            x = valid_points[:, 0]
            y = valid_points[:, 1]
            z = valid_points[:, 2]
            u = (K[0, 0] * x / z) + K[0, 2]
            v = (K[1, 1] * y / z) + K[1, 2]
            uv = np.column_stack((u, v))
        else:
            object_points = valid_points.reshape((-1, 1, 3)).astype(np.float64)
            image_points, _ = cv2.projectPoints(object_points, self._zero_rvec, self._zero_tvec, K, D)
            uv = image_points.reshape((-1, 2))

        for i_local, i_global in enumerate(valid_idx):
            u = float(uv[i_local, 0])
            v = float(uv[i_local, 1])
            if not np.isfinite(u) or not np.isfinite(v):
                continue
            result[i_global] = (int(round(u)), int(round(v)))

        return result

    @staticmethod
    def _clip_camera_edge_to_near_plane(
        p0_cam: np.ndarray,
        p1_cam: np.ndarray,
        near_z: float = 1e-4,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if p0_cam.shape != (3,) or p1_cam.shape != (3,):
            return None

        z0 = float(p0_cam[2])
        z1 = float(p1_cam[2])
        if not np.isfinite(z0) or not np.isfinite(z1):
            return None

        if z0 <= near_z and z1 <= near_z:
            return None

        if z0 > near_z and z1 > near_z:
            return p0_cam, p1_cam

        denom = z1 - z0
        if abs(denom) < 1e-12:
            return None

        if z0 <= near_z < z1:
            t = (near_z - z0) / denom
            p_clip = p0_cam + t * (p1_cam - p0_cam)
            p_clip = np.array([float(p_clip[0]), float(p_clip[1]), float(near_z)], dtype=float)
            return p_clip, p1_cam

        if z1 <= near_z < z0:
            t = (near_z - z0) / denom
            p_clip = p0_cam + t * (p1_cam - p0_cam)
            p_clip = np.array([float(p_clip[0]), float(p_clip[1]), float(near_z)], dtype=float)
            return p0_cam, p_clip

        return None

    @staticmethod
    def _format_age(now_sec: float, stamp_sec: Optional[float]) -> str:
        if stamp_sec is None:
            return "n/a"
        age_sec = max(0.0, now_sec - float(stamp_sec))
        return f"{age_sec:.3f}s"


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = SceneLocalizerDebugNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
