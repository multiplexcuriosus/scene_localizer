#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from scene_localizer.event_ball_geometry import (
    BallLocalizationError,
    camera_ray_from_pixel,
    intersect_camera_ray_with_ball_plane,
    make_ball_point_messages,
)
from scene_localizer.latency_trace import LatencyTracer
from sensor_msgs.msg import CameraInfo


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


@dataclass
class CachedPoseData:
    camera_info: Optional[CameraInfo] = None
    camera_info_time_sec: Optional[float] = None
    table_pose: Optional[PoseStamped] = None
    table_pose_time_sec: Optional[float] = None


class Ball3DPoseEstimatorNode(Node):
    def __init__(self) -> None:
        super().__init__("ball_3d_pose_estimator")

        self._last_warn_time_ns: Dict[str, int] = {}
        self._cached = CachedPoseData()

        self.declare_parameter("ball_px_topic", "/ball_tracker2/ball_2d_px")
        self.declare_parameter("camera_info_topic", "/top_cam/camera/color/camera_info")
        self.declare_parameter("table_pose_topic", "/scene_localizer/top_cam/table_pose_camera")
        self.declare_parameter("ball_3d_camera_topic", "/scene_localizer/top_cam/ball_3d_camera")
        self.declare_parameter("ball_3d_table_topic", "/scene_localizer/top_cam/ball_3d_table")
        self.declare_parameter("table_frame", "table_frame")
        self.declare_parameter("ball_radius", 0.0325)
        self.declare_parameter("table_shortedge", 0.6)
        self.declare_parameter("table_longedge", 1.2)
        # The RGB default preserves the legacy direct-pinhole behavior. Set
        # false only when the supplied pixels are raw and CameraInfo K/D
        # describe those same raw coordinates (as in event_ball_pipeline.yaml).
        self.declare_parameter("input_pixels_are_rectified", True)
        self.declare_parameter("max_table_pose_age_sec", 1.0)
        self.declare_parameter("max_camera_info_age_sec", 5.0)
        self.declare_parameter("reject_outside_table", False)
        self.declare_parameter("outside_table_margin", 0.05)
        self.declare_parameter("debug_log", True)
        self.declare_parameter("enable_latency_trace", False)
        self.declare_parameter("latency_trace_topic", "/intercept_trace/localization_2d_to_3d")
        self.declare_parameter("latency_run_id", "")
        self.declare_parameter("latency_modality", "vision")

        self._latency_tracer = LatencyTracer(
            self,
            enabled=bool(self.get_parameter("enable_latency_trace").value),
            topic=str(self.get_parameter("latency_trace_topic").value),
            run_id=str(self.get_parameter("latency_run_id").value),
            modality=str(self.get_parameter("latency_modality").value),
            stage="localization_2d_to_3d",
        )

        self._ball_3d_camera_pub = self.create_publisher(
            PointStamped,
            str(self.get_parameter("ball_3d_camera_topic").value),
            10,
        )
        self._ball_3d_table_pub = self.create_publisher(
            PointStamped,
            str(self.get_parameter("ball_3d_table_topic").value),
            10,
        )

        self.create_subscription(
            PointStamped,
            str(self.get_parameter("ball_px_topic").value),
            self._ball_px_callback,
            10,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._camera_info_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("table_pose_topic").value),
            self._table_pose_callback,
            10,
        )

        self.get_logger().info("ball_3d_pose_estimator started")
        self.get_logger().info(
            "Topics: "
            f"ball_px={self.get_parameter('ball_px_topic').value}, "
            f"camera_info={self.get_parameter('camera_info_topic').value}, "
            f"table_pose={self.get_parameter('table_pose_topic').value}, "
            f"ball_3d_camera={self.get_parameter('ball_3d_camera_topic').value}, "
            f"ball_3d_table={self.get_parameter('ball_3d_table_topic').value}, "
            "pixels_are_rectified="
            f"{self.get_parameter('input_pixels_are_rectified').value}"
        )

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self._cached.camera_info = msg
        self._cached.camera_info_time_sec = self._stamp_to_sec(msg)

    def _table_pose_callback(self, msg: PoseStamped) -> None:
        self._cached.table_pose = msg
        self._cached.table_pose_time_sec = self._stamp_to_sec(msg)

    def _ball_px_callback(self, msg: PointStamped) -> None:
        trace_span = self._latency_tracer.begin(msg)
        trace_valid = False
        trace_event = "rejected"
        output_stamp_ns = None
        try:
            trace_valid = self._localize_ball_px(msg)
            if trace_valid:
                trace_event = "published"
                output_stamp_ns = self.get_clock().now().nanoseconds
        finally:
            self._latency_tracer.finish(
                trace_span,
                valid=trace_valid,
                event=trace_event,
                detail=(
                    {"output_ros_stamp_ns": output_stamp_ns}
                    if output_stamp_ns is not None
                    else None
                ),
                end_ros_stamp_ns=output_stamp_ns,
            )

    def _localize_ball_px(self, msg: PointStamped) -> bool:
        uv = self._extract_uv(msg)
        if uv is None:
            self._warn_throttled(
                "ball_px_format",
                "Unsupported ball pixel message format; expected PointStamped/Point-like fields",
                2.0,
            )
            return False

        camera_info = self._cached.camera_info
        if camera_info is None or self._cached.camera_info_time_sec is None:
            self._warn_throttled("camera_info_missing", "No camera_info cached yet", 1.0)
            return False

        table_pose_msg = self._cached.table_pose
        if table_pose_msg is None or self._cached.table_pose_time_sec is None:
            self._warn_throttled("table_pose_missing", "No table_pose cached yet", 1.0)
            return False

        ball_stamp_sec = self._stamp_to_sec(msg)
        if self._is_stale(
            ball_stamp_sec,
            self._cached.camera_info_time_sec,
            float(self.get_parameter("max_camera_info_age_sec").value),
        ):
            self._warn_throttled("camera_info_stale", "Cached camera_info is stale", 1.0)
            return False

        if self._is_stale(
            ball_stamp_sec,
            self._cached.table_pose_time_sec,
            float(self.get_parameter("max_table_pose_age_sec").value),
        ):
            self._warn_throttled("table_pose_stale", "Cached table_pose is stale", 1.0)
            return False

        u, v = uv
        try:
            camera_matrix = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
            ray_cam = camera_ray_from_pixel(
                u,
                v,
                camera_matrix,
                np.asarray(camera_info.d, dtype=np.float64),
                pixels_are_rectified=bool(
                    self.get_parameter("input_pixels_are_rectified").value
                ),
                distortion_model=str(camera_info.distortion_model),
            )
        except (BallLocalizationError, TypeError, ValueError) as error:
            self._warn_throttled(
                "invalid_calibrated_ray",
                f"Failed to construct calibrated camera ray: {error}",
                1.0,
            )
            return False

        pose = table_pose_msg.pose
        t_ct = np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=float,
        )
        q_ct = np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=float,
        )
        r_ct = _quaternion_to_matrix_xyzw(q_ct)

        ball_radius = max(0.0, float(self.get_parameter("ball_radius").value))
        T_camera_table = np.eye(4, dtype=np.float64)
        T_camera_table[:3, :3] = r_ct
        T_camera_table[:3, 3] = t_ct
        try:
            p_ball_cam, p_ball_table = intersect_camera_ray_with_ball_plane(
                ray_cam,
                T_camera_table,
                ball_radius,
            )
        except BallLocalizationError as error:
            self._warn_throttled(
                "invalid_ray_plane_intersection",
                f"Failed to intersect ball ray with table plane: {error}",
                1.0,
            )
            return False

        outside_table = self._is_outside_table(p_ball_table)
        if bool(self.get_parameter("reject_outside_table").value) and outside_table:
            self._warn_throttled(
                "outside_table",
                "Rejected ball estimate outside table bounds",
                1.0,
            )
            return False

        camera_frame_id = str(getattr(camera_info.header, "frame_id", "")).strip()
        if not camera_frame_id:
            camera_frame_id = str(getattr(table_pose_msg.header, "frame_id", "")).strip()

        table_frame_id = str(self.get_parameter("table_frame").value)

        ball_stamp_msg = self._stamp_msg(msg)
        camera_msg, table_msg = make_ball_point_messages(
            p_ball_cam,
            p_ball_table,
            ball_stamp_msg,
            camera_frame_id,
            table_frame_id,
        )

        self._ball_3d_camera_pub.publish(camera_msg)
        self._ball_3d_table_pub.publish(table_msg)
        return True

        # self._log_debug(
        #     "ball_2d_px="
        #     f"({u:.3f}, {v:.3f}), "
        #     f"ball_3d_camera=({p_ball_cam[0]:.4f}, {p_ball_cam[1]:.4f}, {p_ball_cam[2]:.4f}), "
        #     "ball_3d_table="
        #     f"({p_ball_table[0]:.4f}, {p_ball_table[1]:.4f}, "
        #     f"{p_ball_table[2]:.4f}), "
        #     f"outside_table={outside_table}"
        # )

    def _extract_intrinsics(self, camera_info: CameraInfo) -> Tuple[float, float, float, float]:
        if len(camera_info.k) < 9:
            return float("nan"), float("nan"), float("nan"), float("nan")
        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        return fx, fy, cx, cy

    def _is_outside_table(self, p_ball_table: np.ndarray) -> bool:
        margin = max(0.0, float(self.get_parameter("outside_table_margin").value))
        table_shortedge = float(self.get_parameter("table_shortedge").value)
        table_longedge = float(self.get_parameter("table_longedge").value)
        return bool(
            p_ball_table[0] < -margin
            or p_ball_table[0] > table_shortedge + margin
            or p_ball_table[1] < -margin
            or p_ball_table[1] > table_longedge + margin
        )

    def _extract_uv(self, msg: Any) -> Optional[Tuple[float, float]]:
        candidates = []

        point = getattr(msg, "point", None)
        if point is not None:
            candidates.append((getattr(point, "x", None), getattr(point, "y", None)))

        candidates.append((getattr(msg, "x", None), getattr(msg, "y", None)))
        candidates.append((getattr(msg, "u", None), getattr(msg, "v", None)))

        for u_raw, v_raw in candidates:
            if u_raw is None or v_raw is None:
                continue
            try:
                u = float(u_raw)
                v = float(v_raw)
            except Exception:
                continue
            if np.isfinite(u) and np.isfinite(v):
                return u, v

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

    def _stamp_msg(self, msg: Any):
        header = getattr(msg, "header", None)
        if header is not None and hasattr(header, "stamp"):
            stamp = header.stamp
            sec = float(getattr(stamp, "sec", 0.0))
            nanosec = float(getattr(stamp, "nanosec", 0.0))
            if sec + nanosec * 1e-9 > 0.0:
                return stamp
        return self.get_clock().now().to_msg()

    @staticmethod
    def _is_stale(now_sec: float, stamp_sec: float, max_age_sec: float) -> bool:
        return (now_sec - stamp_sec) > max(0.0, max_age_sec)

    def _log_debug(self, message: str) -> None:
        if bool(self.get_parameter("debug_log").value):
            self.get_logger().info(message)

    def _warn_throttled(self, key: str, message: str, throttle_sec: float = 1.0) -> None:
        now_ns = self.get_clock().now().nanoseconds
        last_ns = self._last_warn_time_ns.get(key, 0)
        if now_ns - last_ns >= int(throttle_sec * 1e9):
            self._last_warn_time_ns[key] = now_ns
            self.get_logger().warn(message)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = Ball3DPoseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
