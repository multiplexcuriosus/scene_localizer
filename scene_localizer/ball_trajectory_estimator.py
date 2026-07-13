#!/usr/bin/env python3

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
import rclpy.time
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from scene_localizer.msg import BallTrajectory
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker


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

    Convention used here:
    - robot_base_table_pose_topic is interpreted as T_robot_base_table.
    - TF lookup target=robot_base, source=tcp gives T_robot_base_tcp.
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


def _is_finite_vector(v: np.ndarray) -> bool:
    return bool(v.shape == (3,) and np.all(np.isfinite(v)))


@dataclass
class BufferEntry:
    stamp_sec: float
    position: np.ndarray


@dataclass
class FitResult:
    valid: bool
    reason: str
    start_point: np.ndarray
    end_point: np.ndarray
    start_stamp_sec: float
    end_stamp_sec: float
    velocity: np.ndarray
    fit_rms_error: float
    num_observations: int
    middle_anchor_point: Optional[np.ndarray] = None
    middle_direction: Optional[np.ndarray] = None
    intersection_along_ball_m: float = float("nan")
    intersection_along_middle_m: float = float("nan")


class BallTrajectoryEstimator(Node):
    def __init__(self) -> None:
        super().__init__("ball_trajectory_estimator")

        self._last_warn_time_ns: Dict[str, int] = {}
        self._buffer: List[BufferEntry] = []
        self._direction_history: Deque[np.ndarray] = deque()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._latest_robot_base_table_pose: Optional[PoseStamped] = None
        self._latest_robot_base_table_pose_time_sec: Optional[float] = None

        self.declare_parameter("input_topic", "")
        self.declare_parameter("output_topic", "/scene/ball_trajectory_table")
        self.declare_parameter("marker_topic", "/scene/ball_trajectory_marker")
        self.declare_parameter("table_frame", "table_frame")

        # Ball-fit checks.
        self.declare_parameter("min_observations", 3)
        self.declare_parameter("max_buffer_size", 10)
        self.declare_parameter("buffer_time_window_sec", 0.5)
        self.declare_parameter("min_time_span_sec", 0.05)
        self.declare_parameter("max_detection_age_sec", 0.25)
        self.declare_parameter("min_speed_mps", 0.05)
        self.declare_parameter("max_speed_mps", 10.0)
        self.declare_parameter("min_xy_speed_mps", 0.05)
        self.declare_parameter("max_fit_rms_error_m", 0.03)

        # Deprecated/kept for YAML compatibility. The estimator now publishes
        # latest-ball-point -> middle-line-intersection when valid.
        self.declare_parameter("end_point_mode", "intersection")
        self.declare_parameter("trajectory_forward_time_sec", 0.15)

        # Conservative direction gating.
        self.declare_parameter("direction_stability_frame_count", 3)
        self.declare_parameter("direction_stability_max_angle_deg", 15.0)

        # Middle-line / TCP intersection checks.
        self.declare_parameter("ball_radius", 0.035)
        self.declare_parameter("tcp_middle_line_length", 0.6)
        self.declare_parameter("require_intersection_within_middle_line_segment", False)
        self.declare_parameter("middle_line_segment_margin_m", 0.02)
        self.declare_parameter("allow_backward_intersection", False)

        # Robot-frame inputs required for T_table_tcp.
        self.declare_parameter("robot_base_table_pose_topic", "/scene_localizer/table_pose_robot_base")
        self.declare_parameter("robot_base_frame", "base")
        self.declare_parameter("tcp_frame", "right_fr3_hand_tcp")
        self.declare_parameter("robot_base_table_pose_timeout_sec", 1.0)
        self.declare_parameter("tf_lookup_timeout_sec", 0.05)

        self.declare_parameter("publish_invalid_trajectory", True)
        self.declare_parameter("marker_scale", 0.01)
        self.declare_parameter("marker_r", 1.0)
        self.declare_parameter("marker_g", 0.2)
        self.declare_parameter("marker_b", 0.0)
        self.declare_parameter("marker_a", 1.0)
        self.declare_parameter("debug_log", False)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        marker_topic = str(self.get_parameter("marker_topic").value)
        robot_base_table_pose_topic = str(self.get_parameter("robot_base_table_pose_topic").value)

        self._sub = self.create_subscription(PointStamped, input_topic, self.handle_ball_position, 20)
        self._robot_base_table_pose_sub = self.create_subscription(
            PoseStamped,
            robot_base_table_pose_topic,
            self._robot_base_table_pose_callback,
            10,
        )
        self._traj_pub = self.create_publisher(BallTrajectory, output_topic, 10)
        self._marker_pub = self.create_publisher(Marker, marker_topic, 10)
        self._reset_srv = self.create_service(
            Trigger,
            "~/reset",
            self._handle_reset,
        )

        self.get_logger().info(
            "ball_trajectory_estimator started with params: "
            f"input_topic={input_topic}, output_topic={output_topic}, marker_topic={marker_topic}, "
            f"table_frame={self.get_parameter('table_frame').value}, "
            f"robot_base_table_pose_topic={robot_base_table_pose_topic}, "
            f"robot_base_frame={self.get_parameter('robot_base_frame').value}, "
            f"tcp_frame={self.get_parameter('tcp_frame').value}, "
            f"ball_radius={self.get_parameter('ball_radius').value}, "
            f"tcp_middle_line_length={self.get_parameter('tcp_middle_line_length').value}, "
            f"min_observations={self.get_parameter('min_observations').value}, "
            f"min_speed_mps={self.get_parameter('min_speed_mps').value}, "
            f"min_xy_speed_mps={self.get_parameter('min_xy_speed_mps').value}, "
            f"max_fit_rms_error_m={self.get_parameter('max_fit_rms_error_m').value}, "
            f"direction_stability_frame_count={self.get_parameter('direction_stability_frame_count').value}, "
            f"direction_stability_max_angle_deg={self.get_parameter('direction_stability_max_angle_deg').value}, "
            f"publish_invalid_trajectory={self.get_parameter('publish_invalid_trajectory').value}"
        )

    def _handle_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        dropped_samples = len(self._buffer)
        dropped_directions = len(self._direction_history)
        self._buffer.clear()
        self._direction_history.clear()

        response.success = True
        response.message = (
            f"Cleared estimator state: samples={dropped_samples}, "
            f"direction_history={dropped_directions}"
        )
        self.get_logger().info(response.message)
        return response

    def _log_debug(self, message: str) -> None:
        if bool(self.get_parameter("debug_log").value):
            self.get_logger().info(message)

    def _warn_throttled(self, key: str, message: str, throttle_sec: float = 1.0) -> None:
        now_ns = self.get_clock().now().nanoseconds
        last_ns = self._last_warn_time_ns.get(key, 0)
        if now_ns - last_ns >= int(throttle_sec * 1e9):
            self._last_warn_time_ns[key] = now_ns
            self.get_logger().warn(message)

    def _robot_base_table_pose_callback(self, msg: PoseStamped) -> None:
        self._latest_robot_base_table_pose = msg
        self._latest_robot_base_table_pose_time_sec = self._stamp_to_sec_generic(msg)

    def _stamp_to_sec(self, msg: PointStamped) -> Tuple[float, bool]:
        if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
            sec = float(getattr(msg.header.stamp, "sec", 0.0))
            nanosec = float(getattr(msg.header.stamp, "nanosec", 0.0))
            stamp_sec = sec + nanosec * 1e-9
            if stamp_sec > 0.0:
                return stamp_sec, False
        return self.get_clock().now().nanoseconds * 1e-9, True

    def _stamp_to_sec_generic(self, msg) -> float:
        header = getattr(msg, "header", None)
        if header is not None and hasattr(header, "stamp"):
            sec = float(getattr(header.stamp, "sec", 0.0))
            nanosec = float(getattr(header.stamp, "nanosec", 0.0))
            stamp_sec = sec + nanosec * 1e-9
            if stamp_sec > 0.0:
                return stamp_sec
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _sec_to_stamp(stamp_sec: float):
        total_ns = int(max(0.0, float(stamp_sec)) * 1e9)
        return rclpy.time.Time(nanoseconds=total_ns).to_msg()

    def handle_ball_position(self, msg: PointStamped) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        table_frame = str(self.get_parameter("table_frame").value)

        frame_id = str(getattr(msg.header, "frame_id", "")).strip()
        if frame_id and frame_id != table_frame:
            self._warn_throttled(
                "invalid_frame",
                f"Ignoring detection with frame_id='{frame_id}' (expected '{table_frame}')",
                1.0,
            )
            return

        p = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float)
        if not np.all(np.isfinite(p)):
            self._warn_throttled("non_finite_detection", "Ignoring detection with NaN/inf coordinates", 1.0)
            return

        stamp_sec, used_fallback = self._stamp_to_sec(msg)
        if used_fallback:
            self._warn_throttled(
                "zero_stamp_detection",
                "Detection timestamp was zero/invalid; using receive time as fallback",
                2.0,
            )

        max_detection_age_sec = max(0.0, float(self.get_parameter("max_detection_age_sec").value))
        if (now_sec - stamp_sec) > max_detection_age_sec:
            self._warn_throttled(
                "stale_detection",
                f"Ignoring stale detection age={now_sec - stamp_sec:.3f}s (max={max_detection_age_sec:.3f}s)",
                1.0,
            )
            return

        self._buffer.append(BufferEntry(stamp_sec=stamp_sec, position=p))
        self.prune_buffer(now_sec)

        fit = self.fit_trajectory(now_sec)
        if not fit.valid:
            self._warn_throttled(f"fit_invalid_{fit.reason}", f"Trajectory invalid: {fit.reason}", 1.0)

        if fit.valid or bool(self.get_parameter("publish_invalid_trajectory").value):
            self.publish_trajectory(fit)

        self.publish_marker(fit)

    def prune_buffer(self, now_sec: Optional[float] = None) -> None:
        if now_sec is None:
            now_sec = self.get_clock().now().nanoseconds * 1e-9

        window_sec = max(0.0, float(self.get_parameter("buffer_time_window_sec").value))
        cutoff = now_sec - window_sec
        self._buffer = [entry for entry in self._buffer if entry.stamp_sec >= cutoff]

        max_buffer_size = max(1, int(self.get_parameter("max_buffer_size").value))
        if len(self._buffer) > max_buffer_size:
            self._buffer = self._buffer[-max_buffer_size:]

    def _invalid_result(
        self,
        reason: str,
        start_point: np.ndarray,
        end_point: np.ndarray,
        start_stamp_sec: float,
        end_stamp_sec: float,
        velocity: np.ndarray,
        fit_rms_error: float,
        num_observations: int,
    ) -> FitResult:
        return FitResult(
            valid=False,
            reason=reason,
            start_point=start_point,
            end_point=end_point,
            start_stamp_sec=start_stamp_sec,
            end_stamp_sec=end_stamp_sec,
            velocity=velocity,
            fit_rms_error=fit_rms_error,
            num_observations=num_observations,
        )

    def fit_trajectory(self, now_sec: Optional[float] = None) -> FitResult:
        if now_sec is None:
            now_sec = self.get_clock().now().nanoseconds * 1e-9

        num_observations = len(self._buffer)
        zero_vec = np.zeros(3, dtype=float)
        default_stamp = now_sec

        if num_observations == 0:
            self._direction_history.clear()
            return self._invalid_result(
                "insufficient_samples",
                zero_vec,
                zero_vec,
                default_stamp,
                default_stamp,
                zero_vec,
                float("inf"),
                0,
            )

        min_observations = max(1, int(self.get_parameter("min_observations").value))
        if num_observations < min_observations:
            newest = self._buffer[-1]
            return self._invalid_result(
                "insufficient_samples",
                newest.position,
                newest.position,
                newest.stamp_sec,
                newest.stamp_sec,
                zero_vec,
                float("inf"),
                num_observations,
            )

        times = np.array([entry.stamp_sec for entry in self._buffer], dtype=float)
        positions = np.stack([entry.position for entry in self._buffer], axis=0)
        t_oldest = float(times[0])
        t_newest = float(times[-1])
        time_span = t_newest - t_oldest

        min_time_span_sec = max(0.0, float(self.get_parameter("min_time_span_sec").value))
        if time_span < min_time_span_sec:
            newest = self._buffer[-1]
            return self._invalid_result(
                "small_time_span",
                newest.position,
                newest.position,
                t_oldest,
                t_newest,
                zero_vec,
                float("inf"),
                num_observations,
            )

        # Use mean sample time as t_ref to improve numerical stability.
        t_ref = float(np.mean(times))
        dt = times - t_ref
        A = np.column_stack((np.ones_like(dt), dt))
        coeffs, _, _, _ = np.linalg.lstsq(A, positions, rcond=None)
        p_ref = coeffs[0, :]
        velocity = coeffs[1, :]

        predicted = A @ coeffs
        residuals = positions - predicted
        fit_rms_error = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))

        oldest_fit = p_ref + velocity * (t_oldest - t_ref)
        latest_fit = p_ref + velocity * (t_newest - t_ref)

        speed = float(np.linalg.norm(velocity))
        min_speed_mps = max(0.0, float(self.get_parameter("min_speed_mps").value))
        max_speed_mps = max(min_speed_mps, float(self.get_parameter("max_speed_mps").value))
        max_fit_rms_error_m = max(0.0, float(self.get_parameter("max_fit_rms_error_m").value))

        if speed < min_speed_mps:
            self._direction_history.clear()
            return self._invalid_result(
                "speed_too_low",
                oldest_fit,
                latest_fit,
                t_oldest,
                t_newest,
                velocity,
                fit_rms_error,
                num_observations,
            )
        if speed > max_speed_mps:
            self._direction_history.clear()
            return self._invalid_result(
                "speed_too_high",
                oldest_fit,
                latest_fit,
                t_oldest,
                t_newest,
                velocity,
                fit_rms_error,
                num_observations,
            )
        if fit_rms_error > max_fit_rms_error_m:
            self._direction_history.clear()
            return self._invalid_result(
                "high_rms",
                oldest_fit,
                latest_fit,
                t_oldest,
                t_newest,
                velocity,
                fit_rms_error,
                num_observations,
            )

        ball_radius = max(0.0, float(self.get_parameter("ball_radius").value))
        current_on_ball_plane = np.array([latest_fit[0], latest_fit[1], ball_radius], dtype=float)

        xy_velocity = np.array([velocity[0], velocity[1], 0.0], dtype=float)
        xy_speed = float(np.linalg.norm(xy_velocity))
        min_xy_speed_mps = max(0.0, float(self.get_parameter("min_xy_speed_mps").value))
        if xy_speed < min_xy_speed_mps:
            self._direction_history.clear()
            return self._invalid_result(
                "xy_speed_too_low",
                current_on_ball_plane,
                current_on_ball_plane,
                t_newest,
                t_newest,
                velocity,
                fit_rms_error,
                num_observations,
            )

        ball_direction_xy = xy_velocity / xy_speed
        self._push_direction(ball_direction_xy)
        if not self._direction_is_stable(ball_direction_xy):
            return self._invalid_result(
                "unstable_direction",
                current_on_ball_plane,
                current_on_ball_plane,
                t_newest,
                t_newest,
                velocity,
                fit_rms_error,
                num_observations,
            )

        geometry = self._compute_middle_line_geometry(now_sec)
        if geometry is None:
            return self._invalid_result(
                "middle_line_unavailable",
                current_on_ball_plane,
                current_on_ball_plane,
                t_newest,
                t_newest,
                velocity,
                fit_rms_error,
                num_observations,
            )

        middle_anchor, middle_direction = geometry
        hit = self._compute_ball_middle_plane_intersection(
            current_on_ball_plane=current_on_ball_plane,
            ball_direction_xy=ball_direction_xy,
            middle_anchor=middle_anchor,
            middle_direction=middle_direction,
        )
        if hit is None:
            return self._invalid_result(
                "no_middle_line_intersection",
                current_on_ball_plane,
                current_on_ball_plane,
                t_newest,
                t_newest,
                velocity,
                fit_rms_error,
                num_observations,
            )

        hit_point, along_ball_m, along_middle_m = hit

        if bool(self.get_parameter("require_intersection_within_middle_line_segment").value):
            line_length = max(0.0, float(self.get_parameter("tcp_middle_line_length").value))
            margin = max(0.0, float(self.get_parameter("middle_line_segment_margin_m").value))
            if along_middle_m < -margin or along_middle_m > line_length + margin:
                return self._invalid_result(
                    "intersection_outside_middle_segment",
                    current_on_ball_plane,
                    hit_point,
                    t_newest,
                    t_newest,
                    velocity,
                    fit_rms_error,
                    num_observations,
                )

        hit_stamp_sec = t_newest
        if xy_speed > 1e-9:
            hit_stamp_sec = t_newest + max(0.0, along_ball_m) / xy_speed

        self._log_debug(
            f"fit ok: speed={speed:.3f} xy_speed={xy_speed:.3f} rms={fit_rms_error:.4f}, "
            f"hit=({hit_point[0]:.3f},{hit_point[1]:.3f},{hit_point[2]:.3f}), "
            f"s_ball={along_ball_m:.3f}, s_middle={along_middle_m:.3f}"
        )

        return FitResult(
            valid=True,
            reason="ok",
            start_point=current_on_ball_plane,
            end_point=hit_point,
            start_stamp_sec=t_newest,
            end_stamp_sec=hit_stamp_sec,
            velocity=velocity,
            fit_rms_error=fit_rms_error,
            num_observations=num_observations,
            middle_anchor_point=middle_anchor,
            middle_direction=middle_direction,
            intersection_along_ball_m=along_ball_m,
            intersection_along_middle_m=along_middle_m,
        )

    def _push_direction(self, direction: np.ndarray) -> None:
        if not _is_finite_vector(direction):
            return
        stability_count = max(1, int(self.get_parameter("direction_stability_frame_count").value))
        self._direction_history.append(direction.copy())
        while len(self._direction_history) > stability_count:
            self._direction_history.popleft()

    def _direction_is_stable(self, current_direction: np.ndarray) -> bool:
        stability_count = max(1, int(self.get_parameter("direction_stability_frame_count").value))
        if len(self._direction_history) < stability_count:
            return False

        max_angle_deg = max(0.0, float(self.get_parameter("direction_stability_max_angle_deg").value))
        min_dot = float(np.cos(np.deg2rad(max_angle_deg)))
        for direction in self._direction_history:
            dot = float(np.dot(current_direction, direction))
            if not np.isfinite(dot) or dot < min_dot:
                return False
        return True

    def _compute_middle_line_geometry(self, now_sec: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        transforms = self._compute_table_tcp_and_base_table(now_sec)
        if transforms is None:
            return None

        T_table_tcp, T_base_table = transforms
        tcp_origin_table = T_table_tcp[:3, 3].copy()
        ball_radius = max(0.0, float(self.get_parameter("ball_radius").value))

        # Q is the TCP center projected onto the table x-y plane.
        # The middle-line anchor is Q lifted by ball_radius.
        middle_anchor = np.array([tcp_origin_table[0], tcp_origin_table[1], ball_radius], dtype=float)

        # Express robot-base +x in table coordinates. Since T_base_table maps
        # table -> base, its transpose rotation maps base vectors -> table.
        middle_direction = T_base_table[:3, :3].T @ np.array([1.0, 0.0, 0.0], dtype=float)
        middle_direction = middle_direction.astype(float)
        middle_direction[2] = 0.0
        norm = float(np.linalg.norm(middle_direction))
        if norm < 1e-9:
            self._warn_throttled("base_x_degenerate", "robot-base +x projected to table xy is degenerate", 1.0)
            return None
        middle_direction /= norm

        return middle_anchor, middle_direction

    def _compute_ball_middle_plane_intersection(
        self,
        current_on_ball_plane: np.ndarray,
        ball_direction_xy: np.ndarray,
        middle_anchor: np.ndarray,
        middle_direction: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, float, float]]:
        if (
            not _is_finite_vector(current_on_ball_plane)
            or not _is_finite_vector(ball_direction_xy)
            or not _is_finite_vector(middle_anchor)
            or not _is_finite_vector(middle_direction)
        ):
            return None

        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        plane_normal = np.cross(middle_direction, z_axis)
        normal_norm = float(np.linalg.norm(plane_normal))
        if normal_norm < 1e-9:
            return None
        plane_normal /= normal_norm

        denom = float(np.dot(plane_normal, ball_direction_xy))
        if abs(denom) < 1e-9:
            return None

        along_ball_m = float(np.dot(plane_normal, middle_anchor - current_on_ball_plane) / denom)
        if along_ball_m < 0.0 and not bool(self.get_parameter("allow_backward_intersection").value):
            return None

        hit_point = current_on_ball_plane + along_ball_m * ball_direction_xy
        hit_point[2] = middle_anchor[2]

        along_middle_m = float(np.dot(hit_point - middle_anchor, middle_direction))
        return hit_point, along_ball_m, along_middle_m

    def _compute_table_tcp_and_base_table(self, now_sec: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        T_base_table = self._get_robot_base_from_table_transform(now_sec)
        if T_base_table is None:
            return None

        T_base_tcp = self._lookup_robot_base_from_tcp_transform()
        if T_base_tcp is None:
            return None

        T_table_tcp = _invert_transform(T_base_table) @ T_base_tcp
        return T_table_tcp, T_base_table

    def _get_robot_base_from_table_transform(self, now_sec: float) -> Optional[np.ndarray]:
        if self._latest_robot_base_table_pose is None or self._latest_robot_base_table_pose_time_sec is None:
            self._warn_throttled(
                "base_table_pose_missing",
                "robot_base_table_pose unavailable: waiting for PoseStamped on robot_base_table_pose_topic",
                1.0,
            )
            return None

        timeout_sec = max(0.0, float(self.get_parameter("robot_base_table_pose_timeout_sec").value))
        pose_age_sec = now_sec - self._latest_robot_base_table_pose_time_sec
        if pose_age_sec > timeout_sec:
            self._warn_throttled(
                "base_table_pose_stale",
                f"robot_base_table_pose stale: age={pose_age_sec:.3f}s timeout={timeout_sec:.3f}s",
                1.0,
            )
            return None

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

    def _lookup_robot_base_from_tcp_transform(self) -> Optional[np.ndarray]:
        robot_base_frame = str(self.get_parameter("robot_base_frame").value)
        tcp_frame = str(self.get_parameter("tcp_frame").value)
        return self._lookup_transform_matrix(robot_base_frame, tcp_frame, "base_to_tcp")

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

    def publish_trajectory(self, fit: FitResult) -> None:
        table_frame = str(self.get_parameter("table_frame").value)

        msg = BallTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = table_frame

        msg.start_point.x = float(fit.start_point[0])
        msg.start_point.y = float(fit.start_point[1])
        msg.start_point.z = float(fit.start_point[2])
        msg.start_stamp = self._sec_to_stamp(fit.start_stamp_sec)

        msg.end_point.x = float(fit.end_point[0])
        msg.end_point.y = float(fit.end_point[1])
        msg.end_point.z = float(fit.end_point[2])
        msg.end_stamp = self._sec_to_stamp(fit.end_stamp_sec)

        msg.velocity.x = float(fit.velocity[0])
        msg.velocity.y = float(fit.velocity[1])
        msg.velocity.z = float(fit.velocity[2])

        msg.num_observations = int(fit.num_observations)
        msg.fit_rms_error = float(fit.fit_rms_error)
        msg.valid = bool(fit.valid)

        self._traj_pub.publish(msg)

    def publish_marker(self, fit: FitResult) -> None:
        table_frame = str(self.get_parameter("table_frame").value)

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = table_frame
        marker.ns = "ball_trajectory"
        marker.id = 0

        if not fit.valid:
            marker.action = Marker.DELETE
            self._marker_pub.publish(marker)
            return

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = max(1e-5, float(self.get_parameter("marker_scale").value))
        marker.color.r = float(self.get_parameter("marker_r").value)
        marker.color.g = float(self.get_parameter("marker_g").value)
        marker.color.b = float(self.get_parameter("marker_b").value)
        marker.color.a = float(self.get_parameter("marker_a").value)

        start_point = Point()
        start_point.x = float(fit.start_point[0])
        start_point.y = float(fit.start_point[1])
        start_point.z = float(fit.start_point[2])

        end_point = Point()
        end_point.x = float(fit.end_point[0])
        end_point.y = float(fit.end_point[1])
        end_point.z = float(fit.end_point[2])

        marker.points = [start_point, end_point]
        self._marker_pub.publish(marker)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = BallTrajectoryEstimator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
