#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
import rclpy.time
from aruco_opencv_msgs.msg import ArucoDetection
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
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
    debug_image_topic: str
    trajectory_debug_image_topic: str
    debug_pub: Any
    trajectory_debug_pub: Any
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
        self._text_y_by_frame_id: Dict[int, int] = {}
        self._last_warn_time_ns: Dict[str, int] = {}

        # Top camera only. The eef-camera stream and old T_table_eef_cam /
        # T_tcp_end_cam logic were intentionally removed.
        self.declare_parameter("top_image_topic", "/top_cam/camera/color/image_raw")
        self.declare_parameter("top_camera_info_topic", "/top_cam/camera/color/camera_info")
        self.declare_parameter("top_detections_topic", "/aruco_top_cam/aruco_detections")
        self.declare_parameter("top_debug_image_topic", "/scene_localizer/top_cam/reprojection_debug")
        self.declare_parameter("top_trajectory_debug_image_topic", "/scene_localizer/top_cam/ball_trajectory_debug")

        # Publishes/consumes T_top_camera_table:
        # pose of table_frame expressed in top_camera_frame.
        self.declare_parameter("top_table_pose_topic", "/scene_localizer/top_cam/table_pose_camera")

        # Ball trajectory is expressed in table_frame.
        self.declare_parameter("ball_trajectory_topic", "/scene/ball_trajectory_table")

        # Publishes/consumes T_robot_base_table:
        # pose of table_frame expressed in robot_base_frame.
        self.declare_parameter("robot_base_table_pose_topic", "/scene_localizer/table_pose_robot_base")

        # Current frame knowledge from your calibration YAML:
        # /tmp/T_base_cam.yaml currently uses parent_frame: base and
        # child_frame: camera_color_optical_frame. Keep these defaults aligned.

        # robot_base_frame:
        # Parent frame for robot/world-side overlays.
        # With current YAML, T_robot_base_top_camera = T_base_camera_color_optical_frame.
        self.declare_parameter("robot_base_frame", "base")

        # top_camera_frame:
        # Optical camera frame used by image projection and ArUco detections.
        # T_top_camera_table maps table-frame points into this camera frame.
        self.declare_parameter("top_camera_frame", "camera_color_optical_frame")

        # table_frame:
        # Logical table coordinate system used by marker layout and ball trajectory.
        # T_robot_base_table maps table-frame points into robot_base_frame.
        self.declare_parameter("table_frame", "table_frame")

        # tcp_frame:
        # Robot TCP/interceptor frame looked up from TF as T_robot_base_tcp.
        # Middle-line drawing computes T_table_tcp = inverse(T_robot_base_table) * T_robot_base_tcp.
        self.declare_parameter("tcp_frame", "right_fr3_hand_tcp")

        self.declare_parameter("tf_lookup_timeout_sec", 0.05)
        self.declare_parameter("allow_base_table_fallback_from_tf", True)

        self.declare_parameter("physical_marker_size", 0.035)
        self.declare_parameter("axis_length", 0.08)
        self.declare_parameter("detection_timeout_sec", 0.3)
        self.declare_parameter("publish_debug_images", True)
        self.declare_parameter("publish_trajectory_debug_images", True)
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
        self.declare_parameter("draw_tcp_middle_line", True)
        self.declare_parameter("ball_radius", 0.0325)
        self.declare_parameter("tcp_middle_line_length", 0.3)
        self.declare_parameter("tcp_middle_line_half_length", 0.3)
        self.declare_parameter("tcp_middle_line_thickness", 3)
        self.declare_parameter("draw_trajectory_extrapolation", True)
        self.declare_parameter("trajectory_intersection_mode", "tcp_xz_plane")
        self.declare_parameter("allow_backward_trajectory_intersection", False)
        self.declare_parameter("middle_line_intersection_max_distance", 0.02)
        self.declare_parameter("trajectory_intersection_radius_px", 7)

        self._latest_ball_trajectory: Optional[BallTrajectory] = None
        self._latest_ball_trajectory_time_sec: Optional[float] = None
        self._latest_robot_base_table_pose: Optional[PoseStamped] = None
        self._latest_robot_base_table_pose_time_sec: Optional[float] = None

        self._top = self._make_top_camera_state(
            name="top_cam",
            image_topic=str(self.get_parameter("top_image_topic").value),
            camera_info_topic=str(self.get_parameter("top_camera_info_topic").value),
            detections_topic=str(self.get_parameter("top_detections_topic").value),
            table_pose_topic=str(self.get_parameter("top_table_pose_topic").value),
            debug_image_topic=str(self.get_parameter("top_debug_image_topic").value),
            trajectory_debug_image_topic=str(self.get_parameter("top_trajectory_debug_image_topic").value),
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
            10,
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

        self.get_logger().info("scene_localizer_debug started in top-cam-only mode")
        self.get_logger().info(
            f"[top_cam] image={self._top.image_topic}, "
            f"camera_info={self._top.camera_info_topic}, "
            f"detections={self._top.detections_topic}, "
            f"table_pose={self._top.table_pose_topic}, "
            f"debug_pub={self._top.debug_image_topic}, "
            f"trajectory_debug_pub={self._top.trajectory_debug_image_topic}"
        )
        self.get_logger().info(
            "Frame params: "
            f"robot_base_frame={str(self.get_parameter('robot_base_frame').value)}, "
            f"top_camera_frame={str(self.get_parameter('top_camera_frame').value)}, "
            f"table_frame={str(self.get_parameter('table_frame').value)}, "
            f"tcp_frame={str(self.get_parameter('tcp_frame').value)}"
        )
        self.get_logger().info(
            "Middle-line overlay now computes T_table_tcp = inv(T_base_table) * T_base_tcp. "
            "T_base_table comes from robot_base_table_pose_topic or, if enabled, from TF base->camera plus T_camera_table."
        )

    def _make_top_camera_state(
        self,
        name: str,
        image_topic: str,
        camera_info_topic: str,
        detections_topic: str,
        table_pose_topic: str,
        debug_image_topic: str,
        trajectory_debug_image_topic: str,
    ) -> TopCameraState:
        debug_pub = self.create_publisher(Image, debug_image_topic, 10)
        trajectory_debug_pub = self.create_publisher(Image, trajectory_debug_image_topic, 10)
        return TopCameraState(
            name=name,
            image_topic=image_topic,
            camera_info_topic=camera_info_topic,
            detections_topic=detections_topic,
            table_pose_topic=table_pose_topic,
            debug_image_topic=debug_image_topic,
            trajectory_debug_image_topic=trajectory_debug_image_topic,
            debug_pub=debug_pub,
            trajectory_debug_pub=trajectory_debug_pub,
        )

    def _warn_throttled(self, key: str, message: str, throttle_sec: float = 1.0) -> None:
        now_ns = self.get_clock().now().nanoseconds
        last_ns = self._last_warn_time_ns.get(key, 0)
        if now_ns - last_ns >= int(throttle_sec * 1e9):
            self._last_warn_time_ns[key] = now_ns
            self.get_logger().warn(message)

    def _ball_trajectory_callback(self, msg: BallTrajectory) -> None:
        self._latest_ball_trajectory = msg
        self._latest_ball_trajectory_time_sec = self._stamp_to_sec(msg)

    def _robot_base_table_pose_callback(self, msg: PoseStamped) -> None:
        self._latest_robot_base_table_pose = msg
        self._latest_robot_base_table_pose_time_sec = self._stamp_to_sec(msg)

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self._top.latest_camera_info = msg

    def _detections_callback(self, msg: ArucoDetection) -> None:
        self._top.latest_detections = msg
        self._top.latest_detection_time_sec = self._stamp_to_sec(msg)

    def _table_pose_callback(self, msg: PoseStamped) -> None:
        self._top.latest_table_pose = msg
        self._top.latest_table_pose_time_sec = self._stamp_to_sec(msg)

    def _image_callback(self, msg: Image) -> None:
        publish_reprojection_debug = bool(self.get_parameter("publish_debug_images").value)
        publish_trajectory_debug = bool(self.get_parameter("publish_trajectory_debug_images").value)
        if not publish_reprojection_debug and not publish_trajectory_debug:
            return

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"[top_cam] failed to convert image to bgr8: {exc}")
            return

        if publish_trajectory_debug:
            trajectory_frame = frame.copy()
            self._begin_frame_text(trajectory_frame)
            self._draw_projection_debug_text(trajectory_frame, msg)
            if self._top.latest_camera_info is None:
                self._draw_status_text(trajectory_frame, "no camera_info", (0, 140, 255))
            else:
                self._draw_table_rectangle_overlay(trajectory_frame, self._top.latest_camera_info)
                self._draw_ball_trajectory_overlay(trajectory_frame, self._top.latest_camera_info)
            self._publish_trajectory_debug_image(msg, trajectory_frame)

        if not publish_reprojection_debug:
            return

        self._begin_frame_text(frame)
        self._draw_projection_debug_text(frame, msg)

        if self._top.latest_camera_info is None:
            self._draw_status_text(frame, "no camera_info", (0, 140, 255))
            self._publish_debug_image(msg, frame)
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        timeout_sec = max(0.0, float(self.get_parameter("detection_timeout_sec").value))
        stale = (
            self._top.latest_detection_time_sec is None
            or (now_sec - self._top.latest_detection_time_sec) > timeout_sec
        )

        if stale:
            self._draw_status_text(frame, "detections stale", (0, 140, 255))
            self._draw_table_rectangle_overlay(frame, self._top.latest_camera_info)
            self._publish_debug_image(msg, frame)
            return

        if self._top.latest_detections is None:
            self._draw_status_text(frame, "no detections", (0, 140, 255))
            self._draw_table_rectangle_overlay(frame, self._top.latest_camera_info)
            self._publish_debug_image(msg, frame)
            return

        physical_marker_size = max(1e-6, float(self.get_parameter("physical_marker_size").value))
        axis_length = max(1e-6, float(self.get_parameter("axis_length").value))
        self._draw_detection_overlay(
            frame=frame,
            camera_info=self._top.latest_camera_info,
            detections_msg=self._top.latest_detections,
            physical_marker_size=physical_marker_size,
            axis_length=axis_length,
        )
        self._draw_table_rectangle_overlay(frame, self._top.latest_camera_info)
        self._publish_debug_image(msg, frame)

    def _publish_debug_image(self, src_msg: Image, frame: np.ndarray) -> None:
        debug_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        debug_msg.header = src_msg.header
        self._top.debug_pub.publish(debug_msg)

    def _publish_trajectory_debug_image(self, src_msg: Image, frame: np.ndarray) -> None:
        debug_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        debug_msg.header = src_msg.header
        self._top.trajectory_debug_pub.publish(debug_msg)

    def _begin_frame_text(self, frame: np.ndarray) -> None:
        self._text_y_by_frame_id[id(frame)] = 18

    def _draw_status_text(
        self,
        frame: np.ndarray,
        text: str,
        color: Tuple[int, int, int],
        scale: float = 0.48,
        thickness: int = 1,
    ) -> None:
        key = id(frame)
        y = self._text_y_by_frame_id.get(key, 18)
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        self._text_y_by_frame_id[key] = y + 18

    def _draw_projection_debug_text(self, frame: np.ndarray, image_msg: Image) -> None:
        """Draw compact status only; intentionally no camera-info/K/D text."""
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        image_frame_id = str(getattr(image_msg.header, "frame_id", ""))
        configured_camera_frame = str(self.get_parameter("top_camera_frame").value)

        detections_age = self._format_age(now_sec, self._top.latest_detection_time_sec)
        table_pose_age = self._format_age(now_sec, self._top.latest_table_pose_time_sec)
        base_table_age = self._format_age(now_sec, self._latest_robot_base_table_pose_time_sec)

        self._draw_status_text(
            frame,
            f"top_cam image: {int(getattr(image_msg, 'width', 0))}x{int(getattr(image_msg, 'height', 0))} frame={image_frame_id}",
            (255, 255, 255),
        )
        self._draw_status_text(
            frame,
            f"configured camera frame: {configured_camera_frame}",
            (255, 255, 255),
        )
        self._draw_status_text(frame, f"table_pose_camera age: {table_pose_age}", (255, 255, 255))
        self._draw_status_text(frame, f"detections age: {detections_age}", (255, 255, 255))
        self._draw_status_text(frame, f"base_table pose age: {base_table_age}", (255, 255, 255))

    def _draw_detection_overlay(
        self,
        frame: np.ndarray,
        camera_info: CameraInfo,
        detections_msg: ArucoDetection,
        physical_marker_size: float,
        axis_length: float,
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

            t_cm = np.array(
                [pose.position.x, pose.position.y, pose.position.z],
                dtype=float,
            )
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

            square_pixels: List[Tuple[int, int]] = []
            for p_local in square_local:
                p_cam = r_cm @ p_local + t_cm
                uv = self._project_point(camera_info, p_cam)
                if uv is None:
                    square_pixels = []
                    break
                square_pixels.append(uv)

            center_uv = self._project_point(camera_info, t_cm)

            origin_uv = self._project_point(camera_info, r_cm @ origin + t_cm)
            x_uv = self._project_point(camera_info, r_cm @ axis_x + t_cm)
            y_uv = self._project_point(camera_info, r_cm @ axis_y + t_cm)
            z_uv = self._project_point(camera_info, r_cm @ axis_z + t_cm)

            if square_pixels:
                pts = np.array(square_pixels, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

            if center_uv is not None:
                cv2.circle(frame, center_uv, 4, (255, 255, 0), -1)
                label = f"id:{marker_id}" if marker_id is not None else "id:?"
                cv2.putText(
                    frame,
                    label,
                    (center_uv[0] + 6, center_uv[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if origin_uv is not None and x_uv is not None:
                cv2.line(frame, origin_uv, x_uv, (0, 0, 255), 2)
            if origin_uv is not None and y_uv is not None:
                cv2.line(frame, origin_uv, y_uv, (0, 255, 0), 2)
            if origin_uv is not None and z_uv is not None:
                cv2.line(frame, origin_uv, z_uv, (255, 0, 0), 2)

    def _draw_table_rectangle_overlay(self, frame: np.ndarray, camera_info: CameraInfo) -> None:
        if not bool(self.get_parameter("draw_table_rectangle").value):
            return

        T_cam_table = self._get_camera_from_table_transform(self.get_clock().now().nanoseconds * 1e-9)
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

        corners_cam = [_transform_point(T_cam_table, p_table) for p_table in corners_table]
        projected: List[Optional[Tuple[int, int]]] = [
            self._project_point(camera_info, p_cam) for p_cam in corners_cam
        ]
        visible_count = sum(1 for p in projected if p is not None)

        # (0,1): short edge at wall
        # (1,2): long edge away from wall
        # (2,3): short edge away from wall
        # (3,0): long edge at wall
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

        drawn_edge_count = 0
        for i0, i1 in edges:
            clipped = self._clip_camera_edge_to_near_plane(corners_cam[i0], corners_cam[i1])
            if clipped is None:
                continue

            p0_cam, p1_cam = clipped
            p0_uv = self._project_point(camera_info, p0_cam)
            p1_uv = self._project_point(camera_info, p1_cam)
            if p0_uv is None or p1_uv is None:
                continue

            cv2.line(frame, p0_uv, p1_uv, (0, 0, 255), 3)
            drawn_edge_count += 1

        labels = ["origin", "x", "x+y", "y"]
        for idx, point in enumerate(projected):
            if point is None:
                continue
            cv2.circle(frame, point, 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                labels[idx],
                (point[0] + 6, point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        if drawn_edge_count == 0 and visible_count == 0:
            self._draw_status_text(frame, "table rectangle not visible/projectable", (0, 0, 255))

    def _draw_ball_trajectory_overlay(self, frame: np.ndarray, camera_info: CameraInfo) -> None:
        """Project-only trajectory visualization.

        The ball_trajectory_estimator is responsible for deciding whether a
        trajectory is usable. This debug node only draws a fresh, valid
        BallTrajectory message. In the current contract:
            start_point = latest fitted ball point in the z=ball_radius plane
            end_point   = middle-line intersection/hit point
        """
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self._draw_status_text(frame, f"traj_overlay now={now_sec:.3f}s", (200, 200, 200))

        T_cam_table = self._get_camera_from_table_transform(now_sec)
        if T_cam_table is None:
            self._draw_status_text(frame, "T_cam_table: unavailable", (0, 140, 255))
            self._draw_status_text(frame, "no/stale T_camera_table", (0, 140, 255))
            return

        T_table_tcp = self._compute_table_tcp_transform(now_sec, T_cam_table)
        if bool(self.get_parameter("draw_tcp_middle_line").value):
            self._draw_tcp_middle_line_overlay(frame, camera_info, now_sec, T_cam_table, T_table_tcp)

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
            camera_info=camera_info,
            T_cam_table=T_cam_table,
            p0_table=p0_table,
            p1_table=p1_table,
            color=line_color,
            thickness=line_thickness,
            arrow=True,
        )
        if not drawn:
            self._draw_status_text(frame, "valid trajectory not projectable", (0, 140, 255))
            return

        p0_uv = self._project_table_point(camera_info, T_cam_table, p0_table)
        p1_uv = self._project_table_point(camera_info, T_cam_table, p1_table)

        if p0_uv is not None:
            cv2.circle(frame, p0_uv, start_radius, start_color, -1)
            cv2.putText(
                frame,
                "ball",
                (p0_uv[0] + 7, p0_uv[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                start_color,
                2,
                cv2.LINE_AA,
            )

        if p1_uv is not None:
            cv2.circle(frame, p1_uv, hit_radius, hit_color, -1)
            cv2.putText(
                frame,
                "hit",
                (p1_uv[0] + 7, p1_uv[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                hit_color,
                2,
                cv2.LINE_AA,
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

        transform = _pose_to_transform(self._top.latest_table_pose.pose)
        translation = transform[:3, 3]
        self._warn_throttled(
            "top_table_pose_ok",
            "top_table_pose accepted: "
            f"age={pose_age_sec:.3f}s frame='{header_frame or 'n/a'}' "
            f"T_cam_table.xyz=({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f})",
            1.0,
        )

        return transform

    def _compute_table_tcp_transform(
        self,
        now_sec: float,
        T_cam_table: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute T_table_tcp from robot-base table pose and TF base->tcp.

        Correct chain for the current top-cam-only setup:
            T_table_tcp = inverse(T_base_table) * T_base_tcp

        T_base_table comes from /scene_localizer/table_pose_robot_base. If that
        is not available and allow_base_table_fallback_from_tf is true, it is
        reconstructed as:
            T_base_table = T_base_top_camera * T_top_camera_table
        using TF base->camera plus the top table pose.
        """
        T_base_table = self._get_robot_base_from_table_transform(now_sec, T_cam_table)
        if T_base_table is None:
            return None

        T_base_tcp = self._lookup_robot_base_from_tcp_transform()
        if T_base_tcp is None:
            return None

        return _invert_transform(T_base_table) @ T_base_tcp

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

    def _draw_tcp_middle_line_overlay(
        self,
        frame: np.ndarray,
        camera_info: CameraInfo,
        now_sec: float,
        T_cam_table: np.ndarray,
        T_table_tcp: Optional[np.ndarray],
    ) -> None:
        """Draw the TCP-attached middle line at z=ball_radius in table frame.

        Construction:
        1. Take the TCP origin in table frame.
        2. Project it down to the table x-y plane -> Q = [tcp_x, tcp_y, 0].
        3. Lift Q by ball_radius -> anchor = [tcp_x, tcp_y, ball_radius].
        4. Draw from anchor along robot-base +x expressed in table coordinates.
        """
        pink = (180, 105, 255)

        if T_table_tcp is None:
            self._draw_status_text(
                frame,
                "no T_table_tcp: need T_base_table and TF base->tcp",
                pink,
            )
            return

        T_base_table = self._get_robot_base_from_table_transform(now_sec, T_cam_table)
        if T_base_table is None:
            self._draw_status_text(frame, "no T_base_table for robot-base-x middle line", pink)
            return

        tcp_origin_table = T_table_tcp[:3, 3].copy()
        ball_radius = max(0.0, float(self.get_parameter("ball_radius").value))

        # Express robot-base +x in table coordinates. Since T_base_table maps
        # table -> base, its inverse rotation maps base vectors -> table.
        robot_base_x_table = T_base_table[:3, :3].T @ np.array([1.0, 0.0, 0.0], dtype=float)
        robot_base_x_table = robot_base_x_table.astype(float)
        robot_base_x_table[2] = 0.0
        direction_norm = float(np.linalg.norm(robot_base_x_table))
        if direction_norm < 1e-9:
            self._draw_status_text(frame, "robot-base-x projection degenerate in table xy", pink)
            return
        robot_base_x_table /= direction_norm

        anchor_table = np.array(
            [tcp_origin_table[0], tcp_origin_table[1], ball_radius],
            dtype=float,
        )
        q_table = np.array(
            [tcp_origin_table[0], tcp_origin_table[1], 0.0],
            dtype=float,
        )

        configured_length = float(self.get_parameter("tcp_middle_line_length").value)
        if configured_length <= 0.0:
            configured_length = float(self.get_parameter("table_shortedge").value)
        line_length = max(0.0, configured_length)

        p_start_table = anchor_table - line_length * robot_base_x_table
        p_end_table = anchor_table + line_length * robot_base_x_table

        thickness = max(1, int(self.get_parameter("tcp_middle_line_thickness").value))

        drawn = self._draw_table_segment(
            frame=frame,
            camera_info=camera_info,
            T_cam_table=T_cam_table,
            p0_table=p_start_table,
            p1_table=p_end_table,
            color=pink,
            thickness=thickness,
            arrow=False,
        )

        anchor_uv = self._project_table_point(camera_info, T_cam_table, anchor_table)
        if anchor_uv is not None:
            cv2.circle(frame, anchor_uv, 4, pink, -1)
            cv2.putText(
                frame,
                "Q+ball_r",
                (anchor_uv[0] + 7, anchor_uv[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                pink,
                2,
                cv2.LINE_AA,
            )

        q_uv = self._project_table_point(camera_info, T_cam_table, q_table)
        if q_uv is not None:
            cv2.circle(frame, q_uv, 3, (160, 160, 160), -1)

        self._draw_status_text(
            frame,
            f"middle line: Q=({q_table[0]:.3f},{q_table[1]:.3f},0), "
            f"z=ball_r={ball_radius:.3f}, len={line_length:.3f}, "
            f"dir_base_x_table=({robot_base_x_table[0]:.3f},{robot_base_x_table[1]:.3f})",
            (200, 200, 200),
        )

        if not drawn:
            self._draw_status_text(frame, "robot-base-x middle line not projectable", pink)

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

    def _project_table_point(
        self,
        camera_info: CameraInfo,
        T_cam_table: np.ndarray,
        p_table: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        if not _is_finite_vector(p_table):
            return None
        p_cam = _transform_point(T_cam_table, p_table)
        return self._project_point(camera_info, p_cam)

    def _draw_table_segment(
        self,
        frame: np.ndarray,
        camera_info: CameraInfo,
        T_cam_table: np.ndarray,
        p0_table: np.ndarray,
        p1_table: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int,
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
        p0_uv = self._project_point(camera_info, c0_cam)
        p1_uv = self._project_point(camera_info, c1_cam)
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

    @staticmethod
    def _project_point(camera_info: CameraInfo, point_cam: np.ndarray) -> Optional[Tuple[int, int]]:
        if point_cam.shape != (3,):
            return None

        x, y, z = float(point_cam[0]), float(point_cam[1]), float(point_cam[2])
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z) or z <= 1e-9:
            return None

        if len(camera_info.k) < 9:
            return None

        k = np.asarray(camera_info.k, dtype=float).reshape(3, 3)
        if not np.all(np.isfinite(k)):
            return None

        dist_coeffs = np.asarray(camera_info.d, dtype=float).reshape(-1, 1) if len(camera_info.d) > 0 else None
        object_points = np.array([[[x, y, z]]], dtype=np.float64)
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        image_points, _ = cv2.projectPoints(object_points, rvec, tvec, k, dist_coeffs)
        if image_points is None or image_points.shape[0] == 0:
            return None

        u = float(image_points[0, 0, 0])
        v = float(image_points[0, 0, 1])
        if not np.isfinite(u) or not np.isfinite(v):
            return None

        return int(round(u)), int(round(v))

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
