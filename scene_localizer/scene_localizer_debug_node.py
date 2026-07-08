from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from aruco_opencv_msgs.msg import ArucoDetection
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


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
class CameraStreamState:
    name: str
    image_topic: str
    camera_info_topic: str
    detections_topic: str
    table_pose_topic: str
    debug_image_topic: str
    debug_pub: Any
    latest_camera_info: Optional[CameraInfo] = None
    latest_detections: Optional[ArucoDetection] = None
    latest_detection_time_sec: Optional[float] = None
    latest_table_pose: Optional[PoseStamped] = None
    latest_table_pose_time_sec: Optional[float] = None


class SceneLocalizerDebugNode(Node):
    def __init__(self) -> None:
        super().__init__("scene_localizer_debug")

        self._bridge = CvBridge()

        self.declare_parameter("top_image_topic", "/top_cam/camera/color/image_raw")
        self.declare_parameter("top_camera_info_topic", "/top_cam/camera/color/camera_info")
        self.declare_parameter("top_detections_topic", "/aruco_top_cam/aruco_detections")
        self.declare_parameter("top_debug_image_topic", "/scene_localizer/top_cam/reprojection_debug")
        self.declare_parameter("top_table_pose_topic", "/scene_localizer/top_cam/table_pose_camera")

        self.declare_parameter("eef_image_topic", "/eef_cam/camera/color/image_raw")
        self.declare_parameter("eef_camera_info_topic", "/eef_cam/camera/color/camera_info")
        self.declare_parameter("eef_detections_topic", "/aruco_eef_cam/aruco_detections")
        self.declare_parameter("eef_debug_image_topic", "/scene_localizer/eef_cam/reprojection_debug")
        self.declare_parameter("eef_table_pose_topic", "/scene_localizer/eef_cam/table_pose_camera")

        self.declare_parameter("physical_marker_size", 0.035)
        self.declare_parameter("axis_length", 0.08)
        self.declare_parameter("detection_timeout_sec", 0.3)
        self.declare_parameter("publish_debug_images", True)
        self.declare_parameter("table_shortedge", 0.6)
        self.declare_parameter("table_longedge", 1.2)
        self.declare_parameter("table_edge_z", 0.0)
        self.declare_parameter("table_pose_timeout_sec", 1.0)
        self.declare_parameter("draw_table_rectangle", True)

        top_state = self._make_stream_state(
            name="top_cam",
            image_topic=str(self.get_parameter("top_image_topic").value),
            camera_info_topic=str(self.get_parameter("top_camera_info_topic").value),
            detections_topic=str(self.get_parameter("top_detections_topic").value),
            table_pose_topic=str(self.get_parameter("top_table_pose_topic").value),
            debug_image_topic=str(self.get_parameter("top_debug_image_topic").value),
        )
        eef_state = self._make_stream_state(
            name="eef_cam",
            image_topic=str(self.get_parameter("eef_image_topic").value),
            camera_info_topic=str(self.get_parameter("eef_camera_info_topic").value),
            detections_topic=str(self.get_parameter("eef_detections_topic").value),
            table_pose_topic=str(self.get_parameter("eef_table_pose_topic").value),
            debug_image_topic=str(self.get_parameter("eef_debug_image_topic").value),
        )

        self._streams: List[CameraStreamState] = [top_state, eef_state]

        for stream in self._streams:
            self.create_subscription(
                CameraInfo,
                stream.camera_info_topic,
                self._make_camera_info_callback(stream),
                10,
            )
            self.create_subscription(
                ArucoDetection,
                stream.detections_topic,
                self._make_detections_callback(stream),
                10,
            )
            self.create_subscription(
                PoseStamped,
                stream.table_pose_topic,
                self._make_table_pose_callback(stream),
                10,
            )
            self.create_subscription(
                Image,
                stream.image_topic,
                self._make_image_callback(stream),
                10,
            )

        self.get_logger().info("scene_localizer_debug started")
        for stream in self._streams:
            self.get_logger().info(
                f"[{stream.name}] image={stream.image_topic}, "
                f"camera_info={stream.camera_info_topic}, "
                f"detections={stream.detections_topic}, "
                f"table_pose={stream.table_pose_topic}, "
                f"debug_pub={stream.debug_image_topic}"
            )
        self.get_logger().info(
            "Table rectangle params: "
            f"shortedge={float(self.get_parameter('table_shortedge').value):.3f}, "
            f"longedge={float(self.get_parameter('table_longedge').value):.3f}, "
            f"edge_z={float(self.get_parameter('table_edge_z').value):.3f}, "
            f"timeout_sec={float(self.get_parameter('table_pose_timeout_sec').value):.3f}, "
            f"draw={bool(self.get_parameter('draw_table_rectangle').value)}"
        )

    def _make_stream_state(
        self,
        name: str,
        image_topic: str,
        camera_info_topic: str,
        detections_topic: str,
        table_pose_topic: str,
        debug_image_topic: str,
    ) -> CameraStreamState:
        debug_pub = self.create_publisher(Image, debug_image_topic, 10)
        return CameraStreamState(
            name=name,
            image_topic=image_topic,
            camera_info_topic=camera_info_topic,
            detections_topic=detections_topic,
            table_pose_topic=table_pose_topic,
            debug_image_topic=debug_image_topic,
            debug_pub=debug_pub,
        )

    def _make_camera_info_callback(self, stream: CameraStreamState):
        def _callback(msg: CameraInfo) -> None:
            stream.latest_camera_info = msg

        return _callback

    def _make_detections_callback(self, stream: CameraStreamState):
        def _callback(msg: ArucoDetection) -> None:
            stream.latest_detections = msg
            stream.latest_detection_time_sec = self._stamp_to_sec(msg)

        return _callback

    def _make_table_pose_callback(self, stream: CameraStreamState):
        def _callback(msg: PoseStamped) -> None:
            stream.latest_table_pose = msg
            stream.latest_table_pose_time_sec = self._stamp_to_sec(msg)

        return _callback

    def _make_image_callback(self, stream: CameraStreamState):
        def _callback(msg: Image) -> None:
            publish_debug = bool(self.get_parameter("publish_debug_images").value)
            if not publish_debug:
                return

            try:
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as exc:
                self.get_logger().warn(
                    f"[{stream.name}] failed to convert image to bgr8: {exc}"
                )
                return

            self._draw_projection_debug_text(frame, msg, stream)

            if stream.latest_camera_info is None:
                self._draw_status_text(frame, "no camera_info", (0, 140, 255))
                self._publish_debug_image(stream, msg, frame)
                return

            now_sec = self.get_clock().now().nanoseconds * 1e-9
            timeout_sec = max(0.0, float(self.get_parameter("detection_timeout_sec").value))
            stale = (
                stream.latest_detection_time_sec is None
                or (now_sec - stream.latest_detection_time_sec) > timeout_sec
            )

            if stale:
                self._draw_status_text(frame, "detections stale", (0, 140, 255))
                self._draw_table_rectangle_overlay(frame, stream.latest_camera_info, stream)
                self._publish_debug_image(stream, msg, frame)
                return

            if stream.latest_detections is None:
                self._draw_status_text(frame, "no detections", (0, 140, 255))
                self._draw_table_rectangle_overlay(frame, stream.latest_camera_info, stream)
                self._publish_debug_image(stream, msg, frame)
                return

            physical_marker_size = max(1e-6, float(self.get_parameter("physical_marker_size").value))
            axis_length = max(1e-6, float(self.get_parameter("axis_length").value))
            self._draw_detection_overlay(
                frame=frame,
                camera_info=stream.latest_camera_info,
                detections_msg=stream.latest_detections,
                physical_marker_size=physical_marker_size,
                axis_length=axis_length,
            )
            self._draw_table_rectangle_overlay(frame, stream.latest_camera_info, stream)
            self._publish_debug_image(stream, msg, frame)

        return _callback

    def _publish_debug_image(self, stream: CameraStreamState, src_msg: Image, frame: np.ndarray) -> None:
        debug_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        debug_msg.header = src_msg.header
        stream.debug_pub.publish(debug_msg)

    def _draw_projection_debug_text(self, frame: np.ndarray, image_msg: Image, stream: CameraStreamState) -> None:
        image_frame_id = str(getattr(image_msg.header, "frame_id", ""))
        image_text = f"image: {int(getattr(image_msg, 'width', 0))} {int(getattr(image_msg, 'height', 0))} {image_frame_id}"

        camera_info = stream.latest_camera_info
        if camera_info is None:
            camera_info_text = "camera_info: n/a"
            k_text = "K: n/a"
            d_text = "D: n/a"
        else:
            camera_info_frame_id = str(getattr(camera_info.header, "frame_id", ""))
            fx = float(camera_info.k[0]) if len(camera_info.k) >= 1 else float("nan")
            fy = float(camera_info.k[4]) if len(camera_info.k) >= 5 else float("nan")
            cx = float(camera_info.k[2]) if len(camera_info.k) >= 3 else float("nan")
            cy = float(camera_info.k[5]) if len(camera_info.k) >= 6 else float("nan")
            camera_info_text = (
                f"camera_info: {int(camera_info.width)} {int(camera_info.height)} {camera_info_frame_id}"
            )
            k_text = f"K: fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}"
            if len(camera_info.d) > 0:
                d_text = "D: " + " ".join(f"{float(v):.4f}" for v in camera_info.d)
            else:
                d_text = "D: []"

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        detections_age = self._format_age(now_sec, stream.latest_detection_time_sec)
        table_pose_age = self._format_age(now_sec, stream.latest_table_pose_time_sec)

        lines = [
            image_text,
            camera_info_text,
            k_text,
            d_text,
            f"table_pose age: {table_pose_age}",
            f"detections age: {detections_age}",
        ]

        y = 18
        for line in lines:
            cv2.putText(
                frame,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 18

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
                cv2.line(frame, origin_uv, x_uv, (0,0,255), 2)
            if origin_uv is not None and y_uv is not None:
                cv2.line(frame, origin_uv, y_uv, (0, 255, 0), 2)
            if origin_uv is not None and z_uv is not None:
                cv2.line(frame, origin_uv, z_uv, (255, 0, 0), 2)

    def _draw_table_rectangle_overlay(
        self,
        frame: np.ndarray,
        camera_info: CameraInfo,
        stream: CameraStreamState,
    ) -> None:
        if not bool(self.get_parameter("draw_table_rectangle").value):
            return

        if stream.latest_table_pose is None or stream.latest_table_pose_time_sec is None:
            self._draw_status_text(frame, "no table_pose", (0, 0, 255))
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        timeout_sec = max(0.0, float(self.get_parameter("table_pose_timeout_sec").value))
        if (now_sec - stream.latest_table_pose_time_sec) > timeout_sec:
            self._draw_status_text(frame, "table_pose stale", (0, 0, 255))
            return

        table_pose = stream.latest_table_pose.pose
        t_ct = np.array(
            [table_pose.position.x, table_pose.position.y, table_pose.position.z],
            dtype=float,
        )
        q_ct = np.array(
            [
                table_pose.orientation.x,
                table_pose.orientation.y,
                table_pose.orientation.z,
                table_pose.orientation.w,
            ],
            dtype=float,
        )
        r_ct = _quaternion_to_matrix_xyzw(q_ct)

        table_shortedge = float(self.get_parameter("table_shortedge").value)
        table_longedge = float(self.get_parameter("table_longedge").value)
        table_edge_z = float(self.get_parameter("table_edge_z").value)

        corners_table = [
            np.array([0.0, 0.0, table_edge_z], dtype=float),
            np.array([table_shortedge, 0.0, table_edge_z], dtype=float),
            np.array([table_shortedge, table_longedge, table_edge_z], dtype=float),
            np.array([0.0, table_longedge, table_edge_z], dtype=float),
        ]

        corners_cam = [r_ct @ p_table + t_ct for p_table in corners_table]

        projected: List[Optional[Tuple[int, int]]] = []
        for p_cam in corners_cam:
            uv = self._project_point(camera_info, p_cam)
            projected.append(uv)

        visible_count = sum(1 for p in projected if p is not None)


        # (0,1):  short edge at wall
        # (1,2): long edge away from wall
        # (2,3): short edge away from wall
        # (3,0): long edge at wall

        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

        clipped_edge_count = 0
        drawn_edge_count = 0

        for i0, i1 in edges:
            clipped = self._clip_camera_edge_to_near_plane(corners_cam[i0], corners_cam[i1])
            if clipped is None:
                continue

            p0_cam, p1_cam = clipped
            if (corners_cam[i0][2] <= 1e-4 < corners_cam[i1][2]) or (corners_cam[i1][2] <= 1e-4 < corners_cam[i0][2]):
                clipped_edge_count += 1

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

        self.get_logger().debug(
            f"[{stream.name}] table rectangle overlay: visible_corners={visible_count}, "
            f"clipped_edges={clipped_edge_count}, drawn_edges={drawn_edge_count}"
        )

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

    @staticmethod
    def _draw_status_text(frame: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
        cv2.putText(
            frame,
            text,
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )


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
