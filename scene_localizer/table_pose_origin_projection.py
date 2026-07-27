#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np
import rclpy
import rclpy.time
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener


POSE_TOPIC = "/scene_localizer/top_cam/table_pose_camera"
IMAGE_TOPIC = "/top_cam/camera/color/image_raw"
CAMERA_INFO_TOPIC = "/top_cam/camera/color/camera_info"
TABLE_OUTPUT_TOPIC = "/scene_localizer/top_cam/table_frame_image"
TCP_OUTPUT_TOPIC = "/scene_localizer/top_cam/tcp_frame_image"

BASE_CAM_YAML = Path(
    "/home/jau/dyros/calibration/scene_localizer/T_base_cam.yaml"
)
BASE_FRAME = "base"
CAMERA_FRAME = "camera_color_optical_frame"
TCP_FRAME = "right_fr3_hand_tcp"

AXIS_LENGTH_M = 0.10
TABLE_X_LENGTH_M = 0.60
TABLE_Y_LENGTH_M = 1.20
TABLE_EDGE_COLOR = (0, 255, 255)
NEAR_PLANE_M = 1e-3


def quaternion_to_rotation(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion has zero length")
    x, y, z, w = q / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def make_transform(translation, quaternion) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation(quaternion)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def xyz(raw) -> np.ndarray:
    if isinstance(raw, dict):
        return np.array([raw["x"], raw["y"], raw["z"]], dtype=np.float64)
    return np.asarray(raw, dtype=np.float64)


def xyzw(raw) -> np.ndarray:
    if isinstance(raw, dict):
        return np.array(
            [raw["x"], raw["y"], raw["z"], raw["w"]], dtype=np.float64
        )
    return np.asarray(raw, dtype=np.float64)


class TableAndTcpFrameProjection(Node):
    def __init__(self) -> None:
        super().__init__("table_and_tcp_frame_projection")

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion = None
        self.table_points_camera = None
        self.warned_about_tf = False

        self.t_base_camera = self.load_base_camera_transform()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            PoseStamped, POSE_TOPIC, self.pose_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, IMAGE_TOPIC, self.image_callback, qos_profile_sensor_data
        )

        self.table_image_pub = self.create_publisher(
            Image, TABLE_OUTPUT_TOPIC, qos_profile_sensor_data
        )
        self.tcp_image_pub = self.create_publisher(
            Image, TCP_OUTPUT_TOPIC, qos_profile_sensor_data
        )

        self.get_logger().info(
            f"Loaded T_{BASE_FRAME}_{CAMERA_FRAME} from {BASE_CAM_YAML}"
        )
        self.get_logger().info(f"Table overlay: {TABLE_OUTPUT_TOPIC}")
        self.get_logger().info(f"TCP overlay: {TCP_OUTPUT_TOPIC}")

    def load_base_camera_transform(self) -> np.ndarray:
        with BASE_CAM_YAML.open("r", encoding="utf-8") as yaml_file:
            data = yaml.safe_load(yaml_file)

        parent = str(data["parent_frame"]).strip().strip("/")
        child = str(data["child_frame"]).strip().strip("/")
        if parent != BASE_FRAME or child != CAMERA_FRAME:
            raise ValueError(
                f"Expected YAML transform {BASE_FRAME} <- {CAMERA_FRAME}, "
                f"got {parent} <- {child}"
            )

        quaternion = data.get("quaternion", data.get("quaternion_xyzw"))
        if quaternion is None:
            raise ValueError("Calibration YAML has no quaternion")

        return make_transform(xyz(data["translation"]), xyzw(quaternion))

    def pose_callback(self, msg: PoseStamped) -> None:
        # PoseStamped is T_camera_table.
        t_camera_table = make_transform(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
        )

        # Origin, XYZ endpoints, then the four table corners.
        points_table = np.array(
            [
                [0.0, 0.0, 0.0],
                [AXIS_LENGTH_M, 0.0, 0.0],
                [0.0, AXIS_LENGTH_M, 0.0],
                [0.0, 0.0, AXIS_LENGTH_M],
                [0.0, 0.0, 0.0],
                [TABLE_X_LENGTH_M, 0.0, 0.0],
                [TABLE_X_LENGTH_M, TABLE_Y_LENGTH_M, 0.0],
                [0.0, TABLE_Y_LENGTH_M, 0.0],
            ],
            dtype=np.float64,
        )
        self.table_points_camera = self.transform_points(
            t_camera_table, points_table
        )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.distortion = np.asarray(msg.d, dtype=np.float64)

    @staticmethod
    def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
        return (
            transform[:3, :3] @ np.asarray(points, dtype=np.float64).T
        ).T + transform[:3, 3]

    def project_camera_point(self, point_camera):
        image_point, _ = cv2.projectPoints(
            np.asarray(point_camera, dtype=np.float64).reshape(1, 3),
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            self.camera_matrix,
            self.distortion,
        )
        u, v = image_point.reshape(2)
        return int(round(u)), int(round(v))

    def draw_visible_segment(
        self, frame, start_camera, end_camera, color, thickness
    ) -> None:
        start = np.asarray(start_camera, dtype=np.float64).copy()
        end = np.asarray(end_camera, dtype=np.float64).copy()

        if start[2] <= NEAR_PLANE_M and end[2] <= NEAR_PLANE_M:
            return
        if start[2] <= NEAR_PLANE_M:
            alpha = (NEAR_PLANE_M - start[2]) / (end[2] - start[2])
            start += alpha * (end - start)
        elif end[2] <= NEAR_PLANE_M:
            alpha = (NEAR_PLANE_M - end[2]) / (start[2] - end[2])
            end += alpha * (start - end)

        start_pixel = self.project_camera_point(start)
        end_pixel = self.project_camera_point(end)
        height, width = frame.shape[:2]
        visible, clipped_start, clipped_end = cv2.clipLine(
            (0, 0, width, height), start_pixel, end_pixel
        )
        if visible:
            cv2.line(
                frame,
                clipped_start,
                clipped_end,
                color,
                thickness,
                cv2.LINE_AA,
            )

    def draw_frame(self, frame, points_camera, name=None) -> None:
        if points_camera[0, 2] <= NEAR_PLANE_M:
            return

        pixels = [self.project_camera_point(point) for point in points_camera]
        origin = pixels[0]
        height, width = frame.shape[:2]

        axes = (
            ("X", 1, (0, 0, 255)),
            ("Y", 2, (0, 255, 0)),
            ("Z", 3, (255, 0, 0)),
        )
        for label, index, color in axes:
            if points_camera[index, 2] <= NEAR_PLANE_M:
                continue
            endpoint = pixels[index]
            endpoint_is_visible = (
                0 <= endpoint[0] < width and 0 <= endpoint[1] < height
            )
            origin_is_visible = (
                0 <= origin[0] < width and 0 <= origin[1] < height
            )
            if origin_is_visible and endpoint_is_visible:
                cv2.arrowedLine(
                    frame,
                    origin,
                    endpoint,
                    color,
                    3,
                    cv2.LINE_AA,
                    tipLength=0.18,
                )
            else:
                self.draw_visible_segment(
                    frame, points_camera[0], points_camera[index], color, 3
                )
            if endpoint_is_visible:
                cv2.putText(
                    frame,
                    label,
                    (endpoint[0] + 5, endpoint[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if 0 <= origin[0] < width and 0 <= origin[1] < height:
            cv2.circle(frame, origin, 5, (255, 255, 255), -1)
            if name:
                cv2.putText(
                    frame,
                    name,
                    (origin[0] + 8, origin[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

    def lookup_tcp_points_camera(self):
        try:
            # Required TF convention:
            # target=TCP_FRAME, source=BASE_FRAME -> T_tcp_base.
            tf_msg = self.tf_buffer.lookup_transform(
                TCP_FRAME, BASE_FRAME, rclpy.time.Time()
            )
            self.warned_about_tf = False
        except TransformException as exc:
            if not self.warned_about_tf:
                self.get_logger().warn(
                    f"Cannot look up T_{TCP_FRAME}_{BASE_FRAME}: {exc}"
                )
                self.warned_about_tf = True
            return None

        transform = tf_msg.transform
        t_tcp_base = make_transform(
            [
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ],
            [
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            ],
        )

        # Requested transform, followed by the inverse required for projection:
        # T_tcp_cam = T_tcp_base @ T_base_cam
        # T_cam_tcp = inverse(T_tcp_cam)
        t_tcp_camera = t_tcp_base @ self.t_base_camera
        t_camera_tcp = invert_transform(t_tcp_camera)

        points_tcp = np.array(
            [
                [0.0, 0.0, 0.0],
                [AXIS_LENGTH_M, 0.0, 0.0],
                [0.0, AXIS_LENGTH_M, 0.0],
                [0.0, 0.0, AXIS_LENGTH_M],
            ],
            dtype=np.float64,
        )
        return self.transform_points(t_camera_tcp, points_tcp)

    def image_callback(self, msg: Image) -> None:
        if self.camera_matrix is None:
            return

        raw_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if self.table_points_camera is not None:
            table_frame = raw_frame.copy()

            for start_index, end_index in ((4, 5), (5, 6), (6, 7), (7, 4)):
                self.draw_visible_segment(
                    table_frame,
                    self.table_points_camera[start_index],
                    self.table_points_camera[end_index],
                    TABLE_EDGE_COLOR,
                    2,
                )
            self.draw_frame(table_frame, self.table_points_camera[:4])

            table_output = self.bridge.cv2_to_imgmsg(
                table_frame, encoding="bgr8"
            )
            table_output.header = msg.header
            self.table_image_pub.publish(table_output)

        tcp_points_camera = self.lookup_tcp_points_camera()
        if tcp_points_camera is not None:
            tcp_frame = raw_frame.copy()
            self.draw_frame(tcp_frame, tcp_points_camera, "TCP")

            tcp_output = self.bridge.cv2_to_imgmsg(tcp_frame, encoding="bgr8")
            tcp_output.header = msg.header
            self.tcp_image_pub.publish(tcp_output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TableAndTcpFrameProjection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
