#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, PoseStamped


def quaternion_xyzw_to_rotation_matrix(
    x: float,
    y: float,
    z: float,
    w: float,
) -> np.ndarray:
    """Convert a geometry_msgs quaternion (x, y, z, w) to a 3x3 matrix."""
    norm = np.sqrt(x * x + y * y + z * z + w * w)

    if norm < 1.0e-12:
        raise ValueError("Quaternion norm is zero")

    x /= norm
    y /= norm
    z /= norm
    w /= norm

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


def pose_to_matrix(msg: PoseStamped) -> np.ndarray:
    """
    Interpret msg.pose as T_parent_child, where parent is msg.header.frame_id.

    For example:
      camera_pose_robot_base -> T_base_camera
      table_pose_camera      -> T_camera_table
    """
    p = msg.pose.position
    q = msg.pose.orientation

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_xyzw_to_rotation_matrix(
        q.x,
        q.y,
        q.z,
        q.w,
    )
    transform[:3, 3] = [p.x, p.y, p.z]

    return transform


class ExecutedGotoSTableConverter(Node):
    def __init__(self) -> None:
        super().__init__("executed_goto_s_table_converter")

        self.declare_parameter(
            "input_topic",
            "/trajectory_executor/executed_goto_s_target_base",
        )
        self.declare_parameter(
            "output_topic",
            "/trajectory_executor/executed_goto_s_target_table",
        )
        self.declare_parameter(
            "camera_pose_robot_base_topic",
            "/scene_localizer/top_cam/camera_pose_robot_base",
        )
        self.declare_parameter(
            "table_pose_camera_topic",
            "/scene_localizer/top_cam/table_pose_camera",
        )
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("table_frame", "table_frame")

        input_topic = (
            self.get_parameter("input_topic")
            .get_parameter_value()
            .string_value
        )
        output_topic = (
            self.get_parameter("output_topic")
            .get_parameter_value()
            .string_value
        )
        camera_pose_topic = (
            self.get_parameter("camera_pose_robot_base_topic")
            .get_parameter_value()
            .string_value
        )
        table_pose_topic = (
            self.get_parameter("table_pose_camera_topic")
            .get_parameter_value()
            .string_value
        )

        self.base_frame = (
            self.get_parameter("base_frame")
            .get_parameter_value()
            .string_value
        )
        self.camera_frame = (
            self.get_parameter("camera_frame")
            .get_parameter_value()
            .string_value
        )
        self.table_frame = (
            self.get_parameter("table_frame")
            .get_parameter_value()
            .string_value
        )

        self.t_base_camera: np.ndarray | None = None
        self.t_camera_table: np.ndarray | None = None

        self.output_pub = self.create_publisher(
            PointStamped,
            output_topic,
            10,
        )

        self.create_subscription(
            PoseStamped,
            camera_pose_topic,
            self.camera_pose_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            table_pose_topic,
            self.table_pose_callback,
            10,
        )
        self.create_subscription(
            PointStamped,
            input_topic,
            self.target_callback,
            10,
        )

        self.get_logger().info(
            f"Converting {input_topic} from {self.base_frame} "
            f"to {self.table_frame}; publishing {output_topic}"
        )

    def camera_pose_callback(self, msg: PoseStamped) -> None:
        if msg.header.frame_id and msg.header.frame_id != self.base_frame:
            self.get_logger().warning(
                "camera_pose_robot_base has frame_id "
                f"'{msg.header.frame_id}', expected '{self.base_frame}'"
            )

        try:
            self.t_base_camera = pose_to_matrix(msg)
        except ValueError as error:
            self.get_logger().error(
                f"Invalid camera pose quaternion: {error}"
            )

    def table_pose_callback(self, msg: PoseStamped) -> None:
        if msg.header.frame_id and msg.header.frame_id != self.camera_frame:
            self.get_logger().warning(
                "table_pose_camera has frame_id "
                f"'{msg.header.frame_id}', expected '{self.camera_frame}'"
            )

        try:
            self.t_camera_table = pose_to_matrix(msg)
        except ValueError as error:
            self.get_logger().error(
                f"Invalid table pose quaternion: {error}"
            )

    def target_callback(self, msg: PointStamped) -> None:
        if self.t_base_camera is None or self.t_camera_table is None:
            self.get_logger().warning(
                "Received GOTO_S target before both scene transforms "
                "were available.",
                throttle_duration_sec=2.0,
            )
            return

        if msg.header.frame_id and msg.header.frame_id != self.base_frame:
            self.get_logger().warning(
                f"Input point has frame_id '{msg.header.frame_id}', "
                f"expected '{self.base_frame}'"
            )
            return

        # T_base_table = T_base_camera @ T_camera_table
        t_base_table = self.t_base_camera @ self.t_camera_table

        point_base = np.array(
            [
                msg.point.x,
                msg.point.y,
                msg.point.z,
                1.0,
            ],
            dtype=np.float64,
        )

        # p_table = inverse(T_base_table) @ p_base
        point_table = np.linalg.inv(t_base_table) @ point_base

        output = PointStamped()
        output.header.stamp = msg.header.stamp
        output.header.frame_id = self.table_frame
        output.point.x = float(point_table[0])
        output.point.y = float(point_table[1])
        output.point.z = float(point_table[2])

        self.output_pub.publish(output)

        self.get_logger().info(
            "Published executed GOTO_S target in table frame: "
            f"[{output.point.x:.6f}, "
            f"{output.point.y:.6f}, "
            f"{output.point.z:.6f}]"
        )


def main(args=None) -> None:
    rclpy.init(args=args)

    node = ExecutedGotoSTableConverter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()