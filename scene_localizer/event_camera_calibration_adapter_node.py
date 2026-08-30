#!/usr/bin/env python3
"""
Publish event-camera CameraInfo and T_camera_table from solved calibration.

The event calibration package writes T_table_camera using the convention
T_parent_child maps child coordinates into parent coordinates. The existing
ball estimator consumes a PoseStamped encoding T_camera_table, so this adapter
always computes that pose by inverting the solved T_table_camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
import yaml

from scene_localizer.event_ball_geometry import (
    BallLocalizationError,
    invert_transform,
    rotation_matrix_to_quaternion_xyzw,
    transform_from_translation_quaternion,
    validate_transform,
)


class EventCalibrationError(ValueError):
    """Raised when a solved event-camera YAML is absent or unsafe to use."""


@dataclass(frozen=True)
class EventCameraCalibration:
    """Validated values loaded from the event calibration serializer schema."""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    distortion_model: str
    image_width: int
    image_height: int
    T_table_camera: np.ndarray
    T_camera_table: np.ndarray
    calibration_table_frame: str
    calibration_camera_frame: str
    source_path: Path


def _require_mapping(
    payload: Mapping[str, Any], key: str, source: str
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise EventCalibrationError(f"{source} is missing mapping '{key}'")
    return value


def _matrix_data_block(
    payload: Mapping[str, Any], key: str, source: str
) -> tuple[np.ndarray, int, int]:
    block = _require_mapping(payload, key, source)
    try:
        rows = int(block.get("rows", -1))
        columns = int(block.get("cols", -1))
    except (TypeError, ValueError) as error:
        raise EventCalibrationError(f"{source} {key} rows/cols must be integers") from error
    data = block.get("data")
    if not isinstance(data, (list, tuple)):
        raise EventCalibrationError(f"{source} field '{key}.data' must be an array")
    try:
        array = np.asarray(data, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise EventCalibrationError(f"{source} field '{key}.data' is invalid") from error
    if rows <= 0 or columns <= 0 or rows * columns != array.size:
        raise EventCalibrationError(
            f"{source} {key} rows/cols do not match its data"
        )
    if not np.all(np.isfinite(array)):
        raise EventCalibrationError(f"{source} field '{key}.data' must be finite")
    return array, rows, columns


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise EventCalibrationError(f"{name} must be a positive integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise EventCalibrationError(f"{name} must be a positive integer") from error
    if converted <= 0 or converted != value:
        raise EventCalibrationError(f"{name} must be a positive integer")
    return converted


def _load_transform_block(
    payload: Mapping[str, Any], key: str, source: str
) -> np.ndarray:
    block = _require_mapping(payload, key, source)
    matrix_transform: Optional[np.ndarray] = None
    pose_transform: Optional[np.ndarray] = None

    if "matrix" in block:
        try:
            raw_matrix = np.asarray(block["matrix"], dtype=np.float64)
            matrix_transform = validate_transform(raw_matrix, key).copy()
        except (TypeError, ValueError, BallLocalizationError) as error:
            raise EventCalibrationError(
                f"{source} field '{key}.matrix' is not a valid rigid transform: {error}"
            ) from error

    has_translation = "translation_m" in block
    has_quaternion = "quaternion_xyzw" in block
    if has_translation != has_quaternion:
        raise EventCalibrationError(
            f"{source} field '{key}' must contain both translation_m and quaternion_xyzw"
        )
    if has_translation:
        try:
            pose_transform = transform_from_translation_quaternion(
                block["translation_m"],
                block["quaternion_xyzw"],
                key,
            )
        except (TypeError, ValueError, BallLocalizationError) as error:
            raise EventCalibrationError(
                f"{source} field '{key}' has an invalid pose: {error}"
            ) from error

    if matrix_transform is None and pose_transform is None:
        raise EventCalibrationError(
            f"{source} field '{key}' requires matrix or "
            "translation_m/quaternion_xyzw"
        )
    if matrix_transform is not None and pose_transform is not None:
        if not np.allclose(matrix_transform, pose_transform, atol=1e-7):
            raise EventCalibrationError(
                f"{source} field '{key}' matrix and pose representations disagree"
            )
    assert matrix_transform is not None or pose_transform is not None
    return matrix_transform if matrix_transform is not None else pose_transform


def _validate_transform_metadata(
    block: Mapping[str, Any],
    *,
    parent_frame: str,
    child_frame: str,
    source: str,
) -> None:
    serialized_parent = str(block.get("parent_frame", "")).strip()
    serialized_child = str(block.get("child_frame", "")).strip()
    if serialized_parent and parent_frame and serialized_parent != parent_frame:
        raise EventCalibrationError(
            f"{source} T_table_camera.parent_frame='{serialized_parent}' does not "
            f"match table_frame='{parent_frame}'"
        )
    if serialized_child and child_frame and serialized_child != child_frame:
        raise EventCalibrationError(
            f"{source} T_table_camera.child_frame='{serialized_child}' does not "
            f"match camera_frame='{child_frame}'"
        )


def load_event_camera_calibration(
    calibration_file: str | Path,
    *,
    expected_image_width: Optional[int] = None,
    expected_image_height: Optional[int] = None,
) -> EventCameraCalibration:
    """Load the checkerboard calibrator's solved, top-level YAML schema."""
    path = Path(calibration_file).expanduser()
    source = f"event calibration YAML '{path}'"
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except FileNotFoundError as error:
        raise EventCalibrationError(f"{source} does not exist") from error
    except OSError as error:
        raise EventCalibrationError(f"cannot read {source}: {error}") from error
    except yaml.YAMLError as error:
        raise EventCalibrationError(f"invalid YAML in {source}: {error}") from error
    if not isinstance(payload, dict):
        raise EventCalibrationError(f"{source} must contain a top-level mapping")

    camera_data, camera_rows, camera_columns = _matrix_data_block(
        payload, "camera_matrix", source
    )
    if camera_rows != 3 or camera_columns != 3:
        raise EventCalibrationError(
            f"{source} camera_matrix must declare rows=3 and cols=3"
        )
    camera_matrix = camera_data.reshape(3, 3)
    if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
        raise EventCalibrationError(f"{source} camera_matrix has non-positive focal length")

    distortion, _, _ = _matrix_data_block(
        payload, "distortion_coefficients", source
    )
    if distortion.size not in {4, 5, 8, 12, 14}:
        raise EventCalibrationError(
            f"{source} distortion_coefficients must contain 4, 5, 8, 12, or 14 values"
        )
    distortion_model = str(payload.get("distortion_model", "")).strip()
    if distortion_model not in {"plumb_bob", "rational_polynomial"}:
        raise EventCalibrationError(
            f"{source} has unsupported distortion_model='{distortion_model}'; "
            "cv2.undistortPoints requires plumb_bob or rational_polynomial"
        )

    image_width = _positive_int(payload.get("image_width"), f"{source} image_width")
    image_height = _positive_int(payload.get("image_height"), f"{source} image_height")
    if expected_image_width is not None and image_width != int(expected_image_width):
        raise EventCalibrationError(
            f"{source} image_width={image_width} does not match native event width "
            f"{int(expected_image_width)}; coordinates will not be rescaled"
        )
    if expected_image_height is not None and image_height != int(expected_image_height):
        raise EventCalibrationError(
            f"{source} image_height={image_height} does not match native event height "
            f"{int(expected_image_height)}; coordinates will not be rescaled"
        )

    table_frame = str(payload.get("table_frame", "")).strip()
    camera_frame = str(payload.get("camera_frame", "")).strip()
    transform_block = _require_mapping(payload, "T_table_camera", source)
    _validate_transform_metadata(
        transform_block,
        parent_frame=table_frame,
        child_frame=camera_frame,
        source=source,
    )
    T_table_camera = _load_transform_block(payload, "T_table_camera", source)
    T_camera_table = invert_transform(T_table_camera, "T_table_camera")

    # The current serializer also writes this redundant inverse. Validate it
    # when present, but never use it as the source of the published pose.
    if "T_camera_table" in payload:
        serialized_inverse = _load_transform_block(payload, "T_camera_table", source)
        if not np.allclose(serialized_inverse, T_camera_table, atol=1e-7):
            raise EventCalibrationError(
                f"{source} T_camera_table is inconsistent with inverse(T_table_camera)"
            )

    return EventCameraCalibration(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        distortion_model=distortion_model,
        image_width=image_width,
        image_height=image_height,
        T_table_camera=T_table_camera,
        T_camera_table=T_camera_table,
        calibration_table_frame=table_frame,
        calibration_camera_frame=camera_frame,
        source_path=path,
    )


def camera_info_from_calibration(
    calibration: EventCameraCalibration,
    stamp: Any,
    frame_id: str,
) -> CameraInfo:
    """Build CameraInfo for the raw, native-coordinate event calibration."""
    message = CameraInfo()
    message.header.stamp.sec = int(getattr(stamp, "sec", 0))
    message.header.stamp.nanosec = int(getattr(stamp, "nanosec", 0))
    message.header.frame_id = str(frame_id)
    message.width = calibration.image_width
    message.height = calibration.image_height
    message.distortion_model = calibration.distortion_model
    message.d = calibration.distortion_coefficients.reshape(-1).tolist()
    message.k = calibration.camera_matrix.reshape(-1).tolist()
    message.r = np.eye(3, dtype=np.float64).reshape(-1).tolist()
    message.p = np.column_stack(
        (calibration.camera_matrix, np.zeros(3, dtype=np.float64))
    ).reshape(-1).tolist()
    return message


def table_pose_from_calibration(
    calibration: EventCameraCalibration,
    stamp: Any,
    event_camera_frame: str,
) -> PoseStamped:
    """Build a PoseStamped encoding inverse(T_table_camera) = T_camera_table."""
    transform = calibration.T_camera_table
    quaternion = rotation_matrix_to_quaternion_xyzw(transform[:3, :3])
    message = PoseStamped()
    message.header.stamp.sec = int(getattr(stamp, "sec", 0))
    message.header.stamp.nanosec = int(getattr(stamp, "nanosec", 0))
    message.header.frame_id = str(event_camera_frame)
    message.pose.position.x = float(transform[0, 3])
    message.pose.position.y = float(transform[1, 3])
    message.pose.position.z = float(transform[2, 3])
    message.pose.orientation.x = float(quaternion[0])
    message.pose.orientation.y = float(quaternion[1])
    message.pose.orientation.z = float(quaternion[2])
    message.pose.orientation.w = float(quaternion[3])
    return message


class EventCameraCalibrationAdapterNode(Node):
    """Repeatedly publish static solved event-camera calibration as ROS messages."""

    def __init__(self) -> None:
        super().__init__("event_camera_calibration_adapter")

        self.declare_parameter(
            "calibration_file",
            "~/.ros/event_camera_calibration/genx320_calibration.yaml",
        )
        self.declare_parameter("camera_info_topic", "/event_camera/camera_info")
        self.declare_parameter(
            "table_pose_topic",
            "/scene_localizer/event_camera/table_pose_camera",
        )
        self.declare_parameter("event_camera_frame", "event_camera")
        self.declare_parameter("table_frame", "table_frame")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("expected_image_width", 320)
        self.declare_parameter("expected_image_height", 320)

        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if not np.isfinite(publish_rate_hz) or publish_rate_hz <= 0.0:
            raise EventCalibrationError("publish_rate_hz must be finite and positive")
        event_camera_frame = str(
            self.get_parameter("event_camera_frame").value
        ).strip()
        if not event_camera_frame:
            raise EventCalibrationError("event_camera_frame must not be empty")
        self._event_camera_frame = event_camera_frame

        expected_width = int(self.get_parameter("expected_image_width").value)
        expected_height = int(self.get_parameter("expected_image_height").value)
        self._calibration = load_event_camera_calibration(
            str(self.get_parameter("calibration_file").value),
            expected_image_width=expected_width if expected_width > 0 else None,
            expected_image_height=expected_height if expected_height > 0 else None,
        )

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._camera_info_publisher = self.create_publisher(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            qos,
        )
        self._table_pose_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("table_pose_topic").value),
            qos,
        )
        self._timer = self.create_timer(1.0 / publish_rate_hz, self._publish_calibration)
        self._publish_calibration()

        self.get_logger().info(
            f"Loaded event calibration directly from {self._calibration.source_path}; "
            f"native_size={self._calibration.image_width}x"
            f"{self._calibration.image_height}, "
            f"yaml_frames={self._calibration.calibration_table_frame or '<unset>'}->"
            f"{self._calibration.calibration_camera_frame or '<unset>'}, "
            f"published_pose=T_{self._event_camera_frame}_"
            f"{self.get_parameter('table_frame').value}"
        )

    def _publish_calibration(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self._camera_info_publisher.publish(
            camera_info_from_calibration(
                self._calibration,
                stamp,
                self._event_camera_frame,
            )
        )
        self._table_pose_publisher.publish(
            table_pose_from_calibration(
                self._calibration,
                stamp,
                self._event_camera_frame,
            )
        )


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[EventCameraCalibrationAdapterNode] = None
    try:
        node = EventCameraCalibrationAdapterNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except (EventCalibrationError, BallLocalizationError, ValueError) as error:
        rclpy.logging.get_logger("event_camera_calibration_adapter").fatal(str(error))
        raise SystemExit(2) from error
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
