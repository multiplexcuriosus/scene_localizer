"""
Geometry helpers for distortion-aware 2D-to-3D ball localization.

Transform names follow ``T_parent_child``: points in the child frame are
mapped into the parent frame. Pixel coordinates are used exactly as supplied;
this module never rotates, mirrors, resizes, or crops them.
"""

from __future__ import annotations

from typing import Any, Tuple

import cv2
from geometry_msgs.msg import PointStamped
import numpy as np


class BallLocalizationError(ValueError):
    """Raised when calibrated ray construction or intersection is invalid."""


def normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Normalize an xyzw quaternion, preserving the legacy zero-as-identity rule."""
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.size != 4 or not np.all(np.isfinite(q)):
        raise BallLocalizationError("quaternion must contain four finite xyzw values")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def quaternion_to_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix from an xyzw quaternion."""
    x, y, z, w = normalize_quaternion_xyzw(quaternion)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def validate_transform(transform: np.ndarray, name: str = "transform") -> np.ndarray:
    """Validate and return a finite rigid 4x4 homogeneous transform."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise BallLocalizationError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise BallLocalizationError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise BallLocalizationError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise BallLocalizationError(f"{name} rotation determinant is not +1")
    return matrix


def transform_from_translation_quaternion(
    translation: np.ndarray,
    quaternion_xyzw: np.ndarray,
    name: str = "transform",
) -> np.ndarray:
    """Construct a rigid transform from translation and xyzw quaternion."""
    t = np.asarray(translation, dtype=np.float64).reshape(-1)
    q = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(-1)
    if t.size != 3 or not np.all(np.isfinite(t)):
        raise BallLocalizationError(f"{name} translation must contain three finite values")
    if q.size != 4 or not np.all(np.isfinite(q)):
        raise BallLocalizationError(f"{name} quaternion must contain four finite values")
    if float(np.linalg.norm(q)) < 1e-12:
        raise BallLocalizationError(f"{name} quaternion norm must be non-zero")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_matrix_xyzw(q)
    transform[:3, 3] = t
    return validate_transform(transform, name)


def invert_transform(transform: np.ndarray, name: str = "transform") -> np.ndarray:
    """Invert a rigid transform without a generic matrix inverse."""
    parent_child = validate_transform(transform, name)
    child_parent = np.eye(4, dtype=np.float64)
    child_parent[:3, :3] = parent_child[:3, :3].T
    child_parent[:3, 3] = -parent_child[:3, :3].T @ parent_child[:3, 3]
    return child_parent


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized xyzw quaternion."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix = validate_transform(transform, "rotation")[:3, :3]

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    quaternion = normalize_quaternion_xyzw(np.array([x, y, z, w]))
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def camera_ray_from_pixel(
    u: float,
    v: float,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    *,
    pixels_are_rectified: bool,
    distortion_model: str = "plumb_bob",
) -> np.ndarray:
    """
    Return a unit camera ray for one native or already-rectified pixel.

    Raw pixels are undistorted exactly once with ``cv2.undistortPoints`` and
    calibrated K/D. Already-rectified pixels retain the legacy direct-pinhole
    calculation and deliberately skip distortion correction.
    """
    pixel = np.asarray([u, v], dtype=np.float64)
    if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
        raise BallLocalizationError("pixel coordinates must be finite")

    intrinsic = np.asarray(camera_matrix, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        raise BallLocalizationError("camera matrix K must be finite and 3x3")
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    if fx <= 0.0 or fy <= 0.0:
        raise BallLocalizationError("camera matrix K has non-positive focal length")

    if pixels_are_rectified:
        x_normalized = (float(u) - float(intrinsic[0, 2])) / fx
        y_normalized = (float(v) - float(intrinsic[1, 2])) / fy
    else:
        model = str(distortion_model).strip()
        if model not in {"plumb_bob", "rational_polynomial"}:
            raise BallLocalizationError(
                "raw pixels require distortion_model 'plumb_bob' or "
                f"'rational_polynomial', got '{model}'"
            )
        distortion = np.asarray(distortion_coefficients, dtype=np.float64).reshape(-1)
        if distortion.size not in {4, 5, 8, 12, 14} or not np.all(np.isfinite(distortion)):
            raise BallLocalizationError(
                "raw pixels require 4, 5, 8, 12, or 14 finite distortion coefficients"
            )
        source = pixel.reshape(1, 1, 2)
        try:
            undistorted = cv2.undistortPoints(source, intrinsic, distortion)
        except cv2.error as error:
            raise BallLocalizationError(f"cv2.undistortPoints failed: {error}") from error
        x_normalized = float(undistorted[0, 0, 0])
        y_normalized = float(undistorted[0, 0, 1])

    ray = np.array([x_normalized, y_normalized, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(ray))
    if not np.isfinite(norm) or norm < 1e-12:
        raise BallLocalizationError("failed to construct a finite camera ray")
    return ray / norm


def intersect_camera_ray_with_ball_plane(
    ray_camera: np.ndarray,
    T_camera_table: np.ndarray,
    ball_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intersect a camera ray with ``z_table = ball_radius``.

    Returns ``(point_camera, point_table)`` for the ball center.
    """
    ray = np.asarray(ray_camera, dtype=np.float64).reshape(-1)
    if ray.size != 3 or not np.all(np.isfinite(ray)):
        raise BallLocalizationError("camera ray must contain three finite values")
    ray_norm = float(np.linalg.norm(ray))
    if ray_norm < 1e-12:
        raise BallLocalizationError("camera ray norm must be non-zero")
    ray = ray / ray_norm

    camera_table = validate_transform(T_camera_table, "T_camera_table")
    radius = float(ball_radius)
    if not np.isfinite(radius) or radius < 0.0:
        raise BallLocalizationError("ball_radius must be finite and non-negative")

    rotation = camera_table[:3, :3]
    translation = camera_table[:3, 3]
    plane_point_camera = rotation @ np.array([0.0, 0.0, radius]) + translation
    plane_normal_camera = rotation @ np.array([0.0, 0.0, 1.0])

    denominator = float(np.dot(plane_normal_camera, ray))
    if abs(denominator) < 1e-9:
        raise BallLocalizationError("camera ray is nearly parallel to the ball-center plane")
    distance = float(np.dot(plane_normal_camera, plane_point_camera)) / denominator
    if distance <= 0.0 or not np.isfinite(distance):
        raise BallLocalizationError("ray-plane intersection is behind the camera")

    point_camera = distance * ray
    point_table = rotation.T @ (point_camera - translation)
    return point_camera, point_table


def make_ball_point_messages(
    point_camera: np.ndarray,
    point_table: np.ndarray,
    stamp: Any,
    camera_frame: str,
    table_frame: str,
) -> Tuple[PointStamped, PointStamped]:
    """Build camera/table PointStamped outputs with the exact source stamp."""
    camera_point = np.asarray(point_camera, dtype=np.float64).reshape(-1)
    table_point = np.asarray(point_table, dtype=np.float64).reshape(-1)
    if camera_point.size != 3 or not np.all(np.isfinite(camera_point)):
        raise BallLocalizationError("camera point must contain three finite values")
    if table_point.size != 3 or not np.all(np.isfinite(table_point)):
        raise BallLocalizationError("table point must contain three finite values")

    stamp_sec = int(getattr(stamp, "sec", 0))
    stamp_nanosec = int(getattr(stamp, "nanosec", 0))

    camera_message = PointStamped()
    camera_message.header.stamp.sec = stamp_sec
    camera_message.header.stamp.nanosec = stamp_nanosec
    camera_message.header.frame_id = str(camera_frame)
    camera_message.point.x = float(camera_point[0])
    camera_message.point.y = float(camera_point[1])
    camera_message.point.z = float(camera_point[2])

    table_message = PointStamped()
    table_message.header.stamp.sec = stamp_sec
    table_message.header.stamp.nanosec = stamp_nanosec
    table_message.header.frame_id = str(table_frame)
    table_message.point.x = float(table_point[0])
    table_message.point.y = float(table_point[1])
    table_message.point.z = float(table_point[2])
    return camera_message, table_message
