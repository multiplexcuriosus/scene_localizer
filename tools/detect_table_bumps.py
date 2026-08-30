#!/usr/bin/env python3
"""Detect dark table bumps in one RGB image and backproject them to the table.

This module is intentionally independent of ROS.  It uses the same transform
convention as scene_localizer: ``p_camera = T_camera_table @ p_table``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np
import yaml


CAMERA_FRAME = "camera_color_optical_frame"
TABLE_FRAME = "table_frame"
DEFAULT_TABLE_SIZE_M = (0.60, 1.20)

DEFAULT_BRIGHT_BACKGROUND_THRESHOLD = 160.0
DEFAULT_MINIMUM_DARK_CONTRAST = 20.0
DEFAULT_BACKGROUND_BLUR_SIGMA = 15.0
DEFAULT_MORPH_OPEN_KERNEL = 3
DEFAULT_MORPH_CLOSE_KERNEL = 5
DEFAULT_MIN_AREA_PX = 25
DEFAULT_MAX_AREA_PX = 50_000
SEGMENTATION_MODES = ("local_contrast", "hard_threshold", "otsu")
DEFAULT_SEGMENTATION_MODE = "local_contrast"


@dataclass(frozen=True)
class TablePose:
    """A validated camera-from-table transform and its source values."""

    parent_frame: str
    child_frame: str
    translation: np.ndarray
    quaternion_xyzw: np.ndarray
    T_camera_table: np.ndarray


@dataclass(frozen=True)
class ComponentDetection:
    """One accepted image component, expressed in crop-local pixels."""

    label: int
    area_px: int
    bbox_px_crop: tuple[int, int, int, int]
    centroid_px_crop: np.ndarray
    raw_contour_px_crop: np.ndarray
    minimum_area_rectangle_px_crop: np.ndarray


@dataclass(frozen=True)
class SegmentationResult:
    crop_bgr: np.ndarray
    l_channel: np.ndarray
    local_background: Optional[np.ndarray]
    analysis_region_mask: np.ndarray
    upper_right_triangle_px_crop: Optional[np.ndarray]
    brightness_statistics: dict[str, Any]
    segmentation_mode: str
    effective_dark_l_threshold: Optional[float]
    mask: np.ndarray
    raw_component_count: int
    components: tuple[ComponentDetection, ...]
    warnings: tuple[str, ...]


def _load_yaml_mapping(path: Path | str, description: str) -> dict[str, Any]:
    yaml_path = Path(path)
    try:
        with yaml_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid {description} YAML '{yaml_path}': {exc}") from exc
    except OSError as exc:
        raise OSError(f"Cannot read {description} YAML '{yaml_path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{description} YAML root must be a mapping: '{yaml_path}'")
    return data


def _finite_array(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _xyz(value: Any, name: str) -> np.ndarray:
    if isinstance(value, dict):
        try:
            value = [value[key] for key in ("x", "y", "z")]
        except KeyError as exc:
            raise ValueError(f"{name} must contain x, y, and z") from exc
    result = _finite_array(value, name).reshape(-1)
    if result.size != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {result.size}")
    return result


def _xyzw(value: Any, name: str) -> np.ndarray:
    if isinstance(value, dict):
        try:
            value = [value[key] for key in ("x", "y", "z", "w")]
        except KeyError as exc:
            raise ValueError(f"{name} must contain x, y, z, and w") from exc
    result = _finite_array(value, name).reshape(-1)
    if result.size != 4:
        raise ValueError(f"{name} must contain exactly 4 values, got {result.size}")
    norm = float(np.linalg.norm(result))
    if norm < 1e-12:
        raise ValueError(f"{name} has zero length")
    return result / norm


def _clean_frame(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    frame = str(value).strip().strip("/")
    if not frame:
        raise ValueError(f"{field_name} is empty")
    return frame


def quaternion_to_rotation(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    """Convert and normalize an ``[x, y, z, w]`` quaternion."""

    x, y, z, w = _xyzw(quaternion_xyzw, "quaternion_xyzw")
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


def make_transform(
    translation: Sequence[float], quaternion_xyzw: Sequence[float]
) -> np.ndarray:
    """Construct a homogeneous transform from translation and quaternion."""

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation(quaternion_xyzw)
    transform[:3, 3] = _xyz(translation, "translation")
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform using its rotation transpose."""

    transform = _finite_array(transform, "transform")
    if transform.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {transform.shape}")
    rotation = transform[:3, :3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def load_camera_info_yaml(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load ROS CameraInfo-style or simple ``K``/``D`` calibration YAML."""

    data = _load_yaml_mapping(path, "camera-info")

    if "camera_matrix" in data:
        ros_style_matrix = True
        camera_matrix = data["camera_matrix"]
        if not isinstance(camera_matrix, dict) or "data" not in camera_matrix:
            raise ValueError("camera_matrix must be a mapping containing data")
        if "rows" in camera_matrix and int(camera_matrix["rows"]) != 3:
            raise ValueError("camera_matrix.rows must equal 3")
        if "cols" in camera_matrix and int(camera_matrix["cols"]) != 3:
            raise ValueError("camera_matrix.cols must equal 3")
        K_raw = camera_matrix["data"]
    elif "K" in data:
        ros_style_matrix = False
        K_raw = data["K"]
    else:
        raise ValueError("CameraInfo YAML must contain camera_matrix or K")

    K = _finite_array(K_raw, "camera matrix K")
    if ros_style_matrix:
        if K.size != 9:
            raise ValueError(f"Camera matrix K must contain 9 values, got {K.size}")
        K = K.reshape(3, 3)
    elif K.shape != (3, 3):
        raise ValueError(f"Camera matrix K must be 3x3, got {K.shape}")
    if abs(float(K[0, 0])) < 1e-12 or abs(float(K[1, 1])) < 1e-12:
        raise ValueError("Camera matrix focal lengths must be non-zero")

    if "distortion_coefficients" in data:
        distortion = data["distortion_coefficients"]
        if not isinstance(distortion, dict) or "data" not in distortion:
            raise ValueError(
                "distortion_coefficients must be a mapping containing data"
            )
        D_raw = distortion["data"]
    else:
        D_raw = data.get("D")

    if D_raw is None:
        D = np.zeros(5, dtype=np.float64)
    else:
        D = _finite_array(D_raw, "distortion coefficients D").reshape(-1)
        if D.size == 0:
            D = np.zeros(5, dtype=np.float64)
    return K, D


def _pose_frame_metadata(
    data: dict[str, Any], expected_parent_frame: str, expected_child_frame: str
) -> tuple[str, str]:
    header = data.get("header")
    header_frame = None
    if header is not None:
        if not isinstance(header, dict):
            raise ValueError("PoseStamped header must be a mapping")
        header_frame = header.get("frame_id")

    parent_values: list[tuple[str, Any]] = []
    for key, value in (
        ("parent_frame", data.get("parent_frame")),
        ("frame_id", data.get("frame_id")),
        ("header.frame_id", header_frame),
    ):
        if value is not None:
            parent_values.append((key, value))

    if not parent_values:
        raise ValueError(
            "Table-pose YAML must identify its parent frame with parent_frame "
            "or header.frame_id; refusing to guess the transform convention"
        )

    normalized_parents = [
        (name, _clean_frame(value, name)) for name, value in parent_values
    ]
    parent_frame = normalized_parents[0][1]
    for field_name, frame in normalized_parents[1:]:
        if frame != parent_frame:
            raise ValueError(
                "Conflicting table-pose parent frames: "
                f"{normalized_parents[0][0]}='{parent_frame}', "
                f"{field_name}='{frame}'"
            )

    child_values: list[tuple[str, Any]] = []
    for key in ("child_frame", "child_frame_id"):
        if data.get(key) is not None:
            child_values.append((key, data[key]))

    if child_values:
        normalized_children = [
            (name, _clean_frame(value, name)) for name, value in child_values
        ]
        child_frame = normalized_children[0][1]
        for field_name, frame in normalized_children[1:]:
            if frame != child_frame:
                raise ValueError(
                    "Conflicting table-pose child frames: "
                    f"{normalized_children[0][0]}='{child_frame}', "
                    f"{field_name}='{frame}'"
                )
    elif "pose" in data:
        # geometry_msgs/PoseStamped has no child-frame field.  Its validated
        # camera parent plus this tool's table-pose input establish the child.
        child_frame = expected_child_frame
    else:
        raise ValueError(
            "Explicit transform YAML must contain child_frame; refusing to "
            "guess whether it is T_camera_table"
        )

    expected_parent = _clean_frame(expected_parent_frame, "expected parent frame")
    expected_child = _clean_frame(expected_child_frame, "expected child frame")
    if parent_frame != expected_parent or child_frame != expected_child:
        raise ValueError(
            "Incompatible table-pose frames: expected T_camera_table as "
            f"'{expected_parent} <- {expected_child}', got "
            f"'{parent_frame} <- {child_frame}'. A T_base_cam transform is not valid here."
        )
    return parent_frame, child_frame


def load_table_pose_yaml(
    path: Path | str,
    expected_parent_frame: str = CAMERA_FRAME,
    expected_child_frame: str = TABLE_FRAME,
) -> TablePose:
    """Load and validate an explicit transform or PoseStamped-style YAML."""

    data = _load_yaml_mapping(path, "table-pose")
    parent_frame, child_frame = _pose_frame_metadata(
        data, expected_parent_frame, expected_child_frame
    )

    if "pose" in data:
        pose = data["pose"]
        if not isinstance(pose, dict):
            raise ValueError("pose must be a mapping")
        try:
            translation_raw = pose["position"]
            quaternion_raw = pose["orientation"]
        except KeyError as exc:
            raise ValueError("pose must contain position and orientation") from exc
    else:
        try:
            translation_raw = data["translation"]
            quaternion_raw = data["quaternion_xyzw"]
        except KeyError as exc:
            raise ValueError(
                "Explicit table-pose YAML must contain translation and quaternion_xyzw"
            ) from exc

    translation = _xyz(translation_raw, "table-pose translation")
    quaternion = _xyzw(quaternion_raw, "table-pose quaternion_xyzw")
    transform = make_transform(translation, quaternion)
    return TablePose(
        parent_frame=parent_frame,
        child_frame=child_frame,
        translation=translation,
        quaternion_xyzw=quaternion,
        T_camera_table=transform,
    )


def _validate_table_size(table_size_m: Sequence[float]) -> tuple[float, float]:
    result = _finite_array(table_size_m, "table_size_m").reshape(-1)
    if result.size != 2 or np.any(result <= 0.0):
        raise ValueError("table_size_m must contain two positive values")
    return float(result[0]), float(result[1])


def _effective_distortion(D: Optional[np.ndarray], rectified: bool) -> Optional[np.ndarray]:
    if rectified:
        return None
    if D is None:
        return np.zeros(5, dtype=np.float64)
    distortion = _finite_array(D, "distortion coefficients D").reshape(-1)
    return distortion if distortion.size else np.zeros(5, dtype=np.float64)


def project_table_polygon(
    K: np.ndarray,
    D: Optional[np.ndarray],
    T_camera_table: np.ndarray,
    table_size_m: Sequence[float] = DEFAULT_TABLE_SIZE_M,
    *,
    rectified: bool = False,
) -> np.ndarray:
    """Project the z=0 table corners into full-image pixel coordinates."""

    table_x, table_y = _validate_table_size(table_size_m)
    K = _finite_array(K, "camera matrix K")
    transform = _finite_array(T_camera_table, "T_camera_table")
    if K.shape != (3, 3):
        raise ValueError(f"Camera matrix K must be 3x3, got {K.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"T_camera_table must be 4x4, got {transform.shape}")

    corners_table = np.array(
        [
            [0.0, 0.0, 0.0],
            [table_x, 0.0, 0.0],
            [table_x, table_y, 0.0],
            [0.0, table_y, 0.0],
        ],
        dtype=np.float64,
    )
    corners_camera = (
        transform[:3, :3] @ corners_table.T
    ).T + transform[:3, 3]
    if np.any(corners_camera[:, 2] <= 1e-9):
        raise ValueError(
            "Cannot project the table gate: one or more table corners are behind "
            "the camera. Check that the YAML is T_camera_table."
        )

    pixels, _ = cv2.projectPoints(
        corners_camera,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K,
        _effective_distortion(D, rectified),
    )
    pixels = pixels.reshape(-1, 2)
    if not np.all(np.isfinite(pixels)):
        raise ValueError("Projected table polygon contains non-finite pixels")
    return pixels


def crop_pixels_to_full(
    uv_crop: Sequence[Sequence[float]] | np.ndarray,
    crop_xyxy: Sequence[int],
) -> np.ndarray:
    """Add the full-image crop origin to crop-local pixel coordinates."""

    pixels = _finite_array(uv_crop, "crop-local pixels")
    if pixels.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    pixels = pixels.reshape(-1, 2)
    crop = np.asarray(crop_xyxy, dtype=np.int64).reshape(-1)
    if crop.size != 4:
        raise ValueError("crop_xyxy must contain four coordinates")
    return pixels + np.array([crop[0], crop[1]], dtype=np.float64)


def backproject_pixels_to_table_plane(
    uv_full: Sequence[Sequence[float]] | np.ndarray,
    K: np.ndarray,
    D: Optional[np.ndarray],
    T_camera_table: np.ndarray,
    height_m: float,
    table_size_m: Sequence[float] = DEFAULT_TABLE_SIZE_M,
    *,
    rectified: bool = False,
) -> np.ndarray:
    """Backproject full-image pixels to ``z_table = height_m``.

    The returned array has shape ``(N, 3)`` and preserves input ordering.
    Invalid intersections (parallel/behind the camera/non-finite/outside the
    table bounds) are represented by an all-NaN row so callers can reject a
    vertex without losing correspondence to the input polygon.
    """

    try:
        pixels = np.asarray(uv_full, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("full-image pixels must contain numeric values") from exc
    if pixels.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if pixels.size % 2:
        raise ValueError("full-image pixels must contain u/v pairs")
    pixels = pixels.reshape(-1, 2)
    K = _finite_array(K, "camera matrix K")
    transform = _finite_array(T_camera_table, "T_camera_table")
    if K.shape != (3, 3):
        raise ValueError(f"Camera matrix K must be 3x3, got {K.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"T_camera_table must be 4x4, got {transform.shape}")
    if not math.isfinite(float(height_m)):
        raise ValueError("height_m must be finite")
    table_x, table_y = _validate_table_size(table_size_m)

    finite_pixels = np.all(np.isfinite(pixels), axis=1)
    normalized = np.full(pixels.shape, np.nan, dtype=np.float64)
    if np.any(finite_pixels):
        normalized[finite_pixels] = cv2.undistortPoints(
            pixels[finite_pixels].reshape(-1, 1, 2),
            K,
            _effective_distortion(D, rectified),
        ).reshape(-1, 2)
    rays_camera = np.column_stack(
        [normalized, np.ones(normalized.shape[0], dtype=np.float64)]
    )

    rotation_camera_table = transform[:3, :3]
    translation_camera_table = transform[:3, 3]
    rotation_table_camera = rotation_camera_table.T
    origin_table = -rotation_table_camera @ translation_camera_table
    rays_table = (rotation_table_camera @ rays_camera.T).T

    denominator = rays_table[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        lambdas = (float(height_m) - origin_table[2]) / denominator
        intersections = origin_table[None, :] + lambdas[:, None] * rays_table

    valid = (
        np.isfinite(lambdas)
        & (lambdas > 0.0)
        & (np.abs(denominator) > 1e-12)
        & np.all(np.isfinite(intersections), axis=1)
        & (intersections[:, 0] >= 0.0)
        & (intersections[:, 0] <= table_x)
        & (intersections[:, 1] >= 0.0)
        & (intersections[:, 1] <= table_y)
    )
    result = np.full((pixels.shape[0], 3), np.nan, dtype=np.float64)
    result[valid] = intersections[valid]
    # Make the plane contract exact in serialized JSON/CSV, rather than only
    # numerically close after matrix arithmetic.
    result[valid, 2] = float(height_m)
    return result


def _validated_crop(
    crop_xyxy: Sequence[int], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    crop = np.asarray(crop_xyxy, dtype=np.int64).reshape(-1)
    if crop.size != 4:
        raise ValueError("crop must contain X0 Y0 X1 Y1")
    x0, y0, x1, y1 = (int(value) for value in crop)
    if x0 < 0 or y0 < 0 or x1 > image_width or y1 > image_height:
        raise ValueError(
            f"Crop [{x0}, {y0}, {x1}, {y1}] is outside the "
            f"{image_width}x{image_height} image"
        )
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Crop must satisfy X0 < X1 and Y0 < Y1")
    return x0, y0, x1, y1


def _validate_exclude_rectangles(
    rectangles: Optional[Iterable[Sequence[int]]],
) -> tuple[tuple[int, int, int, int], ...]:
    result: list[tuple[int, int, int, int]] = []
    for index, rectangle in enumerate(rectangles or ()):
        values = np.asarray(rectangle, dtype=np.int64).reshape(-1)
        if values.size != 4:
            raise ValueError(f"exclude rectangle {index} must have four coordinates")
        x0, y0, x1, y1 = (int(value) for value in values)
        if x0 >= x1 or y0 >= y1:
            raise ValueError(
                f"exclude rectangle {index} must satisfy X0 < X1 and Y0 < Y1"
            )
        result.append((x0, y0, x1, y1))
    return tuple(result)


def _validate_crop_upper_right(
    crop_upper_right: Optional[Sequence[int]],
    crop_width: int,
    crop_height: int,
) -> Optional[tuple[int, int]]:
    if crop_upper_right is None:
        return None
    values = np.asarray(crop_upper_right, dtype=np.int64).reshape(-1)
    if values.size != 2:
        raise ValueError("crop_upper_right must contain width A and height B")
    width, height = (int(value) for value in values)
    if width <= 0 or height <= 0:
        raise ValueError("crop-upper-right A and B must both be positive")
    if width > crop_width or height > crop_height:
        raise ValueError(
            "crop-upper-right must fit inside the rectangular crop: "
            f"got {width}x{height}, crop is {crop_width}x{crop_height}"
        )
    return width, height


def compute_brightness_statistics(
    l_channel: np.ndarray,
    analysis_region_mask: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Summarize the 8-bit Lab-L histogram in the usable crop region."""

    channel = np.asarray(l_channel)
    if channel.ndim != 2:
        raise ValueError("l_channel must be a two-dimensional array")
    if not np.all(np.isfinite(channel)):
        raise ValueError("l_channel contains non-finite values")
    if np.any(channel < 0) or np.any(channel > 255):
        raise ValueError("l_channel values must be in [0, 255]")
    channel_u8 = channel.astype(np.uint8)

    if analysis_region_mask is None:
        valid = np.ones(channel_u8.shape, dtype=bool)
    else:
        region = np.asarray(analysis_region_mask)
        if region.shape != channel_u8.shape:
            raise ValueError("analysis_region_mask must match l_channel shape")
        valid = region != 0
    values = channel_u8[valid]
    if values.size == 0:
        raise ValueError("No pixels remain in the crop analysis region")

    histogram = np.bincount(values, minlength=256).astype(np.int64)
    otsu_threshold, _ = cv2.threshold(
        values.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    dark = values <= otsu_threshold
    bright = ~dark

    def class_summary(class_values: np.ndarray) -> dict[str, Any]:
        if class_values.size == 0:
            return {
                "count": 0,
                "fraction": 0.0,
                "minimum": None,
                "maximum": None,
                "mean": None,
                "standard_deviation": None,
            }
        return {
            "count": int(class_values.size),
            "fraction": float(class_values.size / values.size),
            "minimum": int(np.min(class_values)),
            "maximum": int(np.max(class_values)),
            "mean": float(np.mean(class_values)),
            "standard_deviation": float(np.std(class_values)),
        }

    dark_values = values[dark]
    bright_values = values[bright]
    dark_summary = class_summary(dark_values)
    bright_summary = class_summary(bright_values)
    total_variance = float(np.var(values))
    if dark_values.size and bright_values.size and total_variance > 0.0:
        dark_fraction = float(dark_values.size / values.size)
        bright_fraction = 1.0 - dark_fraction
        mean_difference = float(np.mean(bright_values) - np.mean(dark_values))
        between_class_variance = (
            dark_fraction * bright_fraction * mean_difference * mean_difference
        )
        variance_explained = between_class_variance / total_variance
        pooled_standard_deviation = math.sqrt(
            0.5 * (float(np.var(dark_values)) + float(np.var(bright_values)))
        )
        standardized_separation = (
            mean_difference / pooled_standard_deviation
            if pooled_standard_deviation > 0.0
            else None
        )
    else:
        variance_explained = 0.0
        standardized_separation = None

    percentile_values = np.percentile(
        values, [1, 5, 10, 25, 50, 75, 90, 95, 99]
    )
    return {
        "channel": "CIELAB_L_uint8",
        "valid_pixel_count": int(values.size),
        "minimum": int(np.min(values)),
        "maximum": int(np.max(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "percentiles": {
            name: float(value)
            for name, value in zip(
                ("p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"),
                percentile_values,
            )
        },
        "histogram_counts_256": histogram.tolist(),
        "otsu_threshold_l": float(otsu_threshold),
        "otsu_dark_class": dark_summary,
        "otsu_bright_class": bright_summary,
        "otsu_between_class_variance_ratio": float(variance_explained),
        "otsu_standardized_mean_separation": (
            None
            if standardized_separation is None
            else float(standardized_separation)
        ),
    }


def _morphology_kernel(size: int, name: str) -> Optional[np.ndarray]:
    if size < 0:
        raise ValueError(f"{name} must be non-negative")
    if size <= 1:
        return None
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def segment_dark_components(
    image_bgr: np.ndarray,
    crop_xyxy: Sequence[int],
    *,
    dark_l_threshold: Optional[float] = None,
    bright_background_threshold: float = DEFAULT_BRIGHT_BACKGROUND_THRESHOLD,
    minimum_dark_contrast: float = DEFAULT_MINIMUM_DARK_CONTRAST,
    background_blur_sigma: float = DEFAULT_BACKGROUND_BLUR_SIGMA,
    morph_open_kernel: int = DEFAULT_MORPH_OPEN_KERNEL,
    morph_close_kernel: int = DEFAULT_MORPH_CLOSE_KERNEL,
    min_area_px: int = DEFAULT_MIN_AREA_PX,
    max_area_px: Optional[int] = DEFAULT_MAX_AREA_PX,
    table_polygon_full: Optional[np.ndarray] = None,
    exclude_rects: Optional[Iterable[Sequence[int]]] = None,
    crop_upper_right: Optional[Sequence[int]] = None,
    segmentation_mode: str = DEFAULT_SEGMENTATION_MODE,
    allow_boundary_components: bool = False,
) -> SegmentationResult:
    """Segment and filter dark connected components inside a full-image crop."""

    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
        raise ValueError("image_bgr must be an HxWx3 array")
    if image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have exactly three color channels")
    image_height, image_width = image_bgr.shape[:2]
    x0, y0, x1, y1 = _validated_crop(crop_xyxy, image_width, image_height)
    if segmentation_mode not in SEGMENTATION_MODES:
        raise ValueError(
            f"segmentation_mode must be one of {', '.join(SEGMENTATION_MODES)}"
        )
    if (
        segmentation_mode == "local_contrast"
        and (
            not math.isfinite(float(background_blur_sigma))
            or background_blur_sigma <= 0.0
        )
    ):
        raise ValueError("background_blur_sigma must be positive and finite")
    if not 0.0 <= float(bright_background_threshold) <= 255.0:
        raise ValueError("bright_background_threshold must be in [0, 255]")
    if not 0.0 <= float(minimum_dark_contrast) <= 255.0:
        raise ValueError("minimum_dark_contrast must be in [0, 255]")
    if dark_l_threshold is not None and not 0.0 <= float(dark_l_threshold) <= 255.0:
        raise ValueError("dark_l_threshold must be in [0, 255]")
    if segmentation_mode == "hard_threshold" and dark_l_threshold is None:
        raise ValueError(
            "hard_threshold segmentation requires --dark-l-threshold"
        )
    if segmentation_mode == "otsu" and dark_l_threshold is not None:
        raise ValueError(
            "--dark-l-threshold is not used in otsu mode; omit it or select "
            "hard_threshold"
        )
    if int(min_area_px) < 0:
        raise ValueError("min_area_px must be non-negative")
    if max_area_px is not None and int(max_area_px) < int(min_area_px):
        raise ValueError("max_area_px must be greater than or equal to min_area_px")

    crop_bgr = image_bgr[y0:y1, x0:x1].copy()
    l_channel = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    crop_height, crop_width = l_channel.shape
    analysis_region_mask = np.full(l_channel.shape, 255, dtype=np.uint8)

    if table_polygon_full is not None:
        table_polygon = _finite_array(
            table_polygon_full, "projected table polygon"
        ).reshape(-1, 2)
        if table_polygon.shape[0] != 4:
            raise ValueError("Projected table polygon must contain four vertices")
        polygon_crop = table_polygon - np.array([x0, y0], dtype=np.float64)
        gate = np.zeros(l_channel.shape, dtype=np.uint8)
        cv2.fillConvexPoly(gate, np.rint(polygon_crop).astype(np.int32), 255)
        analysis_region_mask = cv2.bitwise_and(analysis_region_mask, gate)

    excluded_rectangles = _validate_exclude_rectangles(exclude_rects)
    for ex0, ey0, ex1, ey1 in excluded_rectangles:
        local_x0 = max(0, ex0 - x0)
        local_y0 = max(0, ey0 - y0)
        local_x1 = min(x1 - x0, ex1 - x0)
        local_y1 = min(y1 - y0, ey1 - y0)
        if local_x0 < local_x1 and local_y0 < local_y1:
            analysis_region_mask[local_y0:local_y1, local_x0:local_x1] = 0

    triangle_size = _validate_crop_upper_right(
        crop_upper_right, crop_width, crop_height
    )
    triangle_vertices: Optional[np.ndarray] = None
    triangle_exclusion_mask: Optional[np.ndarray] = None
    triangle_boundary_band: Optional[np.ndarray] = None
    if triangle_size is not None:
        triangle_width, triangle_height = triangle_size
        triangle_vertices = np.array(
            [
                [crop_width - triangle_width, 0],
                [crop_width, 0],
                [crop_width, triangle_height],
            ],
            dtype=np.int32,
        )
        triangle_exclusion_mask = np.zeros(l_channel.shape, dtype=np.uint8)
        cv2.fillConvexPoly(triangle_exclusion_mask, triangle_vertices, 255)
        analysis_region_mask[triangle_exclusion_mask != 0] = 0
        triangle_boundary_band = cv2.dilate(
            triangle_exclusion_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

    brightness_statistics = compute_brightness_statistics(
        l_channel, analysis_region_mask
    )
    effective_dark_l_threshold: Optional[float] = None
    local_background: Optional[np.ndarray] = None
    l_float = l_channel.astype(np.float32)
    if segmentation_mode == "local_contrast":
        local_background = cv2.GaussianBlur(
            l_channel,
            (0, 0),
            sigmaX=float(background_blur_sigma),
            sigmaY=float(background_blur_sigma),
        )
        contrast = local_background.astype(np.float32) - l_float
        mask_bool = (
            (
                local_background.astype(np.float32)
                > float(bright_background_threshold)
            )
            & (contrast > float(minimum_dark_contrast))
        )
        if dark_l_threshold is not None:
            effective_dark_l_threshold = float(dark_l_threshold)
            mask_bool &= l_float < effective_dark_l_threshold
    elif segmentation_mode == "hard_threshold":
        effective_dark_l_threshold = float(dark_l_threshold)
        mask_bool = l_float < effective_dark_l_threshold
    else:
        effective_dark_l_threshold = float(
            brightness_statistics["otsu_threshold_l"]
        )
        mask_bool = l_float <= effective_dark_l_threshold

    mask = mask_bool.astype(np.uint8) * 255
    open_kernel = _morphology_kernel(int(morph_open_kernel), "morph_open_kernel")
    close_kernel = _morphology_kernel(int(morph_close_kernel), "morph_close_kernel")
    if open_kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    if close_kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.bitwise_and(mask, analysis_region_mask)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    raw_component_count = int(count - 1)
    components: list[ComponentDetection] = []
    warnings: list[str] = []
    centroid_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        if area < int(min_area_px):
            warnings.append(
                f"component label {label} rejected: area {area}px is below {int(min_area_px)}px"
            )
            continue
        if max_area_px is not None and area > int(max_area_px):
            warnings.append(
                f"component label {label} rejected: area {area}px exceeds {int(max_area_px)}px"
            )
            continue
        touches_boundary = (
            left == 0
            or top == 0
            or left + width == crop_width
            or top + height == crop_height
        )
        component_pixels = labels == label
        touches_triangle_boundary = bool(
            triangle_boundary_band is not None
            and np.any(triangle_boundary_band[component_pixels] != 0)
        )
        if (
            touches_boundary or touches_triangle_boundary
        ) and not allow_boundary_components:
            boundary_name = (
                "upper-right triangle crop boundary"
                if touches_triangle_boundary
                else "rectangular crop boundary"
            )
            warnings.append(
                f"component label {label} rejected: touches the {boundary_name}"
            )
            continue

        component_mask = component_pixels.astype(np.uint8) * 255
        eroded_core = cv2.erode(
            component_mask,
            centroid_kernel,
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        core_y, core_x = np.nonzero(eroded_core)
        if core_x.size == 0:
            warnings.append(
                f"component label {label} rejected: no pixels remain in the eroded core"
            )
            continue
        centroid = np.array(
            [float(np.mean(core_x)), float(np.mean(core_y))], dtype=np.float64
        )

        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            warnings.append(f"component label {label} rejected: no contour found")
            continue
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        rectangle = cv2.boxPoints(
            cv2.minAreaRect(contour.astype(np.float32).reshape(-1, 1, 2))
        ).astype(np.float64)

        components.append(
            ComponentDetection(
                label=label,
                area_px=area,
                bbox_px_crop=(left, top, left + width - 1, top + height - 1),
                centroid_px_crop=centroid,
                raw_contour_px_crop=contour,
                minimum_area_rectangle_px_crop=rectangle,
            )
        )

    return SegmentationResult(
        crop_bgr=crop_bgr,
        l_channel=l_channel,
        local_background=local_background,
        analysis_region_mask=analysis_region_mask,
        upper_right_triangle_px_crop=(
            None
            if triangle_vertices is None
            else triangle_vertices.astype(np.float64)
        ),
        brightness_statistics=brightness_statistics,
        segmentation_mode=segmentation_mode,
        effective_dark_l_threshold=effective_dark_l_threshold,
        mask=mask,
        raw_component_count=raw_component_count,
        components=tuple(components),
        warnings=tuple(warnings),
    )


def _valid_projected_rows(points_xyz: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return points_xyz[np.all(np.isfinite(points_xyz), axis=1)]


def components_to_bump_records(
    components: Sequence[ComponentDetection],
    crop_xyxy: Sequence[int],
    K: np.ndarray,
    D: Optional[np.ndarray],
    T_camera_table: np.ndarray,
    height_m: float,
    table_size_m: Sequence[float],
    *,
    polygon_source: str = "contour",
    rectified: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Backproject component geometry and build JSON-ready bump records."""

    if polygon_source not in ("contour", "minimum_area_rectangle"):
        raise ValueError(
            "polygon_source must be 'contour' or 'minimum_area_rectangle'"
        )
    crop = np.asarray(crop_xyxy, dtype=np.int64).reshape(-1)
    if crop.size != 4:
        raise ValueError("crop_xyxy must contain four coordinates")
    offset = np.array([int(crop[0]), int(crop[1])], dtype=np.float64)

    bumps: list[dict[str, Any]] = []
    warnings: list[str] = []
    for component in components:
        centroid_full = crop_pixels_to_full(
            component.centroid_px_crop.reshape(1, 2), crop
        )
        centroid_table = backproject_pixels_to_table_plane(
            centroid_full,
            K,
            D,
            T_camera_table,
            height_m,
            table_size_m,
            rectified=rectified,
        )[0]
        if not np.all(np.isfinite(centroid_table)):
            warnings.append(
                f"component label {component.label} rejected: centroid ray does not "
                "intersect the requested height inside the table bounds"
            )
            continue

        raw_contour_crop = component.raw_contour_px_crop
        rectangle_crop = component.minimum_area_rectangle_px_crop
        selected_crop = (
            raw_contour_crop
            if polygon_source == "contour"
            else rectangle_crop
        )
        raw_contour_full = raw_contour_crop + offset
        rectangle_full = rectangle_crop + offset
        selected_full = selected_crop + offset

        selected_table_all = backproject_pixels_to_table_plane(
            selected_full,
            K,
            D,
            T_camera_table,
            height_m,
            table_size_m,
            rectified=rectified,
        )
        selected_table = _valid_projected_rows(selected_table_all)
        rejected_selected_vertices = selected_table_all.shape[0] - selected_table.shape[0]
        if rejected_selected_vertices:
            warnings.append(
                f"component label {component.label}: rejected "
                f"{rejected_selected_vertices}/{selected_table_all.shape[0]} selected "
                "polygon vertices outside the valid table intersection"
            )

        rectangle_table_all = backproject_pixels_to_table_plane(
            rectangle_full,
            K,
            D,
            T_camera_table,
            height_m,
            table_size_m,
            rectified=rectified,
        )
        rectangle_table = _valid_projected_rows(rectangle_table_all)
        rejected_rectangle_vertices = (
            rectangle_table_all.shape[0] - rectangle_table.shape[0]
        )
        if rejected_rectangle_vertices:
            warnings.append(
                f"component label {component.label}: rejected "
                f"{rejected_rectangle_vertices}/4 minimum-area rectangle vertices "
                "outside the valid table intersection"
            )

        bbox_crop = component.bbox_px_crop
        bbox_full = [
            int(bbox_crop[0] + crop[0]),
            int(bbox_crop[1] + crop[1]),
            int(bbox_crop[2] + crop[0]),
            int(bbox_crop[3] + crop[1]),
        ]
        bump_id = len(bumps)
        bumps.append(
            {
                "id": bump_id,
                "component_label": int(component.label),
                "area_px": int(component.area_px),
                "centroid_px_crop": component.centroid_px_crop.tolist(),
                "centroid_px_full": centroid_full[0].tolist(),
                "bbox_px_crop": [int(value) for value in bbox_crop],
                "bbox_px_full": bbox_full,
                "raw_contour_px_crop": raw_contour_crop.tolist(),
                "raw_contour_px_full": raw_contour_full.tolist(),
                "polygon_source": polygon_source,
                "polygon_px_crop": selected_crop.tolist(),
                "polygon_px_full": selected_full.tolist(),
                "minimum_area_rectangle_px_crop": rectangle_crop.tolist(),
                "minimum_area_rectangle_px_full": rectangle_full.tolist(),
                "centroid_table_m": centroid_table.tolist(),
                "polygon_table_xy_m": selected_table[:, :2].tolist(),
                "polygon_table_xyz_m": selected_table.tolist(),
                "minimum_area_rectangle_table_xy_m": rectangle_table[:, :2].tolist(),
                "minimum_area_rectangle_table_xyz_m": rectangle_table.tolist(),
            }
        )
    return bumps, warnings


def _titled_panel(image: np.ndarray, title: str) -> np.ndarray:
    panel = cv2.copyMakeBorder(
        image, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28)
    )
    cv2.putText(
        panel,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def create_segmentation_overlay(
    segmentation: SegmentationResult,
    bumps: Sequence[dict[str, Any]],
    table_polygon_full: Optional[np.ndarray],
    crop_xyxy: Sequence[int],
) -> np.ndarray:
    """Create original/mask/annotated diagnostic panels."""

    original = segmentation.crop_bgr.copy()
    mask_panel = cv2.cvtColor(segmentation.mask, cv2.COLOR_GRAY2BGR)
    annotated = segmentation.crop_bgr.copy()
    crop = np.asarray(crop_xyxy, dtype=np.float64).reshape(4)

    has_triangle_crop = segmentation.upper_right_triangle_px_crop is not None
    if has_triangle_crop:
        triangle = np.asarray(
            segmentation.upper_right_triangle_px_crop, dtype=np.float64
        ).copy()
        triangle[:, 0] = np.clip(triangle[:, 0], 0, original.shape[1] - 1)
        triangle[:, 1] = np.clip(triangle[:, 1], 0, original.shape[0] - 1)
        triangle_i32 = np.rint(triangle).astype(np.int32)
        cv2.fillConvexPoly(original, triangle_i32, (20, 20, 20))
        cv2.fillConvexPoly(annotated, triangle_i32, (20, 20, 20))
        cv2.polylines(
            original,
            [triangle_i32.reshape(-1, 1, 2)],
            True,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.polylines(
            annotated,
            [triangle_i32.reshape(-1, 1, 2)],
            True,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if table_polygon_full is not None:
        polygon_crop = np.asarray(table_polygon_full, dtype=np.float64) - crop[:2]
        cv2.polylines(
            annotated,
            [np.rint(polygon_crop).astype(np.int32).reshape(-1, 1, 2)],
            True,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for bump in bumps:
        contour = np.rint(np.asarray(bump["raw_contour_px_crop"])).astype(np.int32)
        cv2.drawContours(
            annotated,
            [contour.reshape(-1, 1, 2)],
            -1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.drawContours(
            mask_panel,
            [contour.reshape(-1, 1, 2)],
            -1,
            (0, 180, 0),
            1,
            cv2.LINE_AA,
        )
        centroid = tuple(
            int(round(value)) for value in bump["centroid_px_crop"]
        )
        cv2.drawMarker(
            annotated,
            centroid,
            (0, 0, 255),
            cv2.MARKER_CROSS,
            13,
            2,
            cv2.LINE_AA,
        )
        label_origin = (centroid[0] + 7, centroid[1] - 7)
        cv2.putText(
            annotated,
            str(bump["id"]),
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            str(bump["id"]),
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return np.hstack(
        [
            _titled_panel(
                original,
                "Spatially cropped input" if has_triangle_crop else "Original crop",
            ),
            _titled_panel(mask_panel, "Threshold mask"),
            _titled_panel(annotated, "Accepted components"),
        ]
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_bumps_csv(path: Path | str, bumps: Sequence[dict[str, Any]]) -> None:
    """Write the stable centroid-only CSV interchange format."""

    fieldnames = [
        "bump_id",
        "area_px",
        "centroid_u_px",
        "centroid_v_px",
        "bbox_xmin_px",
        "bbox_ymin_px",
        "bbox_xmax_px",
        "bbox_ymax_px",
        "x_table_m",
        "y_table_m",
        "z_table_m",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for bump in bumps:
            centroid_px = bump["centroid_px_full"]
            bbox = bump["bbox_px_full"]
            centroid_table = bump["centroid_table_m"]
            writer.writerow(
                {
                    "bump_id": int(bump["id"]),
                    "area_px": int(bump["area_px"]),
                    "centroid_u_px": f"{float(centroid_px[0]):.6f}",
                    "centroid_v_px": f"{float(centroid_px[1]):.6f}",
                    "bbox_xmin_px": int(bbox[0]),
                    "bbox_ymin_px": int(bbox[1]),
                    "bbox_xmax_px": int(bbox[2]),
                    "bbox_ymax_px": int(bbox[3]),
                    "x_table_m": f"{float(centroid_table[0]):.9f}",
                    "y_table_m": f"{float(centroid_table[1]):.9f}",
                    "z_table_m": f"{float(centroid_table[2]):.9f}",
                }
            )


def save_brightness_histogram(
    path: Path | str,
    statistics: dict[str, Any],
    *,
    segmentation_mode: str,
    effective_dark_l_threshold: Optional[float],
    visualize: bool = False,
) -> None:
    """Save the valid-region Lab-L histogram and threshold diagnostics."""

    import matplotlib

    if not visualize:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.asarray(statistics["histogram_counts_256"], dtype=np.int64)
    bins = np.arange(256)
    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.step(bins, counts, where="mid", color="#252525", linewidth=1.2)
    axis.fill_between(bins, counts, step="mid", color="#8da0cb", alpha=0.45)
    axis.set_yscale("log")

    otsu_threshold = float(statistics["otsu_threshold_l"])
    axis.axvline(
        otsu_threshold,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label=f"Otsu L={otsu_threshold:.0f}",
    )
    if effective_dark_l_threshold is not None:
        axis.axvline(
            float(effective_dark_l_threshold),
            color="#ff7f0e",
            linestyle=":",
            linewidth=2.0,
            label=(
                f"Effective detector L={float(effective_dark_l_threshold):.0f}"
            ),
        )

    dark = statistics["otsu_dark_class"]
    bright = statistics["otsu_bright_class"]
    separation = statistics["otsu_standardized_mean_separation"]
    separation_text = "n/a" if separation is None else f"{float(separation):.2f}"

    def mean_std_text(class_statistics: dict[str, Any]) -> str:
        mean = class_statistics["mean"]
        standard_deviation = class_statistics["standard_deviation"]
        if mean is None or standard_deviation is None:
            return "n/a"
        return f"{float(mean):.1f}/{float(standard_deviation):.1f}"

    summary = (
        f"valid pixels: {statistics['valid_pixel_count']:,}\n"
        f"mean/std: {statistics['mean']:.1f} / "
        f"{statistics['standard_deviation']:.1f}\n"
        f"dark: {100.0 * dark['fraction']:.2f}%  "
        f"mean/std={mean_std_text(dark)}\n"
        f"bright: {100.0 * bright['fraction']:.2f}%  "
        f"mean/std={mean_std_text(bright)}\n"
        f"standardized mean separation: {separation_text}"
    )
    axis.text(
        0.985,
        0.965,
        summary,
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axis.set_xlim(0, 255)
    axis.set_xlabel("Lab L brightness (0=black, 255=white)")
    axis.set_ylabel("Pixel count (log scale)")
    axis.set_title(f"Crop brightness histogram ({segmentation_mode})")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(Path(path), dpi=160)
    if not visualize:
        plt.close(figure)


def save_topdown_bump_map(
    path: Path | str,
    bumps: Sequence[dict[str, Any]],
    table_size_m: Sequence[float],
    *,
    visualize: bool = False,
    segmentation_overlay_bgr: Optional[np.ndarray] = None,
) -> None:
    """Save an equal-aspect table-frame bump map and optionally display it."""

    import matplotlib

    if not visualize:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    table_x, table_y = _validate_table_size(table_size_m)
    figure, axis = plt.subplots(figsize=(6.5, 10.0))
    axis.add_patch(
        Rectangle(
            (0.0, 0.0),
            table_x,
            table_y,
            facecolor="#f4f0df",
            edgecolor="black",
            linewidth=2.0,
            zorder=0,
        )
    )

    color_map = plt.get_cmap("tab10")
    for bump in bumps:
        bump_id = int(bump["id"])
        color = color_map(bump_id % 10)
        polygon_xy = np.asarray(bump["polygon_table_xy_m"], dtype=np.float64)
        if polygon_xy.shape[0] >= 3:
            axis.add_patch(
                Polygon(
                    polygon_xy,
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.35,
                    linewidth=1.5,
                    zorder=2,
                )
            )
        elif polygon_xy.shape[0] >= 2:
            axis.plot(polygon_xy[:, 0], polygon_xy[:, 1], color=color, zorder=2)
        centroid = np.asarray(bump["centroid_table_m"], dtype=np.float64)
        axis.scatter(
            [centroid[0]], [centroid[1]], color=[color], edgecolor="black", zorder=3
        )
        axis.annotate(
            str(bump_id),
            (centroid[0], centroid[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            zorder=4,
        )

    margin = 0.03 * max(table_x, table_y)
    axis.set_xlim(-margin, table_x + margin)
    axis.set_ylim(-margin, table_y + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x_table (m)")
    axis.set_ylabel("y_table (m)")
    axis.set_title("Table bump map")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(Path(path), dpi=160)

    if visualize:
        if segmentation_overlay_bgr is not None:
            overlay_figure, overlay_axis = plt.subplots(figsize=(16, 6))
            overlay_axis.imshow(
                cv2.cvtColor(segmentation_overlay_bgr, cv2.COLOR_BGR2RGB)
            )
            overlay_axis.axis("off")
            overlay_figure.tight_layout()
        plt.show()
    else:
        plt.close(figure)


def _resolved_string(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


def run_detection(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the complete CLI pipeline and write all requested artifacts."""

    image_path = Path(args.image).expanduser()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image '{image_path}'")
    image_height, image_width = image.shape[:2]
    crop_xyxy = _validated_crop(args.crop, image_width, image_height)
    table_size_m = _validate_table_size(args.table_size)
    height_m = float(args.bump_height_m)
    if not math.isfinite(height_m) or height_m < 0.0:
        raise ValueError("bump-height-m must be finite and non-negative")

    K, D = load_camera_info_yaml(args.camera_info)
    table_pose = load_table_pose_yaml(args.table_pose)
    exclude_rects = _validate_exclude_rectangles(args.exclude_rect or ())
    table_polygon_full = None
    if not args.no_table_gate:
        table_polygon_full = project_table_polygon(
            K,
            D,
            table_pose.T_camera_table,
            table_size_m,
            rectified=bool(args.rectified),
        )

    print(f"Image dimensions: {image_width} x {image_height} pixels")
    print(f"Crop coordinates (full-image xyxy): {list(crop_xyxy)}")

    segmentation = segment_dark_components(
        image,
        crop_xyxy,
        dark_l_threshold=args.dark_l_threshold,
        bright_background_threshold=args.bright_background_threshold,
        minimum_dark_contrast=args.minimum_dark_contrast,
        background_blur_sigma=args.background_blur_sigma,
        morph_open_kernel=args.morph_open_kernel,
        morph_close_kernel=args.morph_close_kernel,
        min_area_px=args.min_area_px,
        max_area_px=args.max_area_px,
        table_polygon_full=table_polygon_full,
        exclude_rects=exclude_rects,
        crop_upper_right=args.crop_upper_right,
        segmentation_mode=args.segmentation_mode,
        allow_boundary_components=bool(args.allow_boundary_components),
    )
    bumps, projection_warnings = components_to_bump_records(
        segmentation.components,
        crop_xyxy,
        K,
        D,
        table_pose.T_camera_table,
        height_m,
        table_size_m,
        polygon_source=args.polygon_source,
        rectified=bool(args.rectified),
    )

    all_warnings = list(segmentation.warnings) + projection_warnings
    histogram_statistics = segmentation.brightness_statistics
    otsu_dark = histogram_statistics["otsu_dark_class"]
    otsu_bright = histogram_statistics["otsu_bright_class"]

    def diagnostic_class_text(class_statistics: dict[str, Any]) -> str:
        mean = class_statistics["mean"]
        standard_deviation = class_statistics["standard_deviation"]
        if mean is None or standard_deviation is None:
            return "empty"
        return f"mean={float(mean):.2f}, std={float(standard_deviation):.2f}"

    print(f"Segmentation mode: {segmentation.segmentation_mode}")
    if args.crop_upper_right is not None:
        print(
            "Upper-right triangle crop (width x height): "
            f"{int(args.crop_upper_right[0])} x {int(args.crop_upper_right[1])} pixels"
        )
    print(
        "Brightness histogram (valid crop, Lab L): "
        f"mean={histogram_statistics['mean']:.2f}, "
        f"std={histogram_statistics['standard_deviation']:.2f}, "
        f"Otsu threshold={histogram_statistics['otsu_threshold_l']:.1f}"
    )
    print(
        "Otsu classes: "
        f"dark={100.0 * otsu_dark['fraction']:.2f}% "
        f"({diagnostic_class_text(otsu_dark)}), "
        f"bright={100.0 * otsu_bright['fraction']:.2f}% "
        f"({diagnostic_class_text(otsu_bright)})"
    )
    print(f"Raw connected components: {segmentation.raw_component_count}")
    print(f"Accepted components: {len(bumps)}")
    for bump in bumps:
        pixel = bump["centroid_px_full"]
        point = bump["centroid_table_m"]
        print(
            f"Bump {bump['id']}: centroid_px_full=({pixel[0]:.3f}, {pixel[1]:.3f}), "
            f"centroid_table_m=({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})"
        )
    for warning in all_warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    source_image = _resolved_string(image_path)
    camera_matrix_list = K.tolist()
    distortion_list = D.tolist()
    transform_block = {
        "parent_frame": table_pose.parent_frame,
        "child_frame": table_pose.child_frame,
        "translation": table_pose.translation.tolist(),
        "quaternion_xyzw": table_pose.quaternion_xyzw.tolist(),
    }
    triangle_crop = segmentation.upper_right_triangle_px_crop
    triangle_full = (
        None
        if triangle_crop is None
        else (
            triangle_crop
            + np.array([crop_xyxy[0], crop_xyxy[1]], dtype=np.float64)
        )
    )
    bumps_document: dict[str, Any] = {
        "frame": table_pose.child_frame,
        "height_m": height_m,
        "table_size_m": list(table_size_m),
        "source_image": source_image,
        "crop_xyxy": list(crop_xyxy),
        "crop_upper_right": (
            None
            if args.crop_upper_right is None
            else [int(value) for value in args.crop_upper_right]
        ),
        "camera_matrix": camera_matrix_list,
        "distortion": distortion_list,
        "rectified": bool(args.rectified),
        "T_camera_table": transform_block,
        "bumps": bumps,
    }

    metadata: dict[str, Any] = {
        "source_image": source_image,
        "image_dimensions_px": {
            "width": image_width,
            "height": image_height,
        },
        "crop_xyxy": list(crop_xyxy),
        "thresholds": {
            "segmentation_mode": segmentation.segmentation_mode,
            "dark_l_threshold": args.dark_l_threshold,
            "effective_dark_l_threshold": (
                segmentation.effective_dark_l_threshold
            ),
            "bright_background_threshold": float(args.bright_background_threshold),
            "minimum_dark_contrast": float(args.minimum_dark_contrast),
            "background_blur_sigma_px": float(args.background_blur_sigma),
            "morph_open_kernel_px": int(args.morph_open_kernel),
            "morph_close_kernel_px": int(args.morph_close_kernel),
            "min_area_px": int(args.min_area_px),
            "max_area_px": (
                None if args.max_area_px is None else int(args.max_area_px)
            ),
        },
        "segmentation": {
            "table_gate_enabled": not bool(args.no_table_gate),
            "projected_table_polygon_px_full": (
                None if table_polygon_full is None else table_polygon_full.tolist()
            ),
            "exclude_rectangles_xyxy_full": [list(rect) for rect in exclude_rects],
            "crop_upper_right_width_height_px": (
                None
                if args.crop_upper_right is None
                else [int(value) for value in args.crop_upper_right]
            ),
            "upper_right_triangle_px_crop": (
                None if triangle_crop is None else triangle_crop.tolist()
            ),
            "upper_right_triangle_px_full": (
                None if triangle_full is None else triangle_full.tolist()
            ),
            "allow_boundary_components": bool(args.allow_boundary_components),
            "polygon_source": args.polygon_source,
        },
        "brightness_histogram": histogram_statistics,
        "camera_calibration": {
            "source_yaml": _resolved_string(args.camera_info),
            "camera_matrix": camera_matrix_list,
            "distortion": distortion_list,
            "rectified_input": bool(args.rectified),
        },
        "T_camera_table": {
            "source_yaml": _resolved_string(args.table_pose),
            **transform_block,
            "matrix": table_pose.T_camera_table.tolist(),
        },
        "table_size_m": list(table_size_m),
        "bump_height_m": height_m,
        "raw_component_count": segmentation.raw_component_count,
        "detected_component_count": len(bumps),
        "warnings": all_warnings,
    }

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "bumps.json", bumps_document)
    write_bumps_csv(output_dir / "bumps.csv", bumps)
    overlay = create_segmentation_overlay(
        segmentation, bumps, table_polygon_full, crop_xyxy
    )
    overlay_path = output_dir / "segmentation_overlay.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"Failed to write '{overlay_path}'")
    save_brightness_histogram(
        output_dir / "brightness_histogram.png",
        histogram_statistics,
        segmentation_mode=segmentation.segmentation_mode,
        effective_dark_l_threshold=segmentation.effective_dark_l_threshold,
        visualize=bool(args.visualize),
    )
    save_topdown_bump_map(
        output_dir / "topdown_bump_map.png",
        bumps,
        table_size_m,
        visualize=bool(args.visualize),
        segmentation_overlay_bgr=overlay,
    )
    _write_json(output_dir / "metadata.json", metadata)
    print(f"Wrote bump-map artifacts to: {output_dir.resolve()}")
    return bumps_document, metadata


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect dark raised objects on a bright table and backproject their "
            "known-height tops into table-frame coordinates."
        )
    )
    parser.add_argument("--image", required=True, help="Input RGB/BGR image file")
    parser.add_argument(
        "--crop",
        required=True,
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Half-open crop in full-image pixel coordinates",
    )
    parser.add_argument(
        "--crop-upper-right",
        nargs=2,
        type=int,
        default=None,
        metavar=("A", "B"),
        help=(
            "Exclude an upper-right triangle from the rectangular crop. A is "
            "its top-edge width and B its right-edge height, in crop-local pixels"
        ),
    )
    parser.add_argument("--camera-info", required=True, help="CameraInfo YAML file")
    parser.add_argument(
        "--table-pose", required=True, help="T_camera_table pose YAML file"
    )
    parser.add_argument(
        "--bump-height-m",
        required=True,
        type=float,
        help="Uniform bump-top height above the table surface",
    )
    parser.add_argument("--output-dir", required=True, help="Artifact output directory")
    parser.add_argument(
        "--table-size",
        nargs=2,
        type=float,
        default=DEFAULT_TABLE_SIZE_M,
        metavar=("X", "Y"),
        help="Table x/y extents in meters (default: 0.60 1.20)",
    )
    parser.add_argument(
        "--segmentation-mode",
        choices=SEGMENTATION_MODES,
        default=DEFAULT_SEGMENTATION_MODE,
        help=(
            "Foreground rule: local_contrast preserves locally normalized "
            "segmentation, hard_threshold uses --dark-l-threshold alone, and "
            "otsu uses the valid crop histogram (default: local_contrast)"
        ),
    )
    parser.add_argument(
        "--dark-l-threshold",
        type=float,
        default=None,
        help=(
            "Lab-L hard cutoff; required by hard_threshold and an additional "
            "gate in local_contrast mode"
        ),
    )
    parser.add_argument(
        "--bright-background-threshold",
        type=float,
        default=DEFAULT_BRIGHT_BACKGROUND_THRESHOLD,
        help=f"Minimum local-background Lab L (default: {DEFAULT_BRIGHT_BACKGROUND_THRESHOLD:g})",
    )
    parser.add_argument(
        "--minimum-dark-contrast",
        type=float,
        default=DEFAULT_MINIMUM_DARK_CONTRAST,
        help=f"Minimum local background-minus-pixel L (default: {DEFAULT_MINIMUM_DARK_CONTRAST:g})",
    )
    parser.add_argument(
        "--background-blur-sigma",
        type=float,
        default=DEFAULT_BACKGROUND_BLUR_SIGMA,
        help=(
            "Gaussian local-background sigma in pixels "
            f"(default: {DEFAULT_BACKGROUND_BLUR_SIGMA:g})"
        ),
    )
    parser.add_argument(
        "--morph-open-kernel",
        type=int,
        default=DEFAULT_MORPH_OPEN_KERNEL,
        help=f"Opening kernel size; 0/1 disables (default: {DEFAULT_MORPH_OPEN_KERNEL})",
    )
    parser.add_argument(
        "--morph-close-kernel",
        type=int,
        default=DEFAULT_MORPH_CLOSE_KERNEL,
        help=f"Closing kernel size; 0/1 disables (default: {DEFAULT_MORPH_CLOSE_KERNEL})",
    )
    parser.add_argument(
        "--min-area-px",
        type=int,
        default=DEFAULT_MIN_AREA_PX,
        help=f"Minimum connected-component area (default: {DEFAULT_MIN_AREA_PX})",
    )
    parser.add_argument(
        "--max-area-px",
        type=int,
        default=DEFAULT_MAX_AREA_PX,
        help=f"Maximum connected-component area (default: {DEFAULT_MAX_AREA_PX})",
    )
    parser.add_argument(
        "--no-table-gate",
        action="store_true",
        help="Do not restrict the mask to the projected table polygon",
    )
    parser.add_argument(
        "--rectified",
        action="store_true",
        help="Treat the image as already rectified; do not apply D again",
    )
    parser.add_argument(
        "--exclude-rect",
        action="append",
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Half-open rectangle to mask, in full-image coordinates; repeatable",
    )
    parser.add_argument(
        "--polygon-source",
        choices=("contour", "minimum_area_rectangle"),
        default="contour",
        help="Geometry used for polygon_table_* output (default: contour)",
    )
    parser.add_argument(
        "--allow-boundary-components",
        "--allow-crop-boundary-components",
        dest="allow_boundary_components",
        action="store_true",
        help="Accept components that touch a crop edge",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display the saved segmentation and top-down diagnostics",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        run_detection(args)
    except (OSError, ValueError, cv2.error) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
