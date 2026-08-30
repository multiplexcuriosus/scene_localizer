import math
from pathlib import Path

from builtin_interfaces.msg import Time
import cv2
import numpy as np
import pytest
import yaml

from scene_localizer.event_ball_geometry import (
    camera_ray_from_pixel,
    intersect_camera_ray_with_ball_plane,
    invert_transform,
    make_ball_point_messages,
    quaternion_to_matrix_xyzw,
    rotation_matrix_to_quaternion_xyzw,
    transform_from_translation_quaternion,
)
from scene_localizer.event_camera_calibration_adapter_node import (
    EventCalibrationError,
    camera_info_from_calibration,
    load_event_camera_calibration,
    table_pose_from_calibration,
)


def _transform_block(
    transform: np.ndarray,
    parent_frame: str,
    child_frame: str,
) -> dict:
    quaternion = rotation_matrix_to_quaternion_xyzw(transform[:3, :3])
    return {
        "parent_frame": parent_frame,
        "child_frame": child_frame,
        "maps": f"{child_frame}_coordinates_into_{parent_frame}",
        "translation_m": transform[:3, 3].tolist(),
        "quaternion_xyzw": quaternion.tolist(),
        "matrix": transform.tolist(),
    }


def _calibration_payload(T_table_camera: np.ndarray | None = None) -> dict:
    if T_table_camera is None:
        T_table_camera = transform_from_translation_quaternion(
            [0.4, -0.2, 0.8],
            [0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0)],
            "T_table_camera",
        )
    T_camera_table = invert_transform(T_table_camera, "T_table_camera")
    return {
        "calibration_convention": (
            "T_parent_child maps coordinates from child into parent"
        ),
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [
                205.0,
                0.0,
                160.0,
                0.0,
                207.0,
                158.0,
                0.0,
                0.0,
                1.0,
            ],
        },
        "distortion_coefficients": {
            "rows": 1,
            "cols": 5,
            "data": [0.12, -0.07, -0.01, 0.02, 0.005],
        },
        "distortion_model": "plumb_bob",
        "image_width": 320,
        "image_height": 320,
        "table_frame": "table",
        "camera_frame": "openmv_cam",
        "T_table_camera": _transform_block(
            T_table_camera, "table", "openmv_cam"
        ),
        "T_camera_table": _transform_block(
            T_camera_table, "openmv_cam", "table"
        ),
    }


def _write_payload(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_quaternion_transform_inversion_round_trip():
    T_parent_child = transform_from_translation_quaternion(
        [0.3, -0.4, 1.2],
        [0.2, -0.3, 0.1, 0.9],
        "T_parent_child",
    )
    T_child_parent = invert_transform(T_parent_child, "T_parent_child")

    np.testing.assert_allclose(
        T_child_parent @ T_parent_child,
        np.eye(4),
        atol=1e-12,
    )
    recovered_quaternion = rotation_matrix_to_quaternion_xyzw(
        T_parent_child[:3, :3]
    )
    np.testing.assert_allclose(
        quaternion_to_matrix_xyzw(recovered_quaternion),
        T_parent_child[:3, :3],
        atol=1e-12,
    )


def test_adapter_inverts_T_table_camera_for_published_pose(tmp_path: Path):
    T_table_camera = transform_from_translation_quaternion(
        [0.7, 0.1, 0.9],
        [0.1, -0.2, 0.3, 0.9],
        "T_table_camera",
    )
    calibration_path = tmp_path / "solved.yaml"
    _write_payload(calibration_path, _calibration_payload(T_table_camera))

    calibration = load_event_camera_calibration(
        calibration_path,
        expected_image_width=320,
        expected_image_height=320,
    )
    expected = invert_transform(T_table_camera, "T_table_camera")
    np.testing.assert_allclose(calibration.T_camera_table, expected, atol=1e-12)

    pose_message = table_pose_from_calibration(
        calibration,
        Time(sec=12, nanosec=34),
        "event_camera",
    )
    pose_transform = transform_from_translation_quaternion(
        [
            pose_message.pose.position.x,
            pose_message.pose.position.y,
            pose_message.pose.position.z,
        ],
        [
            pose_message.pose.orientation.x,
            pose_message.pose.orientation.y,
            pose_message.pose.orientation.z,
            pose_message.pose.orientation.w,
        ],
        "published_T_camera_table",
    )
    np.testing.assert_allclose(pose_transform, expected, atol=1e-12)
    assert pose_message.header.frame_id == "event_camera"


def test_known_rectified_pixel_ray():
    camera_matrix = np.array(
        [[200.0, 0.0, 160.0], [0.0, 250.0, 120.0], [0.0, 0.0, 1.0]]
    )
    ray = camera_ray_from_pixel(
        180.0,
        95.0,
        camera_matrix,
        np.zeros(5),
        pixels_are_rectified=True,
    )
    expected = np.array([0.1, -0.1, 1.0])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(ray, expected, atol=1e-12)


def test_known_distorted_pixel_ray():
    camera_matrix = np.array(
        [[210.0, 0.0, 158.0], [0.0, 208.0, 161.0], [0.0, 0.0, 1.0]]
    )
    distortion = np.array([0.18, -0.09, 0.012, -0.02, 0.025])
    ideal = np.array([0.24, -0.17, 1.0])
    distorted_pixel, _ = cv2.projectPoints(
        ideal.reshape(1, 3),
        np.zeros(3),
        np.zeros(3),
        camera_matrix,
        distortion,
    )

    ray = camera_ray_from_pixel(
        float(distorted_pixel[0, 0, 0]),
        float(distorted_pixel[0, 0, 1]),
        camera_matrix,
        distortion,
        pixels_are_rectified=False,
        distortion_model="plumb_bob",
    )
    expected = ideal / np.linalg.norm(ideal)
    np.testing.assert_allclose(ray, expected, atol=1e-8)


def test_ray_intersection_uses_ball_center_plane():
    T_camera_table = np.eye(4)
    T_camera_table[:3, 3] = [0.0, 0.0, 1.0]
    point_camera, point_table = intersect_camera_ray_with_ball_plane(
        np.array([0.0, 0.0, 1.0]),
        T_camera_table,
        0.035,
    )

    np.testing.assert_allclose(point_camera, [0.0, 0.0, 1.035], atol=1e-12)
    np.testing.assert_allclose(point_table, [0.0, 0.0, 0.035], atol=1e-12)


def test_output_messages_preserve_incoming_timestamp():
    source_stamp = Time(sec=123, nanosec=456_789)
    camera_message, table_message = make_ball_point_messages(
        np.array([0.1, 0.2, 1.0]),
        np.array([0.3, 0.4, 0.035]),
        source_stamp,
        "event_camera",
        "table_frame",
    )

    assert (camera_message.header.stamp.sec, camera_message.header.stamp.nanosec) == (
        123,
        456_789,
    )
    assert (table_message.header.stamp.sec, table_message.header.stamp.nanosec) == (
        123,
        456_789,
    )
    assert table_message.header.frame_id == "table_frame"


@pytest.mark.parametrize(
    "missing_key",
    [
        "camera_matrix",
        "distortion_coefficients",
        "distortion_model",
        "image_width",
        "image_height",
        "T_table_camera",
    ],
)
def test_invalid_calibration_yaml_is_rejected(
    tmp_path: Path,
    missing_key: str,
):
    payload = _calibration_payload()
    del payload[missing_key]
    calibration_path = tmp_path / f"missing_{missing_key}.yaml"
    _write_payload(calibration_path, payload)

    with pytest.raises(EventCalibrationError):
        load_event_camera_calibration(calibration_path)


def test_mismatched_native_image_dimensions_are_rejected(tmp_path: Path):
    payload = _calibration_payload()
    payload["image_width"] = 640
    calibration_path = tmp_path / "wrong_size.yaml"
    _write_payload(calibration_path, payload)

    with pytest.raises(EventCalibrationError, match="will not be rescaled"):
        load_event_camera_calibration(
            calibration_path,
            expected_image_width=320,
            expected_image_height=320,
        )


def test_camera_info_contains_native_calibration(tmp_path: Path):
    payload = _calibration_payload()
    calibration_path = tmp_path / "solved.yaml"
    _write_payload(calibration_path, payload)
    calibration = load_event_camera_calibration(calibration_path)

    message = camera_info_from_calibration(
        calibration,
        Time(sec=4, nanosec=5),
        "event_camera",
    )
    assert (message.width, message.height) == (320, 320)
    assert message.header.frame_id == "event_camera"
    assert message.distortion_model == "plumb_bob"
    np.testing.assert_allclose(message.k, payload["camera_matrix"]["data"])
    np.testing.assert_allclose(
        message.d,
        payload["distortion_coefficients"]["data"],
    )


def test_synthetic_event_pixel_recovers_table_ball_center():
    camera_matrix = np.array(
        [
            [201.87239922460154, 0.0, 178.6890327726553],
            [0.0, 207.63465666496444, 131.34359993507243],
            [0.0, 0.0, 1.0],
        ]
    )
    distortion = np.array(
        [
            0.11992561573367166,
            -0.07116164978457495,
            -0.04425024584575582,
            0.058997566824052806,
            0.010208672526730132,
        ]
    )
    T_camera_table = np.eye(4)
    T_camera_table[:3, 3] = [0.0, 0.0, 1.0]
    expected_table_point = np.array([0.18, 0.24, 0.035])

    distorted_pixel, _ = cv2.projectPoints(
        expected_table_point.reshape(1, 3),
        np.zeros(3),
        T_camera_table[:3, 3],
        camera_matrix,
        distortion,
    )
    ray = camera_ray_from_pixel(
        float(distorted_pixel[0, 0, 0]),
        float(distorted_pixel[0, 0, 1]),
        camera_matrix,
        distortion,
        pixels_are_rectified=False,
        distortion_model="plumb_bob",
    )
    _, recovered_table_point = intersect_camera_ray_with_ball_plane(
        ray,
        T_camera_table,
        0.035,
    )
    np.testing.assert_allclose(
        recovered_table_point,
        expected_table_point,
        atol=1e-8,
    )
