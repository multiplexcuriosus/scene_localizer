#!/usr/bin/env python3

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import detect_table_bumps as detector


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array(
            [[200.0, 0.0, 100.0], [0.0, 200.0, 80.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.D = np.zeros(5, dtype=np.float64)
        self.T_camera_table = detector.make_transform(
            [0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]
        )

    def test_quaternion_to_rotation_conversion(self):
        half_angle = math.pi / 4.0
        rotation = detector.quaternion_to_rotation(
            [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]
        )
        expected = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        np.testing.assert_allclose(rotation, expected, atol=1e-12)

    def test_inversion_of_T_camera_table(self):
        transform = detector.make_transform(
            [0.3, -0.2, 1.4], [0.1, -0.2, 0.3, 0.9]
        )
        inverse = detector.invert_transform(transform)
        np.testing.assert_allclose(inverse @ transform, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(transform @ inverse, np.eye(4), atol=1e-12)

    def test_ray_intersects_known_height_plane(self):
        height = 0.10
        expected_table = np.array([0.22, 0.33, height])
        point_camera = expected_table + np.array([0.0, 0.0, 1.0])
        pixel = np.array(
            [
                self.K[0, 0] * point_camera[0] / point_camera[2] + self.K[0, 2],
                self.K[1, 1] * point_camera[1] / point_camera[2] + self.K[1, 2],
            ]
        )
        result = detector.backproject_pixels_to_table_plane(
            [pixel],
            self.K,
            self.D,
            self.T_camera_table,
            height,
            (0.60, 1.20),
        )
        np.testing.assert_allclose(result[0], expected_table, atol=1e-10)

    def test_crop_offset_is_added_before_backprojection(self):
        crop_xyxy = (10, 20, 210, 180)
        crop_local_principal_point = np.array([[90.0, 60.0]])
        full_pixels = detector.crop_pixels_to_full(
            crop_local_principal_point, crop_xyxy
        )
        np.testing.assert_allclose(full_pixels, [[100.0, 80.0]])

        point = detector.backproject_pixels_to_table_plane(
            full_pixels,
            self.K,
            self.D,
            self.T_camera_table,
            0.05,
            (0.60, 1.20),
        )[0]
        np.testing.assert_allclose(point, [0.0, 0.0, 0.05], atol=1e-12)

    def test_points_outside_table_bounds_are_rejected(self):
        height = 0.10
        outside_x = 0.75
        depth = 1.0 + height
        pixel = [
            self.K[0, 0] * outside_x / depth + self.K[0, 2],
            self.K[1, 2],
        ]
        result = detector.backproject_pixels_to_table_plane(
            [pixel, [float("nan"), 80.0]],
            self.K,
            self.D,
            self.T_camera_table,
            height,
            (0.60, 1.20),
        )
        self.assertTrue(np.all(np.isnan(result[0])))
        self.assertTrue(np.all(np.isnan(result[1])))


class CalibrationParsingTests(unittest.TestCase):
    def _write_yaml(self, directory: Path, name: str, value) -> Path:
        path = directory / name
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(value, stream, sort_keys=False)
        return path

    def test_parses_both_camera_info_yaml_formats(self):
        expected_K = np.array(
            [[500.0, 0.0, 320.0], [0.0, 501.0, 240.0], [0.0, 0.0, 1.0]]
        )
        expected_D = np.array([0.1, -0.02, 0.003, -0.004, 0.005])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ros_path = self._write_yaml(
                directory,
                "camera_ros.yaml",
                {
                    "camera_matrix": {
                        "rows": 3,
                        "cols": 3,
                        "data": expected_K.reshape(-1).tolist(),
                    },
                    "distortion_coefficients": {
                        "rows": 1,
                        "cols": 5,
                        "data": expected_D.tolist(),
                    },
                },
            )
            simple_path = self._write_yaml(
                directory,
                "camera_simple.yaml",
                {"K": expected_K.tolist(), "D": expected_D.tolist()},
            )
            no_distortion_path = self._write_yaml(
                directory,
                "camera_without_distortion.yaml",
                {"K": expected_K.tolist()},
            )

            for path in (ros_path, simple_path):
                K, D = detector.load_camera_info_yaml(path)
                np.testing.assert_allclose(K, expected_K)
                np.testing.assert_allclose(D, expected_D)

            K, D = detector.load_camera_info_yaml(no_distortion_path)
            np.testing.assert_allclose(K, expected_K)
            np.testing.assert_allclose(D, np.zeros(5))

    def test_parses_both_table_pose_yaml_formats(self):
        translation = [0.1, -0.2, 1.3]
        quaternion = [0.0, 0.0, 0.0, 1.0]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            explicit_path = self._write_yaml(
                directory,
                "table_explicit.yaml",
                {
                    "parent_frame": detector.CAMERA_FRAME,
                    "child_frame": detector.TABLE_FRAME,
                    "translation": translation,
                    "quaternion_xyzw": quaternion,
                },
            )
            pose_stamped_path = self._write_yaml(
                directory,
                "table_pose_stamped.yaml",
                {
                    "header": {"frame_id": detector.CAMERA_FRAME},
                    "pose": {
                        "position": {"x": 0.1, "y": -0.2, "z": 1.3},
                        "orientation": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0,
                            "w": 1.0,
                        },
                    },
                },
            )

            for path in (explicit_path, pose_stamped_path):
                pose = detector.load_table_pose_yaml(path)
                self.assertEqual(pose.parent_frame, detector.CAMERA_FRAME)
                self.assertEqual(pose.child_frame, detector.TABLE_FRAME)
                np.testing.assert_allclose(pose.translation, translation)
                np.testing.assert_allclose(pose.quaternion_xyzw, quaternion)
                np.testing.assert_allclose(
                    pose.T_camera_table[:3, 3], translation
                )

    def test_wrong_frame_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_yaml(
                Path(temporary),
                "wrong_transform.yaml",
                {
                    "parent_frame": "base",
                    "child_frame": detector.CAMERA_FRAME,
                    "translation": [0.0, 0.0, 1.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            )
            with self.assertRaisesRegex(ValueError, "T_base_cam"):
                detector.load_table_pose_yaml(path)


class SegmentationAndOutputTests(unittest.TestCase):
    @staticmethod
    def _synthetic_image() -> np.ndarray:
        image = np.full((180, 260, 3), 240, dtype=np.uint8)
        for x0, y0, x1, y1 in (
            (35, 35, 53, 53),
            (105, 70, 127, 89),
            (185, 115, 205, 137),
        ):
            cv2.rectangle(image, (x0, y0), (x1, y1), (20, 20, 20), -1)
        return image

    def test_synthetic_dark_rectangles_are_detected(self):
        image = self._synthetic_image()
        result = detector.segment_dark_components(
            image,
            (10, 10, 245, 165),
            bright_background_threshold=180,
            minimum_dark_contrast=20,
            background_blur_sigma=15,
            morph_open_kernel=1,
            morph_close_kernel=3,
            min_area_px=100,
            max_area_px=1_000,
        )
        self.assertEqual(result.raw_component_count, 3)
        self.assertEqual(len(result.components), 3)

    def test_upper_right_triangle_crop_with_hard_threshold(self):
        image = np.full((120, 160, 3), 230, dtype=np.uint8)
        cv2.rectangle(image, (30, 70), (48, 88), (20, 20, 20), -1)
        cv2.rectangle(image, (140, 5), (153, 16), (20, 20, 20), -1)

        result = detector.segment_dark_components(
            image,
            (0, 0, 160, 120),
            crop_upper_right=(50, 60),
            segmentation_mode="hard_threshold",
            dark_l_threshold=100,
            morph_open_kernel=1,
            morph_close_kernel=1,
            min_area_px=50,
            max_area_px=1_000,
        )

        self.assertEqual(result.raw_component_count, 1)
        self.assertEqual(len(result.components), 1)
        np.testing.assert_allclose(
            result.upper_right_triangle_px_crop,
            [[110.0, 0.0], [160.0, 0.0], [160.0, 60.0]],
        )
        self.assertEqual(result.analysis_region_mask[5, 150], 0)
        self.assertNotEqual(result.analysis_region_mask[80, 40], 0)
        self.assertEqual(result.segmentation_mode, "hard_threshold")
        self.assertEqual(result.effective_dark_l_threshold, 100.0)
        self.assertEqual(
            len(result.brightness_statistics["histogram_counts_256"]), 256
        )

    def test_serialized_z_table_equals_requested_height(self):
        image = np.full((160, 220, 3), 240, dtype=np.uint8)
        cv2.rectangle(image, (112, 92), (132, 112), (15, 15, 15), -1)
        segmentation = detector.segment_dark_components(
            image,
            (10, 10, 210, 150),
            bright_background_threshold=180,
            minimum_dark_contrast=20,
            background_blur_sigma=15,
            morph_open_kernel=1,
            morph_close_kernel=3,
            min_area_px=100,
            max_area_px=1_000,
        )
        self.assertEqual(len(segmentation.components), 1)

        K = np.array(
            [[500.0, 0.0, 100.0], [0.0, 500.0, 80.0], [0.0, 0.0, 1.0]]
        )
        transform = detector.make_transform(
            [0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]
        )
        requested_height = 0.073
        bumps, warnings = detector.components_to_bump_records(
            segmentation.components,
            (10, 10, 210, 150),
            K,
            np.zeros(5),
            transform,
            requested_height,
            (0.60, 1.20),
        )
        self.assertEqual(len(bumps), 1, warnings)
        self.assertEqual(bumps[0]["centroid_table_m"][2], requested_height)

        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "bumps.csv"
            detector.write_bumps_csv(csv_path, bumps)
            with csv_path.open("r", encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(row["z_table_m"]), requested_height, places=9)
            self.assertNotIn("s", row)


if __name__ == "__main__":
    unittest.main()
