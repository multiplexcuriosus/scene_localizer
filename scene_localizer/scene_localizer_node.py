from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from aruco_opencv_msgs.msg import ArucoDetection
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


@dataclass
class TransformEstimate:
    translation: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass
class TimedEstimate:
    stamp_sec: float
    estimate: TransformEstimate


@dataclass
class StreamConfig:
    name: str
    detections_topic: str
    camera_frame: str
    pose_topic: str
    min_markers_per_frame: int


@dataclass
class StreamState:
    config: StreamConfig
    marker_layout: Dict[int, TransformEstimate]
    pose_pub: Any
    buffer: List[TimedEstimate] = field(default_factory=list)
    last_accepted_estimate: Optional[TransformEstimate] = None
    last_published_estimate: Optional[TransformEstimate] = None
    last_source_stamp_sec: Optional[float] = None


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


def _matrix_to_quaternion_xyzw(r: np.ndarray) -> np.ndarray:
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
            w = (r[2, 1] - r[1, 2]) / s
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
            w = (r[0, 2] - r[2, 0]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
            w = (r[1, 0] - r[0, 1]) / s
    return _normalize_quaternion_xyzw(np.array([x, y, z, w], dtype=float))


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _invert_transform(translation: np.ndarray, rotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r_inv = rotation.T
    t_inv = -r_inv @ translation
    return t_inv, r_inv


def _compose_transform(
    t_ab: np.ndarray,
    r_ab: np.ndarray,
    t_bc: np.ndarray,
    r_bc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    t_ac = t_ab + r_ab @ t_bc
    r_ac = r_ab @ r_bc
    return t_ac, r_ac


def _quaternion_angular_distance_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    qa = _normalize_quaternion_xyzw(q1)
    qb = _normalize_quaternion_xyzw(q2)
    dot = float(np.clip(np.abs(np.dot(qa, qb)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _average_quaternions_xyzw(quaternions: List[np.ndarray]) -> np.ndarray:
    if not quaternions:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    ref = _normalize_quaternion_xyzw(quaternions[0])
    accum = np.zeros(4, dtype=float)
    for q in quaternions:
        qn = _normalize_quaternion_xyzw(q)
        if float(np.dot(qn, ref)) < 0.0:
            qn = -qn
        accum += qn

    if float(np.linalg.norm(accum)) < 1e-12:
        return ref
    return _normalize_quaternion_xyzw(accum)


def _blend_quaternions_xyzw(prev_q: np.ndarray, curr_q: np.ndarray, alpha: float) -> np.ndarray:
    alpha_clamped = min(1.0, max(0.0, float(alpha)))
    q_prev = _normalize_quaternion_xyzw(prev_q)
    q_curr = _normalize_quaternion_xyzw(curr_q)
    if float(np.dot(q_prev, q_curr)) < 0.0:
        q_curr = -q_curr
    blended = (1.0 - alpha_clamped) * q_prev + alpha_clamped * q_curr
    if float(np.linalg.norm(blended)) < 1e-12:
        return q_curr
    return _normalize_quaternion_xyzw(blended)


class SceneLocalizerNode(Node):
    def __init__(self) -> None:
        super().__init__("scene_localizer")

        self._last_warn_time_ns: Dict[str, int] = {}
        self._tf_broadcaster = TransformBroadcaster(self)

        try:
            share_dir = Path(get_package_share_directory("scene_localizer"))
            default_yaml = share_dir / "config" / "table_marker_layout.yaml"
        except Exception:
            default_yaml = Path(__file__).resolve().parents[1] / "config" / "table_marker_layout.yaml"

        self.declare_parameter("marker_layout_yaml", str(default_yaml))
        self.declare_parameter("window_sec", 5.0)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("min_markers_per_frame", 1)
        self.declare_parameter("top_min_markers_per_frame", -1)
        self.declare_parameter("eef_min_markers_per_frame", -1)
        self.declare_parameter("min_estimates_in_window", 3)
        self.declare_parameter("max_candidate_translation_deviation", 0.05)
        self.declare_parameter("max_window_translation_deviation", 0.08)
        self.declare_parameter("max_translation_jump", 0.12)
        self.declare_parameter("max_rotation_jump_deg", 35.0)
        self.declare_parameter("smoothing_alpha", 0.25)
        self.declare_parameter("publish_tf", False)
        self.declare_parameter("top_detections_topic", "/aruco_top_cam/aruco_detections")
        self.declare_parameter("eef_detections_topic", "/aruco_eef_cam/aruco_detections")
        self.declare_parameter("top_camera_frame", "camera_color_optical_frame")
        self.declare_parameter("eef_camera_frame", "camera_color_optical_frame")
        self.declare_parameter("table_frame", "table_frame")
        self.declare_parameter("debug_log", False)

        self._table_frame = str(self.get_parameter("table_frame").value)

        marker_layout_yaml = str(self.get_parameter("marker_layout_yaml").value)
        stream_layouts = self._load_layout_by_stream(marker_layout_yaml)

        global_min_markers = max(1, int(self.get_parameter("min_markers_per_frame").value))
        top_min_markers_param = int(self.get_parameter("top_min_markers_per_frame").value)
        eef_min_markers_param = int(self.get_parameter("eef_min_markers_per_frame").value)

        top_min_markers = global_min_markers if top_min_markers_param < 1 else top_min_markers_param
        eef_min_markers = global_min_markers if eef_min_markers_param < 1 else eef_min_markers_param

        top_config = StreamConfig(
            name="top_cam",
            detections_topic=str(self.get_parameter("top_detections_topic").value),
            camera_frame=str(self.get_parameter("top_camera_frame").value),
            pose_topic="/scene_localizer/top_cam/table_pose_camera",
            min_markers_per_frame=top_min_markers,
        )
        eef_config = StreamConfig(
            name="eef_cam",
            detections_topic=str(self.get_parameter("eef_detections_topic").value),
            camera_frame=str(self.get_parameter("eef_camera_frame").value),
            pose_topic="/scene_localizer/eef_cam/table_pose_camera",
            min_markers_per_frame=eef_min_markers,
        )

        self._streams: Dict[str, StreamState] = {}
        for config in (top_config, eef_config):
            layout = stream_layouts.get(config.name, {})
            pose_pub = self.create_publisher(PoseStamped, config.pose_topic, 10)
            state = StreamState(config=config, marker_layout=layout, pose_pub=pose_pub)
            self._streams[config.name] = state

            self.create_subscription(
                ArucoDetection,
                config.detections_topic,
                self._make_detections_callback(state),
                10,
            )

        publish_rate_hz = max(0.1, float(self.get_parameter("publish_rate_hz").value))
        self._publish_timer = self.create_timer(1.0 / publish_rate_hz, self._publish_timer_callback)

        self.get_logger().info(f"Loaded marker layout YAML: {marker_layout_yaml}")
        for stream_name, state in self._streams.items():
            self.get_logger().info(
                f"[{stream_name}] markers={len(state.marker_layout)}, "
                f"sub={state.config.detections_topic}, pub={state.config.pose_topic}, "
                f"camera_frame={state.config.camera_frame}, "
                f"min_markers_per_frame={state.config.min_markers_per_frame}"
            )
            if not state.marker_layout:
                self._warn_throttled(
                    f"empty_layout_{stream_name}",
                    f"[{stream_name}] marker layout is empty; estimates will be skipped",
                    5.0,
                )

        self._log_debug(
            "params: "
            f"window_sec={self.get_parameter('window_sec').value}, "
            f"publish_rate_hz={self.get_parameter('publish_rate_hz').value}, "
            f"min_markers_per_frame_global={self.get_parameter('min_markers_per_frame').value}, "
            f"top_min_markers_per_frame={top_min_markers}, "
            f"eef_min_markers_per_frame={eef_min_markers}, "
            f"min_estimates_in_window={self.get_parameter('min_estimates_in_window').value}, "
            f"max_candidate_translation_deviation={self.get_parameter('max_candidate_translation_deviation').value}, "
            f"max_window_translation_deviation={self.get_parameter('max_window_translation_deviation').value}, "
            f"max_translation_jump={self.get_parameter('max_translation_jump').value}, "
            f"max_rotation_jump_deg={self.get_parameter('max_rotation_jump_deg').value}, "
            f"smoothing_alpha={self.get_parameter('smoothing_alpha').value}, "
            f"publish_tf={self.get_parameter('publish_tf').value}"
        )

    def _log_debug(self, message: str) -> None:
        if bool(self.get_parameter("debug_log").value):
            self.get_logger().info(message)

    def _warn_throttled(self, key: str, message: str, throttle_sec: float = 1.0) -> None:
        now_ns = self.get_clock().now().nanoseconds
        last_ns = self._last_warn_time_ns.get(key, 0)
        if now_ns - last_ns >= int(throttle_sec * 1e9):
            self._last_warn_time_ns[key] = now_ns
            self.get_logger().warn(message)

    def _load_layout_by_stream(self, yaml_path: str) -> Dict[str, Dict[int, TransformEstimate]]:
        layout_path = Path(yaml_path)
        if not layout_path.exists():
            self.get_logger().warn(f"Marker layout YAML not found: {yaml_path}")
            return {"top_cam": {}, "eef_cam": {}}

        try:
            with layout_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:
            self.get_logger().warn(f"Failed to read marker layout YAML: {exc}")
            return {"top_cam": {}, "eef_cam": {}}

        streams_data = data.get("streams", {})
        marker_origin = str(data.get("marker_translation_origin", "bottom_left"))
        physical_marker_size = float(data.get("physical_marker_size", 0.035))

        result: Dict[str, Dict[int, TransformEstimate]] = {}
        for stream_name, stream_cfg in streams_data.items():
            markers_data = stream_cfg.get("markers", {})
            parsed: Dict[int, TransformEstimate] = {}
            for marker_id_raw, marker_cfg in markers_data.items():
                try:
                    marker_id = int(marker_id_raw)
                    translation = np.array(marker_cfg.get("translation", [0.0, 0.0, 0.0]), dtype=float)
                    rpy = marker_cfg.get("rpy", [0.0, 0.0, 0.0])
                    if translation.shape != (3,) or len(rpy) != 3:
                        raise ValueError("translation/rpy must be length 3")
                    rotation = _rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
                    if marker_origin in ("bottom_left", "bottom-left", "corner"):
                        corner_to_center_marker = np.array(
                            [-0.5 * physical_marker_size, 0.5 * physical_marker_size, 0.0],
                            dtype=float,
                        )
                        translation = translation + rotation @ corner_to_center_marker
                    elif marker_origin != "center":
                        self._warn_throttled(
                            "unknown_marker_origin",
                            f"Unknown marker_translation_origin '{marker_origin}', assuming center",
                            5.0,
                        )
                    quaternion = _matrix_to_quaternion_xyzw(rotation)
                    parsed[marker_id] = TransformEstimate(translation=translation, quaternion_xyzw=quaternion)
                except Exception as exc:
                    self._warn_throttled(
                        f"parse_marker_{stream_name}_{marker_id_raw}",
                        f"[{stream_name}] failed to parse marker {marker_id_raw}: {exc}",
                        5.0,
                    )
            result[stream_name] = parsed

        if "top_cam" not in result:
            result["top_cam"] = {}
        if "eef_cam" not in result:
            result["eef_cam"] = {}

        total_markers = sum(len(markers) for markers in result.values())
        stream_summaries = []
        for stream_name in sorted(result.keys()):
            marker_items = []
            for marker_id in sorted(result[stream_name].keys()):
                t = result[stream_name][marker_id].translation
                marker_items.append(
                    f"{marker_id}:({float(t[0]):.4f},{float(t[1]):.4f},{float(t[2]):.4f})"
                )
            stream_summaries.append(
                f"{stream_name}:count={len(marker_items)} translations=[{', '.join(marker_items)}]"
            )
        self.get_logger().info(
            "Loaded layout summary: "
            f"streams={len(result)} total_markers={total_markers} "
            f"marker_translation_origin={marker_origin} physical_marker_size={physical_marker_size:.4f}; "
            + "; ".join(stream_summaries)
        )

        return result

    def _make_detections_callback(self, state: StreamState):
        def _callback(msg: ArucoDetection) -> None:
            self._handle_detection_message(state, msg)

        return _callback

    def _handle_detection_message(self, state: StreamState, msg: ArucoDetection) -> None:
        markers = self._extract_markers_from_msg(msg)
        self._log_debug(f"[{state.config.name}] received detections message with markers={len(markers)}")
        if not markers:
            self._warn_throttled(
                f"no_markers_{state.config.name}",
                f"[{state.config.name}] no markers in ArUco detection message",
                1.0,
            )
            return

        if not state.marker_layout:
            self._warn_throttled(
                f"no_layout_{state.config.name}",
                f"[{state.config.name}] marker layout empty; skipping detections",
                2.0,
            )
            return

        min_markers = max(1, int(state.config.min_markers_per_frame))
        max_candidate_dev = max(0.0, float(self.get_parameter("max_candidate_translation_deviation").value))
        max_translation_jump = max(0.0, float(self.get_parameter("max_translation_jump").value))
        max_rotation_jump_deg = max(0.0, float(self.get_parameter("max_rotation_jump_deg").value))

        candidate_translations: List[np.ndarray] = []
        candidate_quaternions: List[np.ndarray] = []
        used_marker_ids: List[int] = []

        for marker in markers:
            marker_id = self._extract_marker_id(marker)
            if marker_id is None or marker_id not in state.marker_layout:
                continue

            pose = self._extract_pose_from_marker(marker)
            if pose is None:
                continue

            t_cm = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
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

            table_marker = state.marker_layout[marker_id]
            t_tm = table_marker.translation
            r_tm = _quaternion_to_matrix_xyzw(table_marker.quaternion_xyzw)

            # T_camera_table = T_camera_marker * inverse(T_table_marker)
            t_mt, r_mt = _invert_transform(t_tm, r_tm)
            t_ct, r_ct = _compose_transform(t_cm, r_cm, t_mt, r_mt)
            q_ct = _matrix_to_quaternion_xyzw(r_ct)

            self._log_debug(
                f"[{state.config.name}] marker_id={marker_id} "
                f"candidate_t=({t_ct[0]:.4f}, {t_ct[1]:.4f}, {t_ct[2]:.4f})"
            )

            candidate_translations.append(t_ct)
            candidate_quaternions.append(q_ct)
            used_marker_ids.append(marker_id)

        if used_marker_ids:
            self._log_debug(
                f"[{state.config.name}] usable markers={used_marker_ids}, "
                f"candidate_count={len(candidate_translations)}"
            )

        if len(candidate_translations) < min_markers:
            self._warn_throttled(
                f"too_few_markers_{state.config.name}",
                f"[{state.config.name}] usable markers below minimum ({len(candidate_translations)} < {min_markers})",
                1.0,
            )
            return

        kept_idx = self._reject_translation_outliers(candidate_translations, max_candidate_dev)
        self._log_debug(
            f"[{state.config.name}] candidate filter kept={len(kept_idx)}/{len(candidate_translations)}"
        )
        if not kept_idx:
            self._warn_throttled(
                f"candidate_reject_all_{state.config.name}",
                f"[{state.config.name}] rejected all frame candidates as outliers",
                1.0,
            )
            return

        frame_t = np.mean(np.stack([candidate_translations[i] for i in kept_idx], axis=0), axis=0)
        frame_q = _average_quaternions_xyzw([candidate_quaternions[i] for i in kept_idx])
        frame_estimate = TransformEstimate(translation=frame_t, quaternion_xyzw=frame_q)

        if state.last_accepted_estimate is not None:
            dt = float(np.linalg.norm(frame_estimate.translation - state.last_accepted_estimate.translation))
            dq = _quaternion_angular_distance_deg(
                frame_estimate.quaternion_xyzw,
                state.last_accepted_estimate.quaternion_xyzw,
            )
            if dt > max_translation_jump or dq > max_rotation_jump_deg:
                self._log_debug(
                    f"[{state.config.name}] jump reject details dt={dt:.4f}m dq={dq:.3f}deg "
                    f"thresholds=({max_translation_jump:.4f}m, {max_rotation_jump_deg:.3f}deg)"
                )
                self._warn_throttled(
                    f"jump_reject_{state.config.name}",
                    f"[{state.config.name}] rejected jump: dt={dt:.3f}m dq={dq:.1f}deg",
                    1.0,
                )
                return

        stamp_sec = self._stamp_to_sec(msg)
        state.last_source_stamp_sec = stamp_sec
        state.last_accepted_estimate = frame_estimate
        state.buffer.append(TimedEstimate(stamp_sec=stamp_sec, estimate=frame_estimate))
        self._prune_buffer(state)
        self._log_debug(
            f"[{state.config.name}] accepted frame estimate "
            f"t=({frame_estimate.translation[0]:.4f}, {frame_estimate.translation[1]:.4f}, {frame_estimate.translation[2]:.4f}) "
            f"q=({frame_estimate.quaternion_xyzw[0]:.4f}, {frame_estimate.quaternion_xyzw[1]:.4f}, "
            f"{frame_estimate.quaternion_xyzw[2]:.4f}, {frame_estimate.quaternion_xyzw[3]:.4f}) "
            f"candidates={len(kept_idx)}/{len(candidate_translations)} buffer_size={len(state.buffer)}"
        )

    def _prune_buffer(self, state: StreamState) -> None:
        window_sec = max(0.1, float(self.get_parameter("window_sec").value))
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        cutoff = now_sec - window_sec
        state.buffer = [item for item in state.buffer if item.stamp_sec >= cutoff]

    @staticmethod
    def _reject_translation_outliers(translations: List[np.ndarray], max_deviation: float) -> List[int]:
        if not translations:
            return []
        arr = np.stack(translations, axis=0)
        median = np.median(arr, axis=0)
        distances = np.linalg.norm(arr - median, axis=1)
        return [int(i) for i, d in enumerate(distances) if float(d) <= max_deviation]

    def _publish_timer_callback(self) -> None:
        for state in self._streams.values():
            self._prune_buffer(state)
            self._publish_stream_pose(state)

    def _publish_stream_pose(self, state: StreamState) -> None:
        min_in_window = max(1, int(self.get_parameter("min_estimates_in_window").value))
        max_window_dev = max(0.0, float(self.get_parameter("max_window_translation_deviation").value))
        smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)
        smoothing_alpha = min(1.0, max(0.0, smoothing_alpha))
        publish_tf = bool(self.get_parameter("publish_tf").value)

        if len(state.buffer) < min_in_window:
            self._warn_throttled(
                f"too_few_window_{state.config.name}",
                f"[{state.config.name}] too few estimates in window ({len(state.buffer)} < {min_in_window})",
                1.0,
            )
            return

        translations = [item.estimate.translation for item in state.buffer]
        quaternions = [item.estimate.quaternion_xyzw for item in state.buffer]
        keep_idx = self._reject_translation_outliers(translations, max_window_dev)
        self._log_debug(
            f"[{state.config.name}] window filter kept={len(keep_idx)}/{len(state.buffer)} "
            f"(min={min_in_window}, max_dev={max_window_dev:.4f})"
        )
        if len(keep_idx) < min_in_window:
            self._warn_throttled(
                f"window_outlier_{state.config.name}",
                f"[{state.config.name}] window outlier rejection left too few estimates ({len(keep_idx)} < {min_in_window})",
                1.0,
            )
            return

        avg_t = np.mean(np.stack([translations[i] for i in keep_idx], axis=0), axis=0)
        avg_q = _average_quaternions_xyzw([quaternions[i] for i in keep_idx])

        current = TransformEstimate(translation=avg_t, quaternion_xyzw=avg_q)
        if state.last_published_estimate is not None:
            smoothed_t = (
                (1.0 - smoothing_alpha) * state.last_published_estimate.translation
                + smoothing_alpha * current.translation
            )
            smoothed_q = _blend_quaternions_xyzw(
                state.last_published_estimate.quaternion_xyzw,
                current.quaternion_xyzw,
                smoothing_alpha,
            )
            current = TransformEstimate(translation=smoothed_t, quaternion_xyzw=smoothed_q)

        state.last_published_estimate = current

        pose_msg = PoseStamped()
        if state.last_source_stamp_sec is not None and state.last_source_stamp_sec > 0.0:
            stamp_nsec = int(state.last_source_stamp_sec * 1e9)
            pose_msg.header.stamp.sec = int(stamp_nsec // 1_000_000_000)
            pose_msg.header.stamp.nanosec = int(stamp_nsec % 1_000_000_000)
        else:
            pose_msg.header.stamp = self.get_clock().now().to_msg()

        pose_msg.header.frame_id = state.config.camera_frame
        pose_msg.pose.position.x = float(current.translation[0])
        pose_msg.pose.position.y = float(current.translation[1])
        pose_msg.pose.position.z = float(current.translation[2])
        pose_msg.pose.orientation.x = float(current.quaternion_xyzw[0])
        pose_msg.pose.orientation.y = float(current.quaternion_xyzw[1])
        pose_msg.pose.orientation.z = float(current.quaternion_xyzw[2])
        pose_msg.pose.orientation.w = float(current.quaternion_xyzw[3])
        state.pose_pub.publish(pose_msg)
        self._log_debug(
            f"[{state.config.name}] published pose frame={pose_msg.header.frame_id} "
            f"stamp={pose_msg.header.stamp.sec}.{pose_msg.header.stamp.nanosec:09d} "
            f"t=({pose_msg.pose.position.x:.4f}, {pose_msg.pose.position.y:.4f}, {pose_msg.pose.position.z:.4f})"
        )

        if publish_tf:
            tf_msg = TransformStamped()
            tf_msg.header = pose_msg.header
            tf_msg.child_frame_id = f"{state.config.name}_{self._table_frame}"
            tf_msg.transform.translation.x = pose_msg.pose.position.x
            tf_msg.transform.translation.y = pose_msg.pose.position.y
            tf_msg.transform.translation.z = pose_msg.pose.position.z
            tf_msg.transform.rotation = pose_msg.pose.orientation
            self._tf_broadcaster.sendTransform(tf_msg)
            self._log_debug(
                f"[{state.config.name}] published tf {tf_msg.header.frame_id} -> {tf_msg.child_frame_id}"
            )

    @staticmethod
    def _extract_markers_from_msg(msg: ArucoDetection) -> List[Any]:
        if hasattr(msg, "markers") and getattr(msg, "markers") is not None:
            return list(getattr(msg, "markers"))
        if hasattr(msg, "detections") and getattr(msg, "detections") is not None:
            return list(getattr(msg, "detections"))
        if (hasattr(msg, "marker_id") or hasattr(msg, "id")) and hasattr(msg, "pose"):
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

    def _stamp_to_sec(self, msg: ArucoDetection) -> float:
        header = getattr(msg, "header", None)
        if header is not None and hasattr(header, "stamp"):
            stamp = header.stamp
            sec = float(getattr(stamp, "sec", 0.0))
            nanosec = float(getattr(stamp, "nanosec", 0.0))
            stamp_sec = sec + nanosec * 1e-9
            if stamp_sec > 0.0:
                return stamp_sec
        return self.get_clock().now().nanoseconds * 1e-9


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = SceneLocalizerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
