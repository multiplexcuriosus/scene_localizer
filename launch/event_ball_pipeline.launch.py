from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config_dir = Path(get_package_share_directory("scene_localizer")) / "config"
    event_config = str(config_dir / "event_ball_pipeline.yaml")
    trajectory_config = str(config_dir / "ball_trajectory_estimator.yaml")

    calibration_file = LaunchConfiguration("calibration_file")
    event_camera_frame = LaunchConfiguration("event_camera_frame")
    publish_rate_hz = LaunchConfiguration("calibration_publish_rate_hz")
    enable_latency_trace = LaunchConfiguration("enable_latency_trace")
    latency_run_id = LaunchConfiguration("latency_run_id")

    calibration_adapter = Node(
        package="scene_localizer",
        executable="event_camera_calibration_adapter",
        name="event_camera_calibration_adapter",
        output="screen",
        parameters=[
            event_config,
            {
                "calibration_file": calibration_file,
                "event_camera_frame": event_camera_frame,
                "publish_rate_hz": ParameterValue(
                    publish_rate_hz,
                    value_type=float,
                ),
            },
        ],
    )

    ball_3d_pose_estimator = Node(
        package="scene_localizer",
        executable="ball_3d_pose_estimator",
        name="ball_3d_pose_estimator",
        output="screen",
        parameters=[
            event_config,
            {
                "enable_latency_trace": enable_latency_trace,
                "latency_trace_topic": LaunchConfiguration(
                    "localization_latency_trace_topic"
                ),
                "latency_run_id": latency_run_id,
                "latency_modality": "event",
            },
        ],
    )

    ball_trajectory_estimator = Node(
        package="scene_localizer",
        executable="ball_trajectory_estimator",
        name="ball_trajectory_estimator",
        output="screen",
        parameters=[
            trajectory_config,
            event_config,
            {
                "enable_latency_trace": enable_latency_trace,
                "latency_trace_topic": LaunchConfiguration(
                    "trajectory_latency_trace_topic"
                ),
                "latency_run_id": latency_run_id,
                "latency_modality": "event",
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "calibration_file",
                default_value=str(
                    Path.home()
                    / ".ros"
                    / "event_camera_calibration"
                    / "genx320_calibration.yaml"
                ),
                description="Solved event_camera_calibration YAML.",
            ),
            DeclareLaunchArgument(
                "event_camera_frame",
                default_value="event_camera",
                description="Runtime camera frame used in CameraInfo and T_camera_table.",
            ),
            DeclareLaunchArgument(
                "calibration_publish_rate_hz",
                default_value="10.0",
                description="Repeated CameraInfo and table-pose publication rate.",
            ),
            DeclareLaunchArgument("enable_latency_trace", default_value="false"),
            DeclareLaunchArgument(
                "localization_latency_trace_topic",
                default_value="/intercept_trace/localization_2d_to_3d",
            ),
            DeclareLaunchArgument(
                "trajectory_latency_trace_topic",
                default_value="/intercept_trace/trajectory_estimation",
            ),
            DeclareLaunchArgument("latency_run_id", default_value=""),
            calibration_adapter,
            ball_3d_pose_estimator,
            ball_trajectory_estimator,
        ]
    )
