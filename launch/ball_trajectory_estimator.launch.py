from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config_path = f"{get_package_share_directory('scene_localizer')}/config/ball_trajectory_estimator.yaml"

    node = Node(
        package="scene_localizer",
        executable="ball_trajectory_estimator",
        name="ball_trajectory_estimator",
        output="screen",
        parameters=[
            config_path,
            {
                "enable_latency_trace": LaunchConfiguration("enable_latency_trace"),
                "latency_trace_topic": LaunchConfiguration("latency_trace_topic"),
                "latency_run_id": LaunchConfiguration("latency_run_id"),
                "latency_modality": LaunchConfiguration("latency_modality"),
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("enable_latency_trace", default_value="false"),
        DeclareLaunchArgument(
            "latency_trace_topic",
            default_value="/intercept_trace/trajectory_estimation",
        ),
        DeclareLaunchArgument("latency_run_id", default_value=""),
        DeclareLaunchArgument("latency_modality", default_value="vision"),
        node,
    ])
