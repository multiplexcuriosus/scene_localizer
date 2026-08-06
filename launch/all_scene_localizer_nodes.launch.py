from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config_dir = f"{get_package_share_directory('scene_localizer')}/config"

    inhibit_scene_localizer = LaunchConfiguration("inhibit_scene_localizer")
    inhibit_scene_localizer_debug = LaunchConfiguration("inhibit_scene_localizer_debug")
    inhibit_ball_3d_pose_estimator = LaunchConfiguration("inhibit_ball_3d_pose_estimator")
    inhibit_ball_trajectory_estimator = LaunchConfiguration("inhibit_ball_trajectory_estimator")
    enable_latency_trace = LaunchConfiguration("enable_latency_trace")
    latency_run_id = LaunchConfiguration("latency_run_id")
    latency_modality = LaunchConfiguration("latency_modality")

    scene_localizer_node = Node(
        package="scene_localizer",
        executable="scene_localizer",
        name="scene_localizer",
        output="screen",
        parameters=[f"{config_dir}/scene_localizer.yaml"],
        condition=UnlessCondition(inhibit_scene_localizer),
    )

    scene_localizer_debug_node = Node(
        package="scene_localizer",
        executable="scene_localizer_debug",
        name="scene_localizer_debug",
        output="screen",
        parameters=[f"{config_dir}/scene_localizer_debug.yaml"],
        condition=UnlessCondition(inhibit_scene_localizer_debug),
    )

    ball_3d_pose_estimator_node = Node(
        package="scene_localizer",
        executable="ball_3d_pose_estimator",
        name="ball_3d_pose_estimator",
        output="screen",
        parameters=[{
            "enable_latency_trace": enable_latency_trace,
            "latency_trace_topic": LaunchConfiguration("localization_latency_trace_topic"),
            "latency_run_id": latency_run_id,
            "latency_modality": latency_modality,
        }],
        condition=UnlessCondition(inhibit_ball_3d_pose_estimator),
    )

    ball_trajectory_estimator_node = Node(
        package="scene_localizer",
        executable="ball_trajectory_estimator",
        name="ball_trajectory_estimator",
        output="screen",
        parameters=[
            f"{config_dir}/ball_trajectory_estimator.yaml",
            {
                "enable_latency_trace": enable_latency_trace,
                "latency_trace_topic": LaunchConfiguration("trajectory_latency_trace_topic"),
                "latency_run_id": latency_run_id,
                "latency_modality": latency_modality,
            },
        ],
        condition=UnlessCondition(inhibit_ball_trajectory_estimator),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "inhibit_scene_localizer",
            default_value="false",
            description="If true, do not launch the scene_localizer node.",
        ),
        DeclareLaunchArgument(
            "inhibit_scene_localizer_debug",
            default_value="false",
            description="If true, do not launch the scene_localizer_debug node.",
        ),
        DeclareLaunchArgument(
            "inhibit_ball_3d_pose_estimator",
            default_value="false",
            description="If true, do not launch the ball_3d_pose_estimator node.",
        ),
        DeclareLaunchArgument(
            "inhibit_ball_trajectory_estimator",
            default_value="false",
            description="If true, do not launch the ball_trajectory_estimator node.",
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
        DeclareLaunchArgument("latency_modality", default_value="vision"),
        scene_localizer_node,
        scene_localizer_debug_node,
        ball_3d_pose_estimator_node,
        ball_trajectory_estimator_node,
    ])
