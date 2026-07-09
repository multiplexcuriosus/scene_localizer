from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config_path = f"{get_package_share_directory('scene_localizer')}/config/ball_trajectory_estimator.yaml"

    node = Node(
        package="scene_localizer",
        executable="ball_trajectory_estimator",
        name="ball_trajectory_estimator",
        output="screen",
        parameters=[config_path],
    )

    return LaunchDescription([node])