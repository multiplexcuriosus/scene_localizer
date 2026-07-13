from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    realsense_launch_path = f"{get_package_share_directory('realsense2_camera')}/launch/rs_launch.py"

    top_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_path),
        launch_arguments={
            'serial_no': '_243722074377',
            'camera_namespace': 'top_cam',
        }.items(),
    )

    top_aruco = Node(
        package='aruco_opencv',
        executable='aruco_tracker_autostart',
        name='aruco_top_cam',
        namespace='/aruco_top_cam',
        output='screen',
        parameters=[{
            'cam_base_topic': '/top_cam/camera/color/image_raw',
            'marker_size': 0.03,
        }],
    )

    return LaunchDescription([
        top_camera,
        top_aruco,
    ])