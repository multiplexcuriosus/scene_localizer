cmake_minimum_required(VERSION 3.8)
project(scene_localizer)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)
find_package(aruco_opencv_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(fr3_husky_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BallTrajectory.msg"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

install(
  PROGRAMS
    scene_localizer/scene_localizer_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME scene_localizer
)

install(
  PROGRAMS
    scene_localizer/table_pose_origin_projection.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME table_pose_origin_projection
)

install(
  PROGRAMS
    scene_localizer/scene_localizer_debug_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME scene_localizer_debug
)

install(
  PROGRAMS
    scene_localizer/ball_3d_pose_estimator_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ball_3d_pose_estimator
)

install(
  PROGRAMS
    scene_localizer/ball_trajectory_estimator.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ball_trajectory_estimator
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()