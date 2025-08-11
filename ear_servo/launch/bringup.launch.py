from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory('ear_servo')
    cfg = os.path.join(pkg, 'config', 'default.yaml')

    cam_arg = DeclareLaunchArgument('camera_topic', default_value='/camera/image_raw')

    pose_node = Node(
        package='ear_servo',
        executable='ear_servo_pose',       
        name='ear_servo_pose_node',
        output='screen',
        parameters=[cfg, {'camera_topic': LaunchConfiguration('camera_topic')}],
    )

    logger = Node(
        package='ear_servo',
        executable='metrics_logger',
        name='metrics_logger',
        output='screen',
    )

    return LaunchDescription([cam_arg, pose_node, logger])
