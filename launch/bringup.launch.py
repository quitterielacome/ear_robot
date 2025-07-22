from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='trajectory_planner', executable='tracking_node'),
        Node(package='trajectory_planner', executable='trajectory_server'),
    ])
