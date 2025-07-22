#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from xarm_msgs.srv import SetCartesianPose
import time

class TrajectoryServer(Node):
    def __init__(self):
        super().__init__('trajectory_server')
        self.traj_sub = self.create_subscription(PoseArray, '/planned_trajectory', self.traj_callback, 10)
        self.cli = self.create_client(SetCartesianPose, '/xarm/set_cartesian_pose')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /xarm/set_cartesian_pose service...')

    def traj_callback(self, msg):
        self.get_logger().info(f"Received trajectory with {len(msg.poses)} waypoints.")
        for pose in msg.poses:
            req = SetCartesianPose.Request()
            req.pose = pose
            req.mvtime = 2.0  # Move duration in seconds
            req.wait = True
            future = self.cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                self.get_logger().info('Waypoint executed.')
            else:
                self.get_logger().error('Failed to execute waypoint.')
            time.sleep(0.1)  # Optional delay between waypoints

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
