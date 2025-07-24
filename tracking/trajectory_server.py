#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from xarm_msgs.srv import SetCartesianPose
import time

class TrajectoryServer(Node):
    def __init__(self):
        super().__init__('trajectory_server')
        self.sub = self.create_subscription(
            PoseArray, '/planned_trajectory', self.on_trajectory, 10)
        self.cli = self.create_client(
            SetCartesianPose, '/xarm/set_cartesian_pose')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for xArm service…')

    def on_trajectory(self, msg: PoseArray):
        self.get_logger().info(f"Got {len(msg.poses)} waypoints")
        for pose in msg.poses:
            req = SetCartesianPose.Request()
            req.pose   = pose
            req.mvtime = 2.0
            req.wait   = True
            future = self.cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is None:
                self.get_logger().error("xArm call failed")
            else:
                self.get_logger().info("Waypoint reached")
            time.sleep(0.1)  # small pause

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
