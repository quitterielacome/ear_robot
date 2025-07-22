#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(PoseArray, '/planned_trajectory', 10)
        self.template = None
        self.needle_c = None
        self.gt_w = None
        self.gt_h = None
        self.last_bbox = None
        self.initialized = False

    def compute_center(self, box):
        x, y, w, h = box
        return (x + w / 2, y + h / 2)

    def search_in_region(self, gray, template, bbox, pad):
        lx, ly, lw, lh = bbox
        sx = max(lx - pad, 0)
        sy = max(ly - pad, 0)
        ex = min(lx + lw + pad, gray.shape[1])
        ey = min(ly + lh + pad, gray.shape[0])
        region = gray[sy:ey, sx:ex]
        res = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(res)
        return val, (loc[0] + sx, loc[1] + sy)

    def interpolate(self, p1, p2, n):
        return [( (1 - (i+1)/(n+1)) * p1[0] + (i+1)/(n+1) * p2[0],
                  (1 - (i+1)/(n+1)) * p1[1] + (i+1)/(n+1) * p2[1] ) for i in range(n)]

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self.initialized:
            roi = cv2.selectROI("Select TARGET", frame, False)
            self.template = gray[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            self.gt_w = int(roi[2])
            self.gt_h = int(roi[3])
            self.last_bbox = (int(roi[0]), int(roi[1]), self.gt_w, self.gt_h)
            needle_roi = cv2.selectROI("Select NEEDLE TIP", frame, False)
            self.needle_c = (int(needle_roi[0] + needle_roi[2]/2), int(needle_roi[1] + needle_roi[3]/2))
            cv2.destroyAllWindows()
            self.initialized = True
            return

        val, (tx, ty) = self.search_in_region(gray, self.template, self.last_bbox, 30)
        if val < 0.8:
            val, (tx, ty) = self.search_in_region(gray, self.template, self.last_bbox, 75)

        if val >= 0.8:
            self.last_bbox = (tx, ty, self.gt_w, self.gt_h)
            tgt_c = self.compute_center(self.last_bbox)

            traj = PoseArray()
            traj.header.stamp = self.get_clock().now().to_msg()
            traj.header.frame_id = 'base_link'

            pts = self.interpolate(self.needle_c, tgt_c, 20)
            for pt in pts:
                pose = Pose()
                pose.position.x = float(pt[0]) / 1000.0  # convert px to meters if needed
                pose.position.y = float(pt[1]) / 1000.0
                pose.position.z = 0.3  # fixed Z for test
                traj.poses.append(pose)

            self.pub.publish(traj)
            self.get_logger().info(f"Published trajectory with {len(traj.poses)} points.")

def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
