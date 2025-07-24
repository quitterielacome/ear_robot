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
        self.sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(
            PoseArray, '/planned_trajectory', 10)

        # ROI / template holders
        self.tgt_center0 = None
        self.lm1_box = None
        self.lm2_box = None
        self.lm1_center0 = None
        self.lm2_center0 = None
        self.tgt_size = None
        self.needle_center = None
        self.template1 = None
        self.template2 = None

        self.initialized = False

        # parameters
        self.match_thresh = 0.8
        self.small_pad   = 30
        self.large_pad   = 75
        self.num_pts     = 20

    def compute_center(self, box):
        x,y,w,h = box
        return np.array([x + w/2, y + h/2], dtype=float)

    def search_in_region(self, gray, template, bbox, pad):
        x,y,w,h = bbox
        sx,sy = max(x-pad,0), max(y-pad,0)
        ex,ey = min(x+w+pad, gray.shape[1]), min(y+h+pad, gray.shape[0])
        region = gray[sy:ey, sx:ex]
        res = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(res)
        return val, (loc[0]+sx, loc[1]+sy)

    def interpolate(self, p1, p2, n):
        pts = []
        for i in range(1, n+1):
            a = i/(n+1)
            x = (1-a)*p1[0] + a*p2[0]
            y = (1-a)*p1[1] + a*p2[1]
            pts.append((x,y))
        return pts

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self.initialized:
            # --- select all 4 ROIs once ---
            # 1) target ROI (only to get its initial center & size)
            tgt = cv2.selectROI("Select TARGET", frame, False)
            cv2.destroyWindow("Select TARGET")
            self.tgt_center0 = self.compute_center(tgt)
            self.tgt_size = (int(tgt[2]), int(tgt[3]))

            # 2) landmark #1
            self.lm1_box = cv2.selectROI("Select LANDMARK 1", frame, False)
            cv2.destroyWindow("Select LANDMARK 1")
            self.lm1_center0 = self.compute_center(self.lm1_box)
            x1,y1,w1,h1 = map(int, self.lm1_box)
            self.template1 = gray[y1:y1+h1, x1:x1+w1]

            # 3) landmark #2
            self.lm2_box = cv2.selectROI("Select LANDMARK 2", frame, False)
            cv2.destroyWindow("Select LANDMARK 2")
            self.lm2_center0 = self.compute_center(self.lm2_box)
            x2,y2,w2,h2 = map(int, self.lm2_box)
            self.template2 = gray[y2:y2+h2, x2:x2+w2]

            # 4) needle tip
            ndl = cv2.selectROI("Select NEEDLE TIP", frame, False)
            cv2.destroyAllWindows()
            self.needle_center = self.compute_center(ndl)

            self.initialized = True
            return

        # --- per-frame: track landmarks only ---
        v1, (nx1,ny1) = self.search_in_region(
            gray, self.template1, self.lm1_box, self.small_pad)
        v2, (nx2,ny2) = self.search_in_region(
            gray, self.template2, self.lm2_box, self.small_pad)

        if v1 < self.match_thresh or v2 < self.match_thresh:
            # try larger pad
            if v1 < self.match_thresh:
                v1, (nx1,ny1) = self.search_in_region(
                    gray, self.template1, self.lm1_box, self.large_pad)
            if v2 < self.match_thresh:
                v2, (nx2,ny2) = self.search_in_region(
                    gray, self.template2, self.lm2_box, self.large_pad)

        if v1>=self.match_thresh and v2>=self.match_thresh:
            # update landmarks
            w1,h1 = self.lm1_box[2], self.lm1_box[3]
            w2,h2 = self.lm2_box[2], self.lm2_box[3]
            self.lm1_box = (nx1,ny1,w1,h1)
            self.lm2_box = (nx2,ny2,w2,h2)
            self.template1 = gray[ny1:ny1+h1, nx1:nx1+w1]
            self.template2 = gray[ny2:ny2+h2, nx2:nx2+w2]

            # compute affine from initial→current landmarks
            curr1 = self.compute_center(self.lm1_box)
            curr2 = self.compute_center(self.lm2_box)
            src = np.vstack([self.lm1_center0, self.lm2_center0]).astype(np.float32)
            dst = np.vstack([curr1, curr2]).astype(np.float32)
            M, _ = cv2.estimateAffinePartial2D(src, dst)

            # warp initial target center
            x0,y0 = self.tgt_center0
            xt,yt,_ = M.dot(np.array([x0,y0,1]))
            tx = int(xt - self.tgt_size[0]/2)
            ty = int(yt - self.tgt_size[1]/2)

            # draw landmarks
            cv2.rectangle(frame, (nx1,ny1), (nx1+w1,ny1+h1), (255,0,255),2)
            cv2.rectangle(frame, (nx2,ny2), (nx2+w2,ny2+h2), (255,0,255),2)

            # draw inferred target
            cv2.rectangle(frame, (tx,ty),
                          (tx+self.tgt_size[0], ty+self.tgt_size[1]),
                          (0,255,0), 2)

            # draw needle tip
            nc = tuple(map(int, self.needle_center))
            cv2.circle(frame, nc, 5, (0,0,255), -1)

            # build & publish trajectory
            pa = PoseArray()
            pa.header.stamp = self.get_clock().now().to_msg()
            pa.header.frame_id = 'base_link'

            pts = self.interpolate(self.needle_center, (xt,yt), self.num_pts)
            for xpt,ypt in pts:
                cv2.circle(frame, (int(xpt),int(ypt)), 3, (255,255,0), -1)
                pose = Pose()
                pose.position.x = float(xpt)/1000.0
                pose.position.y = float(ypt)/1000.0
                pose.position.z = 0.3
                pa.poses.append(pose)

            self.pub.publish(pa)
            self.get_logger().info(f"Published {len(pa.poses)} waypoints")
        else:
            cv2.putText(frame, "LANDMARKS LOST", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        cv2.imshow("Landmark‐Only Tracking", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
