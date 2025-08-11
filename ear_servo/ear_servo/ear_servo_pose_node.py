#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import SetBool
from cv_bridge import CvBridge
import cv2, numpy as np, time, math, csv, os, datetime  # NEW: csv, os, datetime

# ---------- helpers ----------
def compute_center(b):
    x,y,w,h=b; return np.array([x+w/2.0, y+h/2.0], dtype=np.float32)

def glare_mask_hsv(bgr, v_hi=240, s_lo=40, v_hi2=220):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    m1 = (V >= v_hi)
    m2 = (S <= s_lo) & (V >= v_hi2)
    mask = (m1 | m2).astype(np.uint8)*255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 1)
    mask = cv2.dilate(mask, k, 1)
    return mask

def inpaint_if(mask, gray):
    if mask is None or mask.max()==0: return gray
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)

def scale_affine_2x3(M_work, s):
    A = np.vstack([M_work, [0,0,1]]).astype(np.float32)
    S  = np.array([[s,0,0],[0,s,0],[0,0,1]], np.float32)
    Si = np.array([[1/s,0,0],[0,1/s,0],[0,0,1]], np.float32)
    return (Si @ A @ S)[:2,:]
# -------------------------------------------------

class EarServoPoseNode(Node):
    def __init__(self):
        super().__init__('ear_servo_pose_node')

        # --- Parameters ---
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('target_pose_topic', '/ee_target_pose')
        self.declare_parameter('ee_pose_topic', '')
        self.declare_parameter('frame_id', 'tool0')
        self.declare_parameter('work_w', 720)
        self.declare_parameter('n_pts', 150)
        self.declare_parameter('lk_win', 21)
        self.declare_parameter('lk_levels', 3)
        self.declare_parameter('ransac_thr', 3.0)
        self.declare_parameter('ransac_conf', 0.99)
        self.declare_parameter('min_inliers', 6)
        self.declare_parameter('meters_per_px_x', 0.0005)
        self.declare_parameter('meters_per_px_y', 0.0005)
        self.declare_parameter('step_fraction', 0.3)
        self.declare_parameter('max_step_m', 0.01)
        self.declare_parameter('flip_x', False)
        self.declare_parameter('flip_y', True)
        self.declare_parameter('deadband_px', 3.0)
        self.declare_parameter('publish_rate_hz', 30.0)
        self.declare_parameter('show_debug', True)
        self.declare_parameter('inpaint', False)

        # NEW: logging params
        self.declare_parameter('log_enable', True)
        self.declare_parameter('log_csv', '')  # if empty, auto-create in ~/ros2_logs/

        # --- ROS I/O ---
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.bridge = CvBridge()
        cam_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.sub_img = self.create_subscription(Image, cam_topic, self.image_cb, qos)

        tgt_topic = self.get_parameter('target_pose_topic').get_parameter_value().string_value
        self.pub_pose = self.create_publisher(PoseStamped, tgt_topic, 10)

        ee_pose_topic = self.get_parameter('ee_pose_topic').get_parameter_value().string_value
        self._cur_pose = None  # store EE feedback if available
        if ee_pose_topic:
            self.sub_pose = self.create_subscription(PoseStamped, ee_pose_topic, self.ee_pose_cb, 10)

        self.srv_enable = self.create_service(SetBool, 'enable_servo', self.handle_enable)
        self.srv_reset  = self.create_service(SetBool, 'reset_roi',   self.handle_reset)
        self.timer = self.create_timer(1.0/max(1.0, self.get_parameter('publish_rate_hz').value), self.publish_pose)

        # --- State ---
        self.enabled = False
        self.have_roi = False
        self.roi = None
        self.t0 = None
        self.tw = self.th = None
        self.scale = 1.0
        self.frame_W = self.frame_H = None

        self.last_M_full = np.array([[1,0,0],[0,1,0]], np.float32)
        self.prev_gray = None
        self.prev_pts  = None
        self.anchor0   = None

        # cmd/act pose cache
        self.last_cmd_pose = None
        self.last_cmd_time = None
        self.last_act_pose = None
        self.last_act_time = None

        # diagnostics (updated by callbacks)
        self.fps_ema = None
        self.last_inliers = 0            # NEW
        self.last_lost = True            # NEW
        self.last_ex_ey = (0.0, 0.0)     # NEW
        self.last_dx_dy = (0.0, 0.0)     # NEW
        self.last_v_cmd = (0.0, 0.0)     # NEW
        self.last_v_act = (0.0, 0.0)     # NEW

        # --- CSV logger (tiny) ---  # NEW
        self.log_writer = None
        if self.get_parameter('log_enable').value:
            path = self.get_parameter('log_csv').get_parameter_value().string_value
            if not path:
                os.makedirs(os.path.expanduser('~/ros2_logs'), exist_ok=True)
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                path = os.path.expanduser(f'~/ros2_logs/ear_servo_{ts}.csv')
            self.log_fh = open(path, 'w', newline='')
            self.log_writer = csv.writer(self.log_fh)
            self.log_writer.writerow(
                ['stamp', 'ex', 'ey', 'dx', 'dy',
                 'vx_cmd', 'vy_cmd', 'vx_act', 'vy_act',
                 'inliers', 'lost', 'fps'])
            self.get_logger().info(f'Logging to {path}')

        self.get_logger().info("Ear servo (pose mode) ready. Select ROI, enable servo, feed /ee_target_pose to xArm.")

    # ---- services ----
    def handle_enable(self, req, resp):
        self.enabled = bool(req.data)
        resp.success = True
        resp.message = f"Servo {'enabled' if self.enabled else 'disabled'}"
        return resp

    def handle_reset(self, req, resp):
        self.have_roi = False
        self.prev_gray = None
        self.prev_pts = None
        self.anchor0 = None
        self.last_M_full[:] = np.array([[1,0,0],[0,1,0]], np.float32)
        resp.success = True
        resp.message = "ROI reset."
        return resp

    # ---- feedback (optional) ----
    def ee_pose_cb(self, msg: PoseStamped):
        # store act pose + time
        self.last_v_act = self.last_v_act  # keep type happy
        self._cur_pose = msg
        self.last_act_pose = msg
        self.last_act_time = self.get_clock().now()

    # ---- main image callback ----
    def image_cb(self, msg: Image):
        t0 = time.time()
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        H, W = bgr.shape[:2]
        self.frame_W, self.frame_H = W, H

        work_w = int(self.get_parameter('work_w').value)
        self.scale = min(1.0, float(work_w)/W)
        wW, wH = int(W*self.scale), int(H*self.scale)
        work = cv2.resize(bgr, (wW,wH), cv2.INTER_AREA) if self.scale < 1.0 else bgr

        # ROI selection once
        if not self.have_roi:
            r = cv2.selectROI("Select Target ROI", bgr, False)
            cv2.destroyAllWindows()
            if r[2] <= 0 or r[3] <= 0:
                self.get_logger().warn("Empty ROI; waiting for selection.")
                return
            self.roi = r
            self.t0  = compute_center(r)
            self.tw, self.th = r[2], r[3]
            self.have_roi = True
            self.prev_gray = None
            self.prev_pts  = None
            self.anchor0   = None
            self.last_M_full[:] = np.array([[1,0,0],[0,1,0]], np.float32)

        # preprocess
        gmask = glare_mask_hsv(work)
        gray  = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray  = cv2.createCLAHE(2.0, (8,8)).apply(gray)
        gray_i = inpaint_if(gmask, gray) if bool(self.get_parameter('inpaint').value) else gray

        # (re)init features
        if self.prev_pts is None or len(self.prev_pts) < 10 or self.prev_gray is None:
            detect_mask = cv2.bitwise_not(gmask)
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray_i, maxCorners=int(self.get_parameter('n_pts').value),
                qualityLevel=0.01, minDistance=7, blockSize=7,
                mask=detect_mask, useHarrisDetector=False
            )
            self.prev_gray = gray_i.copy()
            if self.prev_pts is not None and len(self.prev_pts) >= 2:
                self.anchor0 = self.prev_pts[:2].reshape(-1,2)

        # track + affine
        tx = ty = None; inl = 0; lost = True
        if self.prev_pts is not None and len(self.prev_pts) >= 10:
            lk_params = dict(winSize=(int(self.get_parameter('lk_win').value),
                                      int(self.get_parameter('lk_win').value)),
                             maxLevel=int(self.get_parameter('lk_levels').value),
                             criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_i, self.prev_pts, None, **lk_params)
            if p1 is not None:
                good_new = p1[st==1].reshape(-1,2)
                good_old = self.prev_pts[st==1].reshape(-1,2)
                M_work, inliers = cv2.estimateAffinePartial2D(
                    good_old, good_new, method=cv2.RANSAC,
                    ransacReprojThreshold=float(self.get_parameter('ransac_thr').value),
                    maxIters=2000, confidence=float(self.get_parameter('ransac_conf').value)
                )
                inl = int(inliers.sum()) if inliers is not None else 0
                if (M_work is not None) and (inl >= int(self.get_parameter('min_inliers').value)):
                    M_full = scale_affine_2x3(M_work, self.scale)
                    H_step  = np.vstack([M_full, [0,0,1]]).astype(np.float32)
                    H_cum   = np.vstack([self.last_M_full, [0,0,1]]).astype(np.float32)
                    H_cum   = H_step @ H_cum
                    self.last_M_full = H_cum[:2,:]
                    p = self.last_M_full.dot(np.array([self.t0[0], self.t0[1], 1.0], dtype=np.float32))
                    tx, ty = int(p[0] - self.tw/2), int(p[1] - self.th/2)
                    lost = False
                    self.prev_gray = gray_i.copy()
                    self.prev_pts  = good_new.reshape(-1,1,2)

        # image-space error (needle=center)
        cx = W/2.0; cy = H/2.0
        if not lost:
            tcx = tx + self.tw/2.0; tcy = ty + self.th/2.0
            ex = float(tcx - cx);  ey = float(tcy - cy)
        else:
            ex = ey = 0.0

        # HUD
        debug = bgr.copy()
        if not lost:
            cv2.rectangle(debug, (tx,ty), (tx+self.tw, ty+self.th), (0,255,0), 2)
        else:
            cv2.putText(debug, "LOST", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        fps = 1.0 / max(1e-6, time.time()-t0)
        self.fps_ema = fps if self.fps_ema is None else (0.9*self.fps_ema + 0.1*fps)
        cv2.putText(debug, f"FPS:{self.fps_ema:4.1f} inl:{inl}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)
        cv2.circle(debug, (int(cx),int(cy)), 4, (255,255,255), -1)
        cv2.imshow("ear_servo_pose_debug", debug); cv2.waitKey(1)

        # stash diagnostics for publisher/log
        self.last_inliers = inl               # NEW
        self.last_lost = lost                 # NEW
        self.last_ex_ey = (ex, ey)            # NEW

    # periodic pose publisher (+ logging)
    def publish_pose(self):
        if not self.enabled or not self.have_roi:
            return

        ex, ey = self.last_ex_ey
        lost = self.last_lost
        dead = float(self.get_parameter('deadband_px').value)
        if abs(ex) < dead and abs(ey) < dead:
            # hold pose; still log with zero dx/dy
            dx = dy = 0.0
            vx_cmd = vy_cmd = 0.0
            vx_act = vy_act = 0.0
            self._log_row(ex, ey, dx, dy, vx_cmd, vy_cmd, vx_act, vy_act, self.last_inliers, lost, self.fps_ema)  # NEW
            if self.cur_pose:
                self.pub_pose.publish(self.cur_pose)
            return

        # px -> meters with optional axis flips
        mppx = float(self.get_parameter('meters_per_px_x').value)
        mppy = float(self.get_parameter('meters_per_px_y').value)
        flip_x = bool(self.get_parameter('flip_x').value)
        flip_y = bool(self.get_parameter('flip_y').value)
        dx = (-ex if flip_x else ex) * mppx
        dy = (-ey if flip_y else ey) * mppy

        # step fraction + clamp
        frac = float(self.get_parameter('step_fraction').value)
        dx *= frac; dy *= frac
        max_step = float(self.get_parameter('max_step_m').value)
        mag = math.hypot(dx, dy)
        if mag > max_step and mag > 1e-9:
            s = max_step / mag; dx *= s; dy *= s
        self.last_dx_dy = (dx, dy)  # NEW

        # base pose
        base_pose = self.cur_pose if self.cur_pose is not None else self.last_cmd_pose
        if base_pose is None:
            base_pose = PoseStamped()
            base_pose.header.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
            base_pose.pose.orientation.w = 1.0

        tgt = PoseStamped()
        now = self.get_clock().now()
        tgt.header.stamp = now.to_msg()
        tgt.header.frame_id = base_pose.header.frame_id
        tgt.pose = base_pose.pose
        tgt.pose.position.x += dx
        tgt.pose.position.y += dy

        # commanded velocity
        vx_cmd = vy_cmd = 0.0
        if self.last_cmd_pose is not None and self.last_cmd_time is not None:
            dt = (now - self.last_cmd_time).nanoseconds * 1e-9
            if dt > 1e-6:
                vx_cmd = (tgt.pose.position.x - self.last_cmd_pose.pose.position.x) / dt
                vy_cmd = (tgt.pose.position.y - self.last_cmd_pose.pose.position.y) / dt
        self.last_cmd_pose = tgt
        self.last_cmd_time = now
        self.last_v_cmd = (vx_cmd, vy_cmd)  # NEW

        # actual velocity (if EE feedback available)
        vx_act = vy_act = 0.0
        if self.last_act_pose is not None and self.last_act_time is not None and self.cur_pose is not None:
            dt = (now - self.last_act_time).nanoseconds * 1e-9
            if dt > 1e-6:
                vx_act = (self.cur_pose.pose.position.x - self.last_act_pose.pose.position.x) / dt
                vy_act = (self.cur_pose.pose.position.y - self.last_act_pose.pose.position.y) / dt
        self.last_v_act = (vx_act, vy_act)  # NEW

        # publish target pose
        self.pub_pose.publish(tgt)

        # log a row
        self._log_row(ex, ey, dx, dy, vx_cmd, vy_cmd, vx_act, vy_act, self.last_inliers, lost, self.fps_ema)  # NEW

    # ---- tiny CSV writer ----  # NEW
    def _log_row(self, ex, ey, dx, dy, vx_cmd, vy_cmd, vx_act, vy_act, inliers, lost, fps):
        if self.log_writer is None:
            return
        stamp = self.get_clock().now().nanoseconds * 1e-9
        self.log_writer.writerow([
            f'{stamp:.6f}', f'{ex:.3f}', f'{ey:.3f}', f'{dx:.5f}', f'{dy:.5f}',
            f'{vx_cmd:.4f}', f'{vy_cmd:.4f}', f'{vx_act:.4f}', f'{vy_act:.4f}',
            int(inliers), int(bool(lost)), f'{(fps if fps else 0.0):.2f}'
        ])

    # keep a copy of the latest EE pose if topic provided
    @property
    def cur_pose(self):
        return getattr(self, '_cur_pose', None)

    @cur_pose.setter
    def cur_pose(self, val):
        self._cur_pose = val

def main():
    rclpy.init()
    node = EarServoPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    # NEW: close logfile politely
    try:
        if getattr(node, 'log_fh', None) is not None:
            node.log_fh.close()
    except Exception:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
