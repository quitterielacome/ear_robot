# ear\_servo — Quick Start (README)

Tiny ROS 2 package that:

* tracks a target ROI in the camera image (KLT + RANSAC, glare-robust),
* publishes incremental **end-effector pose goals** (`geometry_msgs/PoseStamped`) on `/ee_target_pose`,
* streams diagnostics on `/tracking_diag`,
* logs a CSV with pixel error, inliers, FPS, and commanded/actual velocities.

---

# 0) Requirements

* ROS 2 (Humble/Foxy or similar)
* Python 3, OpenCV (`pip install opencv-python`)
* `cv_bridge`, `sensor_msgs`, `geometry_msgs`, `std_msgs`, `std_srvs`
* A camera topic (default `/camera/image_raw`). Quick dummy camera:

  ```bash
  ros2 run image_tools cam2image --ros-args -r image:=/camera/image_raw
  ```

---

# 1) Build

```bash
cd ~/ros2_ws/src
# put this repo here as: src/ear_servo/
cd ..
colcon build --symlink-install
. install/setup.bash
```

---

# 2) Calibrate pixels → meters (once)

Edit `ear_servo/config/default.yaml`:

* `meters_per_px_x`, `meters_per_px_y` — measure on your wall/screen (ruler).
  Example: if 1000 px ≈ 0.5 m, then `0.0005`.
* `flip_x`, `flip_y` — set so a **positive pixel error** commands the **correct EE direction**.
* `step_fraction` and `max_step_m` — how aggressively each pose update moves.

---

# 3) Run

## A) Bring up tracker (+ logger)

```bash
ros2 launch ear_servo bringup.launch.py camera_topic:=/camera/image_raw
```

A debug window will pop up. **Draw the ROI** (target area) and press **Enter**.

## B) Enable / disable the servo

```bash
# start
ros2 service call /enable_servo std_srvs/srv/SetBool "{data: true}"

# stop
ros2 service call /enable_servo std_srvs/srv/SetBool "{data: false}"

# re-pick ROI at any time
ros2 service call /reset_roi std_srvs/srv/SetBool "{}"
```

> The node keeps publishing `/ee_target_pose`. Your xArm follower should subscribe to this topic and execute those incremental pose goals (hold Z/orientation fixed; apply X/Y deltas).

---

# 4) What you should see

* Debug window with **green box** around the tracked target, a **white dot** at image center (needle), and **FPS / inliers** text.
* Terminal prints like: `pix_err=(ex,ey) cmd_v=(vx,vy) m/s`.
* A CSV log path printed (default: `~/ros2_logs/ear_servo_YYYYMMDD_HHMMSS.csv`).

---

# 5) Topics & Services

**Published**

* `/ee_target_pose` — `geometry_msgs/PoseStamped` (incremental pose goals).
* `/tracking_diag` — `std_msgs/Float32MultiArray`
  `data = [ex, ey, inliers, lost(0/1), fps]`

**Subscribed**

* `/camera/image_raw` — `sensor_msgs/Image`
* `/ee_pose` (optional) — `geometry_msgs/PoseStamped` for **actual** EE pose feedback (only for velocity logging).

**Services**

* `/enable_servo` — `std_srvs/SetBool`
* `/reset_roi` — `std_srvs/SetBool`

---

# 6) Parameters (key ones)

Edit in `config/default.yaml` or override at launch.

* `camera_topic` — image source.
* `target_pose_topic` — where pose goals are published (default `/ee_target_pose`).
* `ee_pose_topic` — robot EE pose (optional).
* `meters_per_px_x`, `meters_per_px_y` — **calibration** values.
* `flip_x`, `flip_y` — image→robot axis alignment.
* `step_fraction` — fraction of remaining pixel error moved per update (0–1).
* `max_step_m` — hard clamp per update (safety).
* `deadband_px` — inside this error band, publish “hold” pose.
* `work_w`, `n_pts`, `lk_win`, `lk_levels`, `ransac_thr`, `min_inliers` — tracker tuning.
* `show_debug` — toggle OpenCV window.
* `inpaint` — glare inpainting.
* `publish_rate_hz` — update rate.
* `log_enable`, `log_csv` — CSV logging toggle & path.

---

# 7) CSV Logs

Two writers:

1. **Pose node** (enabled if `log_enable: true`) → `~/ros2_logs/ear_servo_*.csv`

   ```
   stamp, ex, ey, dx, dy, vx_cmd, vy_cmd, vx_act, vy_act, inliers, lost, fps
   ```

2. **Metrics logger** (launched alongside) subscribes to `/ee_target_pose`, `/ee_pose` (optional), `/tracking_diag` → `~/ros2_logs/metrics_*.csv`

   ```
   stamp, cmd_x,cmd_y,cmd_z, act_x,act_y,act_z,
   vx_cmd,vy_cmd,vz_cmd, vx_act,vy_act,vz_act,
   ex,ey,inliers,lost,fps
   ```

---

# 8) Quick robot hookup

Your xArm node should:

* Subscribe to `/ee_target_pose` (PoseStamped in the robot tool frame, e.g., `tool0`).
* Execute each incoming pose goal (small X/Y deltas).
* (Optional) Publish actual EE pose to `/ee_pose` for velocity logging/analysis.

---

# 9) Troubleshooting

* **No window / ROI dialog:** set `show_debug: true`. On headless machines, use VNC or disable the window and set a default ROI in code.
* **No images:** `ros2 topic echo /camera/image_raw` to confirm. Remap `camera_topic` if needed.
* **Pose not moving:** ensure your xArm follower subscribes to `/ee_target_pose` and executes the poses.
* **Direction wrong:** toggle `flip_x` / `flip_y`.
* **Too jumpy / too slow:** tune `step_fraction`, `max_step_m`, `deadband_px`.
* **Tracker drops:** increase `n_pts`, `lk_win`, `lk_levels`, or relax `ransac_thr`; try `inpaint: true`.

---

# 10) Folder Map

```
ear_servo/
  launch/
    bringup.launch.py
  config/
    default.yaml
  ear_servo/
    __init__.py             # (can be empty)
    ear_servo_pose_node.py  # main pose-goal publisher
    metrics_logger.py       # optional extra logger
    tracking.py             # shared OpenCV helpers (no ROS imports)
  resource/
    ear_servo               # ament marker file containing the package name
  package.xml
  setup.py
  setup.cfg
  README.md
```

> `__init__.py` can be empty; it just marks `ear_servo/` as a Python package.

---

# 11) Typical Session

```bash
# 1) build & source
cd ~/ros2_ws
colcon build --symlink-install
. install/setup.bash

# 2) start a camera (example)
ros2 run image_tools cam2image --ros-args -r image:=/camera/image_raw

# 3) bring up tracking + logger
ros2 launch ear_servo bringup.launch.py

# 4) draw ROI in the popup (press Enter)

# 5) start servo
ros2 service call /enable_servo std_srvs/srv/SetBool "{data: true}"

# (optional) reset ROI later
ros2 service call /reset_roi std_srvs/srv/SetBool "{}"
```

---

# 12) Extra: Diagnostics topic snippet (already included)

In `ear_servo_pose_node.py` we publish a small diagnostics array:

```python
from std_msgs.msg import Float32MultiArray  # top of file

self.pub_diag = self.create_publisher(Float32MultiArray, '/tracking_diag', 10)

# ... inside publish_pose() after you compute ex, ey, inliers, lost, fps_ema:
msg = Float32MultiArray()
msg.data = [float(ex), float(ey),
            float(self.last_inliers), float(self.last_lost),
            float(self.fps_ema or 0.0)]
self.pub_diag.publish(msg)
```

The bundled `metrics_logger.py` subscribes to this and merges it with commanded/actual pose data in its CSV.

---

That’s it—tomorrow you should be able to: launch, draw ROI, enable servo, and watch `/ee_target_pose` drive the xArm follower while you log beautiful metrics.
