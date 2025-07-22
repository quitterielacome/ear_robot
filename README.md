# Trajectory Planner for XArm 7  
Vision-guided trajectory planning and tracking system for XArm 7 using ROS 2.

## Prerequisites

- ROS 2 (e.g. Humble or Foxy)
- xarm_ros2 driver installed
- Python dependencies:
  - cv2 (OpenCV)
  - rclpy
  - cv_bridge
  - geometry_msgs
  - sensor_msgs

## Installation and setup

1. Clone this repository into your ROS 2 workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/quitterielacome/ear_robot.git
```
2. Install dependencies:

``` bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

3. Build the workspace:

``` bash
colcon build
```

Source the workspace before running any nodes:

```bash
source install/setup.bash
```

## Workflow

1. Check that a camera is publishing to /camera/image_raw:

``` bash
ros2 run image_tools cam2image
```

2. Check that the topic is live
3. Verify that the topic is live:

``` bash
ros2 topic list
ros2 topic echo /camera/image_raw
```

3. Run the trajectory planner node:

``` bash
ros2 run trajectory_planner tracking_node
```

4. Run the trajectory executor node (in another terminal):

``` bash
ros2 run trajectory_planner trajectory_server
```

5. Run the XArm 7 driver (change the IP address)

``` bash
ros2 launch xarm_bringup xarm7_bringup.launch.py robot_ip:=<ip-address>
```

6. Verify the XArm 7 services are available:

``` bash
ros2 service list
```

## How it works

- Project your video onto a surface.

- The camera on the XArm 7 captures the scene.

- The tracking node detects the target and computes the trajectory.

- The trajectory server executes waypoints on the XArm 7.



