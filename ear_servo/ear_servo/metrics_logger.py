#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
import csv, os, datetime
from typing import Optional

class MetricsLogger(Node):
    def __init__(self):
        super().__init__('metrics_logger')

        # Parameters
        self.declare_parameter('cmd_topic', '/ee_target_pose')
        self.declare_parameter('act_topic', '/ee_pose')          # empty string disables
        self.declare_parameter('diag_topic', '/tracking_diag')   # empty string disables
        self.declare_parameter('log_rate_hz', 30.0)
        self.declare_parameter('csv_path', '')                   # auto path if empty

        # Topics
        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        act_topic = self.get_parameter('act_topic').get_parameter_value().string_value
        diag_topic = self.get_parameter('diag_topic').get_parameter_value().string_value

        self.sub_cmd = self.create_subscription(PoseStamped, cmd_topic, self.on_cmd, 10)
        self.sub_act = None
        if act_topic:
            self.sub_act = self.create_subscription(PoseStamped, act_topic, self.on_act, 10)
        self.sub_diag = None
        if diag_topic:
            self.sub_diag = self.create_subscription(Float32MultiArray, diag_topic, self.on_diag, 10)

        # CSV init
        path = self.get_parameter('csv_path').get_parameter_value().string_value
        if not path:
            os.makedirs(os.path.expanduser('~/ros2_logs'), exist_ok=True)
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.expanduser(f'~/ros2_logs/metrics_{ts}.csv')
        self._fh = open(path, 'w', newline='')
        self._w = csv.writer(self._fh)
        self._w.writerow([
            'stamp',           # logger time (s)
            'cmd_x','cmd_y','cmd_z',
            'act_x','act_y','act_z',
            'vx_cmd','vy_cmd','vz_cmd',
            'vx_act','vy_act','vz_act',
            'ex','ey','inliers','lost','fps'
        ])
        self.get_logger().info(f'Logging to {path}')

        # State
        self.last_cmd: Optional[PoseStamped] = None
        self.prev_cmd: Optional[PoseStamped] = None
        self.last_act: Optional[PoseStamped] = None
        self.prev_act: Optional[PoseStamped] = None

        self.ex = float('nan'); self.ey = float('nan')
        self.inliers = -1; self.lost = 1; self.fps = float('nan')

        # Timer writer
        hz = float(self.get_parameter('log_rate_hz').value)
        self.timer = self.create_timer(1.0/max(1.0, hz), self.write_row)

    # --- subscribers ---
    def on_cmd(self, msg: PoseStamped):
        self.prev_cmd, self.last_cmd = self.last_cmd, msg

    def on_act(self, msg: PoseStamped):
        self.prev_act, self.last_act = self.last_act, msg

    def on_diag(self, msg: Float32MultiArray):
        # expected: [ex, ey, inliers, lost(0/1), fps]
        try:
            data = msg.data
            self.ex = float(data[0])
            self.ey = float(data[1])
            self.inliers = int(data[2])
            self.lost = int(data[3])
            self.fps = float(data[4])
        except Exception:
            pass

    def _vel(self, prev: Optional[PoseStamped], curr: Optional[PoseStamped]):
        if prev is None or curr is None:
            return (float('nan'),)*3
        t_prev = prev.header.stamp.sec + prev.header.stamp.nanosec * 1e-9
        t_curr = curr.header.stamp.sec + curr.header.stamp.nanosec * 1e-9
        dt = t_curr - t_prev
        if dt <= 1e-9:
            return (0.0, 0.0, 0.0)
        dx = curr.pose.position.x - prev.pose.position.x
        dy = curr.pose.position.y - prev.pose.position.y
        dz = curr.pose.position.z - prev.pose.position.z
        return (dx/dt, dy/dt, dz/dt)

    def write_row(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        # command & actual poses
        cmd_x = cmd_y = cmd_z = float('nan')
        if self.last_cmd:
            cmd_x = self.last_cmd.pose.position.x
            cmd_y = self.last_cmd.pose.position.y
            cmd_z = self.last_cmd.pose.position.z
        act_x = act_y = act_z = float('nan')
        if self.last_act:
            act_x = self.last_act.pose.position.x
            act_y = self.last_act.pose.position.y
            act_z = self.last_act.pose.position.z

        vx_cmd, vy_cmd, vz_cmd = self._vel(self.prev_cmd, self.last_cmd)
        vx_act, vy_act, vz_act = self._vel(self.prev_act, self.last_act)

        self._w.writerow([
            f'{now:.6f}',
            _fmt(cmd_x), _fmt(cmd_y), _fmt(cmd_z),
            _fmt(act_x), _fmt(act_y), _fmt(act_z),
            _fmt(vx_cmd), _fmt(vy_cmd), _fmt(vz_cmd),
            _fmt(vx_act), _fmt(vy_act), _fmt(vz_act),
            _fmt(self.ex), _fmt(self.ey), int(self.inliers), int(self.lost), _fmt(self.fps)
        ])
        # optional: flush every N rows if you want durability
        # self._fh.flush()

def _fmt(x):
    return f'{x:.6f}' if isinstance(x, float) else x

def main():
    rclpy.init()
    node = MetricsLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node._fh.close()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
