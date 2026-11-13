"""
turtlesim 的 ROS 2 桥接层。

封装了针对 /turtle1 的 rclpy 发布者、订阅者和服务。
请先启动 turtlesim：pixi run sim
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass


@dataclass
class TurtlePose:
    """供工具和智能体使用的简单位姿容器。"""

    x: float
    y: float
    theta: float  # 弧度，0 = 朝向 +x


class MockTurtleSim:
    """
    内存中的海龟模拟器，无需运行 ROS 2 即可学习。

    用基础运动学模拟 cmd_vel 积分和位姿反馈。
    """

    CANVAS_MIN = 0.0
    CANVAS_MAX = 11.0

    def __init__(self) -> None:
        self.x = 5.5
        self.y = 5.5
        self.theta = 0.0
        self.pen = {"r": 255, "g": 255, "b": 255, "width": 2, "off": False}

    def get_pose(self) -> TurtlePose:
        return TurtlePose(self.x, self.y, self.theta)

    def set_pen(self, r: int, g: int, b: int, width: int, off: bool = False) -> str:
        self.pen = {"r": r, "g": g, "b": b, "width": width, "off": off}
        return f"Pen set to RGB({r},{g},{b}), width={width}, off={off}"

    def move_forward(self, distance: float, linear_speed: float = 1.0) -> str:
        """沿海龟朝向对线性运动进行积分。"""
        if distance <= 0:
            return "Skipped zero or negative forward distance."

        duration = distance / linear_speed
        steps = max(int(duration / 0.05), 1)
        step_distance = distance / steps
        for _ in range(steps):
            self.x += step_distance * math.cos(self.theta)
            self.y += step_distance * math.sin(self.theta)
            time.sleep(duration / steps)

        return f"Moved forward {distance:.2f} m to ({self.x:.2f}, {self.y:.2f})"

    def rotate(
        self,
        angle_degrees: float,
        angular_speed: float = 1.5,
        tolerance_deg: float = 2.0,
    ) -> str:
        """
        利用位姿反馈进行原地旋转（闭环）。

        持续发布 cmd_vel，直到 /turtle1/pose 显示达到所请求的
        朝向变化——比开环的 sleep 定时更可靠。
        """
        if abs(angle_degrees) < 1e-6:
            return "Skipped zero rotation."

        start_theta = self.theta
        target_delta = math.radians(angle_degrees)
        target_theta = self._normalize_angle(start_theta + target_delta)
        tolerance = math.radians(tolerance_deg)
        deadline = time.time() + 15.0

        while time.time() < deadline:
            error = self._normalize_angle(target_theta - self.theta)
            if abs(error) <= tolerance:
                break
            # 模拟器：为了仿真速度直接进行积分
            step = min(abs(error), math.radians(5))
            self.theta = self._normalize_angle(
                self.theta + step * (1 if error > 0 else -1)
            )
            time.sleep(0.02)

        return (
            f"Rotated {angle_degrees:.1f}° (target) — "
            f"heading now {math.degrees(self.theta):.1f}°"
        )

    def teleport(self, x: float, y: float, theta: float) -> str:
        self.x = x
        self.y = y
        self.theta = theta
        return f"Teleported to ({x:.2f}, {y:.2f}, {math.degrees(theta):.1f}°)"

    def clear_canvas(self) -> str:
        return "Canvas cleared (mock — no visual output)."

    def reset_world(self) -> str:
        self.x = 5.5
        self.y = 5.5
        self.theta = 0.0
        return "World reset to default spawn pose."

    @staticmethod
    def _normalize_angle(theta: float) -> float:
        """将朝向保持在 [-pi, pi] 区间内。"""
        while theta > math.pi:
            theta -= 2 * math.pi
        while theta < -math.pi:
            theta += 2 * math.pi
        return theta


class RosTurtleNode:
    """
    与 turtlesim 通信的真实 ROS 2 节点。

    在后台线程中运行 rclpy，使 LangGraph 智能体可以从主线程
    同步调用工具。
    """

    def __init__(self, turtle_name: str = "turtle1") -> None:
        import rclpy
        from geometry_msgs.msg import Twist
        from rclpy.node import Node
        from std_srvs.srv import Empty
        from turtlesim.msg import Pose
        from turtlesim.srv import SetPen, TeleportAbsolute, TeleportRelative

        self._lock = threading.Lock()
        self._pose = TurtlePose(5.5, 5.5, 0.0)
        self.pen = {"r": 255, "g": 255, "b": 255, "width": 2, "off": False}
        self._ready = threading.Event()

        class _TurtleNode(Node):
            pass

        self._node = _TurtleNode("multi_agent_turtle_tools")
        self._turtle = turtle_name

        # 速度命令的发布者
        self._cmd_pub = self._node.create_publisher(
            Twist, f"/{turtle_name}/cmd_vel", 10
        )

        # 订阅位姿反馈
        def _pose_callback(msg: Pose) -> None:
            with self._lock:
                self._pose = TurtlePose(msg.x, msg.y, msg.theta)

        self._node.create_subscription(Pose, f"/{turtle_name}/pose", _pose_callback, 10)

        # 服务客户端
        self._set_pen_client = self._node.create_client(
            SetPen, f"/{turtle_name}/set_pen"
        )
        self._teleport_client = self._node.create_client(
            TeleportAbsolute, f"/{turtle_name}/teleport_absolute"
        )
        self._teleport_relative_client = self._node.create_client(
            TeleportRelative, f"/{turtle_name}/teleport_relative"
        )
        self._clear_client = self._node.create_client(Empty, "/clear")
        self._reset_client = self._node.create_client(Empty, "/reset")

        # 使用可停止的 executor 在后台运行 ROS，这样 shutdown() 可以在
        # 销毁 node / 调用 rclpy.shutdown() 之前干净地停止 spin
        # （裸用 rclpy.spin + rclpy.shutdown 存在竞争，会以
        # "terminate called without an active exception" 中止进程）。
        from rclpy.executors import SingleThreadedExecutor

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True
        )
        self._spin_thread.start()

        # 短暂等待第一条位姿消息
        for _ in range(50):
            if self._wait_for_pose(timeout=0.1):
                break
        self._ready.set()

    def _wait_for_service(self, client, timeout_sec: float = 2.0) -> bool:
        return client.wait_for_service(timeout_sec=timeout_sec)

    def is_connected(self, timeout_sec: float = 2.0) -> bool:
        """当 turtlesim 的服务可达时返回 True（即仿真正在运行）。"""
        return self._wait_for_service(self._set_pen_client, timeout_sec=timeout_sec)

    def _wait_for_future(self, future, timeout_sec: float = 5.0) -> None:
        """
        在后台线程 spin 节点期间，等待一次异步服务调用完成。

        不要在这里调用 rclpy.spin_until_future_complete——它会与
        后台的 spin 线程冲突，可能卡死或抛异常。
        """
        deadline = time.time() + timeout_sec
        while not future.done():
            if time.time() > deadline:
                raise TimeoutError(
                    f"Service call timed out after {timeout_sec:.1f}s"
                )
            time.sleep(0.01)
        future.result()

    def _wait_for_pose(self, timeout: float = 1.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._pose.x != 0.0 or self._pose.y != 0.0:
                    return True
            time.sleep(0.05)
        return True  # 在某些配置下 turtlesim 可能确实从原点开始

    def get_pose(self) -> TurtlePose:
        with self._lock:
            return TurtlePose(self._pose.x, self._pose.y, self._pose.theta)

    def set_pen(self, r: int, g: int, b: int, width: int, off: bool = False) -> str:
        from turtlesim.srv import SetPen

        if not self._wait_for_service(self._set_pen_client):
            return "set_pen service unavailable — is turtlesim running?"

        request = SetPen.Request(r=r, g=g, b=b, width=width, off=off)
        future = self._set_pen_client.call_async(request)
        self._wait_for_future(future, timeout_sec=2.0)
        with self._lock:
            self.pen = {"r": r, "g": g, "b": b, "width": width, "off": off}
        return f"Pen set to RGB({r},{g},{b}), width={width}, off={off}"

    def move_forward(self, distance: float, linear_speed: float = 1.5) -> str:
        """
        利用可见的 cmd_vel 运动，向前移动 `distance` 米。

        使用闭环位姿反馈（反复发布 cmd_vel），随后用一次
        小幅的 TeleportRelative 微调以达到亚厘米级精度。
        """
        from geometry_msgs.msg import Twist
        from turtlesim.srv import TeleportRelative

        if distance <= 0:
            return "Skipped zero or negative forward distance."

        # 先做带动画的闭环 cmd_vel，再用一次极小的 TeleportRelative 微调以提高精度。
        start = self.get_pose()
        sx, sy = start.x, start.y
        snap_distance = min(0.08, distance * 0.05)
        animate_target = max(0.0, distance - snap_distance)
        twist = Twist()
        twist.linear.x = linear_speed
        deadline = time.time() + max(60.0, distance / linear_speed * 3.0)

        while time.time() < deadline:
            self._cmd_pub.publish(twist)
            pose = self.get_pose()
            traveled = math.hypot(pose.x - sx, pose.y - sy)
            if traveled >= animate_target:
                break
            time.sleep(0.05)

        twist.linear.x = 0.0
        self._cmd_pub.publish(twist)
        time.sleep(0.05)

        pose = self.get_pose()
        traveled = math.hypot(pose.x - sx, pose.y - sy)
        remaining = distance - traveled
        if abs(remaining) > 0.005 and self._wait_for_service(
            self._teleport_relative_client, timeout_sec=0.5
        ):
            request = TeleportRelative.Request(linear=float(remaining), angular=0.0)
            future = self._teleport_relative_client.call_async(request)
            self._wait_for_future(future, timeout_sec=5.0)
            pose = self.get_pose()
            traveled = math.hypot(pose.x - sx, pose.y - sy)

        return (
            f"Moved forward {distance:.2f} m (cmd_vel + snap, traveled {traveled:.2f} m) "
            f"to ({pose.x:.2f}, {pose.y:.2f})"
        )

    def rotate(
        self,
        angle_degrees: float,
        angular_speed: float = 1.2,
        tolerance_deg: float = 2.0,
    ) -> str:
        from geometry_msgs.msg import Twist
        from turtlesim.srv import TeleportRelative

        if abs(angle_degrees) < 1e-6:
            return "Skipped zero rotation."

        start_theta = self.get_pose().theta
        target_theta = start_theta + math.radians(angle_degrees)
        snap_tol = math.radians(0.5)
        drive_tol = math.radians(1.5)
        deadline = time.time() + max(30.0, abs(angle_degrees) / 45.0 * 5.0)

        twist = Twist()
        while time.time() < deadline:
            current = self.get_pose().theta
            error = self._angle_error(target_theta, current)
            if abs(error) <= snap_tol:
                break
            if abs(error) <= drive_tol:
                break
            twist.angular.z = angular_speed if error > 0 else -angular_speed
            self._cmd_pub.publish(twist)
            time.sleep(0.05)

        twist.angular.z = 0.0
        self._cmd_pub.publish(twist)
        time.sleep(0.05)

        # 微调最后不足一度的部分，以得到方正的拐角。
        remaining = self._angle_error(target_theta, self.get_pose().theta)
        if abs(remaining) > 0.002 and self._wait_for_service(
            self._teleport_relative_client, timeout_sec=0.5
        ):
            request = TeleportRelative.Request(linear=0.0, angular=float(remaining))
            future = self._teleport_relative_client.call_async(request)
            self._wait_for_future(future, timeout_sec=5.0)

        pose = self.get_pose()
        return (
            f"Rotated {angle_degrees:.1f}° (cmd_vel + snap) — "
            f"heading now {math.degrees(pose.theta):.1f}°"
        )

    @staticmethod
    def _angle_error(target: float, current: float) -> float:
        """从当前朝向到目标朝向的最短有符号角度（弧度）。"""
        error = target - current
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi
        return error

    def teleport(self, x: float, y: float, theta: float) -> str:
        from turtlesim.srv import TeleportAbsolute

        if not self._wait_for_service(self._teleport_client):
            return "teleport_absolute service unavailable — is turtlesim running?"

        request = TeleportAbsolute.Request(x=x, y=y, theta=theta)
        future = self._teleport_client.call_async(request)
        self._wait_for_future(future, timeout_sec=2.0)
        return f"Teleported to ({x:.2f}, {y:.2f}, {math.degrees(theta):.1f}°)"

    def clear_canvas(self) -> str:
        from std_srvs.srv import Empty

        if not self._wait_for_service(self._clear_client):
            return "clear service unavailable — is turtlesim running?"

        request = Empty.Request()
        future = self._clear_client.call_async(request)
        self._wait_for_future(future, timeout_sec=2.0)
        return "Canvas cleared."

    def reset_world(self) -> str:
        from std_srvs.srv import Empty

        if not self._wait_for_service(self._reset_client):
            return "reset service unavailable — is turtlesim running?"

        request = Empty.Request()
        future = self._reset_client.call_async(request)
        self._wait_for_future(future, timeout_sec=2.0)
        time.sleep(0.5)
        return "World reset to default spawn pose."

    def shutdown(self) -> None:
        import rclpy

        # 1) 停止后台 spin，让 executor.spin() 返回
        try:
            self._executor.shutdown()
        except Exception:
            pass
        # 2) 等后台线程真正退出（带超时）
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        # 3) 销毁 node + 关闭 rclpy（保护，避免重复/竞争导致 abort）
        try:
            self._node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


class TurtleBridge:
    """
    在 ROS 后端与模拟后端之间进行选择的工厂。

    启动时调用一次 get_turtle_bridge()，并将返回的实例传给工具。
    """

    def __init__(self, use_mock: bool = False, turtle_name: str = "turtle1") -> None:
        self._use_mock = use_mock
        self._backend: MockTurtleSim | RosTurtleNode

        if use_mock:
            self._backend = MockTurtleSim()
        else:
            import rclpy

            rclpy.init()
            self._backend = RosTurtleNode(turtle_name=turtle_name)

    @property
    def is_mock(self) -> bool:
        return self._use_mock

    def is_connected(self) -> bool:
        """当实时 turtlesim 节点可达时返回 True（模拟模式始终为 True）。"""
        if isinstance(self._backend, RosTurtleNode):
            return self._backend.is_connected()
        return True

    def get_pose(self) -> TurtlePose:
        return self._backend.get_pose()

    def get_pen_state(self) -> dict[str, int | bool]:
        """最近一次已知的画笔颜色、宽度和开/关状态（每次 set_pen 时更新）。"""
        pen = getattr(self._backend, "pen", None)
        if pen is None:
            return {"r": 255, "g": 255, "b": 255, "width": 2, "off": False}
        return dict(pen)

    def set_pen(self, r: int, g: int, b: int, width: int, off: bool = False) -> str:
        return self._backend.set_pen(r, g, b, width, off)

    def move_forward(self, distance: float) -> str:
        return self._backend.move_forward(distance)

    def rotate(self, angle_degrees: float) -> str:
        return self._backend.rotate(angle_degrees)

    def teleport(self, x: float, y: float, theta: float) -> str:
        return self._backend.teleport(x, y, theta)

    def clear_canvas(self) -> str:
        return self._backend.clear_canvas()

    def reset_world(self) -> str:
        return self._backend.reset_world()

    def shutdown(self) -> None:
        if not self._use_mock and isinstance(self._backend, RosTurtleNode):
            self._backend.shutdown()


def get_turtle_bridge(turtle_name: str = "turtle1") -> TurtleBridge:
    """连接到实时的 turtlesim 节点（请先运行 `pixi run sim`）。"""
    return TurtleBridge(use_mock=False, turtle_name=turtle_name)
