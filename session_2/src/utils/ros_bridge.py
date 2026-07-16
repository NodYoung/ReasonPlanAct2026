"""
ROS 2 bridge for turtlesim.

Wraps rclpy publishers, subscribers, and services for /turtle1.
Start turtlesim first: pixi run sim
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass


@dataclass
class TurtlePose:
    """Simple pose container used by tools and agents."""

    x: float
    y: float
    theta: float  # radians, 0 = facing +x


class MockTurtleSim:
    """
    In-memory turtle simulator for learning without ROS 2 running.

    Mimics cmd_vel integration and pose feedback with basic kinematics.
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
        """Integrate linear motion along the turtle heading."""
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
        Rotate in place using pose feedback (closed-loop).

        Keeps publishing cmd_vel until /turtle1/pose shows the requested
        heading change — more reliable than open-loop sleep timing.
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
            # Mock: integrate directly for simulation speed
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
        """Keep heading in [-pi, pi]."""
        while theta > math.pi:
            theta -= 2 * math.pi
        while theta < -math.pi:
            theta += 2 * math.pi
        return theta


class RosTurtleNode:
    """
    Real ROS 2 node that talks to turtlesim.

    Spins rclpy in a background thread so LangGraph agents can call tools
    synchronously from the main thread.
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

        # Publisher for velocity commands
        self._cmd_pub = self._node.create_publisher(
            Twist, f"/{turtle_name}/cmd_vel", 10
        )

        # Subscribe to pose feedback
        def _pose_callback(msg: Pose) -> None:
            with self._lock:
                self._pose = TurtlePose(msg.x, msg.y, msg.theta)

        self._node.create_subscription(Pose, f"/{turtle_name}/pose", _pose_callback, 10)

        # Service clients
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

        # Spin ROS in the background
        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True
        )
        self._spin_thread.start()

        # Wait briefly for first pose message
        for _ in range(50):
            if self._wait_for_pose(timeout=0.1):
                break
        self._ready.set()

    def _wait_for_service(self, client, timeout_sec: float = 2.0) -> bool:
        return client.wait_for_service(timeout_sec=timeout_sec)

    def is_connected(self, timeout_sec: float = 2.0) -> bool:
        """True when turtlesim's services are reachable (i.e. the sim is running)."""
        return self._wait_for_service(self._set_pen_client, timeout_sec=timeout_sec)

    def _wait_for_future(self, future, timeout_sec: float = 5.0) -> None:
        """
        Wait for an async service call while a background thread spins the node.

        Do NOT call rclpy.spin_until_future_complete here — it conflicts with
        the background spin thread and can hang or raise.
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
        return True  # turtlesim may legitimately start at origin in some setups

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
        Move forward by `distance` meters using visible cmd_vel motion.

        Uses closed-loop pose feedback (repeated cmd_vel publish), then a
        small TeleportRelative snap for sub-centimeter accuracy.
        """
        from geometry_msgs.msg import Twist
        from turtlesim.srv import TeleportRelative

        if distance <= 0:
            return "Skipped zero or negative forward distance."

        # Animated closed-loop cmd_vel, then a tiny TeleportRelative snap for accuracy.
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

        # Snap the final fraction of a degree for square corners.
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
        """Shortest signed angle from current to target (radians)."""
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

        self._node.destroy_node()
        rclpy.shutdown()


class TurtleBridge:
    """
    Factory that picks ROS or mock backend.

    Call get_turtle_bridge() once at startup and pass the instance to tools.
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
        """True when the live turtlesim node is reachable (always True for mock)."""
        if isinstance(self._backend, RosTurtleNode):
            return self._backend.is_connected()
        return True

    def get_pose(self) -> TurtlePose:
        return self._backend.get_pose()

    def get_pen_state(self) -> dict[str, int | bool]:
        """Last known pen colour, width, and on/off (updated on each set_pen)."""
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
    """Connect to the live turtlesim node (run `pixi run sim` first)."""
    return TurtleBridge(use_mock=False, turtle_name=turtle_name)
