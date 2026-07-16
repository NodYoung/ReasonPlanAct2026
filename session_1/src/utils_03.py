# ── Exercise 3d ──────────────────────────────────────────────────────────────
# Complete tool suite — ROS backend via TODO blocks.

# ROS imports
import time
import math
from typing import cast
import rclpy
from geometry_msgs.msg import Twist, Vector3
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from std_srvs.srv import Empty
from turtlesim_msgs.msg import Color, Pose
from turtlesim_msgs.srv import SetPen, TeleportAbsolute
from pydantic import ValidationError

# ── Constants ─────────────────────────────────────────────────────────────────
ARENA_MIN = 0.0
ARENA_MAX = 11.0
SERVICE_TIMEOUT = 2.0  # seconds to wait for a ROS service before giving up


# ── Shared helper ─────────────────────────────────────────────────────────────
def _call_service(node, client, request, service_name: str, timeout_sec: float = SERVICE_TIMEOUT):
    """Wait for a service, call it, and return (success, result/error_str)."""
    if not client.wait_for_service(timeout_sec=timeout_sec):
        return False, f"Service {service_name} not available. Is turtlesim running?"

    future = client.call_async(request)

    while rclpy.ok() and not future.done():
        rclpy.spin_once(node, timeout_sec=0.1)

    if future.cancelled():
        return False, f"Service call to {service_name} was cancelled."
    if future.exception() is not None:
        return False, f"Service call to {service_name} failed: {future.exception()}"

    return True, future.result()

# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — get_pose
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_pose(timeout_sec: float = 2.0) -> str:
    """Return the turtle's current pose: x, y position and heading.

    No required arguments. Returns a plain string with labelled fields:
      'x=N, y=N, heading=N° (N rad)'

    Always call this before planning a move to confirm the current state.
    The arena is 11×11 metres; valid positions are roughly x∈[0,11], y∈[0,11].
    """
    if not rclpy.ok():
        rclpy.init()
    node = Node("ros_langchain_get_pose")

    try:
        success, pose = wait_for_message(
            Pose,
            node,
            "/turtle1/pose",
            qos_profile=10,
            time_to_wait=max(0.0, float(timeout_sec)),
        )
        if not success or pose is None:
            return "Timed out waiting for /turtle1/pose. Is turtlesim running?"

        pose = cast(Pose, pose)
        heading_deg = math.degrees(pose.theta)

        return (
            f"x={pose.x:.2f}, y={pose.y:.2f}"
            f", heading={heading_deg:.1f}° ({pose.theta:.3f} rad)"
            # f", linear_vel={pose.linear_velocity:.2f} m/s, "
            # f", angular_vel={pose.angular_velocity:.2f} rad/s"
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — move_forward
# How do we add the constraints?
# ─────────────────────────────────────────────────────────────────────────────

# Expanding the previous MoveArgs with speed and rate parameters, and validation.
class MoveArgs(BaseModel):
    distance_m: float = Field(
        ...,
        gt=0,
        le=10,
        description="Distance to travel in metres. Must be in (0, 10].",
    )
    speed_ms: float = Field(
        default=1.0,
        gt=0,
        le=5.0,
        description="Linear speed in m/s. Defaults to 1.0.",
    )
    rate_hz: float = Field(
        default=10.0,
        gt=0,
        description="Publish rate in Hz. Higher = smoother motion. Defaults to 10.",
    )


@tool(args_schema=MoveArgs)
def move_forward(
    distance_m: float,
    speed_ms: float = 1.0,
    rate_hz: float = 10.0,
) -> str:
    """Move the turtle forward along its current heading by an exact distance.

    Args:
        distance_m: Distance to travel in metres. Must be in (0, 10].
        speed_ms:   Linear speed in m/s (0, 5]. Defaults to 1.0.
        rate_hz:    Publish rate in Hz. Defaults to 10.

    Returns a confirmation with publish stats and the new pose.
    The pen draws a line if it is down (see set_pen).
    Always call get_pose first to confirm the turtle is clear to move.
    """
    if not rclpy.ok():
        rclpy.init()

    node = Node(f"ros_langchain_move_forward_{int(time.monotonic() * 1000)}")
    publisher = node.create_publisher(Twist, "/turtle1/cmd_vel", 10)
    time.sleep(0.5)  # allow publisher to set up before sending messages

    try:
        message = Twist(
            linear=Vector3(x=float(speed_ms)),
            angular=Vector3(z=0.0),   # straight ahead — no rotation
        )
        stop = Twist()                # zero velocity — sent after the loop

        publish_interval = 1.0 / float(rate_hz)
        duration_sec     = distance_m / speed_ms
        deadline         = time.monotonic() + duration_sec
        publish_count    = 0

        while rclpy.ok() and time.monotonic() < deadline:
            publisher.publish(message)
            publish_count += 1
            node.get_logger().debug(
                f"move_forward: linear_x={speed_ms:.2f} m/s "
                f"({publish_count} msgs sent)"
            )
            time.sleep(min(publish_interval, deadline - time.monotonic()))

        # Explicit stop — prevents the turtle coasting on the last velocity
        publisher.publish(stop)

        pose_str = get_pose.invoke({})   # re-use the validated tool

        return (
            f"Moved forward {distance_m:.2f} m at {speed_ms:.2f} m/s "
            f"({publish_count} messages over {duration_sec:.2f} s). "
            f"{pose_str}"
        )

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — rotate
# ─────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field, model_validator

class RotateArgs(BaseModel):
    angle_deg: float = Field(
        ...,
        ge=-360.0,
        le=360.0,
        description=(
            "Target angle to rotate in degrees. "
            "Positive = counter-clockwise, negative = clockwise. Range [-360, 360]."
        ),
    )
    angular_velocity_degs: float = Field(
        default=90.0,
        gt=0,
        le=360.0,
        description=(
            "Rotational speed in deg/s (0, 360]. "
            "Default 90 deg/s (a 90° turn takes 1 second). "
            "Increase for faster turns, decrease for more precise ones."
        ),
    )
    rate_hz: float = Field(
        default=10.0,
        gt=0,
        le=50.0,
        description="Publish rate in Hz. Higher = smoother rotation. Default 10.",
    )


@tool(args_schema=RotateArgs)
def rotate(
    angle_deg: float,
    angular_velocity_degs: float = 90.0,
    rate_hz: float = 10.0,
) -> str:
    """Rotate the turtle in place by a target angle. Does not move forward.

    The tool computes the duration automatically from angle and speed.
    You only need to specify how far to turn and optionally how fast.

    Args:
        angle_deg:              Target angle in degrees [-360, 360].
                                Positive = counter-clockwise, negative = clockwise.
        angular_velocity_degs:  Rotational speed in deg/s (0, 360]. Default 90.
                                Slow (45 deg/s) for precision, fast (180 deg/s) for speed.
        rate_hz:                Publish rate in Hz. Default 10.

    Returns the new heading after the rotation completes.
    Always call get_pose first to confirm the starting heading.

    Examples:
      rotate(angle_deg=90)                                # 90° CCW at default speed
      rotate(angle_deg=-180)                              # 180° CW U-turn
      rotate(angle_deg=45, angular_velocity_degs=45)     # slow precise 45° turn
      rotate(angle_deg=-90, angular_velocity_degs=180)   # fast 90° CW
    """
    if angle_deg == 0.0:
        return "No rotation requested: angle_deg is 0."

    if not rclpy.ok():
        rclpy.init()

    node = Node(f"ros_langchain_rotate_{int(time.monotonic() * 1000)}")
    publisher = node.create_publisher(Twist, "/turtle1/cmd_vel", 10)
    time.sleep(0.5)

    try:
        # Duration is derived — the model never has to compute this
        duration_sec = abs(angle_deg) / angular_velocity_degs

        # Preserve the sign of angle_deg for direction
        signed_velocity = math.copysign(angular_velocity_degs, angle_deg)
        angular_z_rads = math.radians(signed_velocity)

        message = Twist(
            linear=Vector3(x=0.0),                      # explicit zero — no forward motion
            angular=Vector3(z=float(angular_z_rads)),
        )
        stop = Twist()

        publish_interval = 1.0 / float(rate_hz)
        deadline = time.monotonic() + duration_sec
        publish_count = 0

        while rclpy.ok() and time.monotonic() < deadline:
            publisher.publish(message)
            publish_count += 1
            node.get_logger().debug(
                f"rotate: {signed_velocity:.1f} deg/s ({publish_count} msgs)"
            )
            time.sleep(min(publish_interval, deadline - time.monotonic()))

        publisher.publish(stop)

        direction = "CCW" if angle_deg > 0 else "CW"
        pose_str = get_pose.invoke({})

        return (
            f"Rotated {direction} {abs(angle_deg):.1f}° "
            f"at {angular_velocity_degs:.1f} deg/s "
            f"over {duration_sec:.2f} s "
            f"({publish_count} messages at {rate_hz:.0f} Hz). "
            f"{pose_str}"
        )

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

## You now have two tools: move_forward controls distance, rotate controls heading. Each one is simple, correct, and testable on its own.
# What happens if you combine them into a single tool that accepts both linear and angular velocity at the same time?
# Is it more powerful — or does it create a new problem? And if it does create a problem, where does that problem actually live: in the tool, in the agent?



# ─────────────────────────────────────────────────────────────────────────────
# Tool — set_pen
# ─────────────────────────────────────────────────────────────────────────────
class PenArgs(BaseModel):
    r: int = Field(
        default=255,
        ge=0,
        le=255,
        description="Red component [0, 255]. Default 255.",
    )
    g: int = Field(
        default=255,
        ge=0,
        le=255,
        description="Green component [0, 255]. Default 255.",
    )
    b: int = Field(
        default=255,
        ge=0,
        le=255,
        description="Blue component [0, 255]. Default 255.",
    )
    width: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Line width in pixels [1, 10]. Default 2.",
    )
    off: bool = Field(
        default=False,
        description="True = pen up (no drawing). False = pen down (draws). Default False.",
    )


@tool(args_schema=PenArgs)
def set_pen(
    r: int = 255,
    g: int = 255,
    b: int = 255,
    width: int = 2,
    off: bool = False,
) -> str:
    """Control the turtle's drawing pen: colour, width, and up/down state.

    Lift the pen (off=True) before repositioning so you don't draw an
    unwanted line. Lower it again (off=False) when ready to draw.

    Args:
        r:     Red component [0, 255]. Default 255.
        g:     Green component [0, 255]. Default 255.
        b:     Blue component [0, 255]. Default 255.
        width: Line width in pixels [1, 10]. Default 2.
        off:   True = pen up (no drawing). False = pen down (draws).

    Common colours:
      White  r=255, g=255, b=255
      Red    r=255, g=0,   b=0
      Green  r=0,   g=255, b=0
      Blue   r=0,   g=0,   b=255

    Examples:
      set_pen(r=255, g=0, b=0, width=3)   # red pen, width 3
      set_pen(off=True)                    # lift pen before moving
      set_pen(off=False)                   # lower pen to resume drawing
    """
    if not rclpy.ok():
        rclpy.init()

    node = Node(f"ros_langchain_set_pen_{int(time.monotonic() * 1000)}")
    client = node.create_client(SetPen, "/turtle1/set_pen")

    try:
        request = SetPen.Request(r=r, g=g, b=b, width=width, off=int(off))
        success, result = _call_service(node, client, request, "/turtle1/set_pen")

        if not success:
            return result  # error string from helper

        state = "up (not drawing)" if off else "down (drawing)"
        return (
            f"Pen set: colour=rgb({r}, {g}, {b}), "
            f"width={width}px, pen={state}."
        )

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Tool — teleport
# ─────────────────────────────────────────────────────────────────────────────
class TeleportArgs(BaseModel):
    x: float = Field(
        ...,
        ge=ARENA_MIN,
        le=ARENA_MAX,
        description=f"Target x position in metres [{ARENA_MIN}, {ARENA_MAX}].",
    )
    y: float = Field(
        ...,
        ge=ARENA_MIN,
        le=ARENA_MAX,
        description=f"Target y position in metres [{ARENA_MIN}, {ARENA_MAX}].",
    )
    heading_deg: float = Field(
        default=0.0,
        ge=0.0,
        lt=360.0,
        description="New heading in degrees [0, 360). 0=East, 90=North. Default 0.",
    )


@tool(args_schema=TeleportArgs)
def teleport(x: float, y: float, heading_deg: float = 0.0) -> str:
    """Teleport the turtle to an absolute position and heading instantly.

    Does not draw a line regardless of pen state. Use this to:
      - Reposition at the start of a new shape
      - Jump to a known starting point
      - Recover from a navigation error without backtracking

    Args:
        x:           Target x in metres [0, 11].
        y:           Target y in metres [0, 11].
        heading_deg: New heading in degrees [0, 360).
                     0=East, 90=North, 180=West, 270=South. Default 0.

    Note: ROS uses radians internally — the conversion is handled for you.
    Always call get_pose after teleporting to confirm the new position.

    Examples:
      teleport(x=5.5, y=5.5)                  # centre of arena, heading East
      teleport(x=1.0, y=1.0, heading_deg=90)  # bottom-left corner, facing North
    """
    if not rclpy.ok():
        rclpy.init()

    node = Node(f"ros_langchain_teleport_{int(time.monotonic() * 1000)}")
    client = node.create_client(TeleportAbsolute, "/turtle1/teleport_absolute")

    try:
        # Agent works in degrees — convert to radians for ROS
        theta_rad = math.radians(heading_deg)

        request = TeleportAbsolute.Request(x=float(x), y=float(y), theta=theta_rad)
        success, result = _call_service(node, client, request, "/turtle1/teleport_absolute")

        if not success:
            return result  # error string from helper

        pose_str = get_pose.invoke({})

        return (
            f"Teleported to x={x:.2f}, y={y:.2f}, "
            f"heading={heading_deg:.1f}° ({theta_rad:.3f} rad). "
            f"{pose_str}"
        )

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown

# ─────────────────────────────────────────────────────────────────────────────
# Tool — reset
# ─────────────────────────────────────────────────────────────────────────────
@tool
def reset() -> str:
    """Reset the turtle simulator, sending the turtle back to the origin and clearing any drawn lines.

    This is a convenient way to start fresh without having to restart the ROS node.
    Only use this when you want to completely clear the arena and start over. It is not intended for regular navigation, as it disrupts the continuity of the turtle's state.
    """
    if not rclpy.ok():
        rclpy.init()

    node = Node(f"ros_langchain_reset_{int(time.monotonic() * 1000)}")
    client = node.create_client(Empty, "/reset")

    try:
        request = Empty.Request()
        success, result = _call_service(node, client, request, "/reset")

        if not success:
            return result  # error string from helper

        return "Turtle simulator reset successfully."

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown

TOOLS = [get_pose, move_forward, rotate, set_pen, teleport, reset]

print("Tools registered:")
for t in TOOLS:
    first_line = t.description.splitlines()[0]
    print(f"  {t.name:<16} — {first_line}")



def safe_invoke(t, args: dict) -> str:
    """
    Call a tool and return its output as a string.
    If validation fails, return the error message so the agent can self-correct.
    """
    try:
        return str(t.invoke(args))
    except ValidationError as e:
        # Distil pydantic's verbose error into something the model can act on
        msgs = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
        return f"TOOL ERROR — invalid arguments: {'; '.join(msgs)}"
    except Exception as e:
        return f"TOOL ERROR — {type(e).__name__}: {e}"

# ── Exercise 3g ─────────────────────────────────────────────────────────────
# Agent loop with the real tool suite 

global _llm_call_count 

def run_tool_agent(task: str, max_steps: int = 10, verbose: bool = True) -> str:
    """ReAct agent loop with structured tool calls and safe_invoke error handling."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    history = [
        SystemMessage(
            "You are the reasoning core of a TurtleSim robot in an 11 × 11 m arena.\n"
            "The turtle starts at x=5.5, y=5.5, heading=0° (East).\n"
            "Positive heading = counter-clockwise. 90° = North.\n"
            "Use your tools step by step. After completing the task, "
            "call get_pose() and report the final position."
        ),
        HumanMessage(task),
    ]

    for step in range(max_steps):
        response = llm_with_tools.invoke(history)
        history.append(response)

        if verbose:
            print(f"\n── Step {step + 1} ──────────────────────────────────────")
            if response.text:
                print(f"  Model : {response.text}")
            for tc in response.tool_calls:
                print(f"  Tool  : {tc['name']}({tc['args']})")

        if not response.tool_calls:
            if verbose:
                print(f"\n✓ Done: {response.text}")
            return response.text

        for tc in response.tool_calls:
            tool_fn = next((t for t in TOOLS if t.name == tc["name"]), None)
            observation = (
                safe_invoke(tool_fn, tc["args"]) if tool_fn
                else f"TOOL ERROR — unknown tool '{tc['name']}'"
            )
            if verbose:
                print(f"  Obs   : {observation}")

            history.append(ToolMessage(observation, tool_call_id=tc["id"]))
    
    # Mainly for activity 04 to check the number of LLM calls.
    global _llm_call_count
    _llm_call_count = step
    
    return "Max steps reached."


