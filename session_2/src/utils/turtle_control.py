"""通过 ``TurtleBridge`` 进行底层 turtlesim 控制（非 LangChain 工具）。"""

from __future__ import annotations

from src.utils.ros_bridge import TurtleBridge, TurtlePose

# 共享的 turtlesim 限制——从这里导入，不要在别处重复定义。
MIN_STEP_METERS = 0.01  # 实际可行的最小移动量
DEFAULT_PEN_WIDTH = 2


def pose_as_dict(bridge: TurtleBridge) -> dict[str, float]:
    pose: TurtlePose = bridge.get_pose()
    return {"x": pose.x, "y": pose.y, "theta": pose.theta}


def set_pen_color(
    bridge: TurtleBridge,
    r: int,
    g: int,
    b: int,
    width: int = DEFAULT_PEN_WIDTH,
    off: bool = False,
) -> str:
    return bridge.set_pen(r, g, b, width, off)


def move_forward(bridge: TurtleBridge, distance: float) -> str:
    return bridge.move_forward(distance)


def rotate(bridge: TurtleBridge, angle_degrees: float) -> str:
    return bridge.rotate(angle_degrees)


def teleport(bridge: TurtleBridge, x: float, y: float, theta: float) -> str:
    return bridge.teleport(x, y, theta)


def clear_canvas(bridge: TurtleBridge) -> str:
    return bridge.clear_canvas()


def reset_world(bridge: TurtleBridge) -> str:
    return bridge.reset_world()
