"""
用于海龟位姿、画笔和运动的 LangChain ``@tool`` 封装。

由 **Supervisor（主管）** 用于一次性动作（"前进"、"设置画笔"、
"清空画布" 等）。每个工具通过 ``get_agent_runtime`` 解析出 ``AgentRuntime``，
并在 ROS 桥接上调用 ``utils/turtle_control`` 中的辅助函数。形状绘制走的是
确定性的 Navigator 节点，而非这些一次性的运动工具。
"""
from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt import ToolRuntime

from src.utils.agent_runtime import get_agent_runtime
from src.utils.turtle_control import DEFAULT_PEN_WIDTH, MIN_STEP_METERS
from src.utils.turtle_control import (
    clear_canvas,
    move_forward,
    pose_as_dict,
    reset_world,
    rotate,
    set_pen_color,
)


@tool
def get_turtle_pose(runtime: ToolRuntime) -> str:
    """返回海龟当前的 x、y 位置（米）与朝向 heading（弧度）。"""
    return json.dumps(pose_as_dict(get_agent_runtime(runtime).bridge), indent=2)


@tool
def move_forward_tool(distance: float, runtime: ToolRuntime) -> str:
    """让海龟向前移动 ``distance`` 米（最小约 0.01 m）。"""
    if distance < MIN_STEP_METERS:
        return (
            f"Distance {distance} m is below practical minimum ({MIN_STEP_METERS} m). "
            "Use a slightly larger step."
        )
    return move_forward(get_agent_runtime(runtime).bridge, distance)


@tool
def rotate_tool(angle_degrees: float, runtime: ToolRuntime) -> str:
    """让海龟原地旋转。正值 = 逆时针度数。"""
    return rotate(get_agent_runtime(runtime).bridge, angle_degrees)


@tool
def set_pen(
    r: int,
    g: int,
    b: int,
    width: int = DEFAULT_PEN_WIDTH,
    off: bool = False,
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,  # type: ignore[assignment]
) -> str:
    """设置画笔颜色（0–255 RGB）、线宽（1–20），以及抬笔 off=True 或落笔。"""
    return set_pen_color(get_agent_runtime(runtime).bridge, r, g, b, width, off)


@tool
def clear_canvas_tool(runtime: ToolRuntime) -> str:
    """清除画布上所有画笔轨迹。海龟位姿保持不变。"""
    return clear_canvas(get_agent_runtime(runtime).bridge)


@tool
def reset_world_tool(runtime: ToolRuntime) -> str:
    """完整重置 turtlesim —— 海龟回到中心，画布清空。"""
    return reset_world(get_agent_runtime(runtime).bridge)
