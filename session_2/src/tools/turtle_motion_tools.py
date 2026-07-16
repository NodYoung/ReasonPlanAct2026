"""
LangChain ``@tool`` wrappers for turtle pose, pen, and motion.

Used by the **Supervisor** for one-off actions ("move forward", "set pen",
"clear canvas", …). Each tool resolves ``AgentRuntime`` via
``get_agent_runtime`` and calls a ``utils/turtle_control`` helper on the ROS
bridge. Shape drawing goes through the deterministic Navigator node, not
these one-shot motion tools.
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
    """Return the turtle's current x, y position (metres) and heading (radians)."""
    return json.dumps(pose_as_dict(get_agent_runtime(runtime).bridge), indent=2)


@tool
def move_forward_tool(distance: float, runtime: ToolRuntime) -> str:
    """Move the turtle forward by ``distance`` metres (minimum ~0.01 m)."""
    if distance < MIN_STEP_METERS:
        return (
            f"Distance {distance} m is below practical minimum ({MIN_STEP_METERS} m). "
            "Use a slightly larger step."
        )
    return move_forward(get_agent_runtime(runtime).bridge, distance)


@tool
def rotate_tool(angle_degrees: float, runtime: ToolRuntime) -> str:
    """Rotate the turtle in place. Positive = counter-clockwise degrees."""
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
    """Set pen colour (0–255 RGB), line width (1–20), and pen up (off=True) or down."""
    return set_pen_color(get_agent_runtime(runtime).bridge, r, g, b, width, off)


@tool
def clear_canvas_tool(runtime: ToolRuntime) -> str:
    """Clear all pen trails on the canvas. Turtle pose is unchanged."""
    return clear_canvas(get_agent_runtime(runtime).bridge)


@tool
def reset_world_tool(runtime: ToolRuntime) -> str:
    """Full turtlesim reset — turtle returns to centre, canvas cleared."""
    return reset_world(get_agent_runtime(runtime).bridge)
