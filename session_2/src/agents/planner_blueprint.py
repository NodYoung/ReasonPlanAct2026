"""
Blueprint planner（蓝图规划器）—— 把绘图请求拆分为有序的部件。

每个绘图任务运行一次，就在 supervisor 启动流水线之后。一次结构化 LLM 调用，
不使用工具：

    "画一个房子"  →  部件：body（主体）→ roof（屋顶）→ door（门），side_length ≈ 3 m

本节点把蓝图写入任务笔记本；若失败，则标记任务失败，让 supervisor 干净地停止。
"""

from __future__ import annotations

import re
from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm
from src.pydantic_models import ShapeBlueprint
from src.utils.cli_spinner import spinner
from src.utils.mission_state import say
from src.utils.ros_bridge import TurtleBridge
from src.workflow.types import WorkflowState

SYSTEM_PROMPT = """You decompose a turtlesim drawing request into ordered parts.
（你把一个 turtlesim 绘图请求拆解为有序的若干部件。）

规则（Rules）:
- 大的基础形状在前（FIRST），小的细节在后（LAST）（house: body, roof, door）。
- 用简短的 snake_case 部件名。
- 简单形状就是一个部件（square -> square）。用能让物体仍可辨认的最少部件数。
- side_length 是以米为单位的整体尺寸预算。画布为 11x11 m
  （可用范围 0.25-10.75）。除非用户指定了尺寸，否则优先取 2-4 m。

返回一个 ShapeBlueprint。
"""


def _snake(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name.lower()).strip("_") or "shape"


def build_graph_node(bridge: TurtleBridge) -> Callable[[WorkflowState], dict]:
    def node(state: WorkflowState) -> dict:
        ctx = state["ctx"]
        llm = get_llm("blueprint").with_structured_output(ShapeBlueprint)
        try:
            with spinner("Blueprint planning"):
                blueprint = llm.invoke(
                    [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=f"User request: {ctx.user_request}"),
                    ]
                )
        except Exception as exc:  # noqa: BLE001
            blueprint = None
            say(ctx, f"[Blueprint] Planning failed ({exc}).")

        if not isinstance(blueprint, ShapeBlueprint) or not blueprint.parts:
            ctx.mission_failed = True
            return {}

        ctx.shape_name = _snake(blueprint.shape)
        ctx.parts = [_snake(p) for p in blueprint.parts]
        ctx.side_length = float(blueprint.side_length)
        say(
            ctx,
            f"[Blueprint] {ctx.shape_name}: {' → '.join(ctx.parts)} "
            f"(≈{ctx.side_length:g} m).",
        )
        return {}

    return node
