"""外层 LangGraph 工作流的共享状态类型。"""

from __future__ import annotations

from typing import Literal, TypedDict

from src.utils.mission_state import MissionContext

NextAgent = Literal[
    "planner_blueprint",
    "planner_part",
    "plan_reviewer",
    "navigator",
    "finish",
]


class WorkflowState(TypedDict, total=False):
    """在外层 LangGraph 中流转的状态字典。"""

    ctx: MissionContext
    next_agent: NextAgent
