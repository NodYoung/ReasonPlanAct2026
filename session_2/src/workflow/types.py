"""Shared state types for the outer LangGraph workflow."""

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
    """State dictionary flowing through the outer LangGraph."""

    ctx: MissionContext
    next_agent: NextAgent
