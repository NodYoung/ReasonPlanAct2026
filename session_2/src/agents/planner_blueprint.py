"""
Blueprint planner — split a drawing request into ordered parts.

Runs once per drawing mission, right after the supervisor starts the
pipeline. One structured LLM call, no tools:

    "draw a house"  →  parts: body → roof → door, side_length ≈ 3 m

The node writes the blueprint onto the mission notebook; on failure it
marks the mission failed so the supervisor stops cleanly.
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

Rules:
- Large base shapes FIRST, small details LAST (house: body, roof, door).
- Short snake_case part names.
- Simple shapes are ONE part (square -> square). Use the fewest parts that
  still make the object recognisable.
- side_length is the overall size budget in metres. The canvas is 11x11 m
  (usable 0.25-10.75). Prefer 2-4 m unless the user asked for a specific size.

Return a ShapeBlueprint.
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
