"""
Supervisor — the ONLY orchestrator, and the only agent that talks to the user.

## First visit (one LLM ReAct turn)

Every user message lands here first. The supervisor analyses the request:

- Simple turtle task (move, rotate, pen, clear, reset, questions) — it does
  the work itself with its motion tools and replies. Mission over.
- Drawing task (a shape or an object made of shapes) — it calls the
  ``start_drawing`` tool to hand the work to the specialists.

## Revisits (zero LLM calls)

Every specialist (blueprint planner, part planner, reviewer, navigator)
returns here. The drawing pipeline order is fixed, so routing is a plain
deterministic state machine over ``MissionContext`` — no tokens are spent
deciding "who's next":

    blueprint → plan part → review → draw → next part → ... → finish

The reviewer's verdict is also applied here: approve → navigator; reject →
back to the part planner with feedback + the rejected trial image; retries
exhausted → skip the part or draw anyway (``[review].fail_action``).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolRuntime

from src.app_settings import get_config
from src.llm import get_llm
from src.tools.turtle_motion_tools import (
    clear_canvas_tool,
    get_turtle_pose,
    move_forward_tool,
    reset_world_tool,
    rotate_tool,
    set_pen,
)
from src.utils.agent_runtime import AGENT_RUNTIME_CONFIG_KEY, AgentRuntime, get_agent_runtime
from src.utils.cli_spinner import spinner
from src.utils.mission_state import MissionContext, say
from src.utils.ros_bridge import TurtleBridge
from src.utils.turtle_park import park_turtle_for_clear_view
from src.workflow.types import WorkflowState

RECURSION_LIMIT = 16

SYSTEM_PROMPT = """You are the Supervisor of a multi-agent turtlesim drawing system.
You are the ONLY agent that speaks to the user. Be brief and friendly.

Decide how to handle the user's message:

1. DRAWING request — the user wants a shape or object drawn (square, circle,
   house, cat, rocket, ...): call start_drawing ONCE with a short description.
   Specialist agents plan and draw it — do NOT draw it yourself with motion tools.
2. SIMPLE turtle task — one-off actions like "move forward 0.5", "rotate 90",
   "set pen to red" (red=255,0,0 blue=0,0,255 green=0,255,0), "clear the canvas",
   "reset", "where are you?": execute it NOW with your motion tools.
3. QUESTION — answer directly. Useful facts: canvas is 11x11 metres
   (usable 0.25-10.75), smallest practical move is 0.01 m, pen is RGB 0-255.

Always end with ONE short reply to the user (your final message is shown to them).
"""


@tool
def start_drawing(runtime: ToolRuntime, description: str) -> str:
    """Hand a drawing request to the specialist pipeline (planner, reviewer, navigator).

    Call this once for any request to draw a shape or object. ``description``
    is a short summary of what to draw.
    """
    get_agent_runtime(runtime).mission.drawing = True
    return f"Drawing pipeline started for: {description}. Reply with a one-line kickoff."


@lru_cache(maxsize=1)
def build_agent() -> CompiledStateGraph:
    return create_agent(
        get_llm("supervisor"),
        [
            start_drawing,
            get_turtle_pose,
            move_forward_tool,
            rotate_tool,
            set_pen,
            clear_canvas_tool,
            reset_world_tool,
        ],
        system_prompt=SYSTEM_PROMPT,
        name="supervisor",
    )


def _final_text(result: dict) -> str:
    """Extract the assistant's final reply text from a ReAct result."""
    content = result["messages"][-1].content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(parts).strip()
    return ""


def _finish_drawing(ctx: MissionContext, bridge: TurtleBridge) -> None:
    """Park the turtle and print the mission summary."""
    if get_config().park_after_draw and ctx.path_tracker.segments:
        try:
            say(ctx, park_turtle_for_clear_view(bridge, ctx.path_tracker))
        except Exception as exc:  # noqa: BLE001
            say(ctx, f"Could not park turtle ({exc}).")
    done = ", ".join(ctx.parts_completed) or "none"
    if ctx.parts_skipped:
        say(
            ctx,
            f"Drawing finished. Completed: {done}. "
            f"Skipped after failed reviews: {', '.join(ctx.parts_skipped)}.",
        )
    else:
        say(ctx, f"Drawing complete: {done}.")


def _handle_navigator_result(ctx: MissionContext) -> None:
    """Bookkeep after the navigator drew the current part."""
    ctx.execution_pending = False
    part = ctx.current_part() or "?"
    report = ctx.execution_report
    total = len(ctx.parts or [])
    if report and report.success:
        ctx.parts_completed.append(part)
        say(ctx, f"Part [{part}] drawn ({ctx.part_index + 1}/{total}).")
    else:
        detail = report.message if report else "no execution report"
        ctx.parts_skipped.append(part)
        say(ctx, f"Part [{part}] failed while drawing ({detail}) — moving on.")
    ctx.part_index += 1
    ctx.reset_part_pipeline()


def _apply_review_verdict(ctx: MissionContext) -> str:
    """Turn the reviewer's verdict into the next routing step."""
    review = ctx.review
    assert review is not None
    cfg = get_config().review
    part = ctx.current_part() or "?"

    if review.approved:
        ctx.plan_approved = True
        ctx.last_good_image = ctx.trial_image
        return "navigator"

    if ctx.trial >= cfg.max_trials:
        if cfg.fail_action == "draw_anyway":
            say(ctx, f"Part [{part}] review still failing after {ctx.trial} trials — drawing anyway.")
            ctx.plan_approved = True
            return "navigator"
        say(ctx, f"Part [{part}] skipped after {ctx.trial} failed trials.")
        ctx.parts_skipped.append(part)
        ctx.part_index += 1
        ctx.reset_part_pipeline()
        return "continue"  # let the pipeline pick the next step

    ctx.feedback = review.feedback
    ctx.trial += 1
    ctx.movement_plan = None
    ctx.review = None
    return "planner_part"


def _pipeline_next(ctx: MissionContext, bridge: TurtleBridge) -> str:
    """Deterministic drawing pipeline router — runs on every revisit, no LLM."""
    while True:
        if ctx.execution_pending:
            _handle_navigator_result(ctx)

        if ctx.mission_failed:
            say(ctx, "Mission stopped — see messages above.")
            return "finish"

        if ctx.parts is None:
            return "planner_blueprint"

        if ctx.current_part() is None:
            _finish_drawing(ctx, bridge)
            return "finish"

        if ctx.movement_plan is None:
            return "planner_part"

        if not ctx.plan_approved:
            if not get_config().review.enabled:
                ctx.plan_approved = True
                return "navigator"
            if ctx.review is None:
                return "plan_reviewer"
            step = _apply_review_verdict(ctx)
            if step != "continue":
                return step
            continue  # part was skipped — re-evaluate from the top

        return "navigator"


def build_graph_node(bridge: TurtleBridge) -> Callable[[WorkflowState], dict]:
    def node(state: WorkflowState) -> dict:
        ctx = state["ctx"]

        # Revisit during a drawing mission — deterministic routing only.
        if ctx.drawing:
            return {"next_agent": _pipeline_next(ctx, bridge)}

        # First visit — one ReAct turn to analyse and act on the request.
        with spinner("Supervisor thinking"):
            result = build_agent().invoke(
                {"messages": [HumanMessage(content=ctx.user_request)]},
                config={
                    "recursion_limit": RECURSION_LIMIT,
                    "configurable": {AGENT_RUNTIME_CONFIG_KEY: AgentRuntime(bridge, ctx)},
                },
            )
        reply = _final_text(result)
        if reply:
            say(ctx, reply)
        if ctx.drawing:
            return {"next_agent": "planner_blueprint"}
        return {"next_agent": "finish"}

    return node
