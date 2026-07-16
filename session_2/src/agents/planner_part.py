"""
Part planner — turtle movement steps for ONE part of the drawing.

Runs once per planning trial: one structured LLM call, no tools. All the
context the model needs (live pose, canvas bounds, pen guidance, progress)
is computed deterministically and injected into the prompt — no tool
round-trips, no wasted tokens.

- Trial 1 of a later part also gets the last approved trial image so the
  planner can see the ink already on the canvas.
- Replans (after a rejected review) get the reviewer's feedback plus the
  rejected trial image (``captures/{shape}_{part}_trial{N}.png``).

The node normalizes the plan (strip teleports, fix negative moves, align pen
width with existing ink) and stores it on the mission notebook.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm
from src.pydantic_models import MovementPlanOutput
from src.utils.cli_spinner import spinner
from src.utils.drawing_render import png_path_to_data_url
from src.utils.mission_state import MissionContext, say
from src.utils.plan_normalize import align_plan_pen_width, normalize_plan_for_current_pose
from src.utils.ros_bridge import TurtleBridge
from src.utils.turtle_control import DEFAULT_PEN_WIDTH, pose_as_dict
from src.workflow.types import WorkflowState

SYSTEM_PROMPT = """You plan turtle movement steps for ONE part of a turtlesim drawing.

Canvas: 11x11 metres, usable x,y = 0.25-10.75. All ink MUST stay inside.
Allowed step actions: set_pen, move_forward, rotate. Nothing else.
- set_pen: RGB 0-255, width 1-20, off=true lifts the pen (move without drawing).
- rotate: positive degrees = counter-clockwise.
- The turtle starts at the given live pose — plan from there. To reposition
  without ink, use set_pen(off=true), move/rotate, then set_pen(off=false).

Quality rules:
- Draw ONLY the current part. Never redraw parts already on the canvas.
- Attach parts cleanly: endpoints should meet the intended boundaries
  (roof sits on top of body, door on the bottom edge, ...).
- Respect the side_length budget; keep proportions sensible.
- Every step description starts with [part_name].

If reviewer feedback is given, each line of it is a MANDATORY fix — follow it
exactly. The attached image shows the rejected attempt (plus correct existing
ink). Fix only the current part.

Return a MovementPlanOutput.
"""


def _pen_width_target(ctx: MissionContext) -> int:
    """Dominant pen width of ink already drawn, so parts stay consistent."""
    if not ctx.path_tracker.segments:
        return DEFAULT_PEN_WIDTH
    counts = Counter(seg.pen_width for seg in ctx.path_tracker.segments)
    return counts.most_common(1)[0][0]


def build_kickoff_message(ctx: MissionContext, pose: dict[str, float]) -> HumanMessage:
    """Assemble the planning prompt (deterministic context + optional image)."""
    part = ctx.current_part() or "?"
    done = ", ".join(ctx.parts_completed) or "none"
    remaining = ", ".join((ctx.parts or [])[ctx.part_index + 1 :]) or "none"

    lines = [
        f"User request: {ctx.user_request}",
        f"Object: {ctx.shape_name} — DRAW THIS PART NOW: [{part}] (trial {ctx.trial})",
        f"Parts already on canvas: {done}",
        f"Parts still to draw later: {remaining}",
        f"Overall size budget: side_length ≈ {ctx.side_length:g} m",
        f"Live turtle pose: x={pose['x']:.2f}, y={pose['y']:.2f}, theta={pose['theta']:.2f} rad",
        f"Use pen width {_pen_width_target(ctx)} for every set_pen step.",
    ]

    image_path: str | None = None
    if ctx.feedback:
        lines += [
            "",
            f"REVIEWER FEEDBACK — mandatory fixes for [{part}]:",
            ctx.feedback,
        ]
        image_path = ctx.trial_image
        if image_path:
            lines.append("Attached image: the REJECTED attempt (existing ink is correct — fix only the new part).")
    elif ctx.last_good_image:
        image_path = ctx.last_good_image
        lines.append("Attached image: current canvas with all approved parts so far.")

    text = "\n".join(lines)
    image_url = png_path_to_data_url(image_path) if image_path else None
    if image_url:
        return HumanMessage(
            content=[
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
    return HumanMessage(content=text)


def build_graph_node(bridge: TurtleBridge) -> Callable[[WorkflowState], dict]:
    def node(state: WorkflowState) -> dict:
        ctx = state["ctx"]
        part = ctx.current_part() or "?"
        pose = pose_as_dict(bridge)

        llm = get_llm("planner").with_structured_output(MovementPlanOutput)
        try:
            with spinner(f"Planning part [{part}]"):
                output = llm.invoke(
                    [SystemMessage(content=SYSTEM_PROMPT), build_kickoff_message(ctx, pose)]
                )
        except Exception as exc:  # noqa: BLE001
            output = None
            say(ctx, f"[Planner] Planning part [{part}] failed ({exc}).")

        if not isinstance(output, MovementPlanOutput) or not output.steps:
            ctx.mission_failed = True
            return {}

        plan = output.to_movement_plan((pose["x"], pose["y"], pose["theta"]))
        plan = normalize_plan_for_current_pose(plan, pose)
        if ctx.path_tracker.segments:
            plan = align_plan_pen_width(plan, _pen_width_target(ctx))

        ctx.movement_plan = plan
        ctx.feedback = ""
        say(ctx, f"[Planner] Part [{part}] plan ready: {len(plan.steps)} steps (trial {ctx.trial}).")
        return {}

    return node
