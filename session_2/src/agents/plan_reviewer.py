"""
Plan reviewer — draw the plan on a canvas image, look at it, approve or fix.

For each planning trial of each part:

1. Simulate the plan's ink (no turtle moves) and render it ON TOP of the ink
   already drawn — then save it as ``captures/{shape}_{part}_trial{N}.png``.
   These are the ONLY images the app saves, so the captures folder tells the
   story of the drawing: which part went wrong, how it was fixed, and how the
   picture grew part by part.
2. Free deterministic checks first (no LLM tokens): empty plans and
   off-canvas ink are rejected immediately with concrete feedback.
3. Otherwise ONE vision LLM call: does this image make sense for the
   requested drawing? Approve, or reject with fix instructions.

On reject, the supervisor sends the feedback and the saved trial image back
to the part planner for the next trial.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm
from src.pydantic_models import PlanReview
from src.utils.canvas_geometry import CANVAS_MAX, CANVAS_MIN, MARGIN
from src.utils.cli_spinner import spinner
from src.utils.drawing_render import png_path_to_data_url, render_drawing, save_image
from src.utils.mission_state import MissionContext, say
from src.utils.path_simulation import bbox_fits_canvas, simulate_plan_to_path_tracker
from src.utils.path_tracker import PathTracker
from src.utils.ros_bridge import TurtleBridge
from src.utils.turtle_control import pose_as_dict
from src.workflow.types import WorkflowState

SYSTEM_PROMPT = """You review a simulated turtlesim drawing BEFORE the real turtle draws it.

The image shows the 11x11 m canvas (blue background) with the ink already
drawn PLUS the proposed new part. Judge whether the picture makes sense for
the user's request:

- Is the new part in the right place, at the right scale, correctly attached?
  (roof on top of the body, door on the bottom edge, cockpit at the front, ...)
- Does the drawing so far still look like the requested object?
- Reject floating/detached parts, wrong orientation, wrong region, messy
  geometry, or anything you would not call recognisable.

Approve only when the image is a plausible step toward the requested drawing.
On reject, feedback must be imperative fix instructions the planner can follow,
one concrete action per line (what is wrong -> what to draw instead, naming
canvas regions/edges). Never vague criticism.

Return a PlanReview.
"""


def _merged_preview(ctx: MissionContext, start_pose: tuple[float, float, float]) -> tuple[PathTracker, PathTracker]:
    """Simulate the current plan; return (new ink only, existing + new ink)."""
    assert ctx.movement_plan is not None
    new_ink = simulate_plan_to_path_tracker(ctx.movement_plan, start_pose)
    merged = PathTracker()
    merged.segments = list(ctx.path_tracker.segments) + list(new_ink.segments)
    return new_ink, merged


def _reject(ctx: MissionContext, part: str, feedback: str) -> None:
    ctx.review = PlanReview(approved=False, feedback=feedback)
    say(ctx, f"[Reviewer] Part [{part}] trial {ctx.trial} rejected: {feedback.splitlines()[0]}")


def build_graph_node(bridge: TurtleBridge) -> Callable[[WorkflowState], dict]:
    def node(state: WorkflowState) -> dict:
        ctx = state["ctx"]
        part = ctx.current_part() or "?"
        pose = pose_as_dict(bridge)
        start_pose = (pose["x"], pose["y"], pose["theta"])

        new_ink, merged = _merged_preview(ctx, start_pose)

        # Free check 1: a plan that draws nothing.
        if not new_ink.segments:
            _reject(
                ctx,
                part,
                "The plan draws nothing — no pen-down ink. Start with set_pen "
                "(off=false), then move_forward steps that form the part.",
            )
            return {}

        # Save the trial image (the only images this app writes).
        out = Path("captures") / f"{ctx.shape_name}_{part}_trial{ctx.trial}.png"
        save_image(render_drawing(merged), out)
        ctx.trial_image = str(out.resolve())
        say(ctx, f"[Reviewer] Trial image saved: {out}")

        # Free check 2: ink outside the canvas (replaces the old safety agent).
        bbox = new_ink.bounding_box()
        fits, issues = bbox_fits_canvas(
            bbox, canvas_min=CANVAS_MIN, canvas_max=CANVAS_MAX, margin=MARGIN
        )
        if not fits:
            _reject(
                ctx,
                part,
                "Plan goes off-canvas: "
                + "; ".join(issues)
                + ". Keep all ink inside 0.25-10.75 m — use shorter strokes or "
                "reposition (pen up) toward the canvas centre first.",
            )
            return {}

        # One vision LLM call on the trial image.
        context = (
            f"User request: {ctx.user_request}\n"
            f"Object: {ctx.shape_name} | new part in this image: [{part}] "
            f"(trial {ctx.trial})\n"
            f"Parts already drawn: {', '.join(ctx.parts_completed) or 'none'}\n"
            f"Parts coming later: {', '.join((ctx.parts or [])[ctx.part_index + 1:]) or 'none'}"
        )
        llm = get_llm("reviewer").with_structured_output(PlanReview)
        try:
            with spinner(f"Reviewing part [{part}] trial {ctx.trial}"):
                review = llm.invoke(
                    [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(
                            content=[
                                {"type": "text", "text": context},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": png_path_to_data_url(ctx.trial_image)},
                                },
                            ]
                        ),
                    ]
                )
        except Exception as exc:  # noqa: BLE001
            say(ctx, f"[Reviewer] Review call failed ({exc}) — approving without review.")
            review = PlanReview(approved=True, feedback="Review unavailable.")

        if not isinstance(review, PlanReview):
            review = PlanReview(approved=True, feedback="Review unavailable.")
        ctx.review = review
        if review.approved:
            say(ctx, f"[Reviewer] Part [{part}] trial {ctx.trial} approved. {review.feedback}")
        else:
            say(ctx, f"[Reviewer] Part [{part}] trial {ctx.trial} rejected: {review.feedback}")
        return {}

    return node
