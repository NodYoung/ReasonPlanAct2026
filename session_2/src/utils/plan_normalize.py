"""
Normalize MovementPlans so drawing always starts at the turtle's live pose.

The Planner LLM sometimes adds teleport steps to empty corners. We strip those
and anchor feasibility checks to where the turtle actually is.
"""

from __future__ import annotations

from src.pydantic_models import MovementPlan, MovementStep, StepAction


def normalize_plan_for_current_pose(
    plan: MovementPlan,
    pose: dict[str, float],
) -> MovementPlan:
    """
    Remove repositioning teleports and set starting_pose to the live turtle pose.

    In-place heading alignment during execution is unchanged — only location
    teleports before drawing are removed.
    """
    kept_steps = [s for s in plan.steps if s.action != StepAction.TELEPORT]
    renumbered = [
        step.model_copy(update={"step_number": index})
        for index, step in enumerate(kept_steps, start=1)
    ]
    removed = len(plan.steps) - len(renumbered)

    normalized = plan.model_copy(
        update={
            "starting_pose": (pose["x"], pose["y"], pose["theta"]),
            "steps": renumbered,
        }
    )

    if removed:
        print(
            f"  [Plan] Removed {removed} teleport step(s) — drawing from current "
            f"pose ({pose['x']:.2f}, {pose['y']:.2f}, "
            f"θ={pose['theta']:.2f})",
            flush=True,
        )

    return sanitize_negative_move_forward(normalized)


def sanitize_negative_move_forward(plan: MovementPlan) -> MovementPlan:
    """
    Rewrite negative move_forward steps so preview simulation matches turtlesim.

    ROS skips zero/negative forward distances; the simulator does not. Without
    this, plan review can approve geometry the turtle never draws.
    """
    new_steps: list[MovementStep] = []
    rewritten = 0
    step_num = 0
    for step in plan.steps:
        if (
            step.action == StepAction.MOVE_FORWARD
            and step.distance is not None
            and step.distance < 0
        ):
            rewritten += 1
            dist = abs(step.distance)
            step_num += 1
            new_steps.append(
                MovementStep(
                    step_number=step_num,
                    action=StepAction.ROTATE,
                    angle_degrees=180.0,
                    description="Turn around (negative move_forward rewritten)",
                )
            )
            step_num += 1
            new_steps.append(
                step.model_copy(
                    update={
                        "step_number": step_num,
                        "distance": dist,
                    }
                )
            )
        else:
            step_num += 1
            new_steps.append(step.model_copy(update={"step_number": step_num}))

    if rewritten:
        print(
            f"  [Plan] Rewrote {rewritten} negative move_forward step(s) as "
            "rotate-180 + positive distance.",
            flush=True,
        )
        return plan.model_copy(update={"steps": new_steps})
    return plan


def align_plan_pen_width(plan: MovementPlan, target_width: int) -> MovementPlan:
    """
    Force every set_pen step (and plan.pen) to use the same line width.

    Used after part planning so wing_left and wing_right match body thickness.
    """
    width = max(1, min(20, int(target_width)))
    if plan.pen.width == width and all(
        step.pen is None or step.pen.width == width for step in plan.steps
    ):
        return plan

    new_steps = []
    for step in plan.steps:
        if step.action == StepAction.SET_PEN and step.pen:
            new_steps.append(
                step.model_copy(update={"pen": step.pen.model_copy(update={"width": width})})
            )
        else:
            new_steps.append(step)

    aligned = plan.model_copy(
        update={
            "pen": plan.pen.model_copy(update={"width": width}),
            "steps": new_steps,
        }
    )
    print(f"  [Plan] Aligned pen width to {width} for all set_pen steps.", flush=True)
    return aligned
