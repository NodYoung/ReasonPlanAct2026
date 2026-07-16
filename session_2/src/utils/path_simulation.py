"""
Dry-run a ``MovementPlan`` in software — no turtle, no ROS.

Three modules work together for "what's on the canvas?":

| Module | Role |
|--------|------|
| ``path_tracker`` | **Data** — list of line segments (ink memory) |
| ``path_simulation`` (this file) | **Predict** — walk a plan's steps in math only |
| ``drawing_render`` | **Picture** — turn segment data into a PNG |

The plan reviewer uses ``simulate_plan_to_path_tracker`` to build the
proposed part's ink, merges it with the real ink already drawn, renders the
trial PNG, and uses ``bbox_fits_canvas`` for the free out-of-bounds check.
The real turtle never moves during any of this.
"""

from __future__ import annotations

import math

from src.pydantic_models import MovementPlan, StepAction
from src.utils.path_tracker import PathPoint, PathSegment, PathTracker

_DEFAULT_PEN_RGB = (255, 255, 255)


def bbox_fits_canvas(
    bbox: dict[str, float] | None,
    *,
    canvas_min: float,
    canvas_max: float,
    margin: float,
) -> tuple[bool, list[str]]:
    """Return whether a simulated bbox stays inside the canvas (with margin)."""
    if bbox is None:
        return False, ["No pen-down ink found in plan."]
    lo = canvas_min + margin
    hi = canvas_max - margin
    issues: list[str] = []

    if bbox["min_x"] < lo:
        issues.append(f"Drawing extends past left edge (min x={bbox['min_x']:.2f} m)")
    if bbox["max_x"] > hi:
        issues.append(f"Drawing extends past right edge (max x={bbox['max_x']:.2f} m)")
    if bbox["min_y"] < lo:
        issues.append(f"Drawing extends past bottom edge (min y={bbox['min_y']:.2f} m)")
    if bbox["max_y"] > hi:
        issues.append(f"Drawing extends past top edge (max y={bbox['max_y']:.2f} m)")

    return len(issues) == 0, issues


def simulate_plan_to_path_tracker(
    plan: MovementPlan,
    start_pose: tuple[float, float, float],
) -> PathTracker:
    """Dry-run a plan and build a fresh ``PathTracker`` (coloured segments)."""
    x, y, theta = start_pose
    pen_down = False
    pen_color = _DEFAULT_PEN_RGB
    pen_width = 2
    last_point: PathPoint | None = None
    tracker = PathTracker()

    for step in plan.steps:
        if step.action == StepAction.SET_PEN and step.pen is not None:
            pen_color = (step.pen.r, step.pen.g, step.pen.b)
            pen_width = step.pen.width
            pen_down = not step.pen.off
            if not pen_down:
                last_point = None

        elif step.action == StepAction.MOVE_FORWARD and step.distance is not None:
            dist = step.distance
            new_x = x + dist * math.cos(theta)
            new_y = y + dist * math.sin(theta)
            if pen_down:
                start = last_point if last_point else PathPoint(x=x, y=y)
                end = PathPoint(x=new_x, y=new_y)
                tracker.segments.append(
                    PathSegment(
                        start=start,
                        end=end,
                        actual_distance=dist,
                        color=pen_color,
                        pen_width=pen_width,
                    )
                )
                last_point = end
            x, y = new_x, new_y

        elif step.action == StepAction.ROTATE and step.angle_degrees is not None:
            theta += math.radians(step.angle_degrees)
            while theta > math.pi:
                theta -= 2 * math.pi
            while theta < -math.pi:
                theta += 2 * math.pi

        elif step.action == StepAction.TELEPORT:
            continue

    return tracker
