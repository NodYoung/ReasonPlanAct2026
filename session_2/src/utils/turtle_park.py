"""Move the turtle out of the way so the finished drawing is easy to see."""

from __future__ import annotations

from src.utils.turtle_control import set_pen_color, teleport
from src.utils.canvas_geometry import CANVAS_MAX, MARGIN
from src.utils.path_tracker import PathTracker
from src.utils.ros_bridge import TurtleBridge

_VIEW_MARGIN = MARGIN + 0.25
_CORNERS = (
    (_VIEW_MARGIN, _VIEW_MARGIN),
    (CANVAS_MAX - _VIEW_MARGIN, _VIEW_MARGIN),
    (_VIEW_MARGIN, CANVAS_MAX - _VIEW_MARGIN),
    (CANVAS_MAX - _VIEW_MARGIN, CANVAS_MAX - _VIEW_MARGIN),
)


def park_turtle_for_clear_view(
    bridge: TurtleBridge,
    path_tracker: PathTracker | None = None,
) -> str:
    """
    Pen up and teleport to the canvas corner farthest from the drawing centre.
    """
    set_pen_color(bridge, 255, 255, 255, 2, off=True)

    target_x, target_y = _CORNERS[0]
    bbox = path_tracker.bounding_box() if path_tracker else None
    if bbox:
        cx = (bbox["min_x"] + bbox["max_x"]) / 2
        cy = (bbox["min_y"] + bbox["max_y"]) / 2
        target_x, target_y = max(
            _CORNERS,
            key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2,
        )

    teleport(bridge, target_x, target_y, 0.0)
    return (
        f"Turtle parked at ({target_x:.1f}, {target_y:.1f}) pen up "
        f"— drawing should be fully visible."
    )
