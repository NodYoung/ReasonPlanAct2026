"""
Record what the turtle has drawn — the app's memory of ink on the canvas.

## Overview

Turtlesim tells us the turtle's **pose** (x, y, heading) and pen commands.
It does **not** give us a screenshot or a list of lines already on the canvas.

``PathTracker`` rebuilds that picture ourselves: every time the pen is down and
the turtle moves, we store one straight **line segment** (start point, end point,
colour, width).

Think of it as a sketchbook layer that grows during the mission:

- Navigator draws on the real turtle → segments are appended here
- Plan simulation (dry-run) → builds a temporary ``PathTracker`` the same way
- PNG previews, plan review, planner context → all read from segment lists

One ``PathTracker`` lives on ``MissionContext.path_tracker`` for the whole
mission. It accumulates ink across parts (body, then head, then ears, …).

## Why this is necessary

Without it, agents would only know *where the turtle is now*, not *what is
already drawn*. Multi-part shapes need the second thing — e.g. placing ears on
the head requires knowing where the head lines are.

## Main types

- ``PathPoint`` — one (x, y) coordinate in metres
- ``PathSegment`` — one pen-down stroke (line from A to B)
- ``PathTracker`` — list of segments + current pen state; updated by
  ``plan_execution.execute_plan_deterministic`` and ``path_simulation``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_DEFAULT_PEN_RGB = (255, 255, 255)


@dataclass
class PathPoint:
    """A single point on the 11×11 m canvas (world coordinates)."""

    x: float
    y: float


@dataclass
class PathSegment:
    """One pen-down stroke: a straight line from ``start`` to ``end``."""

    start: PathPoint
    end: PathPoint
    requested_distance: float | None = None
    actual_distance: float = 0.0
    color: tuple[int, int, int] = _DEFAULT_PEN_RGB
    pen_width: int = 2


@dataclass
class PathTracker:
    """
    Running log of pen-down ink for one mission.

    Updated as steps execute (or simulate). Use ``segments`` for rendering
    PNGs, bounding-box checks, and “what's already on the canvas?” context.
    """

    pen_down: bool = False
    segments: list[PathSegment] = field(default_factory=list)
    _last_point: PathPoint | None = None
    _pen_color: tuple[int, int, int] = _DEFAULT_PEN_RGB
    _pen_width: int = 2

    def set_pen(self, r: int, g: int, b: int, width: int = 2, *, off: bool = False) -> None:
        """Update active pen colour/width and pen-down state."""
        self._pen_color = (r, g, b)
        self._pen_width = width
        self.pen_down = not off
        if off:
            self._last_point = None

    def set_pen_down(self, down: bool) -> None:
        """Legacy toggle — prefer set_pen() to retain colour."""
        self.pen_down = down
        if not down:
            self._last_point = None

    def mark_pose(self, x: float, y: float) -> None:
        """Call after each movement while pen is down."""
        if not self.pen_down:
            return
        point = PathPoint(x=x, y=y)
        if self._last_point is not None:
            dist = math.hypot(point.x - self._last_point.x, point.y - self._last_point.y)
            if dist > 1e-4:
                self.segments.append(
                    PathSegment(
                        start=self._last_point,
                        end=point,
                        actual_distance=dist,
                        color=self._pen_color,
                        pen_width=self._pen_width,
                    )
                )
        self._last_point = point

    def start_segment(self, x: float, y: float, requested_distance: float | None) -> None:
        """Begin tracking a forward move from (x, y)."""
        if not self.pen_down:
            return
        self._last_point = PathPoint(x=x, y=y)
        self._pending_requested = requested_distance

    def finish_segment(self, x: float, y: float) -> float:
        """End forward move at (x, y); return actual distance traveled."""
        if not self.pen_down or self._last_point is None:
            return 0.0
        end = PathPoint(x=x, y=y)
        dist = math.hypot(end.x - self._last_point.x, end.y - self._last_point.y)
        seg = PathSegment(
            start=self._last_point,
            end=end,
            requested_distance=getattr(self, "_pending_requested", None),
            actual_distance=dist,
            color=self._pen_color,
            pen_width=self._pen_width,
        )
        self.segments.append(seg)
        self._last_point = end
        return dist

    def clear(self) -> None:
        """Drop all segments and reset pen state (e.g. new mission)."""

        self.segments.clear()
        self._last_point = None
        self.pen_down = False
        self._pen_color = _DEFAULT_PEN_RGB
        self._pen_width = 2

    def bounding_box(self) -> dict[str, float] | None:
        """Axis-aligned box around all ink, or ``None`` if the canvas is empty."""

        if not self.segments:
            return None
        xs = [s.start.x for s in self.segments] + [s.end.x for s in self.segments]
        ys = [s.start.y for s in self.segments] + [s.end.y for s in self.segments]
        return {
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
            "width": max(xs) - min(xs),
            "height": max(ys) - min(ys),
        }

    def measured_side_lengths(self) -> list[float]:
        """Length of each drawn segment in metres (for sanity checks / logging)."""

        return [round(s.actual_distance, 3) for s in self.segments if s.actual_distance > 1e-3]
