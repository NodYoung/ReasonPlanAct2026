"""
Turn ``PathTracker`` segment data into PNG images.

## Overview

Turtlesim has no screenshot API. Agents still need **pictures** of the canvas
for plan review, planner context, and saved captures.

This module is the **renderer only**:

1. Input — a ``PathTracker`` (segment list: start/end points, colour, width)
2. Output — PNG bytes (blue background, grid, lines, optional turtle dot)

It does **not** execute movement plans. Something else must fill the tracker first:

| Ink source | How segments get there |
|------------|-------------------------|
| Real drawing | Navigator runs steps → ``plan_execution`` updates ``MissionContext.path_tracker`` |
| Preview / what-if | ``path_simulation.simulate_plan_to_path_tracker`` builds a temporary tracker |

## Typical pipeline (plan review)

::

    simulate_plan_to_path_tracker(new_part_plan)   # path_simulation
    merge with ctx.path_tracker                    # existing ink
    render_drawing(merged)                         # this file → PNG
    save as captures/{shape}_{part}_trial{N}.png   # reviewer looks at it

## Main functions

- ``render_path_to_image`` / ``render_drawing`` — segments → PNG bytes
- ``save_image`` — write bytes to ``captures/`` (or any path)
- ``png_path_to_data_url`` — read a saved PNG for vision-model messages
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from src.utils.canvas_geometry import CANVAS_MAX, CANVAS_MIN
from src.utils.path_tracker import PathTracker

_BG_RGB = (0, 162, 232)
_DEFAULT_LINE_RGB = (255, 255, 255)
_TURTLE_RGB = (255, 165, 0)


def _world_to_pixel(
    x: float,
    y: float,
    *,
    img_size: int,
    margin_px: int,
) -> tuple[int, int]:
    draw_size = img_size - 2 * margin_px
    scale = draw_size / (CANVAS_MAX - CANVAS_MIN)
    px = margin_px + int((x - CANVAS_MIN) * scale)
    py = margin_px + int((CANVAS_MAX - y) * scale)
    return px, py


def render_path_to_image(
    path: PathTracker,
    *,
    turtle_pose: tuple[float, float, float] | None = None,
    line_rgb: tuple[int, int, int] = _DEFAULT_LINE_RGB,
    img_size: int = 600,
    margin_px: int = 20,
) -> bytes:
    """
    Draw all pen-down segments onto a turtlesim-style PNG.

    ``path`` usually comes from ``MissionContext.path_tracker`` (real ink) or
    from ``path_simulation`` (simulated preview). Optional ``turtle_pose`` adds
    an orange dot for the turtle position.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (img_size, img_size), _BG_RGB)
    draw = ImageDraw.Draw(img)

    draw_size = img_size - 2 * margin_px
    scale = draw_size / (CANVAS_MAX - CANVAS_MIN)
    for m in range(int(CANVAS_MIN), int(CANVAS_MAX) + 1):
        px0, py0 = _world_to_pixel(m, CANVAS_MIN, img_size=img_size, margin_px=margin_px)
        px1, py1 = _world_to_pixel(m, CANVAS_MAX, img_size=img_size, margin_px=margin_px)
        draw.line([(px0, py0), (px1, py1)], fill=(0, 140, 200), width=1)
        px0, py0 = _world_to_pixel(CANVAS_MIN, m, img_size=img_size, margin_px=margin_px)
        px1, py1 = _world_to_pixel(CANVAS_MAX, m, img_size=img_size, margin_px=margin_px)
        draw.line([(px0, py0), (px1, py1)], fill=(0, 140, 200), width=1)

    default_width = max(2, int(scale * 0.08))
    for seg in path.segments:
        x0, y0 = _world_to_pixel(seg.start.x, seg.start.y, img_size=img_size, margin_px=margin_px)
        x1, y1 = _world_to_pixel(seg.end.x, seg.end.y, img_size=img_size, margin_px=margin_px)
        rgb = seg.color if seg.color else line_rgb
        seg_width = max(2, int(seg.pen_width * scale * 0.04)) if seg.pen_width else default_width
        draw.line([(x0, y0), (x1, y1)], fill=rgb, width=seg_width)

    if turtle_pose is not None:
        tx, ty, _ = turtle_pose
        tpx, tpy = _world_to_pixel(tx, ty, img_size=img_size, margin_px=margin_px)
        r = max(4, int(scale * 0.15))
        draw.ellipse(
            [(tpx - r, tpy - r), (tpx + r, tpy + r)],
            fill=_TURTLE_RGB,
            outline=(200, 100, 0),
        )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def save_image(png_bytes: bytes, path: str | Path) -> Path:
    """Write PNG bytes to disk (creates parent folders if needed)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png_bytes)
    return out


def png_path_to_data_url(path: str | Path) -> str | None:
    """Read a PNG file and return a data URL for vision-capable chat models."""
    p = Path(path)
    if not p.is_file():
        return None
    b64 = base64.standard_b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def render_drawing(
    path: PathTracker,
    *,
    turtle_pose: tuple[float, float, float] | None = None,
) -> bytes:
    """Convenience wrapper: ``PathTracker`` → PNG bytes (default image size)."""
    return render_path_to_image(path, turtle_pose=turtle_pose)
