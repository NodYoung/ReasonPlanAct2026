"""Turtlesim canvas constants — the world is an 11×11 metre square.

Used by the renderer (``drawing_render``), the reviewer's bounds check
(via ``path_simulation.bbox_fits_canvas``), and turtle parking.
"""

from __future__ import annotations

CANVAS_MIN = 0.0
CANVAS_MAX = 11.0
MARGIN = 0.25
