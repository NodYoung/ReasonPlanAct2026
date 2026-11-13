"""Turtlesim 画布常量——世界是一个 11×11 米的正方形。

由渲染器（``drawing_render``）、评审者的边界检查
（通过 ``path_simulation.bbox_fits_canvas``）以及海龟停靠使用。
"""

from __future__ import annotations

CANVAS_MIN = 0.0
CANVAS_MAX = 11.0
MARGIN = 0.25
