"""
记录海龟已经绘制的内容——应用对画布上墨迹的记忆。

## 概述

Turtlesim 告诉我们海龟的**位姿**（x、y、朝向）和画笔命令。
它**不会**给我们截图，也不会给出画布上已有线条的列表。

``PathTracker`` 由我们自己重建那幅图像：每当画笔落下且海龟移动时，
我们就存储一条直的**线段**（起点、终点、颜色、宽度）。

可以把它看作任务期间不断增长的草图本图层：

- 导航员在真实海龟上绘制 → 线段被追加到这里
- 计划仿真（试运行）→ 以同样方式构建一个临时的 ``PathTracker``
- PNG 预览、计划评审、规划器上下文 → 全都从线段列表读取

整个任务期间，``MissionContext.path_tracker`` 上存在一个 ``PathTracker``。
它跨部件累积墨迹（机身、然后头部、然后耳朵……）。

## 为什么这是必要的

没有它，智能体只会知道*海龟现在在哪里*，而不知道*已经画了什么*。
多部件形状需要后者——例如，在头部上放置耳朵
需要知道头部线条在哪里。

## 主要类型

- ``PathPoint`` —— 一个以米为单位的 (x, y) 坐标
- ``PathSegment`` —— 一条落笔笔画（从 A 到 B 的线段）
- ``PathTracker`` —— 线段列表 + 当前画笔状态；由
  ``plan_execution.execute_plan_deterministic`` 和 ``path_simulation`` 更新
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_DEFAULT_PEN_RGB = (255, 255, 255)


@dataclass
class PathPoint:
    """11×11 米画布上的单个点（世界坐标）。"""

    x: float
    y: float


@dataclass
class PathSegment:
    """一条落笔笔画：从 ``start`` 到 ``end`` 的直线。"""

    start: PathPoint
    end: PathPoint
    requested_distance: float | None = None
    actual_distance: float = 0.0
    color: tuple[int, int, int] = _DEFAULT_PEN_RGB
    pen_width: int = 2


@dataclass
class PathTracker:
    """
    单个任务的落笔墨迹运行日志。

    随步骤执行（或仿真）而更新。使用 ``segments`` 来渲染
    PNG、进行包围盒检查，以及提供「画布上已有什么？」的上下文。
    """

    pen_down: bool = False
    segments: list[PathSegment] = field(default_factory=list)
    _last_point: PathPoint | None = None
    _pen_color: tuple[int, int, int] = _DEFAULT_PEN_RGB
    _pen_width: int = 2

    def set_pen(self, r: int, g: int, b: int, width: int = 2, *, off: bool = False) -> None:
        """更新当前画笔的颜色/宽度以及落笔状态。"""
        self._pen_color = (r, g, b)
        self._pen_width = width
        self.pen_down = not off
        if off:
            self._last_point = None

    def set_pen_down(self, down: bool) -> None:
        """遗留的开关——优先使用 set_pen() 以保留颜色。"""
        self.pen_down = down
        if not down:
            self._last_point = None

    def mark_pose(self, x: float, y: float) -> None:
        """在画笔落下期间每次移动后调用。"""
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
        """开始追踪从 (x, y) 起的一次前进移动。"""
        if not self.pen_down:
            return
        self._last_point = PathPoint(x=x, y=y)
        self._pending_requested = requested_distance

    def finish_segment(self, x: float, y: float) -> float:
        """在 (x, y) 处结束前进移动；返回实际行进距离。"""
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
        """丢弃所有线段并重置画笔状态（例如开始新任务）。"""

        self.segments.clear()
        self._last_point = None
        self.pen_down = False
        self._pen_color = _DEFAULT_PEN_RGB
        self._pen_width = 2

    def bounding_box(self) -> dict[str, float] | None:
        """围绕所有墨迹的轴对齐包围盒，若画布为空则返回 ``None``。"""

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
        """每条已绘制线段的长度（米）（用于合理性检查 / 日志记录）。"""

        return [round(s.actual_distance, 3) for s in self.segments if s.actual_distance > 1e-3]
