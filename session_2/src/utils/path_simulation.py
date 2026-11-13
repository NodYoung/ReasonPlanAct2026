"""
用软件对 ``MovementPlan`` 进行试运行——没有海龟，没有 ROS。

有三个模块协同回答「画布上有什么？」：

| 模块 | 职责 |
|--------|------|
| ``path_tracker`` | **数据**——线段列表（墨迹记忆） |
| ``path_simulation``（本文件） | **预测**——仅用数学遍历计划的步骤 |
| ``drawing_render`` | **图像**——把线段数据变成 PNG |

计划评审者使用 ``simulate_plan_to_path_tracker`` 构建所提议部件的
墨迹，将其与已经绘制的真实墨迹合并，渲染出试次 PNG，
并使用 ``bbox_fits_canvas`` 进行免费的越界检查。
在整个过程中，真实海龟从不移动。
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
    """返回一个模拟出的包围盒是否保持在画布内（含边距）。"""
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
    """试运行一个计划并构建一个全新的 ``PathTracker``（带颜色的线段）。"""
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
