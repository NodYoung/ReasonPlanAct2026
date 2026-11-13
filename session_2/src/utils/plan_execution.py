"""
导航员执行——运行一个 ``MovementPlan``，**不使用 LLM**。

## 概述

一旦评审者批准了计划，每一个海龟步骤就都已确定——
再次调用 LLM 既慢又不可靠。

``execute_plan_deterministic`` 遍历每个步骤并调用 ROS 桥接层：

- ``set_pen`` → 颜色 / 宽度 / 抬笔落笔
- ``move_forward`` → 平移并在 ``PathTracker`` 上记录墨迹
- ``rotate`` → 改变朝向

由导航员（Navigator）节点使用。返回一个写回到
``MissionContext`` 的 ``ExecutionReport``。
"""

from __future__ import annotations

import math

from src.pydantic_models import ExecutionReport, MovementPlan, StepAction, StepLog
from src.utils.turtle_control import (
    clear_canvas,
    move_forward,
    pose_as_dict,
    reset_world,
    rotate,
    set_pen_color,
    teleport,
)
from src.utils.path_tracker import PathTracker
from src.utils.ros_bridge import TurtleBridge
from src.utils.log_utils import log  # log() 等价 print()，但会打印「[文件名:行号]」前缀


def _should_snap_heading(plan: MovementPlan) -> bool:
    """是否在绘制前将朝向对齐到正交方向（当前关闭）。"""
    return False


def _snap_heading_to_cardinal(bridge: TurtleBridge) -> None:
    """当漂移明显时，将朝向瞬移对齐到最近的 90°。"""
    pose = pose_as_dict(bridge)
    quarter = math.pi / 2
    snapped = round(pose["theta"] / quarter) * quarter
    if abs(pose["theta"] - snapped) > 0.03:
        teleport(bridge, pose["x"], pose["y"], snapped)


def execute_plan_deterministic(
    plan: MovementPlan,
    bridge: TurtleBridge,
    path_tracker: PathTracker | None = None,
    *,
    clear_canvas_first: bool = False,
    clear_path_tracker: bool = True,
    step_offset: int = 0,
) -> ExecutionReport:
    """
    在实时海龟上执行 ``plan`` 的每一个步骤。

    提供 ``path_tracker`` 时，会记录落笔运动。返回一个带有成功标志、
    步骤计数以及逐步日志的 ``ExecutionReport``。
    """
    if path_tracker is not None and clear_path_tracker:
        path_tracker.clear()

    logs: list[StepLog] = []
    completed = 0
    heading_snapped = False

    if clear_canvas_first and not any(
        s.action == StepAction.CLEAR_CANVAS for s in plan.steps
    ):
        clear_canvas(bridge)

    for step in plan.steps:
        try:
            pose_before = pose_as_dict(bridge)
            display_step = step_offset + step.step_number

            if step.action == StepAction.SET_PEN and step.pen:
                log(f"  [Exec] step {display_step}: set_pen "
                f"RGB({step.pen.r},{step.pen.g},{step.pen.b})",
                flush=True,)
                msg = set_pen_color(
                    bridge,
                    step.pen.r,
                    step.pen.g,
                    step.pen.b,
                    step.pen.width,
                    step.pen.off,
                )
                if path_tracker is not None:
                    path_tracker.set_pen(
                        step.pen.r,
                        step.pen.g,
                        step.pen.b,
                        step.pen.width,
                        off=step.pen.off,
                    )
                    if not step.pen.off:
                        path_tracker.mark_pose(pose_before["x"], pose_before["y"])

            elif step.action == StepAction.MOVE_FORWARD and step.distance is not None:
                if (
                    path_tracker is not None
                    and path_tracker.pen_down
                    and not heading_snapped
                    and _should_snap_heading(plan)
                ):
                    _snap_heading_to_cardinal(bridge)
                    heading_snapped = True
                    pose_before = pose_as_dict(bridge)

                log(f"  [Exec] step {display_step}: move_forward {step.distance} m",
                flush=True,)
                if path_tracker is not None and path_tracker.pen_down:
                    path_tracker.start_segment(
                        pose_before["x"], pose_before["y"], step.distance
                    )
                msg = move_forward(bridge, step.distance)
                pose_after = pose_as_dict(bridge)
                if path_tracker is not None and path_tracker.pen_down:
                    path_tracker.finish_segment(pose_after["x"], pose_after["y"])

            elif step.action == StepAction.ROTATE and step.angle_degrees is not None:
                log(f"  [Exec] step {display_step}: rotate {step.angle_degrees}°",
                flush=True,)
                msg = rotate(bridge, step.angle_degrees)

            elif step.action == StepAction.TELEPORT:
                msg = "Teleport skipped — draw from current position."
                completed += 1
                logs.append(
                    StepLog(
                        step_number=step.step_number,
                        action=step.action,
                        success=True,
                        message=msg,
                        pose_after=(
                            pose_before["x"],
                            pose_before["y"],
                            pose_before["theta"],
                        ),
                    )
                )
                continue

            elif step.action == StepAction.CLEAR_CANVAS:
                msg = clear_canvas(bridge)

            elif step.action == StepAction.RESET_WORLD:
                msg = reset_world(bridge)
                if path_tracker is not None:
                    path_tracker.clear()

            else:
                logs.append(
                    StepLog(
                        step_number=step.step_number,
                        action=step.action,
                        success=False,
                        message=f"Unknown step: {step.action}",
                    )
                )
                break

            pose = pose_as_dict(bridge)
            logs.append(
                StepLog(
                    step_number=step.step_number,
                    action=step.action,
                    success=True,
                    message=msg,
                    pose_after=(pose["x"], pose["y"], pose["theta"]),
                )
            )
            completed += 1
        except Exception as exc:  # noqa: BLE001
            logs.append(
                StepLog(
                    step_number=step.step_number,
                    action=step.action,
                    success=False,
                    message=str(exc),
                )
            )
            break

    pose = pose_as_dict(bridge)
    return ExecutionReport(
        success=completed == len(plan.steps),
        steps_completed=completed,
        total_steps=len(plan.steps),
        step_logs=logs,
        final_pose=(pose["x"], pose["y"], pose["theta"]),
        message=f"Executed {completed}/{len(plan.steps)} steps.",
    )
