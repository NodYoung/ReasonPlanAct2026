"""
规范化 MovementPlan，使绘图总是从海龟的实时位姿开始。

规划器（Planner）LLM 有时会添加瞬移步骤到空的角落。我们会剥离这些步骤，
并把可行性检查锚定在海龟实际所在的位置。
"""

from __future__ import annotations

from src.pydantic_models import MovementPlan, MovementStep, StepAction
from src.utils.log_utils import log  # log() 等价 print()，但会打印「[文件名:行号]」前缀


def normalize_plan_for_current_pose(
    plan: MovementPlan,
    pose: dict[str, float],
) -> MovementPlan:
    """
    移除重新定位的瞬移步骤，并将 starting_pose 设为海龟的实时位姿。

    执行期间的就地朝向对齐保持不变——只移除绘图前的位置
    瞬移。
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
        log(f"  [Plan] Removed {removed} teleport step(s) — drawing from current "
        f"pose ({pose['x']:.2f}, {pose['y']:.2f}, "
        f"θ={pose['theta']:.2f})",
        flush=True,)

    return sanitize_negative_move_forward(normalized)


def sanitize_negative_move_forward(plan: MovementPlan) -> MovementPlan:
    """
    改写负的 move_forward 步骤，使预览仿真与 turtlesim 一致。

    ROS 会跳过零或负的前进距离；而模拟器不会。如果不这样处理，
    计划评审可能会批准海龟实际上永远不会绘制的几何形状。
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
                    description="掉头（由负向 move_forward 改写而来）",
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
        log(f"  [Plan] Rewrote {rewritten} negative move_forward step(s) as "
        "rotate-180 + positive distance.",
        flush=True,)
        return plan.model_copy(update={"steps": new_steps})
    return plan


def align_plan_pen_width(plan: MovementPlan, target_width: int) -> MovementPlan:
    """
    强制每个 set_pen 步骤（以及 plan.pen）使用相同的线宽。

    在部件规划后使用，使 wing_left 和 wing_right 与机身粗细一致。
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
    log(f"  [Plan] Aligned pen width to {width} for all set_pen steps.", flush=True)
    return aligned
