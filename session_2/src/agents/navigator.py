"""
Navigator（导航器）—— 确定性的海龟执行（不使用 LLM）。

一旦某个部件的计划获批，每一步都已经确定；在这里使用 LLM 会既慢又不可靠。
本节点在真实海龟上逐步执行 ``ctx.movement_plan``（通过 ROS 桥接调用
set_pen / move_forward / rotate），在 ``ctx.path_tracker`` 上记录墨迹，
并将结果标记出来供 supervisor 记账。
"""

from __future__ import annotations

from collections.abc import Callable

from src.pydantic_models import ExecutionReport
from src.utils.mission_state import say
from src.utils.plan_execution import execute_plan_deterministic
from src.utils.ros_bridge import TurtleBridge
from src.utils.turtle_control import pose_as_dict
from src.workflow.types import WorkflowState

def build_graph_node(bridge: TurtleBridge) -> Callable[[WorkflowState], dict]:
    def node(state: WorkflowState) -> dict:
        ctx = state["ctx"]
        plan = ctx.movement_plan
        if plan is None:
            ctx.mission_failed = True
            return {}

        pose = pose_as_dict(bridge)
        plan = plan.model_copy(update={"starting_pose": (pose["x"], pose["y"], pose["theta"])})
        part = ctx.current_part() or "?"
        try:
            report = execute_plan_deterministic(
                plan, bridge, ctx.path_tracker, clear_path_tracker=False
            )
        except Exception as exc:  # noqa: BLE001
            report = ExecutionReport(
                success=False,
                steps_completed=0,
                total_steps=len(plan.steps),
                step_logs=[],
                final_pose=None,
                message=str(exc),
            )
        ctx.execution_report = report
        ctx.execution_pending = True
        say(
            ctx,
            f"[Navigator] Part [{part}]: {report.steps_completed}/{report.total_steps} steps executed.",
        )
        return {}

    return node
