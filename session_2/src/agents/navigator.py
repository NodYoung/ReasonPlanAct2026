"""
Navigator — deterministic turtle execution (no LLM).

Once a part's plan is approved, every step is already known; an LLM would be
slow and unreliable here. This node walks ``ctx.movement_plan`` on the real
turtle (set_pen / move_forward / rotate via the ROS bridge), records ink on
``ctx.path_tracker``, and flags the result for the supervisor's bookkeeping.
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
