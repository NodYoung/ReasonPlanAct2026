"""
Supervisor（主管）—— 唯一的编排者，也是唯一与用户对话的 agent。

## 首次访问（一次 LLM ReAct 回合）

每条用户消息都会先到这里。主管会分析请求：

- 简单的海龟任务（移动、旋转、画笔、清空、重置、提问）—— 它用自己的运动工具
  亲自完成并回复。任务结束。
- 绘图任务（一个形状，或由多个形状组成的物体）—— 它调用 ``start_drawing``
  工具，把工作交给专职 agent。

## 后续访问（零次 LLM 调用）

每个专职 agent（蓝图规划器、部件规划器、审查器、导航器）都会返回到这里。
绘图流水线的顺序是固定的，所以路由就是一个基于 ``MissionContext`` 的纯确定性
状态机 —— 不会花费任何 token 去决定"下一个是谁"：

    蓝图 → 规划部件 → 审查 → 绘制 → 下一部件 → ... → 完成

审查器的结论也在这里应用：批准 → 导航器；拒绝 → 带着反馈和被拒的试验图像
返回部件规划器；重试次数耗尽 → 跳过该部件或直接绘制（``[review].fail_action``）。
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolRuntime

from src.app_settings import get_config
from src.llm import get_llm
from src.tools.turtle_motion_tools import (
    clear_canvas_tool,
    get_turtle_pose,
    move_forward_tool,
    reset_world_tool,
    rotate_tool,
    set_pen,
)
from src.utils.agent_runtime import AGENT_RUNTIME_CONFIG_KEY, AgentRuntime, get_agent_runtime
from src.utils.cli_spinner import spinner
from src.utils.mission_state import MissionContext, say
from src.utils.ros_bridge import TurtleBridge
from src.utils.turtle_park import park_turtle_for_clear_view
from src.workflow.types import WorkflowState

RECURSION_LIMIT = 16

SYSTEM_PROMPT = """你是一个多 agent turtlesim 绘图系统的 Supervisor（主管）。
你是唯一与用户对话的 agent。回答要简短、友好。

判断如何处理用户的消息：

1. DRAWING（绘图）请求 —— 用户想画一个形状或物体（square、circle、house、cat、
   rocket……）：用一句简短描述调用 start_drawing 一次（ONCE）。
   由专职 agent 来规划和绘制 —— 不要（do NOT）自己用运动工具去画。
2. SIMPLE（简单）海龟任务 —— 一次性动作，例如 "move forward 0.5"、"rotate 90"、
   "set pen to red"（red=255,0,0 blue=0,0,255 green=0,255,0）、"clear the canvas"、
   "reset"、"where are you?"：立刻（NOW）用你的运动工具执行。
3. QUESTION（提问）—— 直接回答。有用的事实：画布为 11x11 米
   （可用范围 0.25-10.75），最小可行移动为 0.01 m，画笔为 RGB 0-255。

始终以给用户的一句简短回复结尾（你的最终消息会展示给用户）。
"""


@tool
def start_drawing(runtime: ToolRuntime, description: str) -> str:
    """把一个绘图请求交给专职流水线（planner、reviewer、navigator）。

    对任何「画一个形状或物体」的请求都调用本工具一次。``description``
    是对要画什么的简短概括。
    """
    get_agent_runtime(runtime).mission.drawing = True
    return f"Drawing pipeline started for: {description}. Reply with a one-line kickoff."


@lru_cache(maxsize=1)
def build_agent() -> CompiledStateGraph:
    return create_agent(
        get_llm("supervisor"),
        [
            start_drawing,
            get_turtle_pose,
            move_forward_tool,
            rotate_tool,
            set_pen,
            clear_canvas_tool,
            reset_world_tool,
        ],
        system_prompt=SYSTEM_PROMPT,
        name="supervisor",
    )


def _final_text(result: dict) -> str:
    """从 ReAct 结果中提取助手的最终回复文本。"""
    content = result["messages"][-1].content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(parts).strip()
    return ""


def _finish_drawing(ctx: MissionContext, bridge: TurtleBridge) -> None:
    """停放海龟并打印任务总结。"""
    if get_config().park_after_draw and ctx.path_tracker.segments:
        try:
            say(ctx, park_turtle_for_clear_view(bridge, ctx.path_tracker))
        except Exception as exc:  # noqa: BLE001
            say(ctx, f"Could not park turtle ({exc}).")
    done = ", ".join(ctx.parts_completed) or "none"
    if ctx.parts_skipped:
        say(
            ctx,
            f"Drawing finished. Completed: {done}. "
            f"Skipped after failed reviews: {', '.join(ctx.parts_skipped)}.",
        )
    else:
        say(ctx, f"Drawing complete: {done}.")


def _handle_navigator_result(ctx: MissionContext) -> None:
    """在导航器绘制完当前部件后进行记账。"""
    ctx.execution_pending = False
    part = ctx.current_part() or "?"
    report = ctx.execution_report
    total = len(ctx.parts or [])
    if report and report.success:
        ctx.parts_completed.append(part)
        say(ctx, f"Part [{part}] drawn ({ctx.part_index + 1}/{total}).")
    else:
        detail = report.message if report else "no execution report"
        ctx.parts_skipped.append(part)
        say(ctx, f"Part [{part}] failed while drawing ({detail}) — moving on.")
    ctx.part_index += 1
    ctx.reset_part_pipeline()


def _apply_review_verdict(ctx: MissionContext) -> str:
    """把审查器的结论转化为下一个路由步骤。"""
    review = ctx.review
    assert review is not None
    cfg = get_config().review
    part = ctx.current_part() or "?"

    if review.approved:
        ctx.plan_approved = True
        ctx.last_good_image = ctx.trial_image
        return "navigator"

    if ctx.trial >= cfg.max_trials:
        if cfg.fail_action == "draw_anyway":
            say(ctx, f"Part [{part}] review still failing after {ctx.trial} trials — drawing anyway.")
            ctx.plan_approved = True
            return "navigator"
        say(ctx, f"Part [{part}] skipped after {ctx.trial} failed trials.")
        ctx.parts_skipped.append(part)
        ctx.part_index += 1
        ctx.reset_part_pipeline()
        return "continue"  # 让流水线挑选下一个步骤

    ctx.feedback = review.feedback
    ctx.trial += 1
    ctx.movement_plan = None
    ctx.review = None
    return "planner_part"


def _pipeline_next(ctx: MissionContext, bridge: TurtleBridge) -> str:
    """确定性的绘图流水线路由器 —— 每次重访时运行，不使用 LLM。"""
    while True:
        if ctx.execution_pending:
            _handle_navigator_result(ctx)

        if ctx.mission_failed:
            say(ctx, "Mission stopped — see messages above.")
            return "finish"

        if ctx.parts is None:
            return "planner_blueprint"

        if ctx.current_part() is None:
            _finish_drawing(ctx, bridge)
            return "finish"

        if ctx.movement_plan is None:
            return "planner_part"

        if not ctx.plan_approved:
            if not get_config().review.enabled:
                ctx.plan_approved = True
                return "navigator"
            if ctx.review is None:
                return "plan_reviewer"
            step = _apply_review_verdict(ctx)
            if step != "continue":
                return step
            continue  # 该部件被跳过 —— 从头重新评估

        return "navigator"


def build_graph_node(bridge: TurtleBridge) -> Callable[[WorkflowState], dict]:
    def node(state: WorkflowState) -> dict:
        ctx = state["ctx"]

        # 绘图任务进行中的重访 —— 仅做确定性路由。
        if ctx.drawing:
            return {"next_agent": _pipeline_next(ctx, bridge)}

        # 首次访问 —— 用一次 ReAct 回合来分析并处理请求。
        with spinner("Supervisor thinking"):
            result = build_agent().invoke(
                {"messages": [HumanMessage(content=ctx.user_request)]},
                config={
                    "recursion_limit": RECURSION_LIMIT,
                    "configurable": {AGENT_RUNTIME_CONFIG_KEY: AgentRuntime(bridge, ctx)},
                },
            )
        reply = _final_text(result)
        if reply:
            say(ctx, reply)
        if ctx.drawing:
            return {"next_agent": "planner_blueprint"}
        return {"next_agent": "finish"}

    return node
