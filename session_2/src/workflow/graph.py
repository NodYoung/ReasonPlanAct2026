"""
LangGraph 接线 + ``run_mission`` —— 围绕 supervisor 的中心辐射式结构。

    user message → supervisor ──(简单任务 / 问题)──────→ END
                        │
                        ├→ planner_blueprint ─┐
                        ├→ planner_part ──────┤   每个专职 agent
                        ├→ plan_reviewer ─────┼── 都会返回到
                        └→ navigator ─────────┘   supervisor

supervisor 是唯一的编排者：首次访问时执行一轮 ReAct LLM，
之后在再次访问时按确定性规则在绘图流水线中进行路由。
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents import navigator, plan_reviewer, planner_blueprint, planner_part, supervisor
from src.utils.mission_state import MissionContext
from src.utils.ros_bridge import TurtleBridge
from src.workflow.types import WorkflowState

# 外层图的步数预算：每个部件都会消耗若干次节点访问，重新规划
# 会消耗更多 —— 为多部件绘图保留充裕的余量。
RECURSION_LIMIT = 200


def build_workflow(bridge: TurtleBridge) -> StateGraph:
    """构建这个多 agent 的 StateGraph。"""
    graph = StateGraph(WorkflowState)

    graph.add_node("supervisor", supervisor.build_graph_node(bridge))
    graph.add_node("planner_blueprint", planner_blueprint.build_graph_node(bridge))
    graph.add_node("planner_part", planner_part.build_graph_node(bridge))
    graph.add_node("plan_reviewer", plan_reviewer.build_graph_node(bridge))
    graph.add_node("navigator", navigator.build_graph_node(bridge))

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next_agent", "finish"),
        {
            "planner_blueprint": "planner_blueprint",
            "planner_part": "planner_part",
            "plan_reviewer": "plan_reviewer",
            "navigator": "navigator",
            "finish": END,
        },
    )
    for name in ("planner_blueprint", "planner_part", "plan_reviewer", "navigator"):
        graph.add_edge(name, "supervisor")

    return graph


def run_mission(user_request: str, bridge: TurtleBridge) -> bool:
    """端到端地运行一个用户请求；除非任务失败，否则返回 True。"""
    ctx = MissionContext(user_request=user_request)
    build_workflow(bridge).compile().invoke(
        {"ctx": ctx}, config={"recursion_limit": RECURSION_LIMIT}
    )
    return not ctx.mission_failed
