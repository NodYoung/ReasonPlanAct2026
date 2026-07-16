"""
LangGraph wiring + ``run_mission`` — hub-and-spoke around the supervisor.

    user message → supervisor ──(simple task / question)──→ END
                        │
                        ├→ planner_blueprint ─┐
                        ├→ planner_part ──────┤   every specialist
                        ├→ plan_reviewer ─────┼── returns to the
                        └→ navigator ─────────┘   supervisor

The supervisor is the only orchestrator: one ReAct LLM turn on the first
visit, then deterministic routing through the drawing pipeline on revisits.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents import navigator, plan_reviewer, planner_blueprint, planner_part, supervisor
from src.utils.mission_state import MissionContext
from src.utils.ros_bridge import TurtleBridge
from src.workflow.types import WorkflowState

# Outer graph step budget: every part costs a handful of node visits, replans
# add more — keep generous headroom for multi-part drawings.
RECURSION_LIMIT = 200


def build_workflow(bridge: TurtleBridge) -> StateGraph:
    """Construct the multi-agent StateGraph."""
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
    """Run one user request end-to-end; return True unless the mission failed."""
    ctx = MissionContext(user_request=user_request)
    build_workflow(bridge).compile().invoke(
        {"ctx": ctx}, config={"recursion_limit": RECURSION_LIMIT}
    )
    return not ctx.mission_failed
