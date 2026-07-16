"""
Shared runtime bag for the supervisor's LangChain tools.

When the LLM calls a tool like ``move_forward_tool``, the tool only receives
arguments the model chose. It does not automatically get the live turtlesim
bridge or the mission notebook — and the model must not invent those.

So each agent node packs them into an ``AgentRuntime`` and hides it in the
invoke config::

    agent.invoke(..., config={"configurable": {"agent_runtime": runtime}})

Tools then call ``get_agent_runtime(runtime)`` to open the bag and use
``rt.bridge`` / ``rt.mission``.
"""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.prebuilt import ToolRuntime

from src.utils.mission_state import MissionContext
from src.utils.ros_bridge import TurtleBridge

AGENT_RUNTIME_CONFIG_KEY = "agent_runtime"


@dataclass
class AgentRuntime:
    """Live handles shared across one agent turn and its tool calls."""

    bridge: TurtleBridge
    mission: MissionContext


def get_agent_runtime(runtime: ToolRuntime) -> AgentRuntime:
    """Resolve the ``AgentRuntime`` inside a LangChain ``@tool``."""
    configurable = (runtime.config or {}).get("configurable") or {}
    rt = configurable.get(AGENT_RUNTIME_CONFIG_KEY)
    if isinstance(rt, AgentRuntime):
        return rt
    if isinstance(runtime.context, AgentRuntime):
        return runtime.context
    raise RuntimeError(
        "AgentRuntime missing from invoke config — tool was not called from a "
        "node that passed configurable[agent_runtime]."
    )
