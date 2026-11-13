"""
主管的 LangChain 工具共享的运行时容器。

当 LLM 调用某个工具（如 ``move_forward_tool``）时，工具只会收到
模型所选择的参数。它不会自动获得实时的 turtlesim
桥接层或任务记事本——而且模型绝不能凭空捏造它们。

因此，每个智能体节点把它们打包进一个 ``AgentRuntime``，并藏在
invoke 配置里::

    agent.invoke(..., config={"configurable": {"agent_runtime": runtime}})

然后工具调用 ``get_agent_runtime(runtime)`` 来打开这个容器，并使用
``rt.bridge`` / ``rt.mission``。
"""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.prebuilt import ToolRuntime

from src.utils.mission_state import MissionContext
from src.utils.ros_bridge import TurtleBridge

AGENT_RUNTIME_CONFIG_KEY = "agent_runtime"


@dataclass
class AgentRuntime:
    """在一个智能体回合及其工具调用之间共享的实时句柄。"""

    bridge: TurtleBridge
    mission: MissionContext


def get_agent_runtime(runtime: ToolRuntime) -> AgentRuntime:
    """在 LangChain 的 ``@tool`` 内部解析出 ``AgentRuntime``。"""
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
