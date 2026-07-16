# Tools

Only the **supervisor** uses LangChain `@tool`s — the other agents are single structured LLM calls with all context injected into the prompt (cheaper and simpler than tool round-trips).


| Tool                | Module                         | Purpose                                           |
| ------------------- | ------------------------------ | ------------------------------------------------- |
| `get_turtle_pose`   | `tools/turtle_motion_tools.py` | Current x / y / heading                           |
| `move_forward_tool` | `tools/turtle_motion_tools.py` | Move forward N metres                             |
| `rotate_tool`       | `tools/turtle_motion_tools.py` | Rotate in place (degrees)                         |
| `set_pen`           | `tools/turtle_motion_tools.py` | Pen RGB / width / up-down                         |
| `clear_canvas_tool` | `tools/turtle_motion_tools.py` | Clear all ink                                     |
| `reset_world_tool`  | `tools/turtle_motion_tools.py` | Full turtlesim reset                              |
| `start_drawing`     | `agents/supervisor.py`         | Hand a drawing request to the specialist pipeline |




## How tools get live state

Tools need the ROS bridge and the mission notebook, but the LLM must not (and cannot) supply those as arguments. They travel through `AgentRuntime`(`utils/agent_runtime.py`), hidden in the invoke config:

```python
# agents/supervisor.py — when invoking the ReAct agent
agent.invoke(..., config={"configurable": {"agent_runtime": AgentRuntime(bridge, ctx)}})
```

Each tool declares a `runtime: ToolRuntime` parameter. LangGraph injects it automatically and hides it from the model's tool schema. Inside the tool:

```python
@tool
def get_turtle_pose(runtime: ToolRuntime) -> str:
    rt = get_agent_runtime(runtime)   # rt.bridge, rt.mission
    ...
```



## Adding a capability

1. Add a `@tool` in `src/tools/turtle_motion_tools.py` (or a new module).
2. Add it to the tool list in `src/agents/supervisor.py` `build_agent()`.
3. Mention it in the supervisor `SYSTEM_PROMPT` if the model needs guidance.

ROS, rendering, and geometry helpers live under `src/utils/`. Agent nodes call them as normal Python functions.

An **agent node** is the Python function LangGraph runs for each agent's turn (e.g. the reviewer node). An **LLM** is optional brain *inside* that node — used when the node needs a language or vision decision. Helpers in `src/utils/` are the hands: the node imports and calls them itself (simulate ink, save a PNG, talk to ROS). Only the supervisor exposes some of those actions as `@tool`s so its LLM can choose them; the other agents skip tools and call `utils/` directly from their node code.