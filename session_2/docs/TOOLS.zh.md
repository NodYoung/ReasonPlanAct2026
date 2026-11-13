# 工具

只有 **supervisor** 使用 LangChain 的 `@tool`——其他 agent 都是单次结构化 LLM 调用,所有上下文直接注入 prompt(比工具往返更便宜也更简单)。

| 工具 | 模块 | 用途 |
| --- | --- | --- |
| `get_turtle_pose` | `tools/turtle_motion_tools.py` | 当前 x / y / 朝向 |
| `move_forward_tool` | `tools/turtle_motion_tools.py` | 前进 N 米 |
| `rotate_tool` | `tools/turtle_motion_tools.py` | 原地旋转(度) |
| `set_pen` | `tools/turtle_motion_tools.py` | 画笔 RGB / 宽度 / 抬落 |
| `clear_canvas_tool` | `tools/turtle_motion_tools.py` | 清除所有墨迹 |
| `reset_world_tool` | `tools/turtle_motion_tools.py` | 完整重置 turtlesim |
| `start_drawing` | `agents/supervisor.py` | 把绘图请求交给专职 agent 流水线 |

## 工具如何拿到实时状态

工具需要 ROS bridge 和任务记事本,但 LLM 不应该(也无法)把它们作为参数传进来。它们通过 `AgentRuntime`(`utils/agent_runtime.py`)传递,藏在 invoke 的 config 里:

```python
# agents/supervisor.py —— 调用 ReAct agent 时
agent.invoke(..., config={"configurable": {"agent_runtime": AgentRuntime(bridge, ctx)}})
```

每个工具声明一个 `runtime: ToolRuntime` 参数。LangGraph 会自动注入它,并把它从模型看到的工具 schema 中隐藏掉。在工具内部:

```python
@tool
def get_turtle_pose(runtime: ToolRuntime) -> str:
    rt = get_agent_runtime(runtime)   # rt.bridge, rt.mission
    ...
```

## 新增一项能力

1. 在 `src/tools/turtle_motion_tools.py`(或一个新模块)里加一个 `@tool`。
2. 把它加进 `src/agents/supervisor.py` 的 `build_agent()` 的工具列表。
3. 如果模型需要指引,在 supervisor 的 `SYSTEM_PROMPT` 里提一下。

ROS、渲染和几何相关的辅助函数都放在 `src/utils/` 下。Agent 节点把它们当普通 Python 函数调用。

**agent 节点**是 LangGraph 为每个 agent 的回合运行的那个 Python 函数(例如 reviewer 节点)。**LLM** 是这个节点*内部*可选的大脑——只在节点需要做语言或视觉判断时才用。`src/utils/` 里的辅助函数是手:节点自己 import 并调用它们(模拟墨迹、保存 PNG、和 ROS 通信)。只有 supervisor 把其中一部分动作暴露成 `@tool`,好让它的 LLM 自己挑选;其他 agent 不用工具,直接在节点代码里调 `utils/`。
