# 代码导览:图、工具、ROS bridge 各在哪里

面向刚看完 LangGraph 入门教程、想对照理解 session_2 的读者。教程里「建图 + 绑工具」通常挤在一个文件里,session_2 把它们拆成了多层,这里说明各在何处、与教程的对应关系。

- [1. 构建图](#1-构建图--srcworkflowgraphpy)
- [2. 工具定义](#2-工具定义--srctoolsturtle_motion_toolspy)
- [3. 把工具绑给 agent](#3-把工具绑给-agent--srcagentssupervisorpy-的-build_agent)
- [4. 与入门教程的对应](#4-与入门教程的对应)
- [5. ROS bridge:切换后端](#5-ros-bridge切换后端--srcutilsros_bridgepy)

---

## 1. 构建图 — `src/workflow/graph.py`

`build_workflow()` 做的就是教程里那套 `StateGraph` 操作,只是节点是 5 个 agent:

```python
graph = StateGraph(WorkflowState)
graph.add_node("supervisor", supervisor.build_graph_node(bridge))
graph.add_node("planner_blueprint", ...)          # 共 5 个节点
...
graph.set_entry_point("supervisor")                # = 教程的 add_edge(START, ...)
graph.add_conditional_edges("supervisor", ...)      # 条件边(教程里 tools_condition 的位置)
for name in (...):
    graph.add_edge(name, "supervisor")              # 每个专家都连回 supervisor
```

`add_node` / `set_entry_point` / `add_conditional_edges` / `add_edge` 与教程完全一致;`.compile()` 在同文件的 `run_mission()` 里调用。

唯一区别:条件边**不用** `tools_condition`,而是自己写的路由函数:

```python
graph.add_conditional_edges(
    "supervisor",
    lambda state: state.get("next_agent", "finish"),   # 下一步去谁由 State 里的字段决定
    {...},
)
```

下一步去哪由 `MissionContext` 里的 `next_agent` 字段决定——这就是「supervisor 之后走确定性路由、不再耗 token 决策」的实现。

## 2. 工具定义 — `src/tools/turtle_motion_tools.py`

所有 `@tool` 定义都在这里:

- `get_turtle_pose`
- `move_forward_tool`
- `rotate_tool`
- `set_pen`
- `clear_canvas_tool`
- `reset_world_tool`

(即 `TOOLS.md` 里描述的那组运动工具。)

## 3. 把工具绑给 agent — `src/agents/supervisor.py` 的 `build_agent()`

「工具接到模型上」发生在这里:

```python
def build_agent() -> CompiledStateGraph:
    return create_agent(
        get_llm("supervisor"),
        [start_drawing, get_turtle_pose, move_forward_tool, ...],  # ← 工具列表
        system_prompt=SYSTEM_PROMPT,
    )
```

两个关键点:

1. **用的是 `create_agent`,不是手写的 `bind_tools + ToolNode + tools_condition`。** `create_agent` 是 LangChain 封装好的 ReAct agent,内部把「绑工具、执行工具、路由」三件事全包了,返回的直接是一个编译好的子图(`CompiledStateGraph`)。

2. **所以 session_2 是「图套图」。** 外层大图(`graph.py` 那 5 个节点)里的 `supervisor` 节点,内部又跑了一个 `create_agent` 生成的小 ReAct 图。工具只绑在 supervisor 这一个节点上;其他四个专家(blueprint / part planner / reviewer / navigator)不用工具,是单次结构化 LLM 调用(见 `TOOLS.md`)。

## 4. 与入门教程的对应

| 入门教程(langgraph-basis) | session_2 |
| --- | --- |
| `StateGraph` + `add_node` / `add_edge` | `workflow/graph.py` 的 `build_workflow()` |
| `@tool` 定义 | `tools/turtle_motion_tools.py` |
| `bind_tools` / `ToolNode` / `tools_condition` | `agents/supervisor.py` 里一句 `create_agent(llm, [tools])` 全包 |
| 手写 while 循环或单图 | 图套图:外层多智能体图 + supervisor 内层 ReAct 子图 |

`src/main.py` 是入口:起 ROS bridge、读用户输入、调用 `run_mission()`,本身不含建图和工具逻辑。

## 5. ROS bridge:切换后端 — `src/utils/ros_bridge.py`

这个 bridge 切的**不是**「仿真 vs 真机」,而是「**真实 ROS turtlesim** vs **纯 Python 内存 mock**」。两者都是仿真(turtlesim 本身就是仿真器),边界在于**要不要真的跑 ROS**。

`TurtleBridge` 是工厂,`use_mock` 开关切两条实现:

| 后端 | 条件 | 说明 |
| --- | --- | --- |
| `RosTurtleNode` | `use_mock=False` | 真的 `rclpy.init()`,建 publisher/subscriber/service,发 `cmd_vel`、订阅 `/turtle1/pose`、调 `set_pen`/`teleport` 等服务;对面是 `pixi run sim` 起的 turtlesim |
| `MockTurtleSim` | `use_mock=True` | 纯 Python,几个 `x/y/theta` 变量做运动学积分,完全不依赖 ROS;无图形输出 |

**目的:不装 ROS 也能学。** mock 让没装 ROS、没起 turtlesim 的人也能把整条多智能体流水线(规划→审查→执行)跑通,只是看不到画面。

**对上层透明。** `TurtleBridge` 把两个后端包成同一套接口(`get_pose` / `move_forward` / `rotate` / `set_pen` / ...),agent、工具、graph 都不知道底下是真 ROS 还是 mock——和 `llm.py` 用 provider 抽象是同一思路。

**关于真机:** 当前 bridge 没有「真实机器人」这一档,`RosTurtleNode` 写死了 turtlesim 的话题/服务名。但架构上正是为接真机预留:再加一个后端类(如 `RealRobotNode`,发真机 `/cmd_vel`、订阅真机里程计),`TurtleBridge` 加个分支即可,上层 agent 一行不改。turtlesim 在这里相当于真机的「占位替身」。

一个佐证意图的细节:`RosTurtleNode.move_forward` 用的是**闭环** pose 反馈(反复发 `cmd_vel` 直到走够距离,再用 teleport 补最后几毫米),而非开环 sleep 计时。这种闭环控制是冲着真实硬件去的,纯仿真本不需要这么讲究。
