# 架构

这个多智能体 Turtlesim 原型是一个 **LangGraph 中心辐射式(hub-and-spoke)*** 系统,只有一个编排者。supervisor 是唯一与用户对话的 agent,也是唯一决定下一步由谁工作的 agent;各专职 agent 之间从不互相调用。

*一种路由模式:中间一个**枢纽(hub)**,各专职 agent 作为**辐条(spoke)**只与枢纽通信,彼此之间不通信。

```
User ↔ supervisor ──(简单任务 / 提问:直接处理)──→ END
           │
           ├─→ planner_blueprint ─┐
           ├─→ planner_part ──────┤  每个专职 agent
           ├─→ plan_reviewer ─────┤  都返回 supervisor
           └─→ navigator ─────────┘
```

## 五个 agent

| Agent | LLM 调用次数 | 职责 |
| --- | --- | --- |
| `supervisor` | 每条用户消息 1 次 | 分析请求。简单的 turtle 任务和提问由它自己用运动工具处理。绘图请求通过 `start_drawing` 工具交给流水线。之后每次回到它这里,都按固定的流水线顺序**确定性地**路由(不调 LLM)。 |
| `planner_blueprint` | 每个任务 1 次 | 结构化调用,不用工具:请求 → 有序的 snake_case 部件列表 + `side_length` 尺寸预算。 |
| `planner_part` | 每次试验 1 次 | 结构化调用,不用工具:为当前部件生成运动步骤(`set_pen` / `move_forward` / `rotate`)。实时位姿、画布边界、笔宽建议和进度都是确定性地注入到 prompt 里。后续部件的第 1 次试验会附上上一个通过审查的画布图像;重新规划时会收到审查反馈 + 被驳回的试验图像。生成的计划随后会被归一化(去掉 teleport、修正负向移动、笔宽与已有墨迹对齐)。 |
| `plan_reviewer` | 每次试验 ≤1 次 | 模拟计划将画出的墨迹,叠加在已经画好的墨迹之上渲染,保存为 `captures/{shape}_{part}_trial{N}.png`。空计划和越出画布的墨迹会被免费驳回(不调 LLM)。否则做一次视觉调用:通过,或者驳回并给出命令式的修改指示。如果视觉调用本身出错,该计划直接视为通过、不做审查。 |
| `navigator` | 0 | 通过 ROS bridge 在真实 turtle 上执行通过审查的计划,并把墨迹记录到该任务的 `PathTracker` 上。 |

## 数据流(绘图任务)核心逻辑！！！

1. supervisor 的 LLM 回合调用 `start_drawing` → 路由到 blueprint。
2. blueprint 把 `shape_name`、`parts`、`side_length` 写入 `MissionContext`。
3. 对每个部件,supervisor 确定性地循环:
   - 还没有计划 → `planner_part`
   - 有计划但没有裁决 → `plan_reviewer`
   - 被驳回 → 把反馈 + 试验图像回传给 `planner_part`(第 N+1 次试验)
   - 被驳回达 `max_trials` 次 → 跳过该部件或照画不误(由 `[review].fail_action` 决定)
   - 通过 → `navigator`,然后推进到下一个部件
4. 所有部件完成 → 停靠 turtle,打印总结,END。

状态集中存放在一个可变的 `MissionContext`(`utils/mission_state.py`)中,每个节点都读写它。结构化的 LLM 输出是 Pydantic 模型(`pydantic_models.py`):`ShapeBlueprint`、`MovementPlanOutput`、`PlanReview`。

## 试验图像 = 渐进式的过程记录

`captures/` 目录里只会有 `{shape}_{part}_trial{N}.png`。每张图都显示到目前为止的真实墨迹加上待画的部件,所以整个序列就记录了构建过程:哪个部件没通过审查、是怎么修好的、图案是怎么一点点长出来的。

## 按 agent 配置模型(未测试)

`llm.py` 提供 `get_llm(agent)`,其中 agent ∈ supervisor / blueprint / planner / reviewer。`config.toml` 的 `[llm]` 设置默认值;可选的 `[llm.<agent>]` 表可以按 agent 覆盖 provider 和/或 model(例如给 supervisor 配一个便宜的模型,给 reviewer 配一个视觉模型)。支持的 provider:Gemini、OpenAI、Anthropic、Ollama,均通过 LangChain 官方集成包接入。

## 目录结构

```
src/
  main.py              CLI 入口与交互式循环。
  app_settings.py      config.toml 加载器(+ 环境变量覆盖)。
  llm.py               按 agent 的聊天模型工厂(4 个 provider)。
  pydantic_models.py   结构化输出 / 产物的 schema。
  visualize_graph.py   把 LangGraph 渲染成图片(pixi run graph)。

  agents/
    supervisor.py        编排者:ReAct 回合 + 确定性流水线。
    planner_blueprint.py 为一个请求生成有序部件列表。
    planner_part.py      单个部件的步骤(每次试验一次调用)。
    plan_reviewer.py     试验 PNG + 视觉裁决。
    navigator.py         在 turtle 上执行计划(不用 LLM)。

  tools/
    turtle_motion_tools.py  位姿 / 移动 / 旋转 / 画笔 / 清空 / 重置。

  workflow/
    graph.py           LangGraph 装配 + run_mission()。
    types.py           图状态 + 路由标签。

  utils/
    agent_runtime.py   给 @tool 用的 bridge 与任务记事本。
    mission_state.py   MissionContext(每个任务一份记事本)。
    ros_bridge.py      ROS 2 turtlesim 桥接(含 mock 后端)。
    turtle_control.py  底层 turtle 指令。
    path_tracker.py    墨迹线段记忆。
    path_simulation.py 计划试运行 + 画布边界检查。
    drawing_render.py  线段 → PNG(Pillow)。
    canvas_geometry.py 11×11 画布常量。
    plan_execution.py  确定性的计划执行器。
    plan_normalize.py  去掉 teleport、修正负向移动、对齐笔宽。
    turtle_park.py     绘制结束后停靠 turtle。
    cli_spinner.py     LLM 调用期间的终端加载动画。
```

## 配置

| 来源 | 用途 |
| --- | --- |
| `config.toml` | 默认及按 agent 的 LLM 配置、审查循环(`enabled`、`max_trials`、`fail_action`)、停靠行为 |
| `.env` | API 密钥 + 可选的 LangSmith 追踪(见 [LANGSMITH.zh.md](LANGSMITH.zh.md)) |

用于临时单次运行的环境变量覆盖:`REVIEW_ENABLED`、`MAX_TRIALS`、`FAIL_ACTION`、`PARK_TURTLE_AFTER_DRAW`、`APP_CONFIG`(指定另一个 TOML 路径)。

## 可观测性

在 `.env` 中设置 `LANGSMITH_*` 变量,就能在 LangSmith 里拿到每个任务的完整追踪(每个 agent 的 LLM 调用、prompt、图像、token 数、延迟),无需改动代码。详见 [LANGSMITH.zh.md](LANGSMITH.zh.md)。
