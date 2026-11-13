# LLM Agent 技术栈背景资料

本文整理三个基础问题:JSON Schema 是什么、LangChain 是什么、以及这个领域还有哪些同类工具(重点是机器人方向)。示例均取自本仓库 `session_1` 与 `session_2` 的代码。

- [1. JSON Schema](#1-json-schema)
- [2. LangChain](#2-langchain)
- [3. 同类工具与替代方案](#3-同类工具与替代方案)

---

## 1. JSON Schema

JSON Schema 是用 JSON 写的「数据结构说明书」——它本身也是一份 JSON,用来描述另一份 JSON 应该长什么样:有哪些字段、每个字段什么类型、哪些必填、取值范围、枚举值等。它是一个正式标准,当前常用 draft 2020-12。

### 核心关键字

```json
{
  "type": "object",
  "properties": {
    "r":      { "type": "integer", "minimum": 0, "maximum": 255, "default": 255 },
    "width":  { "type": "integer", "minimum": 1, "maximum": 20 },
    "off":    { "type": "boolean" },
    "action": { "type": "string", "enum": ["move_forward", "rotate", "teleport"] }
  },
  "required": ["width"]
}
```

`type` 定类型,`properties` 定字段,`required` 定必填,`minimum` / `maximum` / `enum` / `items` 之类做约束,`description` 是给人(和给模型)看的说明。

### 三种用途

1. **校验** — 拿 schema 去检查一份数据合不合法。
2. **契约与文档** — OpenAPI 的请求响应体就是 JSON Schema。
3. **告诉 LLM 输出的形状** — 本仓库用的是这一种。

### 在本仓库中的位置

工具调用(tool calling)和结构化输出(structured output)背后传给模型的参数定义,就是 JSON Schema。`session_2/src/pydantic_models.py` 里的 Pydantic 模型不是直接发给模型的,LangChain 会调 `Model.model_json_schema()` 转成 JSON Schema 再发出去。

对应关系:

| Pydantic 写法 | 生成的 JSON Schema |
| --- | --- |
| `r: int = Field(ge=0, le=255)` | `{"type": "integer", "minimum": 0, "maximum": 255}` |
| `action: StepAction`(Enum) | `{"enum": ["set_pen", "move_forward", ...]}` |
| `Field(description=...)` | `description` 字段 |
| `float \| None` | `{"anyOf": [{"type": "number"}, {"type": "null"}]}` |

`Field(description=...)` 是模型真正读到的提示,所以 `ShapeBlueprint.parts` 那句 "Ordered snake_case part names, large base parts first" 实际上起的是 prompt 的作用。

### 一个真实的坑

`MovementPlanOutput` 的注释写着 "some providers reject tuple schema fields",这正是 JSON Schema 层面的限制:Python 的 `tuple[float, float, float]` 会生成带 `prefixItems` 的定长数组 schema,部分厂商的严格模式不支持,所以 `starting_pose` 被挪出结构化输出、在提交时才从实时 turtle 位姿注入。

用 `model_json_schema()` 可以打印完整结果(需要在 pixi 环境里跑,系统自带的 python3 版本过低,不支持 `X | None` 语法)。

---

## 2. LangChain

LangChain 是一个把大模型接进应用的框架。它自己不提供模型,做的是**抽象层加编排**:把「聊天模型、消息、工具、结构化输出」统一成一套接口,这样换厂商、加工具、串流程时不用重写业务代码。

### 生态分层

装的不是一个包,是一组:

| 包 | 职责 | 本仓库中的使用 |
| --- | --- | --- |
| `langchain-core` | 纯抽象,无具体实现。`BaseChatModel`、各类 Message、`@tool`、Runnable | `utils_03.py`、`pydantic_models.py` 只依赖这一层 |
| `langchain-{google-genai,openai,anthropic,ollama}` | 各厂商适配器,一家一个独立包 | `pixi.toml` 里四个都装了 |
| `langchain` | 上层预制件,如 `create_agent` | `session_1/src/02_agents.ipynb` |
| `langgraph` | 有状态的图式编排,多智能体、循环、人工介入 | `session_2/src/workflow/graph.py` |
| LangSmith | 追踪与调试 | 见 `session_2/docs/LANGSMITH.md` |

### 统一的模型接口

`session_1/src/llm_config.py` 是这个价值的直接体现。四个 provider 返回四个不同的类,但对外都是同一套 `.invoke()` / `.stream()` / `.bind_tools()`,所以切 Gemini、Ollama、GPT、Claude 只改一个环境变量 `LLM_PROVIDER`,notebook 里的代码一行不动。底层的鉴权、请求格式、流式解析差异被适配器吃掉了。

代价是抽象层总会漏,某个厂商的独有参数可能要绕过接口。

### 消息与工具

对话状态是一个 message 列表,四种角色:

- `SystemMessage` — 设定
- `HumanMessage` — 用户输入
- `AIMessage` — 模型回复,可能带 `tool_calls`
- `ToolMessage` — 工具执行结果,用 `tool_call_id` 对应回去

工具接上第 1 节:`@tool` 装饰器读函数的 docstring 和类型标注,`args_schema=MoveArgs` 用 Pydantic 覆盖参数定义,最后转成 JSON Schema 发给模型。所以 `MoveArgs` 里 `distance_m: float = Field(gt=0, le=10, description=...)` 的约束和描述,模型是真能看到的。

`llm.bind_tools(TOOLS)` 返回一个知道这些工具存在的新模型对象,调用后从 `response.tool_calls` 拿到模型想调什么、参数是什么。

### Agent 循环

`session_1/src/utils_03.py` 的 `run_tool_agent` 是手写的 ReAct 循环,它揭示了 agent 没什么魔法:

```
invoke → 有 tool_calls? → 执行工具 → 结果塞回 history 当 ToolMessage → 再 invoke
                       → 没有 tool_calls? → 收工
```

`safe_invoke` 那段是关键设计:Pydantic 校验失败时不抛异常,而是把错误信息当作观测结果回给模型,让它自己改参数重试。

`02_agents.ipynb` 里的 `from langchain.agents import create_agent` 是官方封装好的同一个循环。这是 LangChain 1.0 的 API,老教程里的 `create_react_agent` 是它的前身。

### LangGraph

单个 while 循环撑不住复杂流程时就上 LangGraph。`session_2/src/workflow/graph.py` 用 `StateGraph` 把 blueprint → planner → reviewer → navigator 显式画成节点和边,状态在 `workflow/types.py` 里定义。

区别在于控制流是写死的图,而不是靠模型自己决定下一步——reviewer 驳回就连回 planner,这种带条件的回环用裸循环写会很难维护。

### 什么时候值得用

值得:要支持多个 provider;重度用工具调用和结构化输出(Pydantic → JSON Schema 这条链省很多手写代码);流程有分支、循环、多个 agent。

不值得:只是单轮调一次模型拿段文本,直接用厂商 SDK 更短更透明。

> **版本提醒**:这个框架迭代很快(0.1 → 0.3 → 1.0 都有破坏性改动,包也拆过几次),网上教程很容易是过时 API。本仓库 `session_1` 用的是 1.0 风格,而 `session_2/pixi.toml` 还锁在 `langchain >=0.3.0`,两个 session 的 API 版本假设并不完全一致。

---

## 3. 同类工具与替代方案

本仓库的定位是「LLM 当高层规划器 → 输出结构化计划 → 通过 ROS 工具执行」。先按层分清楚,才知道哪些是替代品、哪些是补充。

### 3.1 通用 agent 框架(LangChain 的直接替代)

- **LangGraph** — 已在用。图式编排,做生产级 agent 的主流选择。
- **OpenAI Agents SDK** / **Google ADK** / **Microsoft Agent Framework**(AutoGen 与 Semantic Kernel 合并后的产物)— 厂商自家 SDK,比 LangChain 薄,但多 provider 支持较弱。
- **CrewAI** — 角色化多智能体。session_2 的 blueprint / planner / reviewer / navigator 分工用它表达会很自然,但控制流不如 StateGraph 可控。
- **PydanticAI** / **Mirascope** — 极简派,贴近原生 Python,依赖少。本仓库已重度用 Pydantic,PydanticAI 会很顺手。
- **DSPy** — 不手写 prompt,声明输入输出签名后由框架自动优化。适合有评测集要调准确率的场景。

这一层的[对比资料](https://www.langchain.com/resources/ai-agent-frameworks)很多,但都是通用场景,没有机器人视角。

### 3.2 机器人专用的 LLM agent 框架

与本仓库最对位的一层:

- **[ROSA](https://github.com/nasa-jpl/rosa)**(NASA JPL)— 建在 LangChain 上的 ReAct agent,支持 ROS1 和 ROS2,用自然语言跟机器人交互。跟 session_1 做的事几乎一样,只是它把 ROS 内省(话题、节点、TF、日志)也包成了工具。有[论文](https://arxiv.org/html/2410.06472v1)。**最值得先读的一个。**
- **[RAI](https://arxiv.org/html/2505.07532v1)**(Robotec.ai)— 定位是 embodied AI 的通用 agent 框架,开箱集成 ROS 2,专门处理具身化问题(多模态感知接入、真机加数字孪生联调)。论文在 Husarion ROSBot XL 上做了实机验证。比 ROSA 更重、更完整。
- **[ROS-LLM](https://arxiv.org/html/2406.19741v3)** — 强调任务反馈闭环和结构化推理,非专业人员通过聊天界面下达任务。其「结构化推理 + 反馈」思路与本仓库 reviewer 驳回后回环给 planner 是同一套路。
- **LLM + Nav2** — 学术界有一批把 LLM 接到 Nav2 导航栈的工作,例如这篇做[延迟感知基准测试](https://www.mdpi.com/1424-8220/26/2/608/xml)的,拿多个 LLM 跑自然语言导航并测端到端时延。机器人场景下时延是硬约束,这个角度值得注意。

### 3.3 MCP 路线

比框架更松耦合:把机器人包成 MCP server,任何支持 MCP 的客户端都能驱动。

- **[ros-mcp-server](https://github.com/robotmcp/ros-mcp-server)** — 桥接 LLM 与 ROS/ROS2,卖点是不改机器人现有源码,双向通信。
- **[Yutarop/ros-mcp](https://github.com/Yutarop/ros-mcp)** — 覆盖 topic、service、action 三种通信方式,任意消息类型。

好处是解耦,机器人能力和 agent 逻辑分开演进。坏处是多一跳进程通信,实时性更差,而且工具粒度由 server 决定,没法像现在这样精细控制给模型看到的 schema 和 description——`utils_03.py` 里那些精心写的 docstring 和参数约束恰恰是效果的关键,交给通用 MCP server 就丢掉了。

### 3.4 端到端 VLA(不是替代,是另一层)

这一层直接从图像加语言输出关节动作,绕过「生成计划再调 API」:

- **[π0](https://arxiv.org/abs/2410.24164)**(Physical Intelligence)— 在预训练 VLM 上加 flow matching 输出连续动作,跨多种本体训练,能做叠衣服、清桌子这类任务。
- **[OpenVLA](https://arxiv.org/abs/2406.09246)** — 开源 7B VLA,论文报告在 29 个任务上比 55B 的 RT-2-X 成功率高 16.5%,参数少 7 倍。
- **[GR00T N1.7](https://huggingface.co/blog/nvidia/gr00t-n1-7)**(NVIDIA)— 3B 开放推理 VLA,面向通用人形机器人,已[接入 LeRobot](https://huggingface.co/blog/nvidia/nvidia-isaac-teleop-and-gr00t17-in-lerobot)。
- **[LeRobot](https://huggingface.co/docs/lerobot/main/en/pi0)**(HuggingFace)— 这一层的事实标准工具库,数据集格式、训练、部署都在里面。

关键区别:VLA 解决**低层控制**(怎么动),LangChain 这条线解决**高层任务分解**(先做什么后做什么)。实际系统往往两层叠着用——LLM 出子任务序列,VLA 或传统控制器执行。turtlesim 用不上 VLA,因为动作空间就是 `cmd_vel`,没有需要学习的操作技能。

### 3.5 经典规划层

容易被忽略,但很实用:

- **PDDL + LLM** — 让 LLM 生成 PDDL 问题描述,交给成熟的符号规划器求解,而不是让 LLM 自己排步骤。规划的正确性由求解器保证。
- **行为树**(`py_trees`、BehaviorTree.CPP)— Nav2 本身就是行为树驱动的。让 LLM 生成或选择行为树,而不是生成一串原子动作,可靠性高得多,也天然支持中断和恢复。
- **SkiROS2** — ROS 2 上的技能式规划框架。

这条路线的共识是:**LLM 擅长把模糊需求翻译成形式化表示,不擅长保证长序列执行的正确性**。session_2 加 reviewer 做视觉审查,本质上是用另一个 LLM 补这个缺口;符号规划器和行为树是用确定性方法补同一个缺口。

### 3.6 对本仓库的建议

教学目的下,现在这套手写实现是对的——`run_tool_agent` 那个循环把 agent 的机制暴露得很清楚,换成任何框架都会把这部分藏起来。

值得对照阅读的是 **ROSA**(同样基于 LangChain,能直接对比设计取舍)和 **RAI**(看一个更工业化的 ROS 2 agent 架构长什么样)。如果后面要讲「为什么纯 LLM 规划不可靠」,行为树那条线是很好的对照组。

---

## 分层总览

| 层 | 关心的问题 | 代表工具 | 本仓库对应 |
| --- | --- | --- | --- |
| 数据契约 | 输出该长什么样 | JSON Schema、Pydantic | `pydantic_models.py` |
| 模型接入 | 怎么统一调各家模型 | LangChain、厂商 SDK | `llm_config.py`、`llm.py` |
| Agent 循环 | 什么时候调哪个工具 | LangChain `create_agent`、CrewAI | `utils_03.py` |
| 流程编排 | 多角色怎么协作与回环 | LangGraph、行为树 | `workflow/graph.py` |
| 机器人集成 | 怎么接上 ROS | ROSA、RAI、ros-mcp-server | `utils/ros_bridge.py`、`tools/` |
| 低层控制 | 关节怎么动 | π0、OpenVLA、GR00T | `cmd_vel`(无需学习) |

第 3 节内容为对来源材料的转述与概括,以符合许可要求。
