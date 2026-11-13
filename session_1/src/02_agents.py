#!/usr/bin/env python3
"""
02_agents.py — 智能体（Agent）练习 2
======================================

在练习 1 中，每次 llm.invoke() 都是一次单程往返：输入一个提示、得到一个回复，交互随即结束。

智能体（agent）改变了这一模式：它把模型放进一个循环里。模型不再只生成单条回复，
而是可以调用工具、观察工具结果，并持续推理，直到任务完成。

本脚本你将学到：
- 手写 ReAct 循环（2a）
- 观察不断增长的上下文窗口（2b）
- 用 @tool 装饰器定义工具（2c）
- 用 Pydantic 做输入校验（2d）
- 把工具绑定到模型（2e）
- 完整的工具调用智能体循环（2f）
- 常见失败模式（2g）

运行方式（容器内 conda base 环境）：
    enter_ultron
    cd /home/robot/github/llm-misc/ReasonPlanAct2026/session_1/src
    eval "$(/data/miniforge3/bin/conda shell.bash hook)"
    conda activate base
    python 02_agents.py

关于消息类型（本节新增 ToolMessage）：
    SystemMessage   开发者 → 模型    角色、约束、工具说明
    HumanMessage    用户   → 模型    任务或问题
    AIMessage       模型   → 开发者   回复，可能包含 tool_calls
    ToolMessage     工具   → 模型    工具调用的结果，通过 tool_call_id 关联
"""

from dataclasses import dataclass, field

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from llm_config import get_llm
from log_utils import log  # log() 等价 print()，但会打印「[文件名:行号]」前缀


# =============================================================================
# 背景 —— 什么是智能体？
# =============================================================================
# 普通的 LLM 调用是无状态的：输入文本、输出文本、结束。
# 智能体是一个循环，让模型可以决定调用工具、观察结果，并持续推理直到得出最终答案。
#
# 智能体有两个主要部分：
#   1. 大脑 —— LLM，负责推理与决策。
#   2. 身体 —— 工具与接口，让智能体能与环境交互。
#
# ReAct 模式（Reason + Act，推理 + 行动）把每一步组织为：
#   Thought（思考）: 我需要知道乌龟当前的位姿。
#   Action（行动）:  get_pose
#   Action Input:    {}
#   Observation（观察）: x=5.5, y=5.5, heading=0°
#   Thought: 我已经拿到位姿，可以规划移动了。
#   Final Answer（最终答案）: ...
#
# 循环持续进行，直到模型输出 "Final Answer:"。模型看到的一切（每次观察）
# 都存在于上下文窗口中。没有隐藏状态。记忆 = 消息历史。


# =============================================================================
# 练习 2a —— 手写智能体循环（演示：运行即可，无需修改）
# =============================================================================
# 在使用任何框架之前，这里手动构建 ReAct 循环：模型以纯文本输出 Thought / Action
# 等行，我们用字符串匹配解析它们，调用对应函数，再把结果作为 Observation 回灌。
#
# 这种做法能用，但很脆弱 —— 模型输出里一个意外的空格或大小写就会让解析器失败。
# 正是这种脆弱性促使我们在 2d 转向结构化的工具调用。

REACT_SYSTEM = """你是一个控制 TurtleSim 机器人的推理智能体。
你在一个 11 × 11 米的场地中运行。乌龟初始位置为 x=5.5, y=5.5。

你有以下工具：
  get_pose()                    —— 返回当前 (x, y, heading_deg)
  move_forward(distance_m)      —— 让乌龟向前移动 distance_m 米
  rotate(angle_deg)             —— 旋转乌龟；正值 = 逆时针

每次只输出恰好一个 Thought/Action 组合，然后停下等待。
不要在一条回复里连续执行多个动作。

严格按如下格式回复，一次只走一步：

Thought: <你的推理>
Action: <工具名>
Action Input: <参数，或留空>

当你得出最终答案时，写：
Thought: I now know the answer.
Final Answer: <你的答案>

开始。"""


def fake_get_pose() -> str:
    return "x=5.5, y=5.5, heading_deg=0.0"


def fake_move_forward(distance_m: float) -> str:
    return (
        f"向前移动了 {distance_m} 米。"
        f"新位姿：x=5.5, y={5.5 + distance_m:.2f}, heading_deg=0.0"
    )


def fake_rotate(angle_deg: float) -> str:
    return f"旋转了 {angle_deg}°。新朝向：{angle_deg % 360:.1f}°"


# 工具名 -> 可调用对象（桩实现，stub）
TOOL_MAP = {
    "get_pose": lambda arg: fake_get_pose(),
    "move_forward": lambda arg: fake_move_forward(float(arg)),
    "rotate": lambda arg: fake_rotate(float(arg)),
}


@dataclass
class ReactResult:
    answer: str
    steps_taken: int
    success: bool
    history: list = field(default_factory=list)


def run_react_loop(
    task: str, max_steps: int = 8, verbose: bool = False
) -> ReactResult:
    """手写 ReAct 循环：解析文本 -> 调用工具 -> 回灌观察结果。"""
    llm = get_llm()
    history = [SystemMessage(REACT_SYSTEM), HumanMessage(f"任务：{task}")]

    for step in range(max_steps):
        response = llm.invoke(history)
        text = response.text.strip()

        if verbose:
            log(f"\n── 第 {step + 1} 步 {'─' * 40}")
            log(text)

        history.append(AIMessage(text))

        # 1) 是否给出了最终答案？
        for line in text.splitlines():
            if line.startswith("Final Answer:"):
                answer = line.removeprefix("Final Answer:").strip()
                if verbose:
                    log(f"\n✓ 最终答案：{answer}")
                return ReactResult(answer, step + 1, True, history)

        # 2) 解析 Action / Action Input
        action = action_input = None
        for line in text.splitlines():
            if line.startswith("Action:") and action is None:
                action = line.removeprefix("Action:").strip().rstrip("()")
            if line.startswith("Action Input:") and action is not None:
                action_input = line.removeprefix("Action Input:").strip()
                break

        # 3) 解析失败 —— 要求模型重试
        if not action:
            obs = (
                "[解析错误] 你的回复必须恰好包含一个 Thought/Action 组合，"
                "或一个 Final Answer。请重试。"
            )
            if verbose:
                log(f"\n  ⚠ {obs}")
            history.append(HumanMessage(f"Observation: {obs}"))
            continue

        # 4) 执行工具
        if action in TOOL_MAP:
            try:
                observation = TOOL_MAP[action](action_input or "")
            except (ValueError, TypeError) as e:
                observation = f"[工具错误] '{action}' 的输入无效：{e}"
        else:
            observation = (
                f"[未知工具] '{action}' 不存在。"
                f"可用工具：{list(TOOL_MAP.keys())}"
            )

        if verbose:
            log(f"\n  → Observation: {observation}")
        history.append(HumanMessage(f"Observation: {observation}"))

    return ReactResult("达到最大步数仍未得到最终答案。", max_steps, False, history)


# =============================================================================
# 练习 2b —— 查看不断增长的上下文窗口
# =============================================================================
# 每个 Thought / Action / Observation 循环都会往 history 追加消息。
# 这个列表「就是」智能体的记忆 —— 别无其他。
#
# 观察要点：
# - 每次往返会新增一条 AIMessage（模型的 Thought/Action）和一条 HumanMessage
#   （Observation）。即每步两条消息。
# - SystemMessage 是唯一不变的部分 —— 它是智能体的固定上下文。
# - 任务越长，消息累积越多。上下文窗口有上限，长时间运行的智能体最终会耗尽空间。
#   这也是规划模式（04 讲义）存在的原因之一 —— 它在开头用一次 LLM 调用，
#   而不是每步一次。

def inspect_history(result: ReactResult) -> None:
    history_debug = result.history  # 来自练习 2a 的消息列表

    log(f"history 中的消息数：{len(history_debug)}")
    log(f"类型               ：{[type(m).__name__ for m in history_debug]}")
    log(f"大致字符数         ：{sum(len(m.text) for m in history_debug)}")
    log()
    for i, m in enumerate(history_debug):
        preview = m.text[:60].replace("\n", " ")
        log(f"  [{i}] {type(m).__name__:<15} {preview}...")


# =============================================================================
# 练习 2c —— 用 @tool 定义工具
# =============================================================================
# 在 2a 中，模型以纯文本描述动作（Action: move_forward），由我们自己解析。
# 这能用，但很容易出错。
#
# @tool 装饰器提供了更好的东西：把一个 Python 函数变成带名称、描述和类型化参数
# 模式（schema）的结构化对象。模型据此输出结构化 JSON，而非自由文本。
#
# 装饰器自动做三件事：
#   1. tool.name        —— 取自函数名
#   2. tool.description —— 取自 docstring（文档字符串）← 这是模型读取的内容
#   3. tool.args        —— 根据类型注解构建的 JSON schema
#
# 模型永远看不到你的 Python 代码。它只看到 name、description 和 schema。
# 写这三样东西时，要像写 API 契约一样认真。

@tool
def get_pose() -> str:
    """返回乌龟当前位姿 (x, y, heading_deg)。无需参数。"""
    return "x=5.5, y=5.5, heading_deg=0.0"


@tool
def move_forward(distance_m: float) -> str:
    """沿当前朝向让乌龟向前移动 distance_m 米。

    Args:
        distance_m: 移动距离（米）。只能为正。
    """
    return f"向前移动了 {distance_m} 米。"


@tool
def rotate(angle_deg: float) -> str:
    """让乌龟原地旋转。

    Args:
        angle_deg: 旋转角度（度）。正值 = 逆时针。
    """
    return f"旋转了 {angle_deg}°。"


TOOLS = [move_forward, get_pose, rotate]


def show_tool_schemas() -> None:
    log("模型可见的工具 schema：\n")
    for t in TOOLS:
        log(f"  {t.name}")
        log(f"    描述       : {t.description}")
        log(f"    参数 schema: {t.args}")
        log()
    # 观察要点：
    # - 描述是模型唯一拥有的文档。描述含糊 → 参数值出错。
    # - 单位歧义是常见 bug：distance 指米还是格子？永远在参数名里标明单位
    #   （distance_m、angle_deg）。
    # - JSON schema 是模型调用工具时要填的内容。缺少注解 → 缺少 schema → 模型瞎猜类型。


# =============================================================================
# 练习 2d —— 用 Pydantic 做输入校验
# =============================================================================
# @tool 装饰器接受一个 Pydantic BaseModel 作为参数 schema。
# 这样就能添加范围检查、别名和默认值，在坏的模型输出到达机器人「之前」就拦截它。
#
# ── 这句话到底是什么意思？ ────────────────────────────────────────────────────
# 它说的是：你可以给 @tool 传一个 Pydantic 的 BaseModel 类，用来定义/约束这个
# 工具的参数。这个 BaseModel 就充当工具的「参数 schema（结构说明 + 校验规则）」。
#
# 1) 什么是 Pydantic BaseModel？
#    Pydantic 是一个数据校验库。继承 BaseModel 定义一个类，用「字段 + Field(...)」
#    声明每个参数的类型、取值范围、描述、默认值等约束。例如下面的 MoveArgs：
#        distance_m: float = Field(..., description="...", gt=0, le=10)
#      - ...  表示必填（没有默认值）
#      - gt=0 表示必须 > 0；le=10 表示必须 <= 10
#    用非法值构造它会直接抛错：
#        MoveArgs(distance_m=-1.0)  # → ValidationError: Input should be greater than 0
#    这正是本节 demo_validation() 演示的。
#
# 2) 把它交给 @tool 做「参数 schema」是什么意思？
#    普通写法下，@tool 会从函数的「类型注解」自动推断参数 schema（见练习 2c：
#    move_forward(distance_m: float) → schema 只有 {'type': 'number'}，没有范围限制）。
#    而「@tool 接受一个 BaseModel 作为参数 schema」指的是这种写法
#    （03/04 的 utils_03.py 里真正用到了）：
#        @tool(args_schema=MoveArgs)        # ← 把 MoveArgs 作为参数 schema 传进去
#        def move_forward(distance_m: float, ...) -> str: ...
#    这样这个 BaseModel 同时起两个作用：
#      - 对模型(LLM)：schema 里的范围、描述会一起发给模型，告诉它「这个参数长什么样、
#        合法范围是多少」，帮助它生成更靠谱的参数。
#      - 对执行  ：每次调用工具前，Pydantic 先按这些规则校验模型传来的参数。若越界
#        （如 distance_m=-5），会在工具函数体真正运行「之前」抛出 ValidationError ——
#        也就是在「命令到达机器人之前」拦截掉坏输入。
#    所以上面那句「在坏的模型输出到达机器人之前就拦截它」就是这个意思。
#
# 3) 对比：
#    | 写法                         | 参数 schema 来源              | 能做范围校验吗          |
#    | @tool（仅靠类型注解）        | 从 distance_m: float 推断     | 否：只保证是数字，-5 也放行 |
#    | @tool(args_schema=MoveArgs)  | 由 Pydantic BaseModel 提供    | 是：可加 gt/le/描述/默认值  |
#    练习 2g(demo_failure_modes) 能直观看到区别：无 schema 时 distance_m=-10 被放行；
#    用 @tool(args_schema=MoveArgs) 后 -10 会被 ValidationError 挡下。
#
# 一句话：@tool 接受一个 Pydantic BaseModel 作为参数 schema = 用一个 Pydantic 类来
# 描述并校验工具参数，@tool 用它告诉模型参数规格、并在调用前自动校验参数合法性。

class MoveArgs(BaseModel):
    distance_m: float = Field(
        ...,
        description="向前移动的距离（米）。",
        gt=0,
        le=10,
    )


def demo_validation() -> None:
    # 直接测试校验 —— 不涉及智能体
    try:
        MoveArgs(distance_m=-1.0)  # 应当失败
    except Exception as e:
        log(f"捕获到非法距离：{e}")


# =============================================================================
# 练习 2e —— 把工具绑定到模型
# =============================================================================
# llm.bind_tools(tools) 做两件事：
#   1. 把工具 schema 发给模型，让它知道有哪些可用工具
#   2. 把模型的输出格式从自由文本切换为结构化的 tool_calls
#
# 你不再需要从字符串里解析 "Action: move_forward"，而是直接得到一个 Python 字典：
#   {'name': 'move_forward', 'args': {'distance_m': 2.0}, 'id': '...'}
# 无需解析，也没有脆弱的字符串匹配。
#
# 每个 tool call 上的 id 字段很重要：返回结果时，你把它包进一个带相同
# tool_call_id 的 ToolMessage，模型才知道这个结果属于哪次调用。

def demo_bind_tools() -> None:
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    # 确认：无需把工具写进系统提示词，模型也能知道它们。
    log("绑定的工具 schema：", llm_with_tools.kwargs.get("tools"))
    log()

    messages = [
        SystemMessage(
            "你正在控制一个 TurtleSim 机器人。请使用你的工具来完成用户的请求。"
        ),
        HumanMessage("让乌龟向前移动 1 米，并告诉我它的新位姿。"),
    ]

    response = llm_with_tools.invoke(messages)

    log("response.text      :", response.text)
    log("response.tool_calls:", response.tool_calls)
    log()

    # 手动执行工具调用，以便看到完整的往返过程
    if response.tool_calls:
        for tc in response.tool_calls:
            log(f"  → 调用 : {tc['name']}({tc['args']})")
            match = next((t for t in TOOLS if t.name == tc["name"]), None)
            if match:
                result = match.invoke(tc["args"])
                log(f"  → 结果 : {result}")
            log(f"  → ID   : {tc['id']}   ← 用于关联 ToolMessage")


# =============================================================================
# 练习 2f —— 完整的工具调用智能体循环（主活动）
# =============================================================================
# 把工具调用的响应接入一个完整循环。模式如下：
#   [SystemMessage, HumanMessage]
#     → llm_with_tools.invoke()
#     → AIMessage  ← 可能包含 tool_calls
#     → 对每个 tool call：运行工具 → ToolMessage(result, tool_call_id=...)
#     → 把所有 ToolMessage 追加到 history
#     → 再次 llm_with_tools.invoke()
#     → 重复，直到 response.tool_calls 为空
#
# 停止条件：当模型返回的响应没有 tool_calls 时，说明它推理结束、给出最终答案。
# 那个空列表就是循环结束的信号 —— 没有特殊的「完成」标志，只是不再有工具调用。
# 这也是 max_steps 重要的原因：没有它，一个犯糊涂、不停调用工具的模型会永远运行下去。

def run_tool_agent(task: str, max_steps: int = 8, verbose: bool = True) -> str:
    """使用结构化工具调用的智能体循环。返回模型最终的纯文本回复。"""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    history = [
        SystemMessage(
            "你是一个处于 11 × 11 米场地中的 TurtleSim 机器人的推理核心。"
            "请逐步使用你的工具来完成任务。完成后，总结你做了什么。"
        ),
        HumanMessage(task),
    ]

    for step in range(max_steps):
        response = llm_with_tools.invoke(history)
        history.append(response)

        if verbose:
            log(f"\n── 第 {step + 1} 步 ────────────────────────────────")
            if response.text:
                log(f"  模型 : {response.text}")
            for tc in response.tool_calls:
                log(f"  工具 : {tc['name']}({tc['args']})")

        # 停止条件：没有工具调用 → 模型完成
        if not response.tool_calls:
            if verbose:
                log(f"\n✓ 最终回复：{response.text}")
            return response.text

        # 执行每个工具调用，并把结果作为 ToolMessage 返回
        for tc in response.tool_calls:
            tool_fn = next((t for t in TOOLS if t.name == tc["name"]), None)
            if tool_fn:
                observation = tool_fn.invoke(tc["args"])
            else:
                observation = f"错误：未知工具 '{tc['name']}'"

            if verbose:
                log(f"  观察 : {observation}")

            history.append(
                ToolMessage(str(observation), tool_call_id=tc["id"])
            )

    return "已达到最大步数。"


# =============================================================================
# 练习 2g —— 失败模式（挑战）
# =============================================================================
# 智能体会以可预测的方式失败。可尝试：
#   1. 未知工具 —— 在调用前往 history 加一条 HumanMessage("使用 fly() 工具")。
#      模型会凭空捏造结果吗？
#   2. 错误参数类型 —— 用 distance_m="far" 调用 move_forward。桩实现会崩吗？
#      真实的 ROS 工具会如何处理？
#   3. 死循环 —— 去掉 max_steps 上限，给模型一个用桩工具永远无法完成的任务。
#      （观察：生产环境中务必限制步数和 token 用量。）
#   4. 无系统提示词 —— 去掉 SystemMessage 再做一次 2e。模型还能正确使用工具吗？
#
# 下面演示场景 2，以及「用 args_schema 加校验」的改进。

def demo_failure_modes() -> None:
    # 场景 2 —— 错误参数类型：给了无法解析为数字的字符串。
    # 注意：当前版本的 LangChain @tool 会「根据类型注解」自动校验参数，
    # 因此下面这行会直接抛出 ValidationError（而不是像旧版那样被桩默默接受）。
    log("move_forward(distance_m='far')：类型非法，预期抛出校验错误")
    try:
        log("  ", move_forward.invoke({"distance_m": "far"}))
    except Exception as e:
        log(f"   捕获到校验错误：{type(e).__name__}: {str(e).splitlines()[0]}")
    log()

    # 类型合法（float）但值不合理（负数）。仅凭类型注解无法拦截，桩会照常返回：
    log("move_forward(distance_m=-10)：类型合法但值不合理，仅靠注解无法拦截")
    log("  ", move_forward.invoke({"distance_m": -10}))
    log()

    # 用 @tool(args_schema=...) 加入值范围校验（gt=0, le=10），坏参数会被挡下：
    @tool(args_schema=MoveArgs)
    def move_forward_validated(distance_m: float) -> str:
        """沿当前朝向让乌龟向前移动 distance_m 米。

        Args:
            distance_m: 移动距离（米）。只能为正。
        """
        return f"向前移动了 {distance_m} 米。"

    log("move_forward_validated(distance_m=-10)：带范围校验，预期抛出校验错误")
    try:
        log("  ", move_forward_validated.invoke({"distance_m": -10}))
    except Exception as e:
        log(f"   捕获到校验错误：{type(e).__name__}: {str(e).splitlines()[0]}")


# =============================================================================
# 小结
# =============================================================================
# | 概念            | 工作方式                                                       |
# |-----------------|----------------------------------------------------------------|
# | LLM 即函数      | llm.invoke(messages) -> AIMessage                              |
# | 智能体循环      | 调用(思考) → 执行工具调用(行动) → 追加 ToolMessage(观察) → 重复 |
# | 记忆            | history 列表 —— 别无其他                                       |
# | 工具 schema     | @tool：名称取自函数、描述取自 docstring、参数取自类型注解       |
# | 结构化调用      | llm.bind_tools(tools) → response.tool_calls 是一组字典          |
# | 停止条件        | 当 response.tool_calls 为空时退出循环 —— 不再调用即表示完成     |
# | 安全网          | max_steps 限制循环；没有它，卡住的智能体会永远运行             |
#
# 下一步：03_tools.ipynb —— 用真实的 ROS action 和 service 替换上面的桩，
# 加入输入校验让智能体在参数出错时能恢复而非崩溃，并把完整工具集接回本循环。


def main() -> None:
    log("=" * 60)
    log("练习 2a —— 手写智能体循环")
    log("=" * 60)
    log("正在运行 ReAct 循环...")
    result = run_react_loop(
        "乌龟当前的位姿是多少？然后让它向前移动 2 米。",
        verbose=True,
    )

    log("\n" + "=" * 60)
    log("练习 2b —— 查看不断增长的上下文窗口")
    log("=" * 60)
    inspect_history(result)

    log("\n" + "=" * 60)
    log("练习 2c —— 用 @tool 定义工具")
    log("=" * 60)
    show_tool_schemas()

    log("=" * 60)
    log("练习 2d —— 用 Pydantic 做输入校验")
    log("=" * 60)
    demo_validation()

    log("\n" + "=" * 60)
    log("练习 2e —— 把工具绑定到模型")
    log("=" * 60)
    demo_bind_tools()

    log("\n" + "=" * 60)
    log("练习 2f —— 完整的工具调用智能体循环")
    log("=" * 60)
    log("正在运行结构化工具调用智能体...\n")
    run_tool_agent(
        "获取乌龟当前位姿，让它向前移动 2 米，再旋转 90 度。最后报告最终位姿。"
    )

    log("\n" + "=" * 60)
    log("练习 2g —— 失败模式（挑战）")
    log("=" * 60)
    demo_failure_modes()


if __name__ == "__main__":
    main()
