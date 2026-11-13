#!/usr/bin/env python3
"""
04_planning.py —— 规划（Planning）练习 4
==========================================

在练习 2 中，智能体「边想边做」—— 一次只决定一个工具调用。
对于简单移动这没问题；但画六边形就不行了：模型要在 12 次工具调用里把整个几何
形状「记在脑子里」，任何一步推理出错都会随步数累积放大。

「先规划，后执行」把工作分成两个阶段：
    阶段 1 —— 规划（PLAN）   ：LLM 产出一份完整计划（JSON 或 代码）
    阶段 2 —— 执行（EXECUTE）：Python 执行该计划 —— 不再涉及 LLM

本脚本覆盖两种产出计划的方式：
    4b —— JSON schema  ：由 Pydantic 校验的结构化数据（每个工具定义一次 schema）
    4c–4e —— 代码即动作 ：包在 XML 标签里的 Python 脚本（无需 schema）
两种方式都「只用一次 LLM 调用」来规划，然后用纯 Python 执行。

运行方式（容器内）：
    source /opt/ros/jazzy/setup.bash
    eval "$(/data/miniforge3/bin/conda shell.bash hook)"
    conda activate base
    python 04_planning.py                  # 脚本会自动启动（或复用）turtlesim

说明：本脚本在 main() 执行前会通过 sim_launcher.ensure_turtlesim() 自动启动
turtlesim_node（若已在运行则复用），并在退出时关闭由它启动的实例，因此无需再手动执行
`ros2 run turtlesim turtlesim_node`。（DISPLAY 默认设为 :0 以便 Qt 窗口显示。）
"""

import ast
import re
import math
from typing import Literal, Union

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from llm_config import get_llm
from utils_03 import TOOLS, safe_invoke, MoveArgs, RotateArgs
from log_utils import log  # log() 等价 print()，但会打印「[文件名:行号]」前缀


# LangChain 工具对象 —— 供 safe_invoke 使用（带 .invoke()）
TOOL_OBJECTS = {t.name: t for t in TOOLS}
# 「工具名 -> 底层函数」字典，供后面的 execute_code() 使用
# LangChain 的工具对象可以用 .func 取到被 @tool 装饰的原始函数
TOOL_FUNCTIONS = {t.name: t.func for t in TOOLS}


# =============================================================================
# 练习 4a —— 反应式规划的代价
# =============================================================================
# | 方式                      | 画正方形的 LLM 调用数 | 画六边形的 LLM 调用数 |
# |---------------------------|-----------------------|-----------------------|
# | 反应式（练习 2）          | 9+（每次工具调用一次）| 15+                   |
# | 先规划后执行（任一方式）  | 1                     | 1                     |
#
# 反应式智能体每一步还会让上下文窗口不断增长。
# 两种「先规划」的方式都在开头付出一次固定的 LLM 成本，之后用纯 Python 执行。


# =============================================================================
# 练习 4b —— 用 JSON schema 规划
# =============================================================================
# 第一种方式用 llm.with_structured_output(Schema) 强制模型返回与 Pydantic 模型
# 完全匹配的数据。模型永远不产出自由文本 —— 它直接填充 schema 的各字段。
#
# 你只需定义一次 schema。之后每次调用都保证返回一个合法的 DrawingPlan 对象 ——
# 无需 JSON 解析、无需去掉代码围栏、也不用取 .text。
#
# plan 是一个 Python 对象 —— plan.steps 是一个可迭代、可校验、可重放的列表，
# 全程不碰字符串。

class MoveStep(MoveArgs):
    tool: Literal["move_forward"]


class RotateStep(RotateArgs):
    tool: Literal["rotate"]


class DrawingPlan(BaseModel):
    shape: str
    n_sides: int
    side_m: float
    steps: list[Union[MoveStep, RotateStep]]


SCHEMA_SYSTEM = """你是 TurtleSim 机器人的几何规划器。
乌龟在一个 11 x 11 米的场地中运行。
给定一个绘制任务，产出一份逐步计划。
为形状的每条边交替安排 move_forward 和 rotate 步骤。
外角 = 360 / n_sides 度，逆时针（正值）。
只使用可用工具，并严格按 schema 指定的 JSON 格式输出。
"""


def plan_with_json_schema(task: str) -> DrawingPlan:
    """用 with_structured_output 让模型直接产出结构化计划对象。"""
    llm = get_llm()
    planner = llm.with_structured_output(DrawingPlan)

    plan = planner.invoke([
        SystemMessage(SCHEMA_SYSTEM),
        HumanMessage(task),
    ])

    log(f"形状   : {plan.shape}")
    log(f"边数   : {plan.n_sides}")
    log(f"边长   : {plan.side_m} 米")
    log(f"步数   : {len(plan.steps)}")
    log()
    for i, step in enumerate(plan.steps):
        log(f"  [{i:02d}] {step}")
    return plan


def execute_json_plan(plan: DrawingPlan, verbose: bool = True) -> list[str]:
    """用真实工具套件逐步执行 DrawingPlan。"""
    observations = []
    for i, step in enumerate(plan.steps):
        tool_fn = TOOL_OBJECTS.get(step.tool)
        args = step.model_dump(exclude={"tool"})
        obs = safe_invoke(tool_fn, args) if tool_fn else f"未知工具：{step.tool}"
        observations.append(obs)
        if verbose:
            log(f"  [{i:02d}] {step.tool}({args}) → {obs}")
    return observations


# =============================================================================
# 练习 4c —— 代码即动作（code-as-actions）
# =============================================================================
# 第二种方式让模型直接写一段 Python 脚本，而不是填充 schema。
# 无需 Pydantic 模型 —— 模型用它自己掌握的 Python 和几何知识。
#
# 关键的提示技巧：让模型把代码包在 XML 标签里，使提取毫无歧义 ——
# 无需去代码围栏，也不用猜格式。
#
# 画正方形只是一个 for 循环；用 JSON schema 方式则会是 12 个独立步骤。
# 循环、变量、math 都可用 —— 模型会自然地使用它们。

CODE_SYSTEM = """你是 TurtleSim 机器人的 Python 代码生成器。
乌龟在一个 11 x 11 米的场地中运行，初始位置 x=5.5, y=5.5, heading=0（东）。

你只能调用以下函数：
  move_forward(distance_m: float)  -- 向前移动 distance_m 米
  rotate(angle_deg: float)         -- 旋转；正值 = 逆时针
  get_pose()                       -- 返回当前 (x, y, heading_deg)

规则：
  - 可自由使用 Python：变量、循环、math 都允许。
  - 不要 import 任何东西。
  - 不要调用上面未列出的任何函数。
  - 把你的代码包在 <execute_python> </execute_python> 标签里。
"""


def extract_code(text: str) -> str | None:
    """从模型回复中提取 Python 代码，容错多种常见格式。

    模型「通常」会把代码包在 <execute_python> </execute_python> 标签里（这是系统
    提示的要求），但它偶尔会不听话，改用 Markdown 代码围栏，甚至直接给裸代码。
    这里按优先级依次尝试：
      1. <execute_python> ... </execute_python> 标签（首选、最明确）
      2. ```python ... ``` 或 ``` ... ``` 的 Markdown 代码围栏（回退）
    都匹配不到则返回 None，交由调用方处理。

    注：自由文本 + 正则解析本身就脆弱——这正是练习 4b 用 with_structured_output
    的 schema 强约束更稳的原因。这里的回退只是尽量兼容，而非根治。
    """
    # 1) 首选：<execute_python> 标签
    m = re.search(r"<execute_python>(.*?)</execute_python>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2) 回退：Markdown 代码围栏（```python ... ``` 或 ``` ... ```）
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    return None


def generate_code(task: str, verbose: bool = True) -> str:
    """让 LLM 写一段 Python 脚本，并从回复中提取出来（标签优先，Markdown 围栏回退）。"""
    llm = get_llm()
    messages = [
        SystemMessage(CODE_SYSTEM),
        HumanMessage(f"<task>\n{task}\n</task>"),
    ]
    response = llm.invoke(messages)

    # 提取代码：<execute_python> 标签优先，Markdown 围栏回退
    code = extract_code(response.text)
    if code is None:
        raise ValueError(
            f"响应中未找到 <execute_python> 标签或 Markdown 代码围栏：\n{response.text}"
        )

    if verbose:
        log("生成的代码：")
        log("─" * 50)
        log(code)
        log("─" * 50)

    return code


# =============================================================================
# 练习 4d —— 验证生成的代码
# =============================================================================
# 运行模型生成的代码之前，用 ast 检查两件事（无需执行）：
#   1. 语法 —— 是否是合法的 Python？
#   2. 安全 —— 是否只调用了被许可的函数？
# 这等价于校验 JSON 计划里的工具名和参数类型，但不需要 schema，只需要一份
# 允许调用的函数名集合。

def verify_code(code: str) -> tuple[bool, str]:
    """检查语法与函数白名单。返回 (是否通过, 说明信息)。"""
    ALLOWED = set(TOOL_FUNCTIONS.keys()) | {"print", "range", "len", "abs", "round"}

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"语法错误：{e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "存在未授权的 import 语句"

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in ALLOWED:
                return False, f"未授权的调用：'{node.func.id}'"

    n_calls = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call))
    return True, f"通过 —— 共 {n_calls} 处调用"


def demo_verify_code(hex_code: str) -> None:
    ok, msg = verify_code(hex_code)
    log(f"六边形代码 : {'✓' if ok else '✗'}  {msg}")

    # 测试一个幻觉出的函数名
    ok, msg = verify_code("fly(10)")
    log(f"幻觉函数   : {'✓' if ok else '✗'}  {msg}")

    # 测试 import 注入
    ok, msg = verify_code("import os\nos.system('echo hello')")
    log(f"import 注入: {'✓' if ok else '✗'}  {msg}")


# =============================================================================
# 练习 4e —— 安全执行 + 完整流水线
# =============================================================================
# 完整流水线：生成 → 验证 → 执行。
# 若验证失败，就把错误反馈给 LLM，请它修正后再试。

def execute_code(code: str, verbose: bool = True) -> list[str]:
    """执行已通过验证的代码（在受限的全局命名空间里）。"""
    observations = []

    def tracked(name, fn):
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            observations.append(f"{name}{args} → {result}")
            return result
        return wrapper

    # 只暴露工具函数 + 少量安全内置
    safe_globals = {name: tracked(name, fn) for name, fn in TOOL_FUNCTIONS.items()}
    safe_globals["__builtins__"] = {
        "range": range, "len": len, "abs": abs,
        "round": round, "print": print, "math": math,
    }

    if verbose:
        log("── 执行中 ──────────────────────────────")
    exec(code, safe_globals)
    if verbose:
        log(f"── 共发生 {len(observations)} 次调用 ──────────────────────────────")
    return observations


def plan_and_execute(task: str, max_fix_attempts: int = 2,
                    verbose: bool = True) -> dict:
    """生成 → 验证 → 执行。验证失败时请 LLM 修正。"""
    llm = get_llm()
    messages = [
        SystemMessage(CODE_SYSTEM),
        HumanMessage(f"<task>\n{task}\n</task>"),
    ]

    generated = None
    for attempt in range(1, max_fix_attempts + 2):
        if verbose:
            log(f"── 第 {attempt} 次尝试 ────────────────────────────────────")

        response = llm.invoke(messages)
        code = extract_code(response.text)
        if code is None:
            err = "未找到 <execute_python> 标签或 Markdown 代码围栏。"
            if verbose:
                log(f"   提取: ✗  {err}")
        else:
            ok, msg = verify_code(code)
            if verbose:
                log(f"   验证 : {'✓' if ok else '✗'}  {msg}")
            if ok:
                generated = code
                break
            err = msg

        messages.append(response)
        messages.append(HumanMessage(
            f"你的代码失败了：{err}\n"
            "请修正，并只用 <execute_python> 标签回复。"
        ))

    if generated is None:
        return {"error": f"在 {max_fix_attempts + 1} 次尝试后仍无法产出合法代码。"}

    if verbose:
        log()
        log("最终代码：")
        log("─" * 50)
        log(generated)
        log("─" * 50)
        log()

    observations = execute_code(generated, verbose=verbose)
    return {"code": generated, "observations": observations}


# =============================================================================
# 小结
# =============================================================================
# 方式 1 —— JSON schema 计划：
#   规划  : llm.with_structured_output(DrawingPlan).invoke(messages)
#   输出  : 一个已校验的 Pydantic 对象 —— 无需解析
#   执行  : 迭代 plan.steps，调用对应的工具函数
#   适合  : 需要人类可读、可逐步检查、可重放的计划时
#
# 方式 2 —— 代码即动作：
#   规划  : llm.invoke()，系统提示里要求用 XML 标签包裹代码
#   提取  : re.search(r'<execute_python>(.*?)</execute_python>', ...)
#   验证  : ast.parse() + 函数名白名单
#   执行  : exec(code, safe_globals)，内置受限
#   适合  : 复杂形状、循环、数学 —— 扁平 JSON 难以优雅表达的场景
#
# 两种方式都：只用一次 LLM 调用来规划；用纯 Python 执行（确定、快速、循环中无 LLM）；
# 失败时可把错误回传给模型来重试。


def main() -> None:
    log("工具可用：", list(TOOL_FUNCTIONS.keys()))

    # ── 练习 4b：JSON schema 规划并执行 ──────────────────────────────────
    log("\n" + "=" * 60)
    log("练习 4b —— 用 JSON schema 规划")
    log("=" * 60)
    plan = plan_with_json_schema("画一个边长 2 米的三角形。")

    log(f"\n执行 '{plan.shape}' 计划（{len(plan.steps)} 步）：\n")
    safe_invoke(TOOL_OBJECTS.get("reset"), {})  # 先重置
    execute_json_plan(plan)

    # ── 练习 4c：代码即动作 —— 生成代码 ─────────────────────────────────
    log("\n" + "=" * 60)
    log("练习 4c —— 代码即动作（生成代码）")
    log("=" * 60)
    hex_code = generate_code("画一个边长 2 米的六边形。")

    log("\n附加检查（与绘制无关的任务）：")
    # 这是演示性质的附加检查：看模型对非机器人任务会写出什么。
    # 它偏离系统提示的场景，模型更可能不按标签格式输出；即便提取失败也不该中断脚本。
    try:
        random_code = generate_code("如果输入合法，求任意给定数字的平方根。", verbose=True)
        log(random_code)
    except ValueError as e:
        log(f"（附加检查）未能提取到代码，跳过。原因：{str(e).splitlines()[0]}")

    # ── 练习 4d：验证生成的代码 ─────────────────────────────────────────
    log("\n" + "=" * 60)
    log("练习 4d —— 验证生成的代码")
    log("=" * 60)
    demo_verify_code(hex_code)

    # ── 练习 4e：完整流水线（生成 → 验证 → 执行） ───────────────────────
    log("\n" + "=" * 60)
    log("练习 4e —— 安全执行 + 完整流水线")
    log("=" * 60)
    safe_invoke(TOOL_OBJECTS.get("reset"), {})  # 执行前重置场地
    plan_and_execute("画一个边长 1 米的五边形。")


if __name__ == "__main__":
    # 自动启动（或复用）turtlesim，省去手动执行 `ros2 run turtlesim turtlesim_node`。
    from sim_launcher import ensure_turtlesim

    cleanup = ensure_turtlesim()
    try:
        main()
    finally:
        cleanup()
