#!/usr/bin/env python3
"""
01_chat_model.py — 对话模型（Chat Model）练习 1
================================================

在构建智能体（agent）之前，我们需要先从代码角度理解 LLM 到底是什么：
它本质上是一个「输入一组消息、返回一条消息」的函数。

后面三节课的所有内容 —— 智能体循环、工具调用、规划 —— 都建立在这一次调用之上。

本脚本你将学到：
- 调用对话模型并读取返回结果（1a）
- 观察系统提示词（system prompt）如何塑造模型的行为（1d）
- 构建多轮对话，并理解「为什么消息列表本身就是记忆」（1e）

练习 1b、1c、1f 作为参考一并保留，可自行运行深入了解。

说明：业界有多种构建 AI 智能体的框架，例如 LlamaIndex、AutoGen、CrewAI、PydanticAI。
本教程使用 LangChain 及其 `langchain_core` 抽象层。`langchain_core` 定义了
`BaseMessage` 等基础接口与工具抽象，便于构建模块化、可组合、与框架无关的智能体流程。

运行方式（容器内 conda base 环境）：
    enter_ultron
    cd /home/robot/github/llm-misc/ReasonPlanAct2026/session_1/src
    eval "$(/data/miniforge3/bin/conda shell.bash hook)"
    conda activate base
    python 01_chat_model.py
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from llm_config import get_llm
from log_utils import log  # log() 等价 print()，但会打印「[文件名:行号]」前缀


def exercise_1a(llm):
    """练习 1a —— 最基础的 invoke 调用。"""
    log("=" * 60)
    log("练习 1a —— 基础 invoke 调用")
    log("=" * 60)
    # get_llm() 返回一个 LangChain 对话模型（默认走 codex provider）
    log("LLM 类型：", type(llm).__name__)

    # invoke 接受一个字符串（或消息列表），返回一条 AIMessage
    response = llm.invoke("你好！")
    log(response)
    return response


def exercise_1b(response):
    """练习 1b —— 查看 AIMessage 对象。"""
    log("\n" + "=" * 60)
    log("练习 1b —— 查看 AIMessage 对象")
    log("=" * 60)
    # .text 直接取出模型回复的纯文本内容
    log("输出文本：")
    log(response.text)


def exercise_1c(llm):
    """练习 1c —— 纯字符串 vs 显式 HumanMessage（可选）。

    传入纯字符串是一种简写 —— LangChain 会自动把它包成 HumanMessage。
    传入 [HumanMessage(...)] 则是显式写法，一旦要加 SystemMessage 就必须这么写。
    两者产生的回复是一致的。
    """
    log("\n" + "=" * 60)
    log("练习 1c —— 纯字符串 vs 显式 HumanMessage（可选）")
    log("=" * 60)
    question = "什么是 ROS 节点（node）？请用一句话回答。"

    response_plain = llm.invoke(question)
    log("纯字符串：", response_plain.text)
    log()

    messages = [HumanMessage(question)]
    response_human = llm.invoke(messages)
    log("使用 [HumanMessage]：", response_human.text)


def exercise_1d(llm):
    """练习 1d —— 系统提示词塑造模型行为。

    SystemMessage 是开发者与模型沟通的专用通道。它位于每次对话最前面，用于设定：
    - 角色：模型应该扮演什么身份
    - 约束：模型可以做什么、不能做什么
    - 输出格式：回复应如何组织

    在智能体系统中，系统提示词正是用来描述机器人所处环境、可用工具与决策规则的地方。
    写好系统提示词和写好代码同样重要。

    下面两段「问题完全相同，仅系统提示词不同」。
    观察点：同一个问题，回答在框架、用词、长度上完全不同。模型知识不变，
    是系统提示词控制了它如何使用这些知识。
    """
    log("\n" + "=" * 60)
    log("练习 1d —— 系统提示词塑造模型行为")
    log("=" * 60)
    question = "我该如何让机器人向前移动？请用一句话回答。"

    # 不带系统提示词 —— 得到通用回答
    response_generic = llm.invoke(question)
    log("不带系统提示词：")
    log(response_generic.text)
    log()

    ROBOT_SYSTEM = (
        "你是一个 TurtleSim 机器人的推理核心。\n"
        "你有两个可用工具：'move_forward(distance_m)' 让机器人按指定米数向前移动，"
        "'rotate(angle_deg)' 让机器人按指定角度（度）旋转。\n"
        "回答要简洁，并以你的机器人身体和场地为视角来表述。\n"
    )
    # SystemMessage 永远放在最前面，然后才是 HumanMessage
    messages_with_system = [SystemMessage(ROBOT_SYSTEM), HumanMessage(question)]
    response_robotic = llm.invoke(messages_with_system)
    log("带系统提示词：", response_robotic.text)


def exercise_1d_bonus(llm):
    """思考题 —— SystemMessage 和 HumanMessage 谁说了算？

    把系统提示词设为「始终用法语回答」，而在用户消息里明确要求用英语。
    运行后看看模型最终听谁的。
    如何让模型更可靠地遵循 SystemMessage？见下方最佳实践。
    """
    log("\n" + "=" * 60)
    log("思考题 —— SystemMessage vs HumanMessage 谁优先")
    log("=" * 60)
    ROBOT_SYSTEM = (
        "你是一个 TurtleSim 机器人的推理核心。\n"
        "你有两个可用工具：'move_forward(distance_m)' 让机器人按指定米数向前移动，"
        "'rotate(angle_deg)' 让机器人按指定角度（度）旋转。\n"
        "回答要简洁，并以你的机器人身体和场地为视角来表述。\n"
        "始终用法语回答。"
    )
    question = "我该如何让机器人向前移动？请用一句话、并用英语回答。"

    messages_with_system = [SystemMessage(ROBOT_SYSTEM), HumanMessage(question)]
    response_robotic = llm.invoke(messages_with_system)
    log(response_robotic.text)

    # 让模型更可靠遵循系统提示词的最佳实践：
    # - 明确表述：在指令中加上「无论用户用什么语言书写」。
    # - 使用指令微调（instruct-tuned）模型：它们专门针对遵循系统提示这类结构化指令做了微调。
    # - 使用更大的模型：更大的模型通常遵循指令更稳定。
    # - 注意：具体表现因模型而异。


def exercise_1e(llm):
    """练习 1e —— 多轮对话。

    这是本脚本最重要的练习。后面所有内容 —— 智能体、工具、规划 —— 都依赖于理解它。

    关键事实：LLM 是完全无状态的。

    每次调用 llm.invoke() 都是独立的，模型不记得之前的任何调用。
    如果你想让它「记住」某件事，就必须把它包含在你发送的消息列表里。

    也就是说：消息历史列表就是智能体的记忆 —— 除此之外别无其他。
    没有隐藏状态，服务端也没有会话。只要某条消息不在列表里，对模型而言它就不存在。

    下面的 chat() 函数把这个模式显式化：
      1. 把用户消息追加到 history
      2. 把完整的 history 发给模型
      3. 把模型回复追加回 history
      4. 重复

    观察点：
    - 第 2 轮问 Turbo 的位置，而该信息只在第 1 轮提到过。模型能答对，
      正是因为第 1 轮仍在 history 列表里。
    - 看结尾打印的 history：SystemMessage + 两条 HumanMessage + 两条 AIMessage = 5 条。
      它会随每轮对话增长。02 讲义会演示当它增长过大时会发生什么。
    """
    log("\n" + "=" * 60)
    log("练习 1e —— 多轮对话")
    log("=" * 60)
    history = [
        SystemMessage("记住本次对话中说过的一切。每次只用一句话回答。")
    ]

    def chat(user_text: str) -> str:
        """追加用户消息 -> 带完整 history 调用 LLM -> 追加模型回复。"""
        history.append(HumanMessage(user_text))
        response = llm.invoke(history)
        history.append(response)
        return response.text

    query = "我的机器人叫 Turbo，初始位置在 x=3, y=3。请做个自我介绍。"
    log("第 1 轮（用户）  ：", query)
    log("第 1 轮（助手）  ：", chat(query))
    log()

    query = "Turbo 的初始位置是多少？"
    log("第 2 轮（用户）  ：", query)
    log("第 2 轮（助手）  ：", chat(query))
    log()

    log("完整对话历史：")
    for m in history:
        # :<13 左对齐占 13 字符；:.80 截断文本到 80 字符
        log(f"{type(m).__name__:<13}: {m.text:.80}")


def exercise_1f(llm):
    """练习 1f —— 无历史的无状态调用（可选）。

    同样的问题，不带任何历史发送。模型完全不知道 Turbo 是谁。
    """
    log("\n" + "=" * 60)
    log("练习 1f —— 无历史的无状态调用（可选）")
    log("=" * 60)
    fresh_call = llm.invoke("Turbo 的初始位置是多少？")
    log("不带上下文的全新调用：")
    log(fresh_call.text)


def summary():
    """小结。

    | 概念             | 工作方式                                                   |
    |------------------|------------------------------------------------------------|
    | 调用模型         | llm.invoke(messages) -> AIMessage                          |
    | 读取回复         | response.text                                              |
    | 控制行为         | 在列表最前面放一条 SystemMessage                           |
    | 赋予模型记忆     | 把每条消息追加到 history；每次调用都传入完整列表            |
    | 不同消息类型     | AIMessage、HumanMessage、SystemMessage                     |

    下一步：02_agents.ipynb —— 我们会把这一次调用变成一个循环，
    让它能使用工具、观察结果，并持续推理直到任务完成。
    """
    log("\n" + "=" * 60)
    log("小结：见本函数 docstring")
    log("=" * 60)


def main() -> None:
    # 只创建一次模型，供所有练习复用
    llm = get_llm()

    response = exercise_1a(llm)
    exercise_1b(response)
    exercise_1c(llm)
    exercise_1d(llm)
    exercise_1d_bonus(llm)
    exercise_1e(llm)
    exercise_1f(llm)
    summary()


if __name__ == "__main__":
    main()
