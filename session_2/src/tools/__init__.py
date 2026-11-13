"""
供 supervisor 的 ReAct 循环使用的 LangChain ``@tool`` 定义。

turtle_motion_tools.py
    移动 / 旋转 / 画笔 / 清空 / 位姿 —— 一次性的海龟动作。

``start_drawing`` 工具位于 ``agents/supervisor.py``，紧挨着使用它的 agent。
关于工具如何获取实时桥接和任务笔记本，参见 ``utils/agent_runtime.py``。
"""

__all__: list[str] = []
