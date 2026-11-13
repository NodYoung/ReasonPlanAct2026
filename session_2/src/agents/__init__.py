"""Agents 包 —— 每个 agent 一个模块，各自提供 ``build_graph_node(bridge)``。

supervisor        —— 唯一的编排者；首次访问时走 ReAct，之后转为确定性路由
planner_blueprint —— 一次 LLM 调用：请求 → 有序的部件列表 + 尺寸
planner_part      —— 每次试验一次 LLM 调用：当前部件的运动步骤
plan_reviewer     —— 试验 PNG + 一次视觉 LLM 调用：批准或给出修复说明
navigator         —— 在真实海龟上确定性执行（不使用 LLM）
"""
