"""Agents package — one module per agent, each with ``build_graph_node(bridge)``.

supervisor        — the only orchestrator; ReAct on first visit, deterministic after
planner_blueprint — one LLM call: request → ordered parts + size
planner_part      — one LLM call per trial: movement steps for the current part
plan_reviewer     — trial PNG + one vision LLM call: approve or fix instructions
navigator         — deterministic execution on the real turtle (no LLM)
"""
