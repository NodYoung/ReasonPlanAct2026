"""
LangChain ``@tool`` definitions for the supervisor's ReAct loop.

turtle_motion_tools.py
    Move / rotate / pen / clear / pose — one-off turtle actions.

The ``start_drawing`` tool lives in ``agents/supervisor.py`` next to the
agent that uses it. See ``utils/agent_runtime.py`` for how tools get the
live bridge + mission notebook.
"""

__all__: list[str] = []
