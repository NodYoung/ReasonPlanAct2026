"""
Visualize the LangGraph workflow as a Mermaid diagram.

Prints the Mermaid source to the terminal and saves it to
``docs/workflow_graph.md`` (renders on GitHub / in the IDE markdown preview).

Run:  pixi run graph
"""

from __future__ import annotations

from pathlib import Path

from src.utils.ros_bridge import TurtleBridge
from src.workflow import build_workflow


def main() -> None:
    # Mock backend — no ROS / turtlesim needed just to draw the graph.
    graph = build_workflow(TurtleBridge(use_mock=True)).compile().get_graph()
    mermaid = graph.draw_mermaid()

    print(mermaid)

    out = Path("docs/workflow_graph.md")
    out.write_text(f"# Workflow graph\n\n```mermaid\n{mermaid}```\n")
    print(f"Saved to {out} — open it in a markdown preview to see the diagram.")


if __name__ == "__main__":
    main()
