"""
将 LangGraph 工作流可视化为 Mermaid 图。

把 Mermaid 源码打印到终端，并保存到
``docs/workflow_graph.md``（在 GitHub / IDE 的 markdown 预览中渲染）。

运行：  pixi run graph
"""

from __future__ import annotations

from pathlib import Path

from src.utils.ros_bridge import TurtleBridge
from src.workflow import build_workflow


def main() -> None:
    # Mock 后端 —— 仅为绘制图而无需 ROS / turtlesim。
    graph = build_workflow(TurtleBridge(use_mock=True)).compile().get_graph()
    mermaid = graph.draw_mermaid()

    print(mermaid)

    out = Path("docs/workflow_graph.md")
    out.write_text(f"# Workflow graph\n\n```mermaid\n{mermaid}```\n")
    print(f"Saved to {out} — open it in a markdown preview to see the diagram.")


if __name__ == "__main__":
    main()
