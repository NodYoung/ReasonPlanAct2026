"""
Multi-Agent Turtlesim — start here.

You type requests in the terminal; each one becomes a mission run by the
supervisor agent (see ``workflow/graph.py``):

  - "Draw a house"           → blueprint → plan → review → draw, part by part
  - "Move forward 0.1"       → supervisor does it directly
  - "How big is the canvas?" → supervisor answers, turtle stays put

Before running (本容器 ROS Jazzy + conda base)：
  1. source /opt/ros/jazzy/setup.bash
  2. conda activate base   （已装 langgraph / langchain；codex 读 ~/.codex，无需 .env）
  3. export PYTHONPATH=".:$PYTHONPATH"
  4. python -m src.main    （本文件会自动启动/复用 turtlesim，无需手动 pixi run sim）
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from src.app_settings import load_config
from src.llm import api_key_hint, describe_models, missing_providers
from src.utils.ros_bridge import get_turtle_bridge
from src.utils.sim_launcher import ensure_turtlesim
from src.utils.log_utils import log  # log() 等价 print()，但会打印「[文件名:行号]」前缀
from src.workflow import run_mission


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: one-shot ``--request`` or an interactive loop."""
    load_dotenv()
    load_config()
    missing = missing_providers()
    if missing:
        for provider in missing:
            log(f"Error: no API key for {provider}. {api_key_hint(provider)}")
        return 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", nargs="?", const="", help="One-shot request")
    args = parser.parse_args(argv)

    # 自动启动（或复用）turtlesim，省去手动执行 `ros2 run turtlesim turtlesim_node`。
    sim_cleanup = ensure_turtlesim()

    bridge = get_turtle_bridge(turtle_name="turtle1")
    if bridge.is_connected():
        bridge.rotate(360.0)  # hello spin — visual proof the sim is responding
        log("Connected to turtlesim (/turtle1) — the turtle just spun to say hi.")
    else:
        log("WARNING: turtlesim not reachable. Start `pixi run sim` in another terminal.")
    log(f"LLM: {describe_models()}\n")

    try:
        if args.request is not None:
            req = args.request or "Draw a blue square with side length 2."
            log(f"User: {req}\n")
            return 0 if run_mission(req, bridge) else 1

        log("Multi-Agent Turtlesim — shapes, moves, or questions. Type 'quit' to exit.\n")
        log("Examples:")
        log("  Draw a house.")
        log("  Move forward 0.1")
        log("  Set pen to red\n")
        while True:
            try:
                text = input("User: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text or text.lower() in {"quit", "exit", "q"}:
                break
            run_mission(text, bridge)
            log("-" * 40)
        return 0
    finally:
        bridge.shutdown()
        sim_cleanup()  # 关闭本次自动启动的 turtlesim（若是复用现有实例则为空操作）


if __name__ == "__main__":
    sys.exit(main())
