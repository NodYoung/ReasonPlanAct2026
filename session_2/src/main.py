"""
Multi-Agent Turtlesim — start here.

You type requests in the terminal; each one becomes a mission run by the
supervisor agent (see ``workflow/graph.py``):

  - "Draw a house"           → blueprint → plan → review → draw, part by part
  - "Move forward 0.1"       → supervisor does it directly
  - "How big is the canvas?" → supervisor answers, turtle stays put

Before running:
  1. pixi install
  2. cp .env.example .env  and add your LLM API key (providers: config.toml)
  3. Terminal 1: pixi run sim
  4. Terminal 2: pixi run agents   (this file)
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from src.app_settings import load_config
from src.llm import api_key_hint, describe_models, missing_providers
from src.utils.ros_bridge import get_turtle_bridge
from src.workflow import run_mission


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: one-shot ``--request`` or an interactive loop."""
    load_dotenv()
    load_config()
    missing = missing_providers()
    if missing:
        for provider in missing:
            print(f"Error: no API key for {provider}. {api_key_hint(provider)}")
        return 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", nargs="?", const="", help="One-shot request")
    args = parser.parse_args(argv)

    bridge = get_turtle_bridge(turtle_name="turtle1")
    if bridge.is_connected():
        bridge.rotate(360.0)  # hello spin — visual proof the sim is responding
        print("Connected to turtlesim (/turtle1) — the turtle just spun to say hi.")
    else:
        print("WARNING: turtlesim not reachable. Start `pixi run sim` in another terminal.")
    print(f"LLM: {describe_models()}\n")

    try:
        if args.request is not None:
            req = args.request or "Draw a blue square with side length 2."
            print(f"User: {req}\n")
            return 0 if run_mission(req, bridge) else 1

        print("Multi-Agent Turtlesim — shapes, moves, or questions. Type 'quit' to exit.\n")
        print("Examples:")
        print("  Draw a house.")
        print("  Move forward 0.1")
        print("  Set pen to red\n")
        while True:
            try:
                text = input("User: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text or text.lower() in {"quit", "exit", "q"}:
                break
            run_mission(text, bridge)
            print("-" * 40)
        return 0
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    sys.exit(main())
