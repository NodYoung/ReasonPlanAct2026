"""Tiny terminal spinner shown while waiting on slow calls (LLM requests)."""

from __future__ import annotations

import itertools
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def spinner(label: str = "Thinking") -> Iterator[None]:
    """Animate ``label`` on the current line until the with-block exits."""
    if not sys.stdout.isatty():
        yield
        return

    stop = threading.Event()

    def _spin() -> None:
        for frame in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if stop.wait(0.1):
                break
            sys.stdout.write(f"\r{frame} {label}...")
            sys.stdout.flush()
        sys.stdout.write("\r" + " " * (len(label) + 5) + "\r")  # clear the line
        sys.stdout.flush()

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
