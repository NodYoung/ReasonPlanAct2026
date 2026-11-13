"""在等待缓慢调用（如 LLM 请求）时显示的小型终端旋转指示器。"""

from __future__ import annotations

import itertools
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def spinner(label: str = "Thinking") -> Iterator[None]:
    """在当前行上为 ``label`` 显示动画，直到 with 代码块退出。"""
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
        sys.stdout.write("\r" + " " * (len(label) + 5) + "\r")  # 清除该行
        sys.stdout.flush()

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
