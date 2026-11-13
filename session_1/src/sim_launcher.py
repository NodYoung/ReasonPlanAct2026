"""
sim_launcher.py —— 自动启动 / 复用 turtlesim 模拟器
===================================================

小乌龟的 GUI 窗口由外部进程 `ros2 run turtlesim turtlesim_node` 创建（Qt），
而不是本仓库的 Python 代码。为省去每次手动启动，这里提供一个辅助：

  ensure_turtlesim()  —— 若 turtlesim 未在运行则自动启动；若已运行则复用，不重复启动。
                         返回一个「清理函数」：仅当本次是「我们启动的」才会在退出时关掉它，
                         复用已存在实例时清理函数为空操作（不动别人的进程）。

用法（在脚本 __main__ 里）：
    from sim_launcher import ensure_turtlesim
    cleanup = ensure_turtlesim()
    try:
        main()
    finally:
        cleanup()

设计要点：
  - 通过检查 ROS 图里是否已有 /turtlesim 节点来判断「是否已在运行」。
  - 自动设置 DISPLAY（默认 :0），让 Qt 能连接到 X 显示。
  - 启动后等待 /turtle1/pose 话题出现（或超时），确保就绪再返回。
  - 用 atexit + 返回的 cleanup 双重保证：正常退出时关闭我们启动的子进程。
"""

from __future__ import annotations

import atexit
import os
import subprocess
import time
from typing import Callable

from log_utils import log


def _turtlesim_running() -> bool:
    """检查 ROS 图里是否已存在 /turtlesim 节点。"""
    try:
        import rclpy
        from rclpy.node import Node

        started_here = False
        if not rclpy.ok():
            rclpy.init()
            started_here = True
        probe = Node("ros_langchain_sim_probe")
        try:
            # 给 discovery 一点时间
            time.sleep(0.5)
            names = [name for name, _ns in probe.get_node_names_and_namespaces()]
            return any("turtlesim" in n for n in names)
        finally:
            probe.destroy_node()
            if started_here and rclpy.ok():
                rclpy.shutdown()
    except Exception:
        # 探测失败时保守认为未运行（交给调用方启动）
        return False


def _pose_topic_available() -> bool:
    """检查 /turtle1/pose 话题是否已存在（turtlesim 就绪的标志）。"""
    try:
        import rclpy
        from rclpy.node import Node

        started_here = False
        if not rclpy.ok():
            rclpy.init()
            started_here = True
        probe = Node("ros_langchain_topic_probe")
        try:
            time.sleep(0.3)
            topics = [t for t, _types in probe.get_topic_names_and_types()]
            return "/turtle1/pose" in topics
        finally:
            probe.destroy_node()
            if started_here and rclpy.ok():
                rclpy.shutdown()
    except Exception:
        return False


def ensure_turtlesim(wait_timeout: float = 10.0) -> Callable[[], None]:
    """确保 turtlesim 正在运行。返回一个清理函数。

    - 若已在运行：复用，返回一个空操作清理函数（不关别人的进程）。
    - 若未运行：以子进程启动 turtlesim_node，等待其就绪，返回能关掉它的清理函数，
      并用 atexit 注册兜底。
    """
    # 确保 Qt 能连到显示
    os.environ.setdefault("DISPLAY", ":0")

    if _turtlesim_running():
        log("[sim_launcher] 检测到 turtlesim 已在运行，复用现有实例。")
        return lambda: None

    log("[sim_launcher] 未检测到 turtlesim，正在启动 turtlesim_node ...")
    # stdout/stderr 丢弃，避免干扰脚本输出
    proc = subprocess.Popen(
        ["ros2", "run", "turtlesim", "turtlesim_node"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    _closed = {"done": False}

    def cleanup() -> None:
        if _closed["done"]:
            return
        _closed["done"] = True
        if proc.poll() is None:  # 仍在运行
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        log("[sim_launcher] 已关闭本次启动的 turtlesim。")

    atexit.register(cleanup)

    # 等待就绪：/turtle1/pose 话题出现
    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log("[sim_launcher] 警告：turtlesim 进程意外退出。")
            break
        if _pose_topic_available():
            log("[sim_launcher] turtlesim 已就绪。")
            break
        time.sleep(0.3)
    else:
        log("[sim_launcher] 警告：等待 turtlesim 就绪超时，仍将继续。")

    return cleanup
