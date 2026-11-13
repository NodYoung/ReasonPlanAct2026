"""
sim_launcher.py —— 自动启动 turtlesim 模拟器（先清理旧实例，再起全新实例）
=============================================================================

小乌龟的 GUI 窗口由外部进程 `ros2 run turtlesim turtlesim_node` 创建（Qt），
而不是本仓库的 Python 代码。为省去每次手动启动，这里提供一个辅助：

  ensure_turtlesim()  —— 先杀掉所有已存在的 turtlesim 进程，再启动一个「全新」实例。

为什么「先杀再起」而不是「复用」：
  之前被 Ctrl+C / 超时打断的运行可能残留一个 turtlesim 进程，其状态可能已不稳定
  （例如 context 被外部关过）。若直接复用，绘制阶段可能出现 ExternalShutdownException、
  "rcl node's context is invalid"，表现为「0/N steps executed」只画出半张图。
  每次起一个干净的新实例可彻底规避这类「复用到坏实例」的问题。

用法（在 main() 里）：
    from src.utils.sim_launcher import ensure_turtlesim
    cleanup = ensure_turtlesim()
    try:
        ...  # 连接 bridge、跑任务
    finally:
        cleanup()

设计要点：
  - 启动前用 /proc 扫描并杀掉所有 turtlesim_node 进程（精确匹配可执行名，并排除
    自身 PID，绝不会误杀当前 Python 进程）。
  - 自动设置 DISPLAY（默认 :0），让 Qt 能连接到 X 显示。
  - 启动后等待 /turtle1/pose 话题出现（或超时），确保就绪再返回。
  - 用 atexit + 返回的 cleanup 双重保证：退出时关闭本次启动的子进程。

注意：探测用的临时 rclpy 节点在探测后会把「自己启动的」rclpy context 关掉，
使随后 ros_bridge 的 rclpy.init() 能够干净初始化。
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import time
from typing import Callable

from src.utils.log_utils import log


def _find_turtlesim_pids() -> list[int]:
    """扫描 /proc，返回所有 turtlesim_node 进程的 PID（排除自身）。

    通过读取每个进程的 cmdline 精确匹配可执行名 ``turtlesim_node``，不会匹配到
    本 Python 进程自身（其 cmdline 是 python -m src.main）。
    """
    self_pid = os.getpid()
    pids: list[int] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == self_pid:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().split(b"\x00")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        # 精确匹配：某个 argv 以 turtlesim_node 结尾
        if any(arg.split(b"/")[-1] == b"turtlesim_node" for arg in cmdline if arg):
            pids.append(pid)
    return pids


def _kill_existing_turtlesim() -> None:
    """杀掉所有已存在的 turtlesim_node 进程（先 SIGTERM，必要时 SIGKILL）。"""
    pids = _find_turtlesim_pids()
    if not pids:
        return
    log(f"[sim_launcher] 清理已存在的 turtlesim 进程：{pids}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # 给一点时间优雅退出
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and _find_turtlesim_pids():
        time.sleep(0.2)
    # 仍未退出的强杀
    for pid in _find_turtlesim_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(0.5)


def _pose_topic_available() -> bool:
    """用一个临时 rclpy 节点探测 /turtle1/pose 话题是否已出现。"""
    try:
        import rclpy
        from rclpy.node import Node

        started_here = False
        if not rclpy.ok():
            rclpy.init()
            started_here = True
        probe = Node("ros_langchain_sim_probe")
        try:
            time.sleep(0.4)  # 给 discovery 一点时间
            topics = [t for t, _types in probe.get_topic_names_and_types()]
            return "/turtle1/pose" in topics
        finally:
            probe.destroy_node()
            # 若 rclpy 是在本函数里初始化的，这里关掉它，避免遗留已初始化的 context
            # 导致 ros_bridge 的 rclpy.init() 冲突。
            if started_here and rclpy.ok():
                rclpy.shutdown()
    except Exception:
        return False


def ensure_turtlesim(wait_timeout: float = 10.0) -> Callable[[], None]:
    """先清掉已存在的 turtlesim，再启动一个全新实例。返回清理函数。"""
    os.environ.setdefault("DISPLAY", ":0")  # 确保 Qt 能连到显示

    # 先杀掉所有旧的 turtlesim，保证本次用的是干净的新实例
    _kill_existing_turtlesim()

    log("[sim_launcher] 正在启动全新的 turtlesim_node ...")
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
