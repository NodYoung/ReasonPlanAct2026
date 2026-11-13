#!/usr/bin/env python3
"""
03_tools.py —— 工具（Tools）练习 3
====================================

把机器人能力封装成智能体可以可靠调用的 LangChain 工具。

你将构建：一套 TurtleSim 工具（get_pose / move_forward / rotate / set_pen /
teleport / reset），每个工具都有：
  - 一个用于参数校验的 Pydantic 模型
  - 一个与模拟器交互的 ROS 组件
  - 一个模型可解析的、统一的返回格式

说明：原 notebook 把这些工具的 ROS 实现写在单元格里逐步完成（含 TODO 填空）。
为避免大段重复，本 .py 直接从 `utils_03` 导入「已完成」的工具套件
（utils_03 就是这些工具的最终形态），并在此聚焦于各项练习：
  3e —— 单独测试每个工具
  3f —— 把校验错误当作「观察」而非崩溃（safe_invoke）
  3g —— 把工具接入智能体循环
  3h —— 自己写一个工具（draw_square 挑战）

运行方式（容器内）：
    source /opt/ros/jazzy/setup.bash      # 让 conda base 能导入 rclpy / turtlesim
    eval "$(/data/miniforge3/bin/conda shell.bash hook)"
    conda activate base
    python 03_tools.py                     # 脚本会自动启动（或复用）turtlesim

说明：本脚本在 main() 执行前会通过 sim_launcher.ensure_turtlesim() 自动启动
turtlesim_node（若已在运行则复用），并在退出时关闭由它启动的实例，因此无需再手动执行
`ros2 run turtlesim turtlesim_node`。（DISPLAY 默认设为 :0 以便 Qt 窗口显示。）

环境适配：本容器为 ROS 2 Jazzy，消息/服务包名为 `turtlesim`（而非 Kilted 的
`turtlesim_msgs`）。相关导入已在 utils_03 中改好。
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from utils_03 import (
    TOOLS,
    safe_invoke,
    get_pose,
    move_forward,
    rotate,
    set_pen,
    teleport,
    reset,
    run_tool_agent,
)
from log_utils import log  # log() 等价 print()，但会打印「[文件名:行号]」前缀


# =============================================================================
# 练习 3b —— 完整工具套件（来自 utils_03）
# =============================================================================
def show_registered_tools() -> None:
    log("已注册的工具：")
    for t in TOOLS:
        first_line = t.description.splitlines()[0]
        log(f"  {t.name:<16} — {first_line}")


# =============================================================================
# 练习 3e —— 单独测试每个工具
# =============================================================================
# 切勿第一次就把工具放进智能体循环里测试。
# 如果工具在循环中途崩溃，智能体会收到一个错误观察，并可能「幻觉」出一套恢复方案。
# 请先直接测试每个工具 —— 这等价于写单元测试。
def test_tools_in_isolation() -> None:
    log("── 直接测试各工具 ───────────────────────────────────────────\n")
    # 运行前请确保已在另一进程启动 turtlesim 模拟器。
    log("get_pose       :", get_pose.invoke({}))
    log("move_forward   :", move_forward.invoke({"distance_m": 2.0}))
    log("rotate         :", rotate.invoke({"angle_deg": 90.0}))
    log("move_forward   :", move_forward.invoke({"distance_m": 1.5}))
    log("set_pen (青色) :", set_pen.invoke({"r": 0, "g": 255, "b": 255, "width": 3}))
    log("teleport       :", teleport.invoke({"x": 5.5, "y": 5.5, "heading_deg": 0.0}))
    log("get_pose       :", get_pose.invoke({}))
    log("reset          :", reset.invoke({}))


# =============================================================================
# 练习 3f —— 校验错误是「观察」，不是「崩溃」
# =============================================================================
# 当智能体传入超范围参数时，校验器会在工具主体运行「之前」抛出 ValidationError。
# 如果我们捕获它并返回一个纯字符串，智能体就会收到有用的错误观察，并能够重试。
# 这正是「会崩溃的智能体」与「能自我恢复的智能体」之间的区别。
# safe_invoke 定义在 utils_03 中。
def demo_validation_as_observation() -> None:
    log("超范围距离（不加包装，直接 invoke）：")
    try:
        move_forward.invoke({"distance_m": -5.0})
    except Exception as e:
        log(f"  抛出异常：{str(e).splitlines()[0]}\n")

    log("超范围距离（用 safe_invoke）：")
    log(" ", safe_invoke(move_forward, {"distance_m": -5.0}))
    log()

    log("越界传送：")
    log(" ", safe_invoke(teleport, {"x": 99.0, "y": 5.5}))
    log()

    log("合法调用：")
    log(" ", safe_invoke(move_forward, {"distance_m": 1.0}))


# =============================================================================
# 练习 3g —— 把工具接入智能体循环
# =============================================================================
# run_tool_agent 定义在 utils_03 中：结构化工具调用 + safe_invoke 错误处理的
# ReAct 智能体循环。
def demo_agent_loop() -> None:
    reset.invoke({})  # 先清空场地
    task = "向前移动 2 米，向左旋转 90°，把画笔改成绿色，再向前移动 2 米。"
    log("任务：", task)
    run_tool_agent(task=task)


# =============================================================================
# 练习 3h —— 自己写一个工具（挑战）：draw_square
# =============================================================================
# 增加一个 draw_square(side_m) 工具，让智能体可以作为「单个动作」调用它。
# 要求：
#   1. 接受一个参数：边长（米），范围 (0, 5]
#   2. 内部调用 move_forward 和 rotate 各四次
#   3. 返回最终位姿与累计行进距离
class DrawSquareArgs(BaseModel):
    side_m: float = Field(..., gt=0, le=5, description="正方形边长（米），范围 (0, 5]。")


@tool(args_schema=DrawSquareArgs)
def draw_square(side_m: float) -> str:
    """从当前位姿开始，画一个边长为 side_m 米的正方形。

    Args:
        side_m: 边长（米），范围 (0, 5]。

    内部调用 move_forward 与 rotate 各四次，返回绘制路径摘要与最终位姿。
    """
    total_distance = 0.0
    for _ in range(4):
        move_forward.invoke({"distance_m": side_m})
        total_distance += side_m
        rotate.invoke({"angle_deg": 90.0})

    final_pose = get_pose.invoke({})
    return (
        f"已绘制边长 {side_m:.2f} 米的正方形，"
        f"累计行进 {total_distance:.2f} 米。最终位姿：{final_pose}"
    )


def demo_draw_square() -> None:
    # 直接把 draw_square 作为单个工具调用
    log("直接调用 draw_square(side_m=2)：")
    log(" ", safe_invoke(draw_square, {"side_m": 2.0}))


# =============================================================================
# 小结
# =============================================================================
# | 关注点              | 解决方案                               |
# |---------------------|----------------------------------------|
# | 模型读到错误的文档  | 写精确、带单位的 docstring             |
# | 工具让智能体循环崩溃| safe_invoke 返回错误字符串而非抛异常   |
#
# 下一步：04_planning.py —— 给智能体一个形状目标，让它把任务分解成跨多轮的工具调用序列。


def main() -> None:
    log("=" * 60)
    log("练习 3b —— 完整工具套件")
    log("=" * 60)
    show_registered_tools()

    log("\n" + "=" * 60)
    log("练习 3e —— 单独测试每个工具")
    log("=" * 60)
    test_tools_in_isolation()

    log("\n" + "=" * 60)
    log("练习 3f —— 校验错误是观察，不是崩溃")
    log("=" * 60)
    demo_validation_as_observation()

    log("\n" + "=" * 60)
    log("练习 3g —— 把工具接入智能体循环")
    log("=" * 60)
    demo_agent_loop()

    log("\n" + "=" * 60)
    log("练习 3h —— 自己写一个工具（draw_square 挑战）")
    log("=" * 60)
    reset.invoke({})  # 画正方形前清空场地
    demo_draw_square()


if __name__ == "__main__":
    # 自动启动（或复用）turtlesim，省去手动执行 `ros2 run turtlesim turtlesim_node`。
    from sim_launcher import ensure_turtlesim

    cleanup = ensure_turtlesim()
    try:
        main()
    finally:
        cleanup()
