"""
utils_03.py —— TurtleSim 工具套件（常驻后台单例版）
=====================================================

本模块把机器人能力封装成 LangChain 工具（@tool），供 03/04 讲义复用。

设计要点（相比「每次调用都新建 node」的早期写法，更贴近生产实际）：
  - TurtleSimBackend 单例：进程内只建「一次」node、publisher 和各 service client，
    之后一直复用；discovery 等待（time.sleep）也只付一次。
  - 后台线程 spin：用 SingleThreadedExecutor 在 daemon 线程里持续 spin 该 node，
    使 pose 订阅保持活跃、service 响应得以处理。get_pose 直接读缓存的最新位姿。
  - 运动加锁：发布运动指令用一把锁串行化，避免并发写 /turtle1/cmd_vel。
  - 优雅关闭：提供 shutdown() 并用 atexit 注册 —— 进程正常退出时停止 executor、
    join 后台线程、销毁 node、关闭 rclpy；daemon=True 作为「异常退出不卡进程」的兜底。

分层约定（重要）：
  - 6 个工具（get_pose/move_forward/rotate/set_pen/teleport/reset）保持「模块级普通
    函数」，签名与返回格式不变。这样 @tool 生成的 schema 干净（不含 self），且 04 的
    `t.func` + exec 沙箱用法仍可把它们当普通函数调用。
  - 工具函数体内部委托给 TurtleSimBackend.instance() 的对应方法，由 backend 持有并
    复用 ROS 资源。

环境适配说明：
  本容器使用系统 ROS 2 Jazzy，消息/服务包名为 `turtlesim`（turtlesim.msg / turtlesim.srv），
  而非 RoboStack Kilted 的 `turtlesim_msgs`。

运行前置条件：
  1. source /opt/ros/jazzy/setup.bash   # 让 conda base 的 python 能导入 rclpy / turtlesim
  2. conda activate base                 # langchain_openai / openai 装在 base
  3. 另开进程启动模拟器：DISPLAY=:0 ros2 run turtlesim turtlesim_node
"""

import atexit
import math
import threading
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist, Vector3
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Empty
# Jazzy：turtlesim 的消息/服务在 `turtlesim` 包里（而非 Kilted 的 turtlesim_msgs）
from turtlesim.msg import Pose
from turtlesim.srv import SetPen, TeleportAbsolute

from llm_config import get_llm
from log_utils import log  # log() 等价 print()，但会打印「[文件名:行号]」前缀

# ── 常量 ──────────────────────────────────────────────────────────────────────
ARENA_MIN = 0.0
ARENA_MAX = 11.0
SERVICE_TIMEOUT = 2.0  # 等待 ROS 服务的秒数，超时则放弃


# =============================================================================
# 常驻后台 ROS 节点单例
# =============================================================================
class TurtleSimBackend:
    """常驻的 ROS 节点，持有可复用的 publisher 和 service client，并在后台线程 spin。

    通过 TurtleSimBackend.instance() 获取全局唯一实例。进程退出时自动优雅关闭。
    """

    _instance: Optional["TurtleSimBackend"] = None
    _singleton_lock = threading.Lock()  # 保护单例创建

    # ── 单例入口 ──────────────────────────────────────────────────────────────
    @classmethod
    def instance(cls) -> "TurtleSimBackend":
        # 双重检查锁定：绝大多数调用走无锁快路径
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        # rclpy context 只初始化一次
        if not rclpy.ok():
            rclpy.init()

        self.node = Node("ros_langchain_turtlesim")

        # ── 只建一次、全程复用的 publisher / service client ──────────────────
        self.cmd_vel_pub = self.node.create_publisher(Twist, "/turtle1/cmd_vel", 10)
        self.set_pen_cli = self.node.create_client(SetPen, "/turtle1/set_pen")
        self.teleport_cli = self.node.create_client(
            TeleportAbsolute, "/turtle1/teleport_absolute"
        )
        self.reset_cli = self.node.create_client(Empty, "/reset")

        # ── 订阅 pose，缓存最新值（由后台线程持续更新）──────────────────────
        self._pose: Optional[Pose] = None
        self._pose_lock = threading.Lock()
        self.node.create_subscription(Pose, "/turtle1/pose", self._on_pose, 10)

        # ── 运动指令锁：串行化对 cmd_vel 的发布 ─────────────────────────────
        self._motion_lock = threading.Lock()

        # ── 后台线程 spin ───────────────────────────────────────────────────
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, name="turtlesim_spin", daemon=True
        )
        self._spin_thread.start()

        # ── 优雅关闭 ────────────────────────────────────────────────────────
        self._closed = False
        self._shutdown_lock = threading.Lock()
        atexit.register(self.shutdown)

        # 等待一次 discovery，让 publisher/订阅建立连接；并尽量等到首帧 pose。
        # 注意：turtlesim 的 /turtle1/pose 以 ~62.5 Hz 持续发布，但后台 spin 线程启动、
        # DDS discovery、订阅与 publisher 匹配都需要时间。这里给一个较充裕的预算去等
        # 首帧；即便没等到也不致命 —— 后台 spin 会持续更新缓存，后续调用自然能拿到。
        self._latest_pose(timeout_sec=5.0)

    # ── pose 订阅回调（在后台线程执行）──────────────────────────────────────
    def _on_pose(self, msg: Pose) -> None:
        with self._pose_lock:
            self._pose = msg

    def _latest_pose(self, timeout_sec: float = 2.0) -> Optional[Pose]:
        """返回最新缓存的 pose；若尚未收到，在 timeout_sec 内轮询等待。

        这里用「轮询缓存」而非只等一次性 Event：后台 spin 会持续更新 self._pose，
        只要在预算时间内 pose 不为空即返回。这样即便首帧曾错过，后续调用也能拿到，
        不会一直卡在「从未 set」的状态。
        """
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while True:
            with self._pose_lock:
                if self._pose is not None:
                    return self._pose
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.02)

    # ── 服务调用（适配后台 spin：只 call_async，不自己 spin）────────────────
    def _call_service(self, client, request, service_name: str,
                     timeout_sec: float = SERVICE_TIMEOUT):
        """等待服务就绪、异步调用，并由后台 executor 处理响应。

        返回 (是否成功, 结果或错误字符串)。
        注意：node 已被后台线程持续 spin，这里绝不能再自己 spin_once / spin_until_...
        否则会与后台 executor 抢占回调执行，导致 future 永远不完成或并发异常。
        这里用 future 的同步事件等待结果。
        """
        if not client.wait_for_service(timeout_sec=timeout_sec):
            return False, f"服务 {service_name} 不可用。turtlesim 是否在运行？"

        future = client.call_async(request)

        # future.result() 不阻塞，用一个 Event 等待后台线程完成回调
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout=timeout_sec + 1.0):
            return False, f"对 {service_name} 的服务调用超时。"

        if future.cancelled():
            return False, f"对 {service_name} 的服务调用被取消。"
        if future.exception() is not None:
            return False, f"对 {service_name} 的服务调用失败：{future.exception()}"

        return True, future.result()

    # ── 优雅关闭 ────────────────────────────────────────────────────────────
    def shutdown(self) -> None:
        """停止后台 spin、join 线程、销毁 node、关闭 rclpy。幂等。"""
        with self._shutdown_lock:
            if self._closed:
                return
            self._closed = True

        # 1) 让 executor 停止 spin —— 后台线程的 spin() 会随之返回
        try:
            self._executor.shutdown()
        except Exception:
            pass
        # 2) 等后台线程真正退出（带超时，避免万一卡住）
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        # 3) 销毁 node + 关闭 rclpy context
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()

    # ── 位姿格式化 ──────────────────────────────────────────────────────────
    @staticmethod
    def _format_pose(pose: Optional[Pose]) -> str:
        if pose is None:
            return "等待 /turtle1/pose 超时。turtlesim 是否在运行？"
        heading_deg = math.degrees(pose.theta)
        return (
            f"x={pose.x:.2f}, y={pose.y:.2f}"
            f", heading={heading_deg:.1f}° ({pose.theta:.3f} rad)"
        )

    # =========================================================================
    # 以下为各「能力方法」—— 工具函数会委托到这里，复用上面的资源
    # =========================================================================
    def get_pose(self, timeout_sec: float = 2.0) -> str:
        return self._format_pose(self._latest_pose(timeout_sec))

    def move_forward(self, distance_m: float, speed_ms: float = 1.0,
                    rate_hz: float = 10.0) -> str:
        message = Twist(
            linear=Vector3(x=float(speed_ms)),
            angular=Vector3(z=0.0),   # 直行 —— 不旋转
        )
        stop = Twist()                # 零速度 —— 循环结束后发送

        publish_interval = 1.0 / float(rate_hz)
        duration_sec = distance_m / speed_ms
        publish_count = 0

        # 运动加锁，串行化对 cmd_vel 的写入
        with self._motion_lock:
            deadline = time.monotonic() + duration_sec
            while rclpy.ok() and time.monotonic() < deadline:
                self.cmd_vel_pub.publish(message)
                publish_count += 1
                time.sleep(min(publish_interval, max(0.0, deadline - time.monotonic())))
            # 显式停止 —— 防止乌龟按最后速度继续滑行
            self.cmd_vel_pub.publish(stop)

        # 让后台线程有机会把运动后的最新 pose 更新进缓存
        time.sleep(0.1)
        pose_str = self.get_pose()
        return (
            f"已向前移动 {distance_m:.2f} 米，速度 {speed_ms:.2f} 米/秒 "
            f"（{duration_sec:.2f} 秒内发送 {publish_count} 条消息）。"
            f"{pose_str}"
        )

    def rotate(self, angle_deg: float, angular_velocity_degs: float = 90.0,
              rate_hz: float = 10.0) -> str:
        if angle_deg == 0.0:
            return "未请求旋转：angle_deg 为 0。"

        duration_sec = abs(angle_deg) / angular_velocity_degs
        signed_velocity = math.copysign(angular_velocity_degs, angle_deg)
        angular_z_rads = math.radians(signed_velocity)

        message = Twist(
            linear=Vector3(x=0.0),                      # 显式置零 —— 不前进
            angular=Vector3(z=float(angular_z_rads)),
        )
        stop = Twist()

        publish_interval = 1.0 / float(rate_hz)
        publish_count = 0

        with self._motion_lock:
            deadline = time.monotonic() + duration_sec
            while rclpy.ok() and time.monotonic() < deadline:
                self.cmd_vel_pub.publish(message)
                publish_count += 1
                time.sleep(min(publish_interval, max(0.0, deadline - time.monotonic())))
            self.cmd_vel_pub.publish(stop)

        time.sleep(0.1)
        direction = "逆时针" if angle_deg > 0 else "顺时针"
        pose_str = self.get_pose()
        return (
            f"已{direction}旋转 {abs(angle_deg):.1f}° "
            f"（角速度 {angular_velocity_degs:.1f} 度/秒，"
            f"耗时 {duration_sec:.2f} 秒，"
            f"{rate_hz:.0f} Hz 下发送 {publish_count} 条消息）。"
            f"{pose_str}"
        )

    def set_pen(self, r: int = 255, g: int = 255, b: int = 255,
               width: int = 2, off: bool = False) -> str:
        request = SetPen.Request(r=r, g=g, b=b, width=width, off=int(off))
        success, result = self._call_service(
            self.set_pen_cli, request, "/turtle1/set_pen"
        )
        if not success:
            return result
        state = "抬起（不绘制）" if off else "落下（绘制）"
        return (
            f"画笔已设置：颜色=rgb({r}, {g}, {b})，"
            f"线宽={width}px，画笔={state}。"
        )

    def teleport(self, x: float, y: float, heading_deg: float = 0.0) -> str:
        theta_rad = math.radians(heading_deg)
        request = TeleportAbsolute.Request(x=float(x), y=float(y), theta=theta_rad)
        success, result = self._call_service(
            self.teleport_cli, request, "/turtle1/teleport_absolute"
        )
        if not success:
            return result
        time.sleep(0.1)
        pose_str = self.get_pose()
        return (
            f"已传送到 x={x:.2f}, y={y:.2f}, "
            f"heading={heading_deg:.1f}° ({theta_rad:.3f} rad)。"
            f"{pose_str}"
        )

    def reset(self) -> str:
        request = Empty.Request()
        success, result = self._call_service(self.reset_cli, request, "/reset")
        if not success:
            return result
        # 重置后 pose 会跳回原点，给后台线程一点时间更新缓存
        time.sleep(0.2)
        return "乌龟模拟器重置成功。"


# =============================================================================
# 参数 schema（Pydantic）—— 与工具函数搭配做输入校验
# =============================================================================
class MoveArgs(BaseModel):
    distance_m: float = Field(
        ..., gt=0, le=10,
        description="移动距离（米）。必须在 (0, 10] 范围内。",
    )
    speed_ms: float = Field(
        default=1.0, gt=0, le=5.0,
        description="线速度（米/秒）。默认 1.0。",
    )
    rate_hz: float = Field(
        default=10.0, gt=0,
        description="发布频率（Hz）。越高运动越平滑。默认 10。",
    )


class RotateArgs(BaseModel):
    angle_deg: float = Field(
        ..., ge=-360.0, le=360.0,
        description=(
            "目标旋转角度（度）。正值 = 逆时针，负值 = 顺时针。范围 [-360, 360]。"
        ),
    )
    angular_velocity_degs: float = Field(
        default=90.0, gt=0, le=360.0,
        description=(
            "角速度（度/秒），范围 (0, 360]。默认 90 度/秒（90° 转弯耗时 1 秒）。"
        ),
    )
    rate_hz: float = Field(
        default=10.0, gt=0, le=50.0,
        description="发布频率（Hz）。越高旋转越平滑。默认 10。",
    )


class PenArgs(BaseModel):
    r: int = Field(default=255, ge=0, le=255, description="红色分量 [0, 255]。默认 255。")
    g: int = Field(default=255, ge=0, le=255, description="绿色分量 [0, 255]。默认 255。")
    b: int = Field(default=255, ge=0, le=255, description="蓝色分量 [0, 255]。默认 255。")
    width: int = Field(default=2, ge=1, le=10, description="线宽（像素）[1, 10]。默认 2。")
    off: bool = Field(
        default=False,
        description="True = 抬笔（不绘制）；False = 落笔（绘制）。默认 False。",
    )


class TeleportArgs(BaseModel):
    x: float = Field(
        ..., ge=ARENA_MIN, le=ARENA_MAX,
        description=f"目标 x 位置（米）[{ARENA_MIN}, {ARENA_MAX}]。",
    )
    y: float = Field(
        ..., ge=ARENA_MIN, le=ARENA_MAX,
        description=f"目标 y 位置（米）[{ARENA_MIN}, {ARENA_MAX}]。",
    )
    heading_deg: float = Field(
        default=0.0, ge=0.0, lt=360.0,
        description="新朝向（度）[0, 360)。0=东，90=北。默认 0。",
    )


# =============================================================================
# 工具（模块级函数）—— 保持签名/返回格式不变，内部委托给 backend 单例
# =============================================================================
@tool
def get_pose(timeout_sec: float = 2.0) -> str:
    """返回乌龟当前位姿：x、y 位置以及朝向。

    无必填参数。返回带字段标签的纯字符串：
      'x=N, y=N, heading=N° (N rad)'

    规划移动前务必先调用它以确认当前状态。
    场地为 11×11 米；合法位置大致为 x∈[0,11]、y∈[0,11]。
    """
    return TurtleSimBackend.instance().get_pose(timeout_sec)


@tool(args_schema=MoveArgs)
def move_forward(distance_m: float, speed_ms: float = 1.0,
                rate_hz: float = 10.0) -> str:
    """沿当前朝向让乌龟向前精确移动指定距离。

    Args:
        distance_m: 移动距离（米）。必须在 (0, 10] 范围内。
        speed_ms:   线速度（米/秒），范围 (0, 5]。默认 1.0。
        rate_hz:    发布频率（Hz）。默认 10。

    返回包含发布统计信息和新位姿的确认字符串。
    若画笔处于落下状态（见 set_pen），移动会画出一条线。
    移动前务必先调用 get_pose 确认乌龟可以安全移动。
    """
    return TurtleSimBackend.instance().move_forward(distance_m, speed_ms, rate_hz)


@tool(args_schema=RotateArgs)
def rotate(angle_deg: float, angular_velocity_degs: float = 90.0,
          rate_hz: float = 10.0) -> str:
    """让乌龟原地旋转指定角度，不会向前移动。

    工具会根据角度和速度自动计算持续时间。你只需指定转多少度，以及可选的转速。

    Args:
        angle_deg:              目标角度（度），范围 [-360, 360]。
                                正值 = 逆时针，负值 = 顺时针。
        angular_velocity_degs:  角速度（度/秒），范围 (0, 360]。默认 90。
        rate_hz:                发布频率（Hz）。默认 10。

    返回旋转完成后的新朝向。旋转前务必先调用 get_pose 确认起始朝向。

    示例：
      rotate(angle_deg=90)                                # 默认速度逆时针 90°
      rotate(angle_deg=-180)                              # 顺时针掉头 180°
      rotate(angle_deg=45, angular_velocity_degs=45)      # 慢速精确转 45°
      rotate(angle_deg=-90, angular_velocity_degs=180)    # 快速顺时针 90°
    """
    return TurtleSimBackend.instance().rotate(angle_deg, angular_velocity_degs, rate_hz)


@tool(args_schema=PenArgs)
def set_pen(r: int = 255, g: int = 255, b: int = 255,
           width: int = 2, off: bool = False) -> str:
    """控制乌龟的画笔：颜色、线宽以及抬/落状态。

    重新定位前先抬笔（off=True），以免画出多余的线。准备好绘制时再落笔（off=False）。

    Args:
        r:     红色分量 [0, 255]。默认 255。
        g:     绿色分量 [0, 255]。默认 255。
        b:     蓝色分量 [0, 255]。默认 255。
        width: 线宽（像素）[1, 10]。默认 2。
        off:   True = 抬笔（不绘制）；False = 落笔（绘制）。

    常用颜色：
      白色 r=255,g=255,b=255  红色 r=255,g=0,b=0  绿色 r=0,g=255,b=0  蓝色 r=0,g=0,b=255

    示例：
      set_pen(r=255, g=0, b=0, width=3)   # 红色画笔，线宽 3
      set_pen(off=True)                    # 移动前抬笔
      set_pen(off=False)                   # 落笔继续绘制
    """
    return TurtleSimBackend.instance().set_pen(r, g, b, width, off)


@tool(args_schema=TeleportArgs)
def teleport(x: float, y: float, heading_deg: float = 0.0) -> str:
    """把乌龟瞬间传送到指定的绝对位置和朝向。

    无论画笔状态如何都不会画线。可用于：开始画新形状前重新定位、跳到已知起点、
    或在导航出错后不走回头路直接恢复。

    Args:
        x:           目标 x（米）[0, 11]。
        y:           目标 y（米）[0, 11]。
        heading_deg: 新朝向（度）[0, 360)。0=东，90=北，180=西，270=南。默认 0。

    说明：ROS 内部使用弧度 —— 转换已为你处理好。传送后务必调用 get_pose 确认新位置。

    示例：
      teleport(x=5.5, y=5.5)                  # 场地中心，朝东
      teleport(x=1.0, y=1.0, heading_deg=90)  # 左下角，朝北
    """
    return TurtleSimBackend.instance().teleport(x, y, heading_deg)


@tool
def reset() -> str:
    """重置乌龟模拟器：把乌龟送回原点并清除所有已绘制的线条。

    这是在不重启 ROS 节点的情况下重新开始的便捷方式。
    仅当你想彻底清空场地重来时才使用。它不适合常规导航，因为会打断乌龟状态的连续性。
    """
    return TurtleSimBackend.instance().reset()


# 工具套件（供 03/04 复用）
TOOLS = [get_pose, move_forward, rotate, set_pen, teleport, reset]


def safe_invoke(t, args: dict) -> str:
    """调用一个工具并把其输出作为字符串返回。

    若校验失败，返回错误信息，以便智能体自我纠正（而非崩溃）。
    """
    try:
        return str(t.invoke(args))
    except ValidationError as e:
        msgs = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
        return f"工具错误 —— 参数非法：{'; '.join(msgs)}"
    except Exception as e:
        return f"工具错误 —— {type(e).__name__}: {e}"


# 主要供 04 活动统计 LLM 调用次数
_llm_call_count = 0


def run_tool_agent(task: str, max_steps: int = 10, verbose: bool = True) -> str:
    """使用结构化工具调用 + safe_invoke 错误处理的 ReAct 智能体循环。"""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    history = [
        SystemMessage(
            "你是一个处于 11 × 11 米场地中的 TurtleSim 机器人的推理核心。\n"
            "乌龟初始位置为 x=5.5, y=5.5, heading=0°（东）。\n"
            "正向朝向 = 逆时针，90° = 北。\n"
            "请逐步使用你的工具。完成任务后，调用 get_pose() 并报告最终位置。"
        ),
        HumanMessage(task),
    ]

    step = 0
    for step in range(max_steps):
        response = llm_with_tools.invoke(history)
        history.append(response)

        if verbose:
            log(f"\n── 第 {step + 1} 步 ──────────────────────────────────────")
            if response.text:
                log(f"  模型 : {response.text}")
            for tc in response.tool_calls:
                log(f"  工具 : {tc['name']}({tc['args']})")

        if not response.tool_calls:
            if verbose:
                log(f"\n✓ 完成：{response.text}")
            return response.text

        for tc in response.tool_calls:
            tool_fn = next((t for t in TOOLS if t.name == tc["name"]), None)
            observation = (
                safe_invoke(tool_fn, tc["args"]) if tool_fn
                else f"工具错误 —— 未知工具 '{tc['name']}'"
            )
            if verbose:
                log(f"  观察 : {observation}")

            history.append(ToolMessage(observation, tool_call_id=tc["id"]))

    global _llm_call_count
    _llm_call_count = step

    return "已达到最大步数。"


if __name__ == "__main__":
    log("已注册的工具：")
    for t in TOOLS:
        first_line = t.description.splitlines()[0]
        log(f"  {t.name:<16} — {first_line}")
