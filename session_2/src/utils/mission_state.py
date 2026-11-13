"""
单个任务的可变记事本：``MissionContext``。

每个用户请求会创建一个实例（``workflow.run_mission``），并作为 ``ctx``
贯穿 LangGraph 状态进行传递。每个智能体节点都读取并修改同一个
对象——专家节点就是通过这种方式把结果交还给主管（supervisor）的。

它是 dataclass（而非 Pydantic），因为它持有活跃的 Python 状态
（``PathTracker`` 墨迹记忆、计数器、标志位），工具会就地更新这些状态。
其中带类型的产物（``MovementPlan``、``PlanReview`` 等）则是来自
``pydantic_models.py`` 的 Pydantic 模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.pydantic_models import ExecutionReport, MovementPlan, PlanReview
from src.utils.path_tracker import PathTracker
from src.utils.log_utils import log  # log() 等价 print()，但会打印「[文件名:行号]」前缀


@dataclass
class MissionContext:
    """单个用户请求的共享可变状态。"""

    user_request: str

    # 由主管的 start_drawing 工具设置。
    drawing: bool = False

    # 蓝图（每个绘图任务只设置一次）。
    shape_name: str = ""
    parts: list[str] | None = None
    side_length: float = 3.0

    # 单个部件的流水线状态。
    part_index: int = 0
    trial: int = 1  # 当前部件的当前规划试次（从 1 开始）
    movement_plan: MovementPlan | None = None
    plan_approved: bool = False
    review: PlanReview | None = None
    feedback: str = ""  # 评审者给下一试次的修正指令
    trial_image: str | None = None  # 最近保存的 {shape}_{part}_trialN.png
    last_good_image: str | None = None  # 上一个已批准计划的试次图像

    # 进度。
    parts_completed: list[str] = field(default_factory=list)
    parts_skipped: list[str] = field(default_factory=list)
    execution_report: ExecutionReport | None = None
    execution_pending: bool = False  # 导航员已运行；主管必须记账
    mission_failed: bool = False

    # 真实画布上已有的墨迹。
    path_tracker: PathTracker = field(default_factory=PathTracker)

    def current_part(self) -> str | None:
        """当前正在规划/绘制的部件名称，完成后为 None。"""
        if self.parts and self.part_index < len(self.parts):
            return self.parts[self.part_index]
        return None

    def reset_part_pipeline(self) -> None:
        """切换到下一个部件时，清除单个部件的状态。"""
        self.trial = 1
        self.movement_plan = None
        self.plan_approved = False
        self.review = None
        self.feedback = ""
        self.trial_image = None
        self.execution_report = None


def say(ctx: MissionContext, message: str) -> None:
    """为 CLI 打印一行面向用户的状态信息。"""
    log(f"Supervisor: {message}", flush=True)
