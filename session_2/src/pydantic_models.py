"""LLM 输出与任务产物的结构化数据形态。

这些是 LLM agent 填写的、对 JSON 友好的 Pydantic schema
（``ShapeBlueprint``、``MovementPlanOutput``、``PlanReview``），外加它们
生成的执行产物（``MovementPlan``、``ExecutionReport``）。
运行时的实时状态位于 ``utils/mission_state.MissionContext`` 中。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PenSettings(BaseModel):
    """绘制步骤所用的 RGB 画笔颜色和宽度。"""

    r: int = Field(default=255, ge=0, le=255)
    g: int = Field(default=255, ge=0, le=255)
    b: int = Field(default=255, ge=0, le=255)
    width: int = Field(default=2, ge=1, le=20)
    off: bool = False


class StepAction(str, Enum):
    """移动计划中的一种海龟运动或画布动作。"""

    SET_PEN = "set_pen"
    MOVE_FORWARD = "move_forward"
    ROTATE = "rotate"
    TELEPORT = "teleport"
    CLEAR_CANVAS = "clear_canvas"
    RESET_WORLD = "reset_world"


class MovementStep(BaseModel):
    """移动计划中的单个步骤（前进、旋转、画笔）。"""

    step_number: int = Field(ge=1)
    action: StepAction
    distance: float | None = None
    angle_degrees: float | None = None
    x: float | None = None
    y: float | None = None
    theta: float | None = None
    pen: PenSettings | None = None
    description: str = ""


class MovementPlanOutput(BaseModel):
    """Planner 的结构化输出 —— 不含 ``starting_pose``（在提交时
    从实时海龟注入；某些 provider 会拒绝元组类型的 schema 字段）。"""

    shape: str
    side_length: float
    pen: PenSettings
    steps: list[MovementStep]

    def to_movement_plan(self, start_pose: tuple[float, float, float]) -> MovementPlan:
        return MovementPlan(**self.model_dump(), starting_pose=start_pose)


class MovementPlan(BaseModel):
    """可供评审和执行的完整移动计划。"""

    shape: str
    side_length: float
    pen: PenSettings
    steps: list[MovementStep]
    starting_pose: tuple[float, float, float] = (5.5, 5.5, 0.0)


class StepLog(BaseModel):
    """在海龟上执行某一个移动步骤的结果。"""

    step_number: int
    action: StepAction
    success: bool
    message: str
    pose_after: tuple[float, float, float] | None = None


class ExecutionReport(BaseModel):
    """navigator 运行已批准计划时发生的情况。"""

    success: bool
    steps_completed: int
    total_steps: int
    step_logs: list[StepLog]
    final_pose: tuple[float, float, float] | None = None
    message: str


class ShapeBlueprint(BaseModel):
    """Blueprint agent 的输出 —— 所请求绘图的有序部件列表。"""

    shape: str = Field(description="要绘制物体的简短 snake_case 名称。")
    parts: list[str] = Field(
        min_length=1,
        description="有序的 snake_case 部件名，较大的基础部件在前。",
    )
    side_length: float = Field(
        default=3.0, gt=0, description="整幅绘图的尺寸预算，单位米。"
    )


class PlanReview(BaseModel):
    """reviewer 对某一张试画图像的裁决。"""

    approved: bool
    feedback: str = Field(
        description=(
            "拒绝时：给 planner 的祈使式修复说明，每行一个具体动作"
            "（例如 'Draw the door on the bottom edge of the body'）。"
            "批准时：一句简短的话。"
        ),
    )
