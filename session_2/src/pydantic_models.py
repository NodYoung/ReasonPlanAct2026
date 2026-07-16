"""Structured data shapes for LLM outputs and mission artefacts.

These are JSON-friendly Pydantic schemas the LLM agents fill out
(``ShapeBlueprint``, ``MovementPlanOutput``, ``PlanReview``) plus the
execution artefacts they produce (``MovementPlan``, ``ExecutionReport``).
Live runtime state lives in ``utils/mission_state.MissionContext``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PenSettings(BaseModel):
    """RGB pen color and width for drawing steps."""

    r: int = Field(default=255, ge=0, le=255)
    g: int = Field(default=255, ge=0, le=255)
    b: int = Field(default=255, ge=0, le=255)
    width: int = Field(default=2, ge=1, le=20)
    off: bool = False


class StepAction(str, Enum):
    """One kind of turtle motion or canvas action in a movement plan."""

    SET_PEN = "set_pen"
    MOVE_FORWARD = "move_forward"
    ROTATE = "rotate"
    TELEPORT = "teleport"
    CLEAR_CANVAS = "clear_canvas"
    RESET_WORLD = "reset_world"


class MovementStep(BaseModel):
    """Single step in a movement plan (forward, rotate, pen)."""

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
    """Planner structured output — no ``starting_pose`` (injected at commit
    time from the live turtle; some providers reject tuple schema fields)."""

    shape: str
    side_length: float
    pen: PenSettings
    steps: list[MovementStep]

    def to_movement_plan(self, start_pose: tuple[float, float, float]) -> MovementPlan:
        return MovementPlan(**self.model_dump(), starting_pose=start_pose)


class MovementPlan(BaseModel):
    """Full movement plan ready for review and execution."""

    shape: str
    side_length: float
    pen: PenSettings
    steps: list[MovementStep]
    starting_pose: tuple[float, float, float] = (5.5, 5.5, 0.0)


class StepLog(BaseModel):
    """Result of executing one movement step on the turtle."""

    step_number: int
    action: StepAction
    success: bool
    message: str
    pose_after: tuple[float, float, float] | None = None


class ExecutionReport(BaseModel):
    """What happened when the navigator ran the approved plan."""

    success: bool
    steps_completed: int
    total_steps: int
    step_logs: list[StepLog]
    final_pose: tuple[float, float, float] | None = None
    message: str


class ShapeBlueprint(BaseModel):
    """Blueprint agent output — ordered part list for the requested drawing."""

    shape: str = Field(description="Short snake_case name of the object to draw.")
    parts: list[str] = Field(
        min_length=1,
        description="Ordered snake_case part names, large base parts first.",
    )
    side_length: float = Field(
        default=3.0, gt=0, description="Overall drawing size budget in metres."
    )


class PlanReview(BaseModel):
    """Reviewer verdict on one trial image."""

    approved: bool
    feedback: str = Field(
        description=(
            "On reject: imperative fix instructions for the planner, one concrete "
            "action per line (e.g. 'Draw the door on the bottom edge of the body'). "
            "On approve: one short sentence."
        ),
    )
