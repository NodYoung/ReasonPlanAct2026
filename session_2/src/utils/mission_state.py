"""
Mutable notebook for one mission: ``MissionContext``.

One instance is created per user request (``workflow.run_mission``) and passed
through LangGraph state as ``ctx``. Every agent node reads and mutates the
same object — this is how specialists hand results back to the supervisor.

It is a dataclass (not Pydantic) because it holds live Python state
(``PathTracker`` ink memory, counters, flags) that tools update in place.
The typed artefacts inside (``MovementPlan``, ``PlanReview``, ...) are
Pydantic models from ``pydantic_models.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.pydantic_models import ExecutionReport, MovementPlan, PlanReview
from src.utils.path_tracker import PathTracker


@dataclass
class MissionContext:
    """Shared mutable state for one user request."""

    user_request: str

    # Set by the supervisor's start_drawing tool.
    drawing: bool = False

    # Blueprint (set once per drawing mission).
    shape_name: str = ""
    parts: list[str] | None = None
    side_length: float = 3.0

    # Per-part pipeline state.
    part_index: int = 0
    trial: int = 1  # current planning trial for the current part (1-based)
    movement_plan: MovementPlan | None = None
    plan_approved: bool = False
    review: PlanReview | None = None
    feedback: str = ""  # reviewer fix instructions for the next trial
    trial_image: str | None = None  # latest saved {shape}_{part}_trialN.png
    last_good_image: str | None = None  # trial image of the last approved plan

    # Progress.
    parts_completed: list[str] = field(default_factory=list)
    parts_skipped: list[str] = field(default_factory=list)
    execution_report: ExecutionReport | None = None
    execution_pending: bool = False  # navigator ran; supervisor must bookkeep
    mission_failed: bool = False

    # Ink already on the real canvas.
    path_tracker: PathTracker = field(default_factory=PathTracker)

    def current_part(self) -> str | None:
        """Name of the part being planned/drawn now, or None when finished."""
        if self.parts and self.part_index < len(self.parts):
            return self.parts[self.part_index]
        return None

    def reset_part_pipeline(self) -> None:
        """Clear per-part state when moving to the next part."""
        self.trial = 1
        self.movement_plan = None
        self.plan_approved = False
        self.review = None
        self.feedback = ""
        self.trial_image = None
        self.execution_report = None


def say(ctx: MissionContext, message: str) -> None:
    """Print a user-facing status line for the CLI."""
    print(f"Supervisor: {message}", flush=True)
