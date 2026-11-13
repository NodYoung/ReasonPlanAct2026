"""
Plan reviewer（计划审查器）—— 把计划画到一张画布图像上，看一看，然后批准或修复。

对每个部件的每次规划试验：

1. 模拟计划产生的墨迹（海龟不移动），并把它渲染在已绘墨迹之上 —— 然后保存为
   ``captures/{shape}_{part}_trial{N}.png``。这些是本应用唯一保存的图像，
   因此 captures 文件夹讲述了整幅画的故事：哪个部件出了问题、是如何修复的，
   以及画面是如何一个部件接一个部件生长起来的。
2. 先做免费的确定性检查（不消耗 LLM token）：空计划和画布外的墨迹会立即被拒绝，
   并给出具体反馈。
3. 否则进行一次视觉 LLM 调用：这张图像对于所请求的绘图而言是否合理？批准，
   或给出修复说明后拒绝。

被拒时，supervisor 会把反馈和已保存的试验图像送回部件规划器，用于下一次试验。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm
from src.pydantic_models import PlanReview
from src.utils.canvas_geometry import CANVAS_MAX, CANVAS_MIN, MARGIN
from src.utils.cli_spinner import spinner
from src.utils.drawing_render import png_path_to_data_url, render_drawing, save_image
from src.utils.mission_state import MissionContext, say
from src.utils.path_simulation import bbox_fits_canvas, simulate_plan_to_path_tracker
from src.utils.path_tracker import PathTracker
from src.utils.ros_bridge import TurtleBridge
from src.utils.turtle_control import pose_as_dict
from src.workflow.types import WorkflowState

SYSTEM_PROMPT = """You review a simulated turtlesim drawing BEFORE the real turtle draws it.
（在真实海龟绘制之前，你审查一幅模拟出的 turtlesim 绘图。）

图像展示的是 11x11 m 画布（蓝色背景）上「已经画好的墨迹」加上「拟新增的部件」。
判断这幅画面对用户的请求而言是否合理：

- 新部件的位置、比例、衔接是否正确？
  （屋顶在主体顶部、门在底边、驾驶舱在前部，……）
- 目前为止的画面是否仍然像用户请求的那个物体？
- 对以下情况应拒绝（Reject）：漂浮/脱离的部件、朝向错误、区域错误、
  几何杂乱，或任何你认为不可辨认的东西。

仅当图像是朝着用户所求绘图迈出的一个合理步骤时才批准（Approve）。
拒绝时，feedback 必须是 planner 能照做的祈使式修复说明，
每行一个具体动作（指出哪里错了 -> 应改画什么，并点明画布的区域/边）。
绝不要含糊的批评。

返回一个 PlanReview。
"""


def _merged_preview(ctx: MissionContext, start_pose: tuple[float, float, float]) -> tuple[PathTracker, PathTracker]:
    """模拟当前计划；返回 (仅新墨迹, 已有墨迹 + 新墨迹)。"""
    assert ctx.movement_plan is not None
    new_ink = simulate_plan_to_path_tracker(ctx.movement_plan, start_pose)
    merged = PathTracker()
    merged.segments = list(ctx.path_tracker.segments) + list(new_ink.segments)
    return new_ink, merged


def _reject(ctx: MissionContext, part: str, feedback: str) -> None:
    ctx.review = PlanReview(approved=False, feedback=feedback)
    say(ctx, f"[Reviewer] Part [{part}] trial {ctx.trial} rejected: {feedback.splitlines()[0]}")


def build_graph_node(bridge: TurtleBridge) -> Callable[[WorkflowState], dict]:
    def node(state: WorkflowState) -> dict:
        ctx = state["ctx"]
        part = ctx.current_part() or "?"
        pose = pose_as_dict(bridge)
        start_pose = (pose["x"], pose["y"], pose["theta"])

        new_ink, merged = _merged_preview(ctx, start_pose)

        # 免费检查 1：什么都不画的计划。
        if not new_ink.segments:
            _reject(
                ctx,
                part,
                "The plan draws nothing — no pen-down ink. Start with set_pen "
                "(off=false), then move_forward steps that form the part.",
            )
            return {}

        # 保存试验图像（本应用唯一写入的图像）。
        out = Path("captures") / f"{ctx.shape_name}_{part}_trial{ctx.trial}.png"
        save_image(render_drawing(merged), out)
        ctx.trial_image = str(out.resolve())
        say(ctx, f"[Reviewer] Trial image saved: {out}")

        # 免费检查 2：画布外的墨迹（取代了旧的安全 agent）。
        bbox = new_ink.bounding_box()
        fits, issues = bbox_fits_canvas(
            bbox, canvas_min=CANVAS_MIN, canvas_max=CANVAS_MAX, margin=MARGIN
        )
        if not fits:
            _reject(
                ctx,
                part,
                "Plan goes off-canvas: "
                + "; ".join(issues)
                + ". Keep all ink inside 0.25-10.75 m — use shorter strokes or "
                "reposition (pen up) toward the canvas centre first.",
            )
            return {}

        # 对试验图像进行一次视觉 LLM 调用。
        context = (
            f"User request: {ctx.user_request}\n"
            f"Object: {ctx.shape_name} | new part in this image: [{part}] "
            f"(trial {ctx.trial})\n"
            f"Parts already drawn: {', '.join(ctx.parts_completed) or 'none'}\n"
            f"Parts coming later: {', '.join((ctx.parts or [])[ctx.part_index + 1:]) or 'none'}"
        )
        llm = get_llm("reviewer").with_structured_output(PlanReview)
        try:
            with spinner(f"Reviewing part [{part}] trial {ctx.trial}"):
                review = llm.invoke(
                    [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(
                            content=[
                                {"type": "text", "text": context},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": png_path_to_data_url(ctx.trial_image)},
                                },
                            ]
                        ),
                    ]
                )
        except Exception as exc:  # noqa: BLE001
            say(ctx, f"[Reviewer] Review call failed ({exc}) — approving without review.")
            review = PlanReview(approved=True, feedback="Review unavailable.")

        if not isinstance(review, PlanReview):
            review = PlanReview(approved=True, feedback="Review unavailable.")
        ctx.review = review
        if review.approved:
            say(ctx, f"[Reviewer] Part [{part}] trial {ctx.trial} approved. {review.feedback}")
        else:
            say(ctx, f"[Reviewer] Part [{part}] trial {ctx.trial} rejected: {review.feedback}")
        return {}

    return node
