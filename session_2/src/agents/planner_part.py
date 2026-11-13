"""
Part planner（部件规划器）—— 为绘图中的某个部件生成海龟运动步骤。

每次规划试验运行一次：一次结构化 LLM 调用，不使用工具。模型所需的全部上下文
（实时位姿、画布边界、画笔指引、进度）都是确定性计算出来并注入提示词的 ——
没有工具往返，不浪费 token。

- 较靠后部件的第 1 次试验还会附上上一个获批的试验图像，让规划器能看到画布上
  已有的墨迹。
- 重新规划（在审查被拒之后）会得到审查器的反馈，外加被拒的试验图像
  （``captures/{shape}_{part}_trial{N}.png``）。

本节点会对计划做归一化处理（剔除瞬移、修正负向移动、把画笔宽度与已有墨迹对齐）
并将其存入任务笔记本。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm
from src.pydantic_models import MovementPlanOutput
from src.utils.cli_spinner import spinner
from src.utils.drawing_render import png_path_to_data_url
from src.utils.mission_state import MissionContext, say
from src.utils.plan_normalize import align_plan_pen_width, normalize_plan_for_current_pose
from src.utils.ros_bridge import TurtleBridge
from src.utils.turtle_control import DEFAULT_PEN_WIDTH, pose_as_dict
from src.workflow.types import WorkflowState

SYSTEM_PROMPT = """You plan turtle movement steps for ONE part of a turtlesim drawing.
（你为一幅 turtlesim 绘图中的「一个」部件规划海龟运动步骤。）

画布：11x11 米，可用 x,y = 0.25-10.75。所有墨迹必须（MUST）留在范围内。
允许的 step action：set_pen、move_forward、rotate。除此以外都不允许。
- set_pen：RGB 0-255，width 1-20，off=true 抬笔（移动但不绘制）。
- rotate：正度数 = 逆时针。
- 海龟从给定的实时位姿开始 —— 从那里规划。要在不留墨迹的情况下重新定位，
  使用 set_pen(off=true)、move/rotate，然后 set_pen(off=false)。

质量规则：
- 只（ONLY）画当前这个部件。绝不重画画布上已有的部件。
- 部件要干净地衔接：端点应与预期边界相接
  （屋顶坐落在主体顶部、门在底边，……）。
- 尊重 side_length 预算；保持合理的比例。
- 每个 step 的 description 都以 [part_name] 开头。

如果给出了 reviewer 反馈，其中每一行都是必须（MANDATORY）执行的修复 —— 严格照做。
所附图像展示的是被拒的尝试（以及正确的已有墨迹）。只修复当前部件。

返回一个 MovementPlanOutput。
"""


def _pen_width_target(ctx: MissionContext) -> int:
    """已绘制墨迹中占主导的画笔宽度，以保持各部件一致。"""
    if not ctx.path_tracker.segments:
        return DEFAULT_PEN_WIDTH
    counts = Counter(seg.pen_width for seg in ctx.path_tracker.segments)
    return counts.most_common(1)[0][0]


def build_kickoff_message(ctx: MissionContext, pose: dict[str, float]) -> HumanMessage:
    """组装规划提示词（确定性上下文 + 可选图像）。"""
    part = ctx.current_part() or "?"
    done = ", ".join(ctx.parts_completed) or "none"
    remaining = ", ".join((ctx.parts or [])[ctx.part_index + 1 :]) or "none"

    lines = [
        f"User request: {ctx.user_request}",
        f"Object: {ctx.shape_name} — DRAW THIS PART NOW: [{part}] (trial {ctx.trial})",
        f"Parts already on canvas: {done}",
        f"Parts still to draw later: {remaining}",
        f"Overall size budget: side_length ≈ {ctx.side_length:g} m",
        f"Live turtle pose: x={pose['x']:.2f}, y={pose['y']:.2f}, theta={pose['theta']:.2f} rad",
        f"Use pen width {_pen_width_target(ctx)} for every set_pen step.",
    ]

    image_path: str | None = None
    if ctx.feedback:
        lines += [
            "",
            f"REVIEWER FEEDBACK — mandatory fixes for [{part}]:",
            ctx.feedback,
        ]
        image_path = ctx.trial_image
        if image_path:
            lines.append("Attached image: the REJECTED attempt (existing ink is correct — fix only the new part).")
    elif ctx.last_good_image:
        image_path = ctx.last_good_image
        lines.append("Attached image: current canvas with all approved parts so far.")

    text = "\n".join(lines)
    image_url = png_path_to_data_url(image_path) if image_path else None
    if image_url:
        return HumanMessage(
            content=[
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
    return HumanMessage(content=text)


def build_graph_node(bridge: TurtleBridge) -> Callable[[WorkflowState], dict]:
    def node(state: WorkflowState) -> dict:
        ctx = state["ctx"]
        part = ctx.current_part() or "?"
        pose = pose_as_dict(bridge)

        llm = get_llm("planner").with_structured_output(MovementPlanOutput)
        try:
            with spinner(f"Planning part [{part}]"):
                output = llm.invoke(
                    [SystemMessage(content=SYSTEM_PROMPT), build_kickoff_message(ctx, pose)]
                )
        except Exception as exc:  # noqa: BLE001
            output = None
            say(ctx, f"[Planner] Planning part [{part}] failed ({exc}).")

        if not isinstance(output, MovementPlanOutput) or not output.steps:
            ctx.mission_failed = True
            return {}

        plan = output.to_movement_plan((pose["x"], pose["y"], pose["theta"]))
        plan = normalize_plan_for_current_pose(plan, pose)
        if ctx.path_tracker.segments:
            plan = align_plan_pen_width(plan, _pen_width_target(ctx))

        ctx.movement_plan = plan
        ctx.feedback = ""
        say(ctx, f"[Planner] Part [{part}] plan ready: {len(plan.steps)} steps (trial {ctx.trial}).")
        return {}

    return node
