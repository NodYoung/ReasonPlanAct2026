"""
把 ``PathTracker`` 的线段数据变成 PNG 图像。

## 概述

Turtlesim 没有截图 API。但智能体仍然需要画布的**图像**，
用于计划评审、规划器上下文和已保存的截图。

本模块**仅是渲染器**：

1. 输入——一个 ``PathTracker``（线段列表：起点/终点、颜色、宽度）
2. 输出——PNG 字节（蓝色背景、网格、线条、可选的海龟圆点）

它**不**执行移动计划。必须先由其他东西填充该 tracker：

| 墨迹来源 | 线段如何进来 |
|------------|-------------------------|
| 真实绘制 | 导航员运行步骤 → ``plan_execution`` 更新 ``MissionContext.path_tracker`` |
| 预览 / 假设推演 | ``path_simulation.simulate_plan_to_path_tracker`` 构建一个临时 tracker |

## 典型流程（计划评审）

::

    simulate_plan_to_path_tracker(new_part_plan)   # path_simulation
    merge with ctx.path_tracker                    # 已有墨迹
    render_drawing(merged)                          # 本文件 → PNG
    save as captures/{shape}_{part}_trial{N}.png   # 评审者查看它

## 主要函数

- ``render_path_to_image`` / ``render_drawing`` —— 线段 → PNG 字节
- ``save_image`` —— 将字节写入 ``captures/``（或任意路径）
- ``png_path_to_data_url`` —— 读取已保存的 PNG，供视觉模型消息使用
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from src.utils.canvas_geometry import CANVAS_MAX, CANVAS_MIN
from src.utils.path_tracker import PathTracker

_BG_RGB = (0, 162, 232)
_DEFAULT_LINE_RGB = (255, 255, 255)
_TURTLE_RGB = (255, 165, 0)


def _world_to_pixel(
    x: float,
    y: float,
    *,
    img_size: int,
    margin_px: int,
) -> tuple[int, int]:
    draw_size = img_size - 2 * margin_px
    scale = draw_size / (CANVAS_MAX - CANVAS_MIN)
    px = margin_px + int((x - CANVAS_MIN) * scale)
    py = margin_px + int((CANVAS_MAX - y) * scale)
    return px, py


def render_path_to_image(
    path: PathTracker,
    *,
    turtle_pose: tuple[float, float, float] | None = None,
    line_rgb: tuple[int, int, int] = _DEFAULT_LINE_RGB,
    img_size: int = 600,
    margin_px: int = 20,
) -> bytes:
    """
    将所有落笔线段绘制到一张 turtlesim 风格的 PNG 上。

    ``path`` 通常来自 ``MissionContext.path_tracker``（真实墨迹）或
    来自 ``path_simulation``（模拟预览）。可选的 ``turtle_pose`` 会为海龟
    位置添加一个橙色圆点。
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (img_size, img_size), _BG_RGB)
    draw = ImageDraw.Draw(img)

    draw_size = img_size - 2 * margin_px
    scale = draw_size / (CANVAS_MAX - CANVAS_MIN)
    for m in range(int(CANVAS_MIN), int(CANVAS_MAX) + 1):
        px0, py0 = _world_to_pixel(m, CANVAS_MIN, img_size=img_size, margin_px=margin_px)
        px1, py1 = _world_to_pixel(m, CANVAS_MAX, img_size=img_size, margin_px=margin_px)
        draw.line([(px0, py0), (px1, py1)], fill=(0, 140, 200), width=1)
        px0, py0 = _world_to_pixel(CANVAS_MIN, m, img_size=img_size, margin_px=margin_px)
        px1, py1 = _world_to_pixel(CANVAS_MAX, m, img_size=img_size, margin_px=margin_px)
        draw.line([(px0, py0), (px1, py1)], fill=(0, 140, 200), width=1)

    default_width = max(2, int(scale * 0.08))
    for seg in path.segments:
        x0, y0 = _world_to_pixel(seg.start.x, seg.start.y, img_size=img_size, margin_px=margin_px)
        x1, y1 = _world_to_pixel(seg.end.x, seg.end.y, img_size=img_size, margin_px=margin_px)
        rgb = seg.color if seg.color else line_rgb
        seg_width = max(2, int(seg.pen_width * scale * 0.04)) if seg.pen_width else default_width
        draw.line([(x0, y0), (x1, y1)], fill=rgb, width=seg_width)

    if turtle_pose is not None:
        tx, ty, _ = turtle_pose
        tpx, tpy = _world_to_pixel(tx, ty, img_size=img_size, margin_px=margin_px)
        r = max(4, int(scale * 0.15))
        draw.ellipse(
            [(tpx - r, tpy - r), (tpx + r, tpy + r)],
            fill=_TURTLE_RGB,
            outline=(200, 100, 0),
        )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def save_image(png_bytes: bytes, path: str | Path) -> Path:
    """将 PNG 字节写入磁盘（如有需要会创建父文件夹）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png_bytes)
    return out


def png_path_to_data_url(path: str | Path) -> str | None:
    """读取一个 PNG 文件，并返回供支持视觉的聊天模型使用的 data URL。"""
    p = Path(path)
    if not p.is_file():
        return None
    b64 = base64.standard_b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def render_drawing(
    path: PathTracker,
    *,
    turtle_pose: tuple[float, float, float] | None = None,
) -> bytes:
    """便捷封装：``PathTracker`` → PNG 字节（使用默认图像尺寸）。"""
    return render_path_to_image(path, turtle_pose=turtle_pose)
