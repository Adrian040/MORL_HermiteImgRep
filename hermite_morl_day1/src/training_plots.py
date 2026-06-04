from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PALETTE = {
    "return": "#1f4e79",
    "loss": "#8c2d04",
    "mse": "#756bb1",
    "ssim": "#238b45",
    "cost": "#636363",
    "k": "#b15928",
    "grid": "#d9d9d9",
    "axis": "#222222",
    "text": "#222222",
    "muted": "#666666",
}


def _font(size: int = 14, bold: bool = False):
    candidates = ["DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold else ["DejaVuSans.ttf", "arial.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _hex(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _as_array(history: Iterable[dict], key: str, default: float = np.nan) -> np.ndarray:
    return np.asarray([float(h.get(key, default)) for h in history], dtype=float)


def _fill_missing(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.any(np.isfinite(values)):
        fill = float(np.nanmedian(values[np.isfinite(values)]))
    else:
        fill = 0.0
    return np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = _fill_missing(values)
    window = max(1, min(int(window), len(values)))
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _nice_limits(values: list[np.ndarray], pad_frac: float = 0.06) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(v, dtype=float)[np.isfinite(v)] for v in values if len(v) > 0])
    if len(finite) == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if abs(hi - lo) < 1e-12:
        span = max(abs(hi), 1.0)
        return lo - 0.1 * span, hi + 0.1 * span
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, ylabel: str) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=_hex("#c7c7c7"), width=1)
    draw.text((x0 + 8, y0 + 6), title, fill=_hex(PALETTE["text"]), font=_font(14, bold=True))
    draw.text((x0 + 8, y0 + 27), ylabel, fill=_hex(PALETTE["muted"]), font=_font(10))


def _draw_line_plot(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    x: np.ndarray,
    series: list[tuple[str, np.ndarray, str, int]],
    y_limits: tuple[float, float] | None = None,
) -> None:
    x0, y0, x1, y1 = box
    left, top, right, bottom = x0 + 54, y0 + 52, x1 - 18, y1 - 34
    draw.rectangle((left, top, right, bottom), outline=_hex(PALETTE["axis"]), width=1)

    y_limits = y_limits or _nice_limits([s[1] for s in series])
    y_min, y_max = y_limits
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if abs(x_max - x_min) < 1e-12:
        x_max = x_min + 1.0

    for i in range(1, 4):
        yy = top + i * (bottom - top) / 4
        draw.line((left, yy, right, yy), fill=_hex(PALETTE["grid"]), width=1)
    for i in range(1, 4):
        xx = left + i * (right - left) / 4
        draw.line((xx, top, xx, bottom), fill=_hex(PALETTE["grid"]), width=1)

    tick_font = _font(9)
    for frac in [0.0, 0.5, 1.0]:
        value = y_max - frac * (y_max - y_min)
        yy = top + frac * (bottom - top)
        draw.text((x0 + 8, yy - 6), f"{value:.3g}", fill=_hex(PALETTE["muted"]), font=tick_font)
    for frac in [0.0, 0.5, 1.0]:
        value = x_min + frac * (x_max - x_min)
        xx = left + frac * (right - left)
        draw.text((xx - 12, bottom + 8), f"{int(round(value))}", fill=_hex(PALETTE["muted"]), font=tick_font)

    def _points(values: np.ndarray) -> list[tuple[float, float]]:
        values = _fill_missing(values)
        xs = left + (x - x_min) / (x_max - x_min) * (right - left)
        ys = bottom - (values - y_min) / max(y_max - y_min, 1e-12) * (bottom - top)
        ys = np.clip(ys, top, bottom)
        return list(map(tuple, np.stack([xs, ys], axis=1)))

    for label, values, color, width in series:
        points = _points(values)
        if len(points) > 1:
            draw.line(points, fill=_hex(color), width=width, joint="curve")
        elif points:
            px, py = points[0]
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=_hex(color))

    legend_x = right - 168
    legend_y = y0 + 8
    for idx, (label, _values, color, width) in enumerate(series):
        yy = legend_y + 16 * idx
        draw.line((legend_x, yy + 7, legend_x + 22, yy + 7), fill=_hex(color), width=max(2, width))
        draw.text((legend_x + 28, yy), label, fill=_hex(PALETTE["text"]), font=_font(10))


def save_training_curves(history, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = list(history)
    if len(history) == 0:
        Image.new("RGB", (900, 520), "white").save(output_path)
        return

    episodes = _as_array(history, "episode", 0.0)
    scalar_return = _fill_missing(_as_array(history, "scalar_return"))
    loss = _fill_missing(_as_array(history, "loss"))
    mse = _fill_missing(_as_array(history, "mse"))
    ssim = _fill_missing(_as_array(history, "ssim"))
    cost = _fill_missing(_as_array(history, "cost"))
    k = _fill_missing(_as_array(history, "k"))
    epsilon = _fill_missing(_as_array(history, "epsilon"))
    window = max(5, min(75, len(history) // 12 if len(history) >= 120 else len(history) // 4 or 1))

    width, height = 1380, 860
    canvas = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22, bold=True)
    subtitle_font = _font(12)
    draw.text((42, 26), "Envelope-DQN training diagnostics", fill=_hex(PALETTE["text"]), font=title_font)
    draw.text(
        (42, 57),
        f"Episodes: {int(np.nanmax(episodes))} | smoothing window: {window} episodes",
        fill=_hex(PALETTE["muted"]),
        font=subtitle_font,
    )

    gap = 28
    panel_w = (width - 84 - gap) // 2
    panel_h = 350
    boxes = [
        (42, 96, 42 + panel_w, 96 + panel_h),
        (42 + panel_w + gap, 96, 42 + 2 * panel_w + gap, 96 + panel_h),
        (42, 96 + panel_h + gap, 42 + panel_w, 96 + 2 * panel_h + gap),
        (42 + panel_w + gap, 96 + panel_h + gap, 42 + 2 * panel_w + gap, 96 + 2 * panel_h + gap),
    ]

    _panel(draw, boxes[0], "Scalarized return", "Raw trajectory and rolling mean")
    _draw_line_plot(
        draw,
        boxes[0],
        episodes,
        [
            ("raw return", scalar_return, "#9ecae1", 1),
            ("rolling mean", _rolling_mean(scalar_return, window), PALETTE["return"], 3),
        ],
    )

    loss_log = np.log10(np.maximum(loss, 1e-8))
    _panel(draw, boxes[1], "Optimization loss", "log10(loss + eps), with rolling mean")
    _draw_line_plot(
        draw,
        boxes[1],
        episodes,
        [
            ("log loss", loss_log, "#fdd0a2", 1),
            ("rolling mean", _rolling_mean(loss_log, window), PALETTE["loss"], 3),
        ],
    )

    _panel(draw, boxes[2], "Reconstruction quality", "MSE lower is better; SSIM higher is better")
    mse_norm = (mse - np.min(mse)) / (np.max(mse) - np.min(mse) + 1e-12)
    _draw_line_plot(
        draw,
        boxes[2],
        episodes,
        [
            ("SSIM", _rolling_mean(ssim, window), PALETTE["ssim"], 3),
            ("MSE normalized", _rolling_mean(mse_norm, window), PALETTE["mse"], 3),
        ],
        y_limits=(0.0, 1.0),
    )

    _panel(draw, boxes[3], "Exploration and compactness", "Epsilon decay, cost, and selected K")
    k_norm = k / max(float(np.nanmax(k)), 1.0)
    _draw_line_plot(
        draw,
        boxes[3],
        episodes,
        [
            ("epsilon", epsilon, "#3182bd", 2),
            ("cost", _rolling_mean(cost, window), PALETTE["cost"], 3),
            ("K normalized", _rolling_mean(k_norm, window), PALETTE["k"], 3),
        ],
        y_limits=(0.0, 1.0),
    )

    draw.text((width - 365, height - 38), "Generated from results/tables/Days9-14_training_log.csv", fill=_hex(PALETTE["muted"]), font=_font(10))
    canvas.save(output_path)
