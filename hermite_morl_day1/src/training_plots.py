from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def save_training_curves(history, output_path: str | Path) -> None:
    """Dibuja curvas simples sin depender de matplotlib ni modificar src/plots.py."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(history) == 0:
        Image.new("RGB", (640, 360), "white").save(output_path)
        return

    episodes = np.asarray([h["episode"] for h in history], dtype=float)
    reward = np.asarray([h.get("scalar_return", 0.0) for h in history], dtype=float)
    loss = np.asarray([h.get("loss", np.nan) for h in history], dtype=float)
    loss = np.nan_to_num(loss, nan=np.nanmedian(loss[np.isfinite(loss)]) if np.any(np.isfinite(loss)) else 0.0)

    width, height, margin = 820, 460, 70
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    small = _font(12)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    draw.text((width // 2 - 85, height - 45), "Episodio", fill="black", font=small)
    draw.text((margin, 25), "Reward escalarizado y loss normalizados", fill="black", font=small)

    def scale(vals):
        vals = np.asarray(vals, dtype=float)
        return (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)

    def points(vals):
        xs = margin + (episodes - episodes.min()) / (episodes.max() - episodes.min() + 1e-12) * (width - 2 * margin)
        ys = height - margin - scale(vals) * (height - 2 * margin)
        return list(map(tuple, np.stack([xs, ys], axis=1)))

    reward_points = points(reward)
    loss_points = points(-loss)
    if len(reward_points) > 1:
        draw.line(reward_points, fill="black", width=3)
        draw.line(loss_points, fill="gray", width=3)
    draw.text((width - 270, margin), "— retorno escalarizado", fill="black", font=font)
    draw.text((width - 270, margin + 25), "— -loss", fill="gray", font=font)
    canvas.save(output_path)
