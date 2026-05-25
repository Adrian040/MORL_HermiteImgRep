from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .baselines import _evaluate_selected, greedy_selection
from .metrics import to_float01


def _font(size: int = 13):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _scale(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    return (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)


def save_quality_cost_curve(summary: pd.DataFrame, output_path: str | Path, y_col: str = "ssim_mean") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height, margin = 860, 520, 70
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(13)
    small = _font(11)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    draw.text((width // 2 - 80, height - 45), "K / presupuesto", fill="black", font=small)
    draw.text((margin, 30), y_col, fill="black", font=small)

    methods = list(summary["method"].drop_duplicates())
    dash_patterns = [None, (5, 4), (2, 4), (8, 3), (1, 3)]
    for idx, method in enumerate(methods):
        sub = summary[summary["method"] == method].sort_values("k_budget")
        if sub.empty:
            continue
        xs = sub["k_budget"].to_numpy(dtype=float)
        ys = sub[y_col].to_numpy(dtype=float)
        x_plot = margin + _scale(xs) * (width - 2 * margin)
        y_plot = height - margin - _scale(ys) * (height - 2 * margin)
        points = list(map(tuple, np.stack([x_plot, y_plot], axis=1)))
        if len(points) > 1:
            draw.line(points, fill="black" if idx % 2 == 0 else "gray", width=2 + (idx % 2))
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="black" if idx % 2 == 0 else "gray")
        draw.text((width - 260, margin + 22 * idx), method, fill="black" if idx % 2 == 0 else "gray", font=font)
    canvas.save(output_path)


def _array_to_pil(image: np.ndarray, size: int = 180) -> Image.Image:
    arr = (255 * to_float01(image)).astype(np.uint8)
    return Image.fromarray(arr, mode="L").resize((size, size), Image.Resampling.BILINEAR).convert("RGB")


def save_visual_baseline_comparison(env, image_id: int, output_path: str | Path, k: int = 5, seed: int = 0) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    image_id = int(image_id) % len(env.images)
    image = env.images[image_id]
    energy_order = np.argsort(env.representation.analyze(image).energies)[::-1].astype(int).tolist()
    random_selected = rng.choice(env.n_components, size=min(k, env.n_components), replace=False).astype(int).tolist()
    greedy_selected = greedy_selection(env, image, max_k=min(k, env.n_components))
    selections = {
        "Original": [],
        f"Random k={k}": random_selected,
        f"Energy k={k}": energy_order[:k],
        f"Top-k k={k}": energy_order[:k],
        f"Greedy k={k}": greedy_selected[:k],
    }
    images = [image]
    titles = ["Original"]
    for title, selected in list(selections.items())[1:]:
        analysis = env.representation.analyze(image)
        rec = env.representation.reconstruct(image, analysis.coefficients, selected, calibrated=bool(getattr(env, "calibrated_reconstruction", True)))
        images.append(rec)
        titles.append(title)

    tile, title_h, pad = 180, 34, 10
    cols = len(images)
    canvas = Image.new("RGB", (cols * (tile + pad) + pad, tile + title_h + 2 * pad), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(12)
    for idx, (img, title) in enumerate(zip(images, titles)):
        x = pad + idx * (tile + pad)
        draw.text((x, pad), title[:24], fill="black", font=font)
        canvas.paste(_array_to_pil(img, tile), (x, pad + title_h))
    canvas.save(output_path)
