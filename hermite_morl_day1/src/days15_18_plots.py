from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    _NUMPY_MAJOR = int(np.__version__.split(".", 1)[0])
except Exception:
    _NUMPY_MAJOR = 1

try:
    if _NUMPY_MAJOR >= 2:
        raise ImportError("Skip matplotlib fallback under NumPy 2 runtime.")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

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
    if plt is None:
        width, height, margin = 860, 520, 70
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        font = _font(13)
        small = _font(11)
        draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
        draw.line((margin, margin, margin, height - margin), fill="black", width=2)
        draw.text((width // 2 - 110, height - 45), "Effective K / budget", fill="black", font=small)
        draw.text((margin, 30), y_col, fill="black", font=small)
        methods = list(summary["method"].drop_duplicates())
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
            draw.text((width - 260, margin + 22 * idx), str(method), fill="black" if idx % 2 == 0 else "gray", font=font)
        canvas.save(output_path)
        return

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    markers = ["o", "s", "^", "D", "P"]
    linestyles = ["-", "--", "-.", ":", (0, (5, 2))]
    colors = {
        "random": "#6B6B6B",
        "top-k energy": "#1F77B4",
        "greedy": "#B23A48",
        "Envelope-DQN": "#4C3B8F",
    }
    for idx, method in enumerate(summary["method"].drop_duplicates()):
        sub = summary[summary["method"] == method].sort_values("k_budget")
        if sub.empty:
            continue
        color = next((v for k, v in colors.items() if k.lower() in str(method).lower()), "#333333")
        yerr_col = y_col.replace("_mean", "_std")
        ax.errorbar(
            sub["k_budget"],
            sub[y_col],
            yerr=sub[yerr_col].fillna(0.0) if yerr_col in sub.columns else None,
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2.2,
            markersize=6.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            capsize=3,
            elinewidth=1.0,
            alpha=0.95,
            color=color,
            label=str(method),
        )
    ylabel_map = {
        "ssim_mean": "Mean SSIM (higher is better)",
        "mse_mean": "Mean MSE (lower is better)",
    }
    title_map = {
        "ssim_mean": "Baseline perceptual quality by component budget",
        "mse_mean": "Baseline reconstruction error by component budget",
    }
    ylabel = ylabel_map.get(y_col, y_col.replace("_", " "))
    ax.set_title(title_map.get(y_col, f"{ylabel} vs effective K"), fontsize=13.5, fontweight="bold", pad=11)
    ax.set_xlabel("Effective component budget K", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, color="#D8D8D8", linewidth=0.75, alpha=0.75)
    ax.set_axisbelow(True)
    if y_col == "mse_mean":
        ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9.5, colors="#303030")
    ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="#CFCFCF",
        facecolor="white",
        framealpha=0.96,
        fontsize=9,
        loc="best",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=340, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
        f"Top-k energy k={k}": energy_order[:k],
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
