from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .hermite_filters import component_labels
from .metrics import to_float01


def _font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _array_to_pil(image: np.ndarray, size: int = 180) -> Image.Image:
    image = to_float01(image)
    image_uint8 = (255 * image).astype(np.uint8)
    return Image.fromarray(image_uint8, mode="L").resize((size, size), Image.Resampling.BILINEAR).convert("RGB")


def save_image_grid(images: Sequence[np.ndarray], titles: Sequence[str], output_path: str | Path, cols: int = 4) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tile = 180
    title_h = 30
    pad = 10
    n = len(images)
    rows = int(np.ceil(n / cols))
    canvas = Image.new("RGB", (cols * (tile + pad) + pad, rows * (tile + title_h + pad) + pad), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(13)

    for idx, (image, title) in enumerate(zip(images, titles)):
        row, col = divmod(idx, cols)
        x = pad + col * (tile + pad)
        y = pad + row * (tile + title_h + pad)
        draw.text((x, y), str(title), fill="black", font=font)
        canvas.paste(_array_to_pil(image, size=tile), (x, y + title_h))

    canvas.save(output_path)


def save_filter_grid(filters: np.ndarray, components: Sequence[tuple[int, int]], output_path: str | Path, cols: int = 5) -> None:
    vmax = float(np.max(np.abs(filters)))
    scaled = [(kernel / (2 * vmax) + 0.5) if vmax > 1e-12 else np.zeros_like(kernel) for kernel in filters]
    save_image_grid(scaled, component_labels(components), output_path, cols=cols)


def save_coefficient_maps(coefficients: np.ndarray, components: Sequence[tuple[int, int]], output_path: str | Path, cols: int = 5) -> None:
    maps = []
    for coeff in coefficients:
        low, high = np.percentile(coeff, [1, 99])
        if high - low < 1e-12:
            maps.append(np.zeros_like(coeff))
        else:
            maps.append(np.clip((coeff - low) / (high - low), 0, 1))
    save_image_grid(maps, component_labels(components), output_path, cols=cols)


def _scale_points(xs, ys, width, height, margin):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    if x_max - x_min < 1e-12:
        x_max = x_min + 1.0
    if y_max - y_min < 1e-12:
        y_max = y_min + 1.0
    px = margin + (xs - x_min) / (x_max - x_min) * (width - 2 * margin)
    py = height - margin - (ys - y_min) / (y_max - y_min) * (height - 2 * margin)
    return list(map(tuple, np.stack([px, py], axis=1)))


def _scale_x(x, x_min, x_max, width, margin):
    if x_max - x_min < 1e-12:
        x_max = x_min + 1.0
    return margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin)


def save_metric_curve(summary_df, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height, margin = 760, 460, 80
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    small = _font(12)

    # Colores mejorados
    mse_color = (220, 20, 60)  # Crimson
    ssim_color = (30, 144, 255)  # Dodger blue
    axis_color = (0, 0, 0)

    draw.line((margin, height - margin, width - margin, height - margin), fill=axis_color, width=2)
    draw.line((margin, margin, margin, height - margin), fill=axis_color, width=2)
    draw.text((width // 2 - 120, height - 45), "Numero de componentes seleccionados (k)", fill=axis_color, font=small)
    draw.text((margin, 25), "MSE y SSIM normalizados", fill=axis_color, font=small)

    xs = summary_df["k"].to_numpy()
    mse_vals = summary_df["mse_mean"].to_numpy()
    ssim_vals = summary_df["ssim_mean"].to_numpy()
    mse_norm = (mse_vals - mse_vals.min()) / (mse_vals.max() - mse_vals.min() + 1e-12)
    ssim_norm = (ssim_vals - ssim_vals.min()) / (ssim_vals.max() - ssim_vals.min() + 1e-12)

    mse_points = _scale_points(xs, mse_norm, width, height, margin)
    ssim_points = _scale_points(xs, ssim_norm, width, height, margin)
    if len(mse_points) > 1:
        draw.line(mse_points, fill=mse_color, width=3)
        draw.line(ssim_points, fill=ssim_color, width=3)
    for p in mse_points:
        draw.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=mse_color)
    for p in ssim_points:
        draw.rectangle((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=ssim_color)

    draw.ellipse((width - 200, margin + 4, width - 190, margin + 14), fill=mse_color)
    draw.text((width - 184, margin), "MSE medio (norm.)", fill=mse_color, font=font)
    draw.rectangle((width - 200, margin + 31, width - 190, margin + 41), fill=ssim_color)
    draw.text((width - 184, margin + 25), "SSIM medio (norm.)", fill=ssim_color, font=font)
    
    for tick in np.linspace(0.0, 1.0, 6):
        y = height - margin - tick * (height - 2 * margin)
        draw.line((margin - 5, y, margin, y), fill=axis_color, width=1)
        draw.text((margin - 42, y - 7), f"{tick:.1f}", fill=axis_color, font=small)

    # Espaciar etiquetas del eje X para evitar amontonamiento
    x_min, x_max = float(xs.min()), float(xs.max())
    step = max(1, len(xs) // 5)  # Mostrar ~5 etiquetas máximo
    for i, k in enumerate(xs):
        if i % step == 0 or i == len(xs) - 1:  # Mostrar cada 'step' valores y siempre el último
            x = _scale_x(float(k), x_min, x_max, width, margin)
            draw.line((x, height - margin, x, height - margin + 5), fill=axis_color, width=1)
            label = str(int(k))
            bbox = draw.textbbox((0, 0), label, font=small)
            draw.text((x - (bbox[2] - bbox[0]) / 2, height - margin + 10), label, fill=axis_color, font=small)

    canvas.save(output_path)
