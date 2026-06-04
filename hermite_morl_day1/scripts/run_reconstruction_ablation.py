from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.filters import sobel

from src.days9_14_env_adapter import make_selection_env_from_images
from src.data_utils import prepare_dataset
from src.env_utils import load_config
from src.metrics import reconstruction_metrics
from src.plots import save_image_grid


def _base(env) -> list[int]:
    return list(getattr(env, "base_components", [])) if getattr(env, "detail_only_h00_free", False) else []


def _selectable(env) -> list[int]:
    base = set(_base(env))
    return [i for i in range(env.n_components) if i not in base]


def _selected_sets(env, image: np.ndarray) -> dict[str, list[int]]:
    analysis = env.representation.analyze(image)
    selectable = _selectable(env)
    energy_order = [i for i in np.argsort(analysis.energies)[::-1].astype(int).tolist() if i in selectable]
    out = {"all_components": selectable}
    for k in [1, 3, 5]:
        out[f"top_{k}"] = energy_order[: min(k, len(energy_order))]
    out["energy_top_k"] = energy_order[: min(5, len(energy_order))]
    out["greedy_k"] = energy_order[: min(5, len(energy_order))]
    return out


def _evaluate(env, image: np.ndarray, image_id: int, selected: list[int], mode: str, selection_method: str) -> dict:
    selected_all = sorted(set(selected + _base(env)))
    analysis = env.representation.analyze(image)
    reconstruction = env.representation.reconstruct(image, analysis.coefficients, selected_all, calibrated=(mode == "least_squares"), mode=mode)
    metrics = reconstruction_metrics(image, reconstruction)
    paid_k = len(env.paid_components(selected_all)) if hasattr(env, "paid_components") else len(selected_all)
    return {
        "reconstruction_mode": mode,
        "selection_method": selection_method,
        "image_id": int(image_id),
        "k": int(paid_k),
        "paid_k": int(paid_k),
        "total_k": int(len(selected_all)),
        "cost": env.compute_cost_from_selected(selected_all) if hasattr(env, "compute_cost_from_selected") else 0.0,
        **metrics,
    }


def _save_summary_figure(summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (860, 420), "white")
    draw = ImageDraw.Draw(canvas)
    if summary.empty:
        draw.text((30, 30), "No data", fill="black")
    else:
        draw.text((30, 20), "Reconstruction ablation: SSIM mean", fill="black")
        vals = summary["ssim_mean"].fillna(0).to_numpy(dtype=float)
        labels = (summary["reconstruction_mode"] + "/" + summary["selection_method"]).astype(str).tolist()
        vmax = max(float(vals.max()), 1e-8)
        bar_w = max(18, int(760 / max(1, len(vals))))
        for i, (label, val) in enumerate(zip(labels, vals)):
            x = 40 + i * bar_w
            h = int(val / vmax * 280)
            draw.rectangle((x, 350 - h, x + bar_w - 6, 350), fill=(65, 137, 98))
            draw.text((x, 360), label[:14], fill="black")
    canvas.save(output)


def _save_detail_only_example(env, output: Path) -> None:
    image = env.images[0]
    analysis = env.representation.analyze(image)
    base = _base(env)
    selectable = _selected_sets(env, image)["all_components"]
    order = _selected_sets(env, image)["energy_top_k"]
    base_rec = env.representation.reconstruct(image, analysis.coefficients, base, calibrated=True)
    compact = env.representation.reconstruct(image, analysis.coefficients, sorted(set(base + order[:3])), calibrated=True)
    high = env.representation.reconstruct(image, analysis.coefficients, sorted(set(base + selectable)), calibrated=True)
    save_image_grid(
        [image, base_rec, high, compact, sobel(image), sobel(high)],
        ["Original", "H00 base", "High fidelity", "Compact", "Original edges", "Reconstruction edges"],
        output,
        cols=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config.get("project_root", "."))
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    splits = prepare_dataset(config)
    env = make_selection_env_from_images(splits[args.split], config, split=args.split)
    rows = []
    for image_id, image in enumerate(env.images):
        selections = _selected_sets(env, image)
        for mode in ["direct_sum", "least_squares"]:
            for name, selected in selections.items():
                rows.append(_evaluate(env, image, image_id, selected, mode, name))
    df = pd.DataFrame(rows)
    summary = df.groupby(["reconstruction_mode", "selection_method"]).agg(
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        ssim_mean=("ssim", "mean"),
        ssim_std=("ssim", "std"),
        psnr_mean=("psnr", "mean"),
        psnr_std=("psnr", "std"),
        gradient_mse_mean=("gradient_mse", "mean"),
        gradient_mse_std=("gradient_mse", "std"),
        edge_corr_mean=("edge_corr", "mean"),
        edge_corr_std=("edge_corr", "std"),
    ).reset_index()
    df.to_csv(tables_dir / "reconstruction_ablation.csv", index=False)
    summary.to_csv(tables_dir / "reconstruction_ablation_summary.csv", index=False)
    _save_summary_figure(summary, figures_dir / "reconstruction_ablation.png")
    _save_detail_only_example(env, figures_dir / "detail_only_example.png")
    print(f"Ablacion de reconstruccion guardada en {tables_dir}")


if __name__ == "__main__":
    main()
