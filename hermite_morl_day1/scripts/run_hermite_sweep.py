from __future__ import annotations

import argparse
import copy
import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image, ImageDraw

from scripts.run_reconstruction_ablation import _evaluate, _selected_sets
from src.days9_14_env_adapter import make_selection_env_from_images
from src.data_utils import prepare_dataset
from src.env_utils import load_config


def _save_sweep_figure(summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (920, 460), "white")
    draw = ImageDraw.Draw(canvas)
    if summary.empty:
        draw.text((30, 30), "No data", fill="black")
        canvas.save(output)
        return
    draw.text((30, 20), "Hermite sweep: SSIM mean by config", fill="black")
    compact = summary.groupby(["sigma", "kernel_size", "max_order"]).agg(ssim_mean=("ssim_mean", "mean")).reset_index()
    vals = compact["ssim_mean"].fillna(0).to_list()
    vmax = max(max(vals), 1e-8)
    bar_w = max(16, int(820 / max(1, len(vals))))
    for i, row in compact.iterrows():
        x = 40 + i * bar_w
        h = int(float(row["ssim_mean"]) / vmax * 300)
        draw.rectangle((x, 370 - h, x + bar_w - 5, 370), fill=(118, 83, 154))
        label = f"s{row['sigma']} k{int(row['kernel_size'])} N{int(row['max_order'])}"
        draw.text((x, 382), label[:15], fill="black")
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    config = load_config(args.config)
    sweep = config.get("hermite_sweep", {})
    root = Path(config.get("project_root", "."))
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not bool(sweep.get("enabled", True)):
        print("Hermite sweep disabled in config.")
        return

    rows = []
    for sigma, kernel_size, max_order in product(
        sweep.get("sigmas", [1.5]),
        sweep.get("kernel_sizes", [13]),
        sweep.get("max_orders", [3]),
    ):
        run_config = copy.deepcopy(config)
        run_config.setdefault("hermite", {})
        run_config["hermite"].update({"sigma": float(sigma), "kernel_size": int(kernel_size), "max_order": int(max_order)})
        run_config.setdefault("hermite_multi_config", {})
        run_config["hermite_multi_config"]["enabled"] = False
        splits = prepare_dataset(run_config)
        env = make_selection_env_from_images(splits[args.split], run_config, split=args.split)
        for image_id, image in enumerate(env.images):
            for selection_method, selected in _selected_sets(env, image).items():
                row = _evaluate(env, image, image_id, selected, str(run_config.get("reconstruction", {}).get("mode", "least_squares")), selection_method)
                row.update({
                    "sigma": float(sigma),
                    "kernel_size": int(kernel_size),
                    "max_order": int(max_order),
                    "n_components": int(env.n_components),
                    "method": "baseline",
                })
                rows.append(row)
    df = pd.DataFrame(rows)
    summary = df.groupby(["sigma", "kernel_size", "max_order", "n_components", "method", "selection_method"]).agg(
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
        paid_k_mean=("paid_k", "mean"),
        paid_k_std=("paid_k", "std"),
        cost_mean=("cost", "mean"),
        cost_std=("cost", "std"),
    ).reset_index()
    df.to_csv(tables_dir / "hermite_sweep_results.csv", index=False)
    summary.to_csv(tables_dir / "hermite_sweep_summary.csv", index=False)
    _save_sweep_figure(summary, figures_dir / "hermite_sweep_summary.png")
    print(f"Hermite sweep guardado en {tables_dir}")


if __name__ == "__main__":
    main()
