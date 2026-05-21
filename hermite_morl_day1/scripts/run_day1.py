from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.data_utils import prepare_dataset, save_processed_dataset
from src.hermite_filters import build_hermite_filter_bank, component_labels
from src.hermite_representation import HermiteRepresentation
from src.metrics import mse, ssim
from src.plots import save_coefficient_maps, save_filter_grid, save_image_grid, save_metric_curve


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_top_energy(images: np.ndarray, representation: HermiteRepresentation, ks: list[int]) -> pd.DataFrame:
    rows = []
    for image_id, image in enumerate(images):
        analysis = representation.analyze(image)
        order = representation.order_by_energy(analysis.energies)
        for k in ks:
            selected = order[:k].tolist()
            rec = representation.reconstruct(image, analysis.coefficients, selected, calibrated=True)
            rows.append({
                "image_id": image_id,
                "k": int(k),
                "selected_indices": " ".join(map(str, selected)),
                "mse": mse(image, rec),
                "ssim": ssim(image, rec),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config.get("project_root", "."))
    figures_dir = root / "results" / "figures"
    tables_dir = root / "results" / "tables"
    recon_dir = root / "results" / "reconstructions"
    processed_dir = root / "data" / "processed"
    for d in [figures_dir, tables_dir, recon_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    splits = prepare_dataset(config)
    save_processed_dataset(splits, processed_dir / "dataset_day1.npz")
    save_image_grid(
        splits["train"][:6],
        [f"train_{i}" for i in range(min(6, len(splits["train"])))],
        figures_dir / "dataset_examples.png",
        cols=3,
    )

    hermite_cfg = config["hermite"]
    bank = build_hermite_filter_bank(
        max_order=int(hermite_cfg.get("max_order", 3)),
        sigma=float(hermite_cfg.get("sigma", 1.5)),
        kernel_size=int(hermite_cfg.get("kernel_size", 11)),
    )
    representation = HermiteRepresentation(bank)

    save_filter_grid(bank.filters, bank.components, figures_dir / f"hermite_filters_N{bank.max_order}.png")
    components_df = pd.DataFrame({
        "component_index": list(range(bank.n_components)),
        "component": component_labels(bank.components),
        "m": [m for m, _ in bank.components],
        "n": [n for _, n in bank.components],
        "order_sum": [m + n for m, n in bank.components],
    })
    components_df.to_csv(tables_dir / "hermite_components.csv", index=False)

    image = splits["train"][0]
    analysis = representation.analyze(image)
    save_coefficient_maps(analysis.coefficients, bank.components, figures_dir / "coefficient_maps_example.png")

    energy_df = components_df.copy()
    energy_df["energy"] = analysis.energies
    energy_df["normalized_energy"] = analysis.normalized_energies
    energy_df.sort_values("energy", ascending=False).to_csv(tables_dir / "component_energy_example.csv", index=False)

    ks = [int(k) for k in config["evaluation"].get("ks", [1, 3, 5, bank.n_components])]
    ks = sorted(set([k for k in ks if 1 <= k <= bank.n_components]))
    order = representation.order_by_energy(analysis.energies)
    reconstructions = [image]
    titles = ["Original"]
    for k in ks:
        selected = order[:k].tolist()
        rec = representation.reconstruct(image, analysis.coefficients, selected, calibrated=True)
        reconstructions.append(rec)
        titles.append(f"Top-energy k={k}")
        np.save(recon_dir / f"example_top_energy_k{k}.npy", rec)
    save_image_grid(reconstructions, titles, figures_dir / "reconstruction_top_energy_example.png", cols=3)

    all_images = np.concatenate([splits["train"], splits["val"], splits["test"]], axis=0)
    metrics_df = evaluate_top_energy(all_images, representation, ks=ks)
    metrics_df.to_csv(tables_dir / "top_energy_metrics_by_image.csv", index=False)

    summary = metrics_df.groupby("k").agg(
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        ssim_mean=("ssim", "mean"),
        ssim_std=("ssim", "std"),
    ).reset_index()
    summary.to_csv(tables_dir / "top_energy_metrics_summary.csv", index=False)
    save_metric_curve(summary, figures_dir / "top_energy_metrics_curve.png")

    manifest = {
        "n_train": int(len(splits["train"])),
        "n_val": int(len(splits["val"])),
        "n_test": int(len(splits["test"])),
        "image_size": int(config["dataset"].get("image_size", 64)),
        "max_order": bank.max_order,
        "sigma": bank.sigma,
        "kernel_size": bank.kernel_size,
        "n_components": bank.n_components,
        "outputs": {
            "dataset": str(processed_dir / "dataset_day1.npz"),
            "filters_figure": str(figures_dir / f"hermite_filters_N{bank.max_order}.png"),
            "reconstruction_figure": str(figures_dir / "reconstruction_top_energy_example.png"),
            "summary_table": str(tables_dir / "top_energy_metrics_summary.csv"),
        },
    }
    with open(root / "results" / "day1_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Día 1 completado.")
    print(summary.round(4).to_string(index=False))
    print(f"Resultados guardados en: {root / 'results'}")


if __name__ == "__main__":
    main()
