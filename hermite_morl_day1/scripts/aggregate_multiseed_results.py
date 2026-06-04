from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


METRICS = ["mse", "ssim", "psnr", "gradient_mse", "edge_corr", "paid_k", "k", "cost", "score"]


def _seed_id(path: Path) -> int:
    try:
        return int(path.name.split("_")[-1])
    except Exception:
        return -1


def _wide_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    spec = {"n_seeds": ("seed", "nunique")}
    for metric in METRICS:
        if metric in df.columns:
            spec[f"{metric}_mean"] = (metric, "mean")
            spec[f"{metric}_std"] = (metric, "std")
    return df.groupby(group_cols).agg(**spec).reset_index()


def _load_seed_tables(results_dir: Path, filename: str) -> pd.DataFrame:
    rows = []
    for seed_dir in sorted(results_dir.glob("seed_*")):
        path = seed_dir / "tables" / filename
        if path.exists():
            df = pd.read_csv(path)
            df["seed"] = _seed_id(seed_dir)
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _save_training_curve(results_dir: Path, output: Path) -> None:
    curves = []
    for seed_dir in sorted(results_dir.glob("seed_*")):
        path = seed_dir / "tables" / "Days9-14_training_log.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "episode" in df.columns and "scalar_return" in df.columns:
                curves.append(df[["episode", "scalar_return"]].assign(seed=_seed_id(seed_dir)))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (760, 420), "white")
    draw = ImageDraw.Draw(canvas)
    if not curves:
        draw.text((30, 30), "No training curves found", fill="black")
        canvas.save(output)
        return
    all_df = pd.concat(curves, ignore_index=True)
    summary = all_df.groupby("episode").agg(mean=("scalar_return", "mean"), std=("scalar_return", "std")).reset_index()
    x = summary["episode"].to_numpy(dtype=float)
    y = summary["mean"].to_numpy(dtype=float)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    y_max = y_min + 1.0 if abs(y_max - y_min) < 1e-12 else y_max
    points = []
    for xi, yi in zip(x, y):
        px = 60 + (xi - x_min) / (x_max - x_min + 1e-12) * 660
        py = 350 - (yi - y_min) / (y_max - y_min + 1e-12) * 280
        points.append((px, py))
    if len(points) > 1:
        draw.line(points, fill=(38, 99, 174), width=3)
    draw.text((30, 20), "Multi-seed training curve: scalar_return mean", fill="black")
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/seeds")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    root = results_dir.parent
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    eval_df = _load_seed_tables(results_dir, "Days9-14_eval_by_preference.csv")
    if not eval_df.empty:
        _wide_summary(eval_df, ["preference_id", "preference_name"]).to_csv(tables_dir / "multiseed_agent_summary.csv", index=False)
        _wide_summary(eval_df.assign(method="Envelope-DQN"), ["method"]).to_csv(tables_dir / "multiseed_method_summary.csv", index=False)
        _wide_summary(eval_df, ["preference_id", "preference_name", "preference"]).to_csv(tables_dir / "multiseed_preference_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(tables_dir / "multiseed_agent_summary.csv", index=False)
        pd.DataFrame().to_csv(tables_dir / "multiseed_method_summary.csv", index=False)
        pd.DataFrame().to_csv(tables_dir / "multiseed_preference_summary.csv", index=False)
    _save_training_curve(results_dir, figures_dir / "multiseed_training_curves.png")
    print(f"Resumen multi-seed guardado en {tables_dir}")


if __name__ == "__main__":
    main()
