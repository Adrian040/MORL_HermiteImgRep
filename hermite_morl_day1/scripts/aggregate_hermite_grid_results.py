from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["mse", "ssim", "psnr", "gradient_mse", "edge_corr", "paid_k", "total_k", "cost", "score", "scalar_return"]


def _seed_id(seed_dir: Path) -> int:
    try:
        return int(seed_dir.name.split("_")[-1])
    except Exception:
        return -1


def _load_grid_eval(results_dir: Path) -> pd.DataFrame:
    rows = []
    for seed_dir in sorted(results_dir.rglob("seed_*")):
        table = seed_dir / "tables" / "Days9-14_eval_by_preference.csv"
        if not table.exists():
            continue
        df = pd.read_csv(table)
        df["seed"] = df["seed"] if "seed" in df.columns else _seed_id(seed_dir)
        if "hermite_config" not in df.columns:
            df["hermite_config"] = seed_dir.parent.name
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    spec = {"n_seeds": ("seed", "nunique"), "n_rows": ("seed", "size")}
    for metric in METRICS:
        if metric in df.columns:
            spec[f"{metric}_mean"] = (metric, "mean")
            spec[f"{metric}_std"] = (metric, "std")
    return df.groupby(group_cols).agg(**spec).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/mixed_hermite_multiseed")
    parser.add_argument("--output-root", type=str, default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_root = Path(args.output_root)
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = _load_grid_eval(results_dir)
    if df.empty:
        print(f"No se encontraron evaluaciones en {results_dir}.")
        return

    df.to_csv(tables_dir / "mixed_hermite_all_eval_by_preference.csv", index=False)
    config_cols = ["hermite_config", "max_order"]
    for optional in ["sigma", "kernel_size"]:
        if optional in df.columns:
            config_cols.append(optional)
    _summary(df, config_cols).to_csv(tables_dir / "mixed_hermite_multiseed_summary.csv", index=False)
    pref_cols = config_cols + ["preference_id", "preference_name", "preference"]
    _summary(df, pref_cols).to_csv(tables_dir / "mixed_hermite_preference_summary.csv", index=False)
    best = _summary(df, config_cols).sort_values("score_mean", ascending=False) if "score" in df.columns else _summary(df, config_cols)
    best.to_csv(tables_dir / "mixed_hermite_ranked_configs.csv", index=False)
    print(f"Agregacion del banco Hermite mixto guardada en {tables_dir}")


if __name__ == "__main__":
    main()
