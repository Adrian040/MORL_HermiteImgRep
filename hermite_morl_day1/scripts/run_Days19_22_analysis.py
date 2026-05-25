from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.days19_22_analysis import run_days19_22_analysis
from src.days19_22_plots import save_all_report_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, default="results/tables/Days15-18_all_methods_by_image.csv")
    parser.add_argument("--output-root", type=str, default="results")
    parser.add_argument("--prefix", type=str, default="Days19-22")
    parser.add_argument("--mse-reference", type=float, default=None)
    parser.add_argument("--hv-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decimals", type=int, default=4)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(
            f"No existe {input_csv}. Primero corre Days15-18 o usa --input-csv con una tabla compatible."
        )

    manifest = run_days19_22_analysis(
        input_csv=input_csv,
        output_root=args.output_root,
        prefix=args.prefix,
        mse_reference=args.mse_reference,
        hv_samples=args.hv_samples,
        seed=args.seed,
        decimals=args.decimals,
    )

    tables_dir = Path(args.output_root) / "tables"
    figures_dir = Path(args.output_root) / "figures"
    objective_space = pd.read_csv(tables_dir / f"{args.prefix}_objective_space.csv")
    hv_summary = pd.read_csv(tables_dir / f"{args.prefix}_hypervolume_summary.csv")
    method_summary = pd.read_csv(tables_dir / f"{args.prefix}_method_summary.csv")
    figure_paths = save_all_report_figures(objective_space, hv_summary, method_summary, figures_dir, prefix=args.prefix)
    manifest["figures"] = figure_paths

    manifest_path = Path(args.output_root) / f"{args.prefix}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Days19-22 completado.")
    print(f"Filas analizadas: {manifest['n_rows']}")
    print(f"Soluciones Pareto globales: {manifest['n_pareto_global']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
