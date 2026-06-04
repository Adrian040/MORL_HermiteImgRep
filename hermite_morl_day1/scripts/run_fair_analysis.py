from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.days19_22_analysis import hypervolume_2d_exact, hypervolume_pymoo_or_mc
from src.env_utils import load_config


def _load_solutions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Corre primero la evaluacion Days15-18 o usa --input-csv.")
    df = pd.read_csv(path)
    if "paid_k" not in df.columns:
        df["paid_k"] = df["k"]
    if "k_norm" not in df.columns:
        denom = max(float(df["paid_k"].max()), 1.0)
        df["k_norm"] = df["paid_k"] / denom
    return df


def _normalize_objectives(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mse_ref = max(float(out["mse"].max()), 1e-8)
    cost_ref = max(float(out["cost"].max()), 1e-8)
    k_ref = max(float(out["paid_k"].max()), 1.0)
    out["obj_mse"] = 1.0 - np.clip(out["mse"].astype(float) / mse_ref, 0.0, 1.0)
    out["obj_ssim"] = np.clip(out["ssim"].astype(float), 0.0, 1.0)
    out["obj_cost"] = 1.0 - np.clip(out["cost"].astype(float) / cost_ref, 0.0, 1.0)
    out["obj_k"] = 1.0 - np.clip(out["paid_k"].astype(float) / k_ref, 0.0, 1.0)
    return out


def run_equal_budget(df: pd.DataFrame, n_solutions: int, repetitions: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    objective_cols = ["obj_mse", "obj_ssim", "obj_cost", "obj_k"]
    for method, group in df.groupby("method"):
        replace = len(group) < n_solutions
        if replace:
            print(f"[WARN] {method} tiene {len(group)} soluciones; se muestrea con reemplazo.")
        indices = group.index.to_numpy()
        for rep in range(repetitions):
            sampled = group.loc[rng.choice(indices, size=n_solutions, replace=replace)]
            points4d = sampled[objective_cols].to_numpy(dtype=float)
            hv, hv_type = hypervolume_pymoo_or_mc(points4d, samples=50000, seed=seed + rep)
            if hv_type == "monte_carlo":
                hv_type = "4d_monte_carlo"
            if hv_type == "empty":
                points2d = sampled[["obj_k", "obj_ssim"]].to_numpy(dtype=float)
                hv = hypervolume_2d_exact(points2d)
                hv_type = "2d_fallback"
            rows.append({
                "method": method,
                "n_solutions": int(n_solutions),
                "repetition": int(rep),
                "hypervolume": float(hv),
                "hv_type": hv_type,
                "sampled_with_replacement": bool(replace),
                "available_solutions": int(len(group)),
            })
    hv_df = pd.DataFrame(rows)
    summary = hv_df.groupby("method").agg(
        hv_mean=("hypervolume", "mean"),
        hv_std=("hypervolume", "std"),
        hv_min=("hypervolume", "min"),
        hv_max=("hypervolume", "max"),
        n_solutions=("n_solutions", "first"),
        repetitions=("repetition", "count"),
        hv_type=("hv_type", lambda x: ",".join(sorted(set(map(str, x))))),
    ).reset_index()
    return hv_df, summary


def run_same_k(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agent = df[df["method"].astype(str).str.contains("Envelope|DQN|agent", case=False, regex=True, na=False)].copy()
    baselines = df[~df.index.isin(agent.index)].copy()
    rows = []
    if agent.empty or baselines.empty:
        return pd.DataFrame(), pd.DataFrame()
    metrics = ["mse", "ssim", "psnr", "gradient_mse", "edge_corr"]
    for _, a in agent.iterrows():
        same_image = baselines[baselines["image_id"] == a["image_id"]]
        for method, group in same_image.groupby("method"):
            exact = group[group["paid_k"] == a["paid_k"]]
            candidate_pool = exact if not exact.empty else group.iloc[(group["paid_k"].astype(float) - float(a["paid_k"])).abs().argsort()[:1]]
            if method == "random" and "repeat" in candidate_pool.columns:
                candidate = candidate_pool.mean(numeric_only=True)
                baseline_paid_k = float(candidate["paid_k"])
            else:
                candidate = candidate_pool.iloc[0]
                baseline_paid_k = float(candidate["paid_k"])
            row = {
                "image_id": int(a["image_id"]),
                "preference": a.get("preference", ""),
                "agent_paid_k": int(a["paid_k"]),
                "baseline_method": method,
                "baseline_paid_k": baseline_paid_k,
                "k_gap": float(abs(float(a["paid_k"]) - baseline_paid_k)),
                "agent_cost": float(a["cost"]),
                "baseline_cost": float(candidate["cost"]),
            }
            for metric in metrics:
                if metric in df.columns:
                    row[f"agent_{metric}"] = float(a[metric])
                    row[f"baseline_{metric}"] = float(candidate[metric])
                    row[f"delta_{metric}"] = float(a[metric] - candidate[metric])
            rows.append(row)
    comp = pd.DataFrame(rows)
    if comp.empty:
        return comp, pd.DataFrame()
    summary_spec = {
        "n": ("image_id", "count"),
        "mean_k_gap": ("k_gap", "mean"),
    }
    for metric in metrics:
        delta = f"delta_{metric}"
        if delta in comp.columns:
            summary_spec[f"mean_{delta}"] = (delta, "mean")
            summary_spec[f"std_{delta}"] = (delta, "std")
    group_cols = [c for c in ["preference", "baseline_method"] if c in comp.columns]
    summary = comp.groupby(group_cols).agg(**summary_spec).reset_index()
    return comp, summary


def _save_bar(summary: pd.DataFrame, value_col: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (760, 420), "white")
    draw = ImageDraw.Draw(canvas)
    if summary.empty or value_col not in summary.columns:
        draw.text((30, 30), "No data", fill="black")
        canvas.save(output)
        return
    vals = summary[value_col].fillna(0).to_numpy(dtype=float)
    labels = summary.iloc[:, 0].astype(str).tolist()
    lo, hi = min(0.0, float(vals.min())), max(1e-8, float(vals.max()))
    bar_w = max(20, int(620 / max(1, len(vals))))
    for i, (label, val) in enumerate(zip(labels, vals)):
        x = 70 + i * bar_w
        h = int((val - lo) / (hi - lo + 1e-12) * 280)
        draw.rectangle((x, 350 - h, x + bar_w - 8, 350), fill=(62, 112, 181))
        draw.text((x, 360), label[:12], fill="black")
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--input-csv", type=str, default="results/tables/Days15-18_all_methods_by_image.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config.get("project_root", "."))
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = _normalize_objectives(_load_solutions(root / args.input_csv))
    eq_cfg = config.get("evaluation", {}).get("equal_budget", {})
    hv_df, hv_summary = run_equal_budget(
        df,
        n_solutions=int(eq_cfg.get("n_solutions", 50)),
        repetitions=int(eq_cfg.get("repetitions", 20)),
        seed=int(eq_cfg.get("seed", 123)),
    )
    hv_df.to_csv(tables_dir / "equal_budget_hypervolume.csv", index=False)
    hv_summary.to_csv(tables_dir / "equal_budget_hypervolume_summary.csv", index=False)
    _save_bar(hv_summary, "hv_mean", figures_dir / "equal_budget_hypervolume.png")

    same_k, same_k_summary = run_same_k(df)
    same_k.to_csv(tables_dir / "same_k_comparison.csv", index=False)
    same_k_summary.to_csv(tables_dir / "same_k_comparison_summary.csv", index=False)
    value_col = "mean_delta_ssim" if "mean_delta_ssim" in same_k_summary.columns else "mean_k_gap"
    _save_bar(same_k_summary, value_col, figures_dir / "same_k_comparison.png")
    print(f"Analisis justo guardado en {tables_dir}")


if __name__ == "__main__":
    main()
