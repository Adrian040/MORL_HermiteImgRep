from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

OBJECTIVE_COLUMNS = ["obj_mse", "obj_ssim", "obj_cost", "obj_k"]
METRIC_COLUMNS = ["mse", "ssim", "k", "cost", "k_norm"]


def ensure_objective_columns(df: pd.DataFrame, mse_reference: float | None = None) -> tuple[pd.DataFrame, float]:
    """Asegura objetivos normalizados en formato mayor-es-mejor."""
    df = df.copy()
    if "k_norm" not in df.columns and "k" in df.columns:
        max_k = float(df["k"].max()) if float(df["k"].max()) > 0 else 1.0
        df["k_norm"] = df["k"] / max_k
    if "cost" not in df.columns and "k_norm" in df.columns:
        df["cost"] = df["k_norm"]

    if mse_reference is None:
        if "mse" not in df.columns:
            raise ValueError("La tabla debe contener una columna 'mse' o columnas objetivo ya preparadas.")
        mse_reference = float(max(df["mse"].max(), 1e-8))

    if "obj_mse" not in df.columns:
        df["obj_mse"] = 1.0 - np.clip(df["mse"].astype(float) / mse_reference, 0.0, 1.0)
    if "obj_ssim" not in df.columns:
        df["obj_ssim"] = np.clip(df["ssim"].astype(float), 0.0, 1.0)
    if "obj_cost" not in df.columns:
        df["obj_cost"] = 1.0 - np.clip(df["cost"].astype(float), 0.0, 1.0)
    if "obj_k" not in df.columns:
        df["obj_k"] = 1.0 - np.clip(df["k_norm"].astype(float), 0.0, 1.0)
    return df, float(mse_reference)


def is_dominated(point: np.ndarray, others: np.ndarray, atol: float = 1e-12) -> bool:
    """Dominancia de Pareto para maximización."""
    if len(others) == 0:
        return False
    better_or_equal = np.all(others >= point - atol, axis=1)
    strictly_better = np.any(others > point + atol, axis=1)
    return bool(np.any(better_or_equal & strictly_better))


def pareto_mask(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    mask = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if is_dominated(p, np.delete(points, i, axis=0)):
            mask[i] = False
    return mask


def add_pareto_flags(df: pd.DataFrame, objective_cols: Sequence[str] = OBJECTIVE_COLUMNS) -> pd.DataFrame:
    df = df.copy()
    points = df[list(objective_cols)].to_numpy(dtype=float)
    df["is_pareto_global"] = pareto_mask(points)
    df["is_pareto_by_method"] = False
    for method, idx in df.groupby("method").groups.items():
        local_points = df.loc[idx, list(objective_cols)].to_numpy(dtype=float)
        df.loc[idx, "is_pareto_by_method"] = pareto_mask(local_points)
    return df


def hypervolume_2d_exact(points: np.ndarray, ref: tuple[float, float] = (0.0, 0.0)) -> float:
    """Hipervolumen exacto 2D para maximización respecto a ref=(0,0)."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = np.clip(pts - np.asarray(ref, dtype=float), 0.0, None)
    pts = pts[pareto_mask(pts)]
    if len(pts) == 0:
        return 0.0
    xs = np.unique(np.sort(pts[:, 0]))
    prev_x = 0.0
    area = 0.0
    for x in xs:
        y = float(np.max(pts[pts[:, 0] >= x - 1e-12, 1]))
        area += max(0.0, float(x) - prev_x) * max(0.0, y)
        prev_x = float(x)
    return float(area)


def hypervolume_mc(points: np.ndarray, samples: int = 100_000, seed: int = 0) -> float:
    """Estimación Monte Carlo de HV 4D en [0,1]^d si pymoo no está disponible."""
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = np.clip(pts, 0.0, 1.0)
    pts = pts[pareto_mask(pts)]
    rng = np.random.default_rng(seed)
    sample_points = rng.random((int(samples), pts.shape[1]))
    dominated = np.zeros(len(sample_points), dtype=bool)
    chunk = 4096
    for start in range(0, len(sample_points), chunk):
        s = sample_points[start:start + chunk]
        dominated[start:start + chunk] = np.any(np.all(pts[:, None, :] >= s[None, :, :], axis=2), axis=0)
    return float(np.mean(dominated))


def hypervolume_pymoo_or_mc(points: np.ndarray, samples: int = 100_000, seed: int = 0) -> tuple[float, str]:
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return 0.0, "empty"
    pts = np.clip(pts, 0.0, 1.0)
    try:
        from pymoo.indicators.hv import HV  # type: ignore
        hv = HV(ref_point=np.zeros(pts.shape[1]))
        return float(hv(pts)), "pymoo_exact_or_indicator"
    except Exception:
        return hypervolume_mc(pts, samples=samples, seed=seed), "monte_carlo"


def build_hypervolume_summary(
    df: pd.DataFrame,
    objective_cols: Sequence[str] = OBJECTIVE_COLUMNS,
    hv_samples: int = 100_000,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    grouped = list(df.groupby("method")) + [("GLOBAL", df)]
    for method, group in grouped:
        points4d = group[list(objective_cols)].to_numpy(dtype=float)
        hv4d, hv4d_method = hypervolume_pymoo_or_mc(points4d, samples=hv_samples, seed=seed)
        points2d = group[["obj_k", "obj_ssim"]].to_numpy(dtype=float)
        hv2d = hypervolume_2d_exact(points2d)
        rows.append({
            "method": method,
            "n_solutions": int(len(group)),
            "n_pareto_global": int(group.get("is_pareto_global", pd.Series(False, index=group.index)).sum()),
            "n_pareto_by_method": int(group.get("is_pareto_by_method", pd.Series(False, index=group.index)).sum()),
            "hv_4d": hv4d,
            "hv_4d_method": hv4d_method,
            "hv_2d_quality_parsimony": hv2d,
        })
    return pd.DataFrame(rows)


def build_method_summary(df: pd.DataFrame, hv_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    agg = df.groupby("method").agg(
        n_solutions=("method", "size"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        ssim_mean=("ssim", "mean"),
        ssim_std=("ssim", "std"),
        k_mean=("k", "mean"),
        k_std=("k", "std"),
        cost_mean=("cost", "mean"),
        cost_std=("cost", "std"),
        obj_mse_mean=("obj_mse", "mean"),
        obj_ssim_mean=("obj_ssim", "mean"),
        obj_cost_mean=("obj_cost", "mean"),
        obj_k_mean=("obj_k", "mean"),
    ).reset_index()
    if hv_summary is not None:
        hv_cols = hv_summary[hv_summary["method"] != "GLOBAL"][["method", "hv_4d", "hv_2d_quality_parsimony"]]
        agg = agg.merge(hv_cols, on="method", how="left")
    return agg


def build_preference_summary(df: pd.DataFrame) -> pd.DataFrame:
    pref_like = df[df["method"].astype(str).str.contains("Envelope|agent|DQN|preference", case=False, regex=True, na=False)].copy()
    if pref_like.empty:
        pref_like = df[df.get("preference_name", pd.Series("", index=df.index)).astype(str).str.len() > 0].copy()
    if pref_like.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["method", "preference_id", "preference_name", "preference"] if c in pref_like.columns]
    if not group_cols:
        group_cols = ["method"]
    summary = pref_like.groupby(group_cols).agg(
        mse_mean=("mse", "mean"),
        ssim_mean=("ssim", "mean"),
        k_mean=("k", "mean"),
        cost_mean=("cost", "mean"),
        common_components=("selected_labels", lambda x: most_common_nonempty(x)),
    ).reset_index()
    return summary


def most_common_nonempty(values: Iterable) -> str:
    vals = [str(v) for v in values if str(v) not in {"nan", "", "None"}]
    if not vals:
        return ""
    return pd.Series(vals).value_counts().index[0]


def format_report_table(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include=[float]).columns:
        out[c] = out[c].round(decimals)
    return out


def save_report_tables(
    method_summary: pd.DataFrame,
    preference_summary: pd.DataFrame,
    output_dir: str | Path,
    decimals: int = 4,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method_report = format_report_table(method_summary, decimals=decimals)
    method_report.to_csv(output_dir / "Days19-22_table1_method_summary_report.csv", index=False)
    method_report.to_markdown(output_dir / "Days19-22_table1_method_summary_report.md", index=False)
    try:
        method_report.to_latex(output_dir / "Days19-22_table1_method_summary_report.tex", index=False, escape=False)
    except Exception:
        pass
    if preference_summary is not None and not preference_summary.empty:
        pref_report = format_report_table(preference_summary, decimals=decimals)
        pref_report.to_csv(output_dir / "Days19-22_table2_agent_preferences_report.csv", index=False)
        pref_report.to_markdown(output_dir / "Days19-22_table2_agent_preferences_report.md", index=False)
        try:
            pref_report.to_latex(output_dir / "Days19-22_table2_agent_preferences_report.tex", index=False, escape=False)
        except Exception:
            pass


def run_days19_22_analysis(
    input_csv: str | Path,
    output_root: str | Path = "results",
    prefix: str = "Days19-22",
    mse_reference: float | None = None,
    hv_samples: int = 100_000,
    seed: int = 0,
    decimals: int = 4,
) -> dict:
    input_csv = Path(input_csv)
    output_root = Path(output_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df, mse_ref = ensure_objective_columns(df, mse_reference=mse_reference)
    df = add_pareto_flags(df)

    pareto_global = df[df["is_pareto_global"]].copy()
    pareto_by_method = df[df["is_pareto_by_method"]].copy()
    hv_summary = build_hypervolume_summary(df, hv_samples=hv_samples, seed=seed)
    method_summary = build_method_summary(df, hv_summary=hv_summary)
    preference_summary = build_preference_summary(df)

    df.to_csv(tables_dir / f"{prefix}_objective_space.csv", index=False)
    pareto_global.to_csv(tables_dir / f"{prefix}_pareto_front_global.csv", index=False)
    pareto_by_method.to_csv(tables_dir / f"{prefix}_pareto_front_by_method.csv", index=False)
    hv_summary.to_csv(tables_dir / f"{prefix}_hypervolume_summary.csv", index=False)
    method_summary.to_csv(tables_dir / f"{prefix}_method_summary.csv", index=False)
    if not preference_summary.empty:
        preference_summary.to_csv(tables_dir / f"{prefix}_agent_preference_summary.csv", index=False)
    save_report_tables(method_summary, preference_summary, tables_dir, decimals=decimals)

    return {
        "input_csv": str(input_csv),
        "mse_reference": mse_ref,
        "n_rows": int(len(df)),
        "n_pareto_global": int(len(pareto_global)),
        "objective_space": str(tables_dir / f"{prefix}_objective_space.csv"),
        "pareto_front_global": str(tables_dir / f"{prefix}_pareto_front_global.csv"),
        "pareto_front_by_method": str(tables_dir / f"{prefix}_pareto_front_by_method.csv"),
        "hypervolume_summary": str(tables_dir / f"{prefix}_hypervolume_summary.csv"),
        "method_summary": str(tables_dir / f"{prefix}_method_summary.csv"),
    }
