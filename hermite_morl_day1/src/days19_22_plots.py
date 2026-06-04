from __future__ import annotations

from pathlib import Path

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

METHOD_COLORS = {
    "random": "#7f7f7f",
    "top-k energy": "#1f77b4",
    "greedy": "#d62728",
    "Envelope-DQN": "#9467bd",
    "agent": "#9467bd",
}
METHOD_MARKERS = ["o", "s", "^", "D", "P", "X", "v"]
METHOD_LINESTYLES = ["-", "--", "-.", ":", (0, (4, 1, 1, 1)), (0, (2, 1))]


def _method_color(method: str) -> str:
    for key, color in METHOD_COLORS.items():
        if key.lower() in str(method).lower():
            return color
    return "#333333"


def _setup_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.22, linestyle="-", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _k_column(df: pd.DataFrame) -> str:
    return "k_effective" if "k_effective" in df.columns else "k"


def _plot_kwargs(method: str, idx: int) -> dict:
    return {
        "marker": METHOD_MARKERS[idx % len(METHOD_MARKERS)],
        "linestyle": METHOD_LINESTYLES[idx % len(METHOD_LINESTYLES)],
        "linewidth": 2.0,
        "markersize": 5.5,
        "label": str(method),
        "color": _method_color(method),
        "alpha": 0.9,
    }


def _fallback_plot(df: pd.DataFrame, output_path: str | Path, title: str, y_col: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height, margin = 960, 560, 80
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        small = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        small = font
    draw.text((margin, 24), title, fill="black", font=font)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    if y_col not in df.columns:
        canvas.save(output_path)
        return
    grouped = df.groupby("method")[y_col].mean().sort_values(ascending=False)
    max_y = max(float(grouped.max()), 1e-8)
    bar_w = (width - 2 * margin) / max(1, len(grouped))
    for i, (method, val) in enumerate(grouped.items()):
        x0 = margin + i * bar_w + bar_w * 0.18
        x1 = margin + (i + 1) * bar_w - bar_w * 0.18
        y1 = height - margin
        y0 = y1 - (float(val) / max_y) * (height - 2 * margin)
        draw.rectangle((x0, y0, x1, y1), fill=_method_color(str(method)))
        draw.text((x0, y1 + 8), str(method)[:18], fill="black", font=small)
    canvas.save(output_path)


def save_quality_cost_curve(df: pd.DataFrame, output_path: str | Path) -> None:
    if plt is None:
        _fallback_plot(df, output_path, "Quality vs cost", "ssim")
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for idx, (method, group) in enumerate(df.groupby("method")):
        curve = group.groupby(_k_column(group)).agg(ssim=("ssim", "mean"), cost=("cost", "mean")).reset_index().sort_values("cost")
        ax.plot(curve["cost"], curve["ssim"], **_plot_kwargs(method, idx))
    _setup_ax(ax, "Quality vs cost", "Normalized cost", "Mean SSIM (higher is better)")
    ax.set_ylim(max(0.0, float(df["ssim"].min()) - 0.05), min(1.0, float(df["ssim"].max()) + 0.05))
    ax.legend(frameon=True, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_mse_k_curve(df: pd.DataFrame, output_path: str | Path) -> None:
    if plt is None:
        _fallback_plot(df, output_path, "Reconstruction error vs components", "mse")
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    k_col = _k_column(df)
    for idx, (method, group) in enumerate(df.groupby("method")):
        curve = group.groupby(k_col).agg(mse=("mse", "mean")).reset_index().sort_values(k_col)
        ax.plot(curve[k_col], curve["mse"], **_plot_kwargs(method, idx))
    _setup_ax(ax, "Reconstruction error vs components", "Effective components K", "Mean MSE (lower is better)")
    ax.legend(frameon=True, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_edge_corr_k_curve(df: pd.DataFrame, output_path: str | Path) -> None:
    if plt is None:
        _fallback_plot(df, output_path, "Edge preservation vs components", "edge_corr")
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    k_col = _k_column(df)
    for idx, (method, group) in enumerate(df.groupby("method")):
        curve = group.groupby(k_col).agg(edge_corr=("edge_corr", "mean")).reset_index().sort_values(k_col)
        ax.plot(curve[k_col], curve["edge_corr"], **_plot_kwargs(method, idx))
    _setup_ax(ax, "Edge preservation vs components", "Effective components K", "Mean edge correlation")
    ax.set_ylim(max(-1.0, float(df["edge_corr"].min()) - 0.05), min(1.0, float(df["edge_corr"].max()) + 0.05))
    ax.legend(frameon=True, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_edge_corr_cost_curve(df: pd.DataFrame, output_path: str | Path) -> None:
    if plt is None:
        _fallback_plot(df, output_path, "Edge preservation vs cost", "edge_corr")
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for idx, (method, group) in enumerate(df.groupby("method")):
        curve = group.groupby(_k_column(group)).agg(edge_corr=("edge_corr", "mean"), cost=("cost", "mean")).reset_index().sort_values("cost")
        ax.plot(curve["cost"], curve["edge_corr"], **_plot_kwargs(method, idx))
    _setup_ax(ax, "Edge preservation vs cost", "Normalized cost", "Mean edge correlation")
    ax.set_ylim(max(-1.0, float(df["edge_corr"].min()) - 0.05), min(1.0, float(df["edge_corr"].max()) + 0.05))
    ax.legend(frameon=True, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_pareto_scatter(df: pd.DataFrame, output_path: str | Path) -> None:
    if plt is None:
        _fallback_plot(df, output_path, "Projected Pareto front", "obj_ssim")
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for idx, (method, group) in enumerate(df.groupby("method")):
        ax.scatter(
            group["obj_k"],
            group["obj_ssim"],
            s=42,
            alpha=0.76,
            label=str(method),
            color=_method_color(method),
            marker=METHOD_MARKERS[idx % len(METHOD_MARKERS)],
            edgecolor="white",
            linewidth=0.5,
        )
    pareto = df[df.get("is_pareto_global", False).astype(bool)] if "is_pareto_global" in df.columns else pd.DataFrame()
    if not pareto.empty:
        pareto_sorted = pareto.sort_values("obj_k")
        ax.plot(pareto_sorted["obj_k"], pareto_sorted["obj_ssim"], color="#000000", linewidth=2.0, linestyle="--", label="Global Pareto front")
        ax.scatter(pareto_sorted["obj_k"], pareto_sorted["obj_ssim"], s=90, facecolors="none", edgecolors="#000000", linewidth=1.5)
    _setup_ax(ax, "Projected Pareto front", "Parsimony: 1 - K_norm (higher is better)", "Perceptual quality: SSIM")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(max(0.0, float(df["obj_ssim"].min()) - 0.05), min(1.0, float(df["obj_ssim"].max()) + 0.05))
    ax.legend(frameon=True, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_hypervolume_bar(hv_summary: pd.DataFrame, output_path: str | Path, column: str = "hv_2d_quality_parsimony") -> None:
    if plt is None:
        _fallback_plot(hv_summary[hv_summary["method"] != "GLOBAL"], output_path, "2D quality-parsimony hypervolume", column)
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = hv_summary[hv_summary["method"] != "GLOBAL"].sort_values(column, ascending=False)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = [_method_color(m) for m in data["method"]]
    bars = ax.bar(data["method"].astype(str), data[column], color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)
    _setup_ax(ax, "2D quality-parsimony hypervolume", "Method", "2D HV (higher is better)")
    ax.tick_params(axis="x", rotation=25)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_summary_radar_like(method_summary: pd.DataFrame, output_path: str | Path) -> None:
    if plt is None:
        _fallback_plot(method_summary, output_path, "Normalized objective comparison", "obj_ssim_mean")
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["obj_mse_mean", "obj_ssim_mean", "obj_cost_mean", "obj_k_mean"]
    labels = ["1-MSE_norm", "SSIM", "1-Cost", "1-K_norm"]
    data = method_summary.set_index("method")[cols].fillna(0.0)
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(data))
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    for i, (method, row) in enumerate(data.iterrows()):
        ax.bar(x - 0.4 + width / 2 + i * width, row.values, width=width, label=str(method), color=_method_color(method), alpha=0.9)
    _setup_ax(ax, "Normalized objective comparison", "Objective", "Mean value (higher is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_ablation_plot(ablation_summary: pd.DataFrame, output_path: str | Path) -> None:
    if plt is None:
        _fallback_plot(ablation_summary, output_path, "Hermite-Gauss order ablation", "ssim_mean")
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = ablation_summary.copy().sort_values(["N", "method"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    metrics = [("ssim_mean", "Mean SSIM"), ("mse_mean", "Mean MSE"), ("hv_2d_quality_parsimony", "2D HV")]
    for ax, (col, ylabel) in zip(axes, metrics):
        for idx, (method, group) in enumerate(data.groupby("method")):
            ax.plot(group["N"], group[col], **_plot_kwargs(method, idx))
        _setup_ax(ax, f"Hermite order ablation: {ylabel}", "Maximum order N", ylabel)
        ax.set_xticks(sorted(data["N"].unique()))
    axes[0].legend(frameon=True, fontsize=8, loc="best")
    fig.suptitle("Hermite-Gauss order ablation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_all_report_figures(df: pd.DataFrame, hv_summary: pd.DataFrame, method_summary: pd.DataFrame, figures_dir: str | Path, prefix: str = "Days19-22") -> dict:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "quality_cost": figures_dir / f"{prefix}_fig3_quality_cost.png",
        "mse_k": figures_dir / f"{prefix}_fig4_mse_vs_k.png",
        "pareto": figures_dir / f"{prefix}_fig5_pareto_front.png",
        "hypervolume": figures_dir / f"{prefix}_hypervolume_bar.png",
        "objectives": figures_dir / f"{prefix}_objectives_summary.png",
    }
    save_quality_cost_curve(df, paths["quality_cost"])
    save_mse_k_curve(df, paths["mse_k"])
    save_pareto_scatter(df, paths["pareto"])
    save_hypervolume_bar(hv_summary, paths["hypervolume"])
    save_summary_radar_like(method_summary, paths["objectives"])
    if "edge_corr" in df.columns and df["edge_corr"].notna().any():
        paths["edge_corr_k"] = figures_dir / "edge_corr_vs_k.png"
        paths["edge_corr_cost"] = figures_dir / "edge_corr_vs_cost.png"
        save_edge_corr_k_curve(df, paths["edge_corr_k"])
        save_edge_corr_cost_curve(df, paths["edge_corr_cost"])
    return {k: str(v) for k, v in paths.items()}
