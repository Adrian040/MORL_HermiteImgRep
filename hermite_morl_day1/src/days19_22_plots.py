from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_COLORS = {
    "random": "#7f7f7f",
    "energy": "#1f77b4",
    "topk": "#2ca02c",
    "top-k": "#2ca02c",
    "greedy": "#d62728",
    "Envelope-DQN": "#9467bd",
    "agent": "#9467bd",
}


def _method_color(method: str) -> str:
    for key, color in METHOD_COLORS.items():
        if key.lower() in str(method).lower():
            return color
    return "#333333"


def _setup_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.7)
    ax.tick_params(axis="both", labelsize=10)


def save_quality_cost_curve(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method, group in df.groupby("method"):
        curve = group.groupby("k").agg(ssim=("ssim", "mean"), cost=("cost", "mean")).reset_index().sort_values("cost")
        ax.plot(curve["cost"], curve["ssim"], marker="o", linewidth=2.0, markersize=5, label=str(method), color=_method_color(method))
    _setup_ax(ax, "Curva calidad–costo", "Costo normalizado", "SSIM promedio ↑")
    ax.set_ylim(max(0.0, df["ssim"].min() - 0.05), min(1.0, df["ssim"].max() + 0.05))
    ax.legend(frameon=True, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_mse_k_curve(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method, group in df.groupby("method"):
        curve = group.groupby("k").agg(mse=("mse", "mean")).reset_index().sort_values("k")
        ax.plot(curve["k"], curve["mse"], marker="o", linewidth=2.0, markersize=5, label=str(method), color=_method_color(method))
    _setup_ax(ax, "Error de reconstrucción vs número de componentes", "Número de componentes K", "MSE promedio ↓")
    ax.legend(frameon=True, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_pareto_scatter(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for method, group in df.groupby("method"):
        ax.scatter(group["obj_k"], group["obj_ssim"], s=38, alpha=0.72, label=str(method), color=_method_color(method), edgecolor="white", linewidth=0.5)
    pareto = df[df.get("is_pareto_global", False).astype(bool)] if "is_pareto_global" in df.columns else pd.DataFrame()
    if not pareto.empty:
        pareto_sorted = pareto.sort_values("obj_k")
        ax.plot(pareto_sorted["obj_k"], pareto_sorted["obj_ssim"], color="#000000", linewidth=2.0, linestyle="--", label="Frente Pareto global")
        ax.scatter(pareto_sorted["obj_k"], pareto_sorted["obj_ssim"], s=90, facecolors="none", edgecolors="#000000", linewidth=1.5)
    _setup_ax(ax, "Frente de Pareto proyectado", "Parsimonia: 1 - K_norm ↑", "Calidad perceptual: SSIM ↑")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(max(0.0, df["obj_ssim"].min() - 0.05), min(1.0, df["obj_ssim"].max() + 0.05))
    ax.legend(frameon=True, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_hypervolume_bar(hv_summary: pd.DataFrame, output_path: str | Path, column: str = "hv_2d_quality_parsimony") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = hv_summary[hv_summary["method"] != "GLOBAL"].sort_values(column, ascending=False)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = [_method_color(m) for m in data["method"]]
    bars = ax.bar(data["method"].astype(str), data[column], color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)
    _setup_ax(ax, "Hipervolumen 2D calidad–parsimonia", "Método", "HV 2D ↑")
    ax.tick_params(axis="x", rotation=25)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary_radar_like(method_summary: pd.DataFrame, output_path: str | Path) -> None:
    """Resumen formal en barras agrupadas de objetivos normalizados."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["obj_mse_mean", "obj_ssim_mean", "obj_cost_mean", "obj_k_mean"]
    labels = ["1-MSE_norm", "SSIM", "1-Costo", "1-K_norm"]
    data = method_summary.set_index("method")[cols].fillna(0.0)
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(data))
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    for i, (method, row) in enumerate(data.iterrows()):
        ax.bar(x - 0.4 + width / 2 + i * width, row.values, width=width, label=str(method), color=_method_color(method), alpha=0.9)
    _setup_ax(ax, "Comparación de objetivos normalizados", "Objetivo", "Valor promedio ↑")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_ablation_plot(ablation_summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = ablation_summary.copy().sort_values(["N", "method"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    metrics = [("ssim_mean", "SSIM promedio ↑"), ("mse_mean", "MSE promedio ↓"), ("hv_2d_quality_parsimony", "HV 2D ↑")]
    for ax, (col, ylabel) in zip(axes, metrics):
        for method, group in data.groupby("method"):
            ax.plot(group["N"], group[col], marker="o", linewidth=2, label=str(method), color=_method_color(method))
        _setup_ax(ax, f"Ablación N: {ylabel}", "Orden máximo N", ylabel)
        ax.set_xticks(sorted(data["N"].unique()))
    axes[0].legend(frameon=True, fontsize=8, loc="best")
    fig.suptitle("Ablación de orden Hermite-Gauss", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
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
    return {k: str(v) for k, v in paths.items()}
