from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


METHOD_COLORS = {
    "Envelope-DQN": "#4C3B8F",
    "greedy": "#B23A48",
    "top-k energy": "#1F77B4",
    "random": "#6B6B6B",
    "agent": "#4C3B8F",
}
METHOD_MARKERS = ["o", "s", "^", "D", "P", "X", "v"]
METHOD_LINESTYLES = ["-", "--", "-.", ":", (0, (5, 2)), (0, (2, 2))]


def _method_color(method: str) -> str:
    for key, color in METHOD_COLORS.items():
        if key.lower() in str(method).lower():
            return color
    return "#2F2F2F"


def _method_style(method: str, idx: int) -> dict:
    return {
        "color": _method_color(method),
        "marker": METHOD_MARKERS[idx % len(METHOD_MARKERS)],
        "linestyle": METHOD_LINESTYLES[idx % len(METHOD_LINESTYLES)],
        "linewidth": 2.2,
        "markersize": 6.0,
        "markeredgecolor": "white",
        "markeredgewidth": 0.7,
        "label": str(method),
    }


def _k_column(df: pd.DataFrame) -> str:
    return "k_effective" if "k_effective" in df.columns else "k"


def _setup_report_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, color="#D8D8D8", linewidth=0.75, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9.5, colors="#303030")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#303030")
    ax.spines["bottom"].set_color("#303030")


def _new_figure(figsize: tuple[float, float] = (8.2, 5.2)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def _legend(ax, ncol: int = 1) -> None:
    ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="#CFCFCF",
        facecolor="white",
        framealpha=0.96,
        fontsize=9,
        ncol=ncol,
        loc="best",
    )


def _save(fig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=340, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _fallback_plot(df: pd.DataFrame, output_path: str | Path, title: str, y_col: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height, margin = 1100, 640, 90
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small = font
    draw.text((margin, 28), title, fill="black", font=font)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    if y_col in df.columns and "method" in df.columns:
        grouped = df.groupby("method")[y_col].mean().sort_values(ascending=False)
        max_y = max(float(grouped.max()), 1e-8)
        bar_w = (width - 2 * margin) / max(1, len(grouped))
        for i, (method, val) in enumerate(grouped.items()):
            x0 = margin + i * bar_w + bar_w * 0.18
            x1 = margin + (i + 1) * bar_w - bar_w * 0.18
            y1 = height - margin
            y0 = y1 - (float(val) / max_y) * (height - 2 * margin)
            draw.rectangle((x0, y0, x1, y1), fill=_method_color(str(method)))
            draw.text((x0, y1 + 10), str(method)[:18], fill="black", font=small)
    canvas.save(output_path)


def _curve_by_k(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    k_col = _k_column(df)
    return df.groupby(k_col).agg(
        y_mean=(y_col, "mean"),
        y_std=(y_col, "std"),
    ).reset_index().rename(columns={k_col: "k"})


def save_quality_cost_curve(df: pd.DataFrame, output_path: str | Path) -> None:
    if plt is None:
        _fallback_plot(df, output_path, "Quality vs cost", "ssim")
        return
    fig, ax = _new_figure((8.4, 5.2))
    for idx, (method, group) in enumerate(df.groupby("method")):
        curve = group.groupby(_k_column(group)).agg(
            ssim_mean=("ssim", "mean"),
            ssim_std=("ssim", "std"),
            cost_mean=("cost", "mean"),
            cost_std=("cost", "std"),
        ).reset_index().sort_values("cost_mean")
        style = _method_style(method, idx)
        ax.errorbar(
            curve["cost_mean"],
            curve["ssim_mean"],
            xerr=curve["cost_std"].fillna(0.0),
            yerr=curve["ssim_std"].fillna(0.0),
            capsize=3,
            elinewidth=1.0,
            alpha=0.95,
            **style,
        )
    _setup_report_ax(ax, "Quality-cost trade-off", "Normalized cost", "Mean SSIM (higher is better)")
    ax.set_ylim(max(0.0, float(df["ssim"].min()) - 0.04), min(1.0, float(df["ssim"].max()) + 0.04))
    _legend(ax)
    _save(fig, output_path)


def save_mse_k_curve(df: pd.DataFrame, output_path: str | Path) -> None:
    if plt is None:
        _fallback_plot(df, output_path, "Reconstruction error vs components", "mse")
        return
    fig, ax = _new_figure((8.4, 5.2))
    for idx, (method, group) in enumerate(df.groupby("method")):
        curve = _curve_by_k(group, "mse").sort_values("k")
        style = _method_style(method, idx)
        ax.plot(curve["k"], curve["y_mean"], **style)
        if curve["y_std"].notna().any():
            ax.fill_between(
                curve["k"],
                np.maximum(curve["y_mean"] - curve["y_std"].fillna(0.0), 0.0),
                curve["y_mean"] + curve["y_std"].fillna(0.0),
                color=style["color"],
                alpha=0.12,
                linewidth=0,
            )
    _setup_report_ax(ax, "Reconstruction error by component budget", "Effective components K", "Mean MSE (lower is better)")
    ax.set_yscale("log")
    _legend(ax)
    _save(fig, output_path)


def save_edge_corr_k_curve(df: pd.DataFrame, output_path: str | Path) -> None:
    if plt is None:
        _fallback_plot(df, output_path, "Edge preservation vs components", "edge_corr")
        return
    fig, ax = _new_figure((8.4, 5.2))
    for idx, (method, group) in enumerate(df.groupby("method")):
        curve = _curve_by_k(group, "edge_corr").sort_values("k")
        style = _method_style(method, idx)
        ax.plot(curve["k"], curve["y_mean"], **style)
        if curve["y_std"].notna().any():
            ax.fill_between(
                curve["k"],
                curve["y_mean"] - curve["y_std"].fillna(0.0),
                curve["y_mean"] + curve["y_std"].fillna(0.0),
                color=style["color"],
                alpha=0.12,
                linewidth=0,
            )
    _setup_report_ax(ax, "Edge preservation by component budget", "Effective components K", "Mean edge correlation")
    ax.set_ylim(max(-1.0, float(df["edge_corr"].min()) - 0.04), min(1.0, float(df["edge_corr"].max()) + 0.04))
    _legend(ax)
    _save(fig, output_path)


def save_edge_corr_cost_curve(df: pd.DataFrame, output_path: str | Path) -> None:
    if plt is None:
        _fallback_plot(df, output_path, "Edge preservation vs cost", "edge_corr")
        return
    fig, ax = _new_figure((8.4, 5.2))
    for idx, (method, group) in enumerate(df.groupby("method")):
        curve = group.groupby(_k_column(group)).agg(
            edge_mean=("edge_corr", "mean"),
            edge_std=("edge_corr", "std"),
            cost_mean=("cost", "mean"),
        ).reset_index().sort_values("cost_mean")
        style = _method_style(method, idx)
        ax.plot(curve["cost_mean"], curve["edge_mean"], **style)
        if curve["edge_std"].notna().any():
            ax.fill_between(
                curve["cost_mean"],
                curve["edge_mean"] - curve["edge_std"].fillna(0.0),
                curve["edge_mean"] + curve["edge_std"].fillna(0.0),
                color=style["color"],
                alpha=0.12,
                linewidth=0,
            )
    _setup_report_ax(ax, "Edge preservation vs cost", "Normalized cost", "Mean edge correlation")
    ax.set_ylim(max(-1.0, float(df["edge_corr"].min()) - 0.04), min(1.0, float(df["edge_corr"].max()) + 0.04))
    _legend(ax)
    _save(fig, output_path)


def save_pareto_scatter(df: pd.DataFrame, output_path: str | Path) -> None:
    if plt is None:
        _fallback_plot(df, output_path, "Projected Pareto front", "obj_ssim")
        return
    fig, ax = _new_figure((8.1, 5.6))
    for idx, (method, group) in enumerate(df.groupby("method")):
        ax.scatter(
            group["obj_k"],
            group["obj_ssim"],
            s=52,
            alpha=0.68,
            label=str(method),
            color=_method_color(method),
            marker=METHOD_MARKERS[idx % len(METHOD_MARKERS)],
            edgecolor="white",
            linewidth=0.7,
        )
    if "is_pareto_global" in df.columns:
        pareto = df[df["is_pareto_global"].astype(bool)].copy()
    else:
        pareto = pd.DataFrame()
    if not pareto.empty:
        pareto_sorted = pareto.sort_values("obj_k")
        ax.plot(
            pareto_sorted["obj_k"],
            pareto_sorted["obj_ssim"],
            color="#111111",
            linewidth=2.0,
            linestyle="--",
            label="Global Pareto front",
            zorder=5,
        )
        ax.scatter(
            pareto_sorted["obj_k"],
            pareto_sorted["obj_ssim"],
            s=96,
            facecolors="none",
            edgecolors="#111111",
            linewidth=1.45,
            zorder=6,
        )
    _setup_report_ax(ax, "Projected Pareto front", "Parsimony objective: 1 - K_norm", "Quality objective: SSIM")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(max(0.0, float(df["obj_ssim"].min()) - 0.04), min(1.0, float(df["obj_ssim"].max()) + 0.04))
    _legend(ax)
    _save(fig, output_path)


def save_hypervolume_bar(hv_summary: pd.DataFrame, output_path: str | Path, column: str = "hv_2d_quality_parsimony") -> None:
    if plt is None:
        _fallback_plot(hv_summary[hv_summary["method"] != "GLOBAL"], output_path, "2D quality-parsimony hypervolume", column)
        return
    data = hv_summary[hv_summary["method"] != "GLOBAL"].copy().sort_values(column, ascending=True)
    fig, ax = _new_figure((8.2, 4.8))
    bars = ax.barh(
        data["method"].astype(str),
        data[column],
        color=[_method_color(m) for m in data["method"]],
        edgecolor="#2F2F2F",
        linewidth=0.75,
        alpha=0.93,
    )
    _setup_report_ax(ax, "Equal projection hypervolume summary", "2D quality-parsimony hypervolume (higher is better)", "Method")
    ax.grid(True, axis="x", color="#D8D8D8", linewidth=0.75)
    ax.grid(False, axis="y")
    xmax = max(float(data[column].max()), 1e-8)
    ax.set_xlim(0.0, xmax * 1.14)
    for bar in bars:
        value = bar.get_width()
        ax.text(value + xmax * 0.025, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", ha="left", fontsize=9.5)
    _save(fig, output_path)


def save_summary_radar_like(method_summary: pd.DataFrame, output_path: str | Path) -> None:
    if plt is None:
        _fallback_plot(method_summary, output_path, "Normalized objective comparison", "obj_ssim_mean")
        return
    cols = ["obj_mse_mean", "obj_ssim_mean", "obj_cost_mean", "obj_k_mean"]
    labels = ["1-MSE", "SSIM", "1-Cost", "1-K"]
    data = method_summary.set_index("method")[cols].fillna(0.0)
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    fig.patch.set_facecolor("white")
    im = ax.imshow(data.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("Normalized objective profile by method", fontsize=13.5, fontweight="bold", pad=12)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index.astype(str), fontsize=10)
    ax.tick_params(length=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = float(data.iloc[i, j])
            color = "white" if value > 0.62 else "#1F1F1F"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9.5, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Mean normalized objective", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, output_path)


def save_ablation_plot(ablation_summary: pd.DataFrame, output_path: str | Path) -> None:
    if plt is None:
        _fallback_plot(ablation_summary, output_path, "Hermite-Gauss order ablation", "ssim_mean")
        return
    data = ablation_summary.copy().sort_values(["N", "method"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    fig.patch.set_facecolor("white")
    metrics = [("ssim_mean", "Mean SSIM"), ("mse_mean", "Mean MSE"), ("hv_2d_quality_parsimony", "2D HV")]
    for ax, (col, ylabel) in zip(axes, metrics):
        for idx, (method, group) in enumerate(data.groupby("method")):
            ax.plot(group["N"], group[col], **_method_style(method, idx))
        _setup_report_ax(ax, f"Hermite order ablation: {ylabel}", "Maximum order N", ylabel)
        ax.set_xticks(sorted(data["N"].unique()))
    _legend(axes[0])
    fig.suptitle("Hermite-Gauss order ablation", fontsize=14, fontweight="bold")
    _save(fig, output_path)


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
