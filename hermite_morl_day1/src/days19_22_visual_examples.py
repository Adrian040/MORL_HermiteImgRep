from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data_utils import prepare_dataset
from .hermite_filters import build_hermite_filter_bank, component_labels
from .hermite_representation import HermiteRepresentation
from .metrics import mse, ssim


@dataclass(frozen=True)
class VisualConfig:
    max_order: int = 3
    sigma: float = 1.5
    kernel_size: int = 11
    image_size: int = 64
    seed: int = 0
    split: str = "test"
    fig2_image_id: int = 0
    fig6_images: int = 3
    fig6_k: int = 5
    compact_k: int = 3
    balanced_k: int = 5
    high_fidelity_k: int | None = None
    require_agent: bool = False


METHOD_COLORS = {
    "Random": "#6c757d",
    "Energy top-k": "#1f77b4",
    "Greedy": "#d62728",
    "Envelope-DQN": "#9467bd",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _raw_has_images(raw_dir: str | Path) -> bool:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return False
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in raw_path.rglob("*"))


def load_project_images(project_config: dict, split: str = "test") -> Tuple[np.ndarray, str]:
    """Usa la misma lógica del proyecto: data/raw si hay imágenes; si no, fallback skimage."""
    splits = prepare_dataset(project_config)
    if split not in splits:
        raise ValueError(f"Split inválido: {split}. Opciones: {list(splits.keys())}")
    dataset_cfg = project_config.get("dataset", {})
    raw_dir = dataset_cfg.get("raw_dir", "data/raw")
    source = "data/raw" if _raw_has_images(raw_dir) else "skimage_fallback"
    images = np.asarray(splits[split], dtype=np.float32)
    if len(images) == 0:
        raise ValueError(f"El split '{split}' no contiene imágenes.")
    return images, source


def make_representation(config: VisualConfig) -> Tuple[HermiteRepresentation, List[str]]:
    bank = build_hermite_filter_bank(
        max_order=config.max_order,
        sigma=config.sigma,
        kernel_size=config.kernel_size,
    )
    return HermiteRepresentation(bank), component_labels(bank.components)


def _analysis_and_reconstruction(
    representation: HermiteRepresentation,
    image: np.ndarray,
    selected: Sequence[int],
) -> Tuple[np.ndarray, Dict[str, float]]:
    analysis = representation.analyze(image)
    selected = [int(i) for i in selected if 0 <= int(i) < representation.filter_bank.n_components]
    reconstruction = representation.reconstruct(image, analysis.coefficients, selected, calibrated=True)
    k = len(selected)
    n_components = representation.filter_bank.n_components
    cost = float(k / n_components)
    return reconstruction, {
        "mse": mse(image, reconstruction),
        "ssim": ssim(image, reconstruction),
        "k": float(k),
        "cost": cost,
        "k_norm": cost,
    }


def energy_order(representation: HermiteRepresentation, image: np.ndarray) -> np.ndarray:
    analysis = representation.analyze(image)
    return np.argsort(analysis.energies)[::-1]


def greedy_order(
    representation: HermiteRepresentation,
    image: np.ndarray,
    max_k: int,
    alpha: float = 0.5,
    beta: float = 0.5,
    lambda_cost: float = 0.1,
) -> List[int]:
    selected: List[int] = []
    n_components = representation.filter_bank.n_components
    _, current = _analysis_and_reconstruction(representation, image, selected)
    for _ in range(min(max_k, n_components)):
        remaining = [i for i in range(n_components) if i not in selected]
        best_component = remaining[0]
        best_score = -np.inf
        best_metrics = None
        for component in remaining:
            _, candidate = _analysis_and_reconstruction(representation, image, selected + [component])
            score = (
                alpha * (candidate["ssim"] - current["ssim"])
                + beta * (current["mse"] - candidate["mse"])
                - lambda_cost * (candidate["cost"] - current["cost"])
            )
            if score > best_score:
                best_score = score
                best_component = int(component)
                best_metrics = candidate
        selected.append(best_component)
        current = best_metrics if best_metrics is not None else current
    return selected


def random_selection(n_components: int, k: int, seed: int = 0) -> List[int]:
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_components, size=min(k, n_components), replace=False).astype(int).tolist())


def parse_selected_indices(value) -> List[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value]
    text = str(value).replace(",", " ").strip()
    if not text:
        return []
    selected = []
    for token in text.split():
        try:
            selected.append(int(token))
        except ValueError:
            continue
    return selected


def load_agent_selections_from_csv(results_csv: str | Path | None, image_id: int = 0) -> Dict[str, List[int]]:
    if not results_csv or not Path(results_csv).exists():
        return {}
    df = pd.read_csv(results_csv)
    if "method" not in df.columns or "selected_indices" not in df.columns:
        return {}
    agent = df[df["method"].astype(str).str.contains("Envelope|DQN|agent", case=False, regex=True, na=False)].copy()
    if agent.empty:
        return {}
    if "image_id" in agent.columns:
        agent = agent[agent["image_id"].astype(int) == int(image_id)]
    selections: Dict[str, List[int]] = {}
    for _, row in agent.iterrows():
        name = str(row.get("preference_name", row.get("preference", "Envelope-DQN")))
        selected = parse_selected_indices(row.get("selected_indices", ""))
        if selected:
            selections[name] = selected
    return selections


def _choose_agent_like_selection(
    agent_selections: Dict[str, List[int]],
    key_words: Sequence[str],
    fallback: List[int],
) -> Tuple[str, List[int], bool]:
    for name, selected in agent_selections.items():
        low = name.lower()
        if any(k.lower() in low for k in key_words):
            return f"Envelope-DQN: {name}", selected, True
    if agent_selections:
        name, selected = next(iter(agent_selections.items()))
        return f"Envelope-DQN: {name}", selected, True
    return "fallback", fallback, False


def _mask_image(selected_sets: Dict[str, Sequence[int]], n_components: int) -> np.ndarray:
    img = np.zeros((len(selected_sets), n_components), dtype=np.float32)
    for r, selected in enumerate(selected_sets.values()):
        for idx in selected:
            if 0 <= int(idx) < n_components:
                img[r, int(idx)] = 1.0
    return img


def _safe_image_id(image_id: int, n_images: int) -> int:
    if n_images <= 0:
        raise ValueError("No hay imágenes disponibles.")
    return int(np.clip(int(image_id), 0, n_images - 1))


def _safe_image_ids(image_ids: Sequence[int] | None, n_images: int, max_count: int) -> List[int]:
    if n_images <= 0:
        raise ValueError("No hay imágenes disponibles.")
    if image_ids:
        ids = [int(i) for i in image_ids if 0 <= int(i) < n_images]
    else:
        ids = list(range(min(max_count, n_images)))
    if not ids:
        ids = [0]
    return ids[:max_count]


def save_fig2_selection_example(
    images: np.ndarray,
    representation: HermiteRepresentation,
    labels: Sequence[str],
    output_path: str | Path,
    results_csv: str | Path | None = None,
    image_id: int = 0,
    config: VisualConfig | None = None,
) -> pd.DataFrame:
    """Figura 2: original, alta fidelidad, balanceada, compacta y máscara de selección."""
    config = config or VisualConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_id = _safe_image_id(image_id, len(images))
    image = images[image_id]
    n_components = representation.filter_bank.n_components
    high_k = n_components if config.high_fidelity_k is None else min(config.high_fidelity_k, n_components)
    balanced_k = min(config.balanced_k, n_components)
    compact_k = min(config.compact_k, n_components)

    order = energy_order(representation, image)
    fallback_high = order[:high_k].astype(int).tolist()
    fallback_balanced = greedy_order(representation, image, max_k=balanced_k)
    fallback_compact = order[:compact_k].astype(int).tolist()
    agent_selections = load_agent_selections_from_csv(results_csv, image_id=image_id)

    high_label, high_sel, high_is_agent = _choose_agent_like_selection(agent_selections, ["high", "fidelity", "alta"], fallback_high)
    bal_label, bal_sel, bal_is_agent = _choose_agent_like_selection(agent_selections, ["balance", "general"], fallback_balanced)
    comp_label, comp_sel, comp_is_agent = _choose_agent_like_selection(agent_selections, ["compact", "parsimon"], fallback_compact)

    rec_high, met_high = _analysis_and_reconstruction(representation, image, high_sel)
    rec_bal, met_bal = _analysis_and_reconstruction(representation, image, bal_sel)
    rec_comp, met_comp = _analysis_and_reconstruction(representation, image, comp_sel)

    fig = plt.figure(figsize=(12.8, 7.35))
    gs = fig.add_gridspec(2, 4, height_ratios=[3.15, 1.50], hspace=0.36, wspace=0.08)
    panels = [
        (image, "Original", None),
        (rec_high, "Alta fidelidad" if high_is_agent else "Alta fidelidad\n(fallback energy)", met_high),
        (rec_bal, "Balanceada" if bal_is_agent else "Balanceada\n(fallback greedy)", met_bal),
        (rec_comp, "Compacta" if comp_is_agent else "Compacta\n(fallback energy)", met_comp),
    ]
    for i, (img, title, metrics) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight="bold")
        if metrics is not None:
            ax.set_xlabel(f"MSE={metrics['mse']:.4f} | SSIM={metrics['ssim']:.3f} | K={int(metrics['k'])}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    selected_sets = {
        "Alta fidelidad": high_sel,
        "Balanceada": bal_sel,
        "Compacta": comp_sel,
    }
    mask = _mask_image(selected_sets, n_components)
    axm = fig.add_subplot(gs[1, :])
    im = axm.imshow(mask, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    axm.set_title("Máscara de componentes seleccionados", fontsize=12, fontweight="bold")
    axm.set_yticks(np.arange(len(selected_sets)))
    axm.set_yticklabels(list(selected_sets.keys()), fontsize=10)
    axm.set_xticks(np.arange(n_components))
    axm.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axm.set_xlabel("Componente Hermite-Gauss", fontsize=10)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] > 0.5:
                axm.text(x, y, "✓", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=axm, fraction=0.025, pad=0.02)
    cbar.set_label("Seleccionado", fontsize=9)
    fig.suptitle("Figura 2 — Ejemplo de selección adaptativa de componentes", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for name, selected, metrics, source in [
        ("high_fidelity", high_sel, met_high, high_label),
        ("balanced", bal_sel, met_bal, bal_label),
        ("compact", comp_sel, met_comp, comp_label),
    ]:
        rows.append({
            "figure": "fig2_selection_example",
            "image_id": int(image_id),
            "variant": name,
            "source": source,
            "mse": round(metrics["mse"], 6),
            "ssim": round(metrics["ssim"], 6),
            "k": int(metrics["k"]),
            "cost": round(metrics["cost"], 6),
            "selected_indices": " ".join(map(str, selected)),
            "selected_labels": " ".join(labels[i] for i in selected if 0 <= i < len(labels)),
        })
    return pd.DataFrame(rows)


def save_fig6_method_comparison(
    images: np.ndarray,
    representation: HermiteRepresentation,
    labels: Sequence[str],
    output_path: str | Path,
    results_csv: str | Path | None = None,
    image_ids: Sequence[int] | None = None,
    config: VisualConfig | None = None,
) -> pd.DataFrame:
    """Figura 6: comparación visual de métodos para 2 o 3 imágenes."""
    config = config or VisualConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_components = representation.filter_bank.n_components
    k = min(config.fig6_k, n_components)
    image_ids = _safe_image_ids(image_ids, len(images), config.fig6_images)
    method_names = ["Original", "Random", "Energy top-k", "Greedy", "Envelope-DQN"]

    fig, axes = plt.subplots(len(image_ids), len(method_names), figsize=(14.7, 3.45 * len(image_ids)), squeeze=False)
    rows = []
    for row_idx, image_id in enumerate(image_ids):
        image = images[image_id]
        order = energy_order(representation, image)
        selections = {
            "Random": random_selection(n_components, k=k, seed=config.seed + 100 * image_id),
            "Energy top-k": order[:k].astype(int).tolist(),
            "Greedy": greedy_order(representation, image, max_k=k),
        }
        agent_selections = load_agent_selections_from_csv(results_csv, image_id=image_id)
        agent_label, agent_sel, agent_ok = _choose_agent_like_selection(
            agent_selections,
            ["balance", "general", "high", "compact"],
            selections["Energy top-k"],
        )
        selections["Envelope-DQN"] = agent_sel

        for col_idx, method in enumerate(method_names):
            ax = axes[row_idx, col_idx]
            if method == "Original":
                ax.imshow(image, cmap="gray", vmin=0, vmax=1)
                ax.set_title("Original" if row_idx == 0 else "", fontsize=11, fontweight="bold")
                ax.set_ylabel(f"Imagen {image_id}", fontsize=11, fontweight="bold")
            else:
                selected = selections[method]
                rec, metrics = _analysis_and_reconstruction(representation, image, selected)
                ax.imshow(rec, cmap="gray", vmin=0, vmax=1)
                title = method
                if method == "Envelope-DQN" and not agent_ok:
                    title = "Envelope-DQN\n(no disponible; fallback)"
                ax.set_title(title if row_idx == 0 else "", fontsize=11, fontweight="bold", color=METHOD_COLORS.get(method, "#111111"))
                ax.set_xlabel(f"MSE={metrics['mse']:.4f}\nSSIM={metrics['ssim']:.3f}, K={int(metrics['k'])}", fontsize=8.8)
                rows.append({
                    "figure": "fig6_method_comparison",
                    "image_id": image_id,
                    "method": method,
                    "source": agent_label if method == "Envelope-DQN" else method,
                    "mse": round(metrics["mse"], 6),
                    "ssim": round(metrics["ssim"], 6),
                    "k": int(metrics["k"]),
                    "cost": round(metrics["cost"], 6),
                    "selected_indices": " ".join(map(str, selected)),
                    "selected_labels": " ".join(labels[i] for i in selected if 0 <= i < len(labels)),
                })
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
    fig.suptitle("Figura 6 — Comparación visual de métodos", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def run_visual_examples(
    project_config: dict,
    output_root: str | Path = "results",
    prefix: str = "Days19-22",
    config: VisualConfig | None = None,
    results_csv: str | Path | None = "results/tables/Days15-18_all_methods_by_image.csv",
    image_ids: Sequence[int] | None = None,
) -> Dict[str, str]:
    config = config or VisualConfig()
    output_root = Path(output_root)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    images, dataset_source = load_project_images(project_config, split=config.split)
    representation, labels = make_representation(config)

    fig2_path = figures_dir / f"{prefix}_fig2_selection_example.png"
    fig6_path = figures_dir / f"{prefix}_fig6_method_comparison.png"
    fig2_image_id = _safe_image_id(config.fig2_image_id, len(images))
    fig6_image_ids = _safe_image_ids(image_ids, len(images), config.fig6_images)
    meta2 = save_fig2_selection_example(
        images,
        representation,
        labels,
        fig2_path,
        results_csv=results_csv,
        image_id=fig2_image_id,
        config=config,
    )
    meta6 = save_fig6_method_comparison(
        images,
        representation,
        labels,
        fig6_path,
        results_csv=results_csv,
        image_ids=fig6_image_ids,
        config=config,
    )
    metadata = pd.concat([meta2, meta6], ignore_index=True)
    metadata_path = tables_dir / f"{prefix}_visual_examples_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    manifest = {
        "dataset_source": dataset_source,
        "split": config.split,
        "n_images_available": int(len(images)),
        "max_order": config.max_order,
        "sigma": config.sigma,
        "kernel_size": config.kernel_size,
        "image_size": config.image_size,
        "fig2_image_id": int(fig2_image_id),
        "fig6_image_ids": [int(i) for i in fig6_image_ids],
        "fig2": str(fig2_path),
        "fig6": str(fig6_path),
        "metadata": str(metadata_path),
    }
    return manifest
