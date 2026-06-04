from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .metrics import edge_correlation, mse, ssim


@dataclass(frozen=True)
class BaselineConfig:
    ks: Sequence[int]
    topk_budgets: Sequence[int]
    random_repeats: int = 20
    greedy_alpha: float = 0.5
    greedy_beta: float = 0.5
    greedy_lambda: float = 0.1
    seed: int = 0


def _component_labels(env) -> list[str]:
    if hasattr(env, "action_labels"):
        return list(env.action_labels[: env.n_components])
    return [f"c{i}" for i in range(env.n_components)]


def _component_costs(env) -> np.ndarray:
    if hasattr(env, "component_costs"):
        return np.asarray(env.component_costs, dtype=np.float32)
    return np.ones(env.n_components, dtype=np.float32)


def _base_indices(env) -> list[int]:
    return [int(i) for i in getattr(env, "base_indices", [])] if bool(getattr(env, "detail_only", False)) else []


def _detail_candidates(env) -> list[int]:
    base = set(_base_indices(env))
    return [i for i in range(env.n_components) if i not in base]


def _with_base(env, selected: Sequence[int]) -> list[int]:
    merged = list(_base_indices(env)) + [int(i) for i in selected if int(i) not in set(_base_indices(env))]
    return sorted(set(merged))


def _effective_selected(env, selected: Sequence[int]) -> list[int]:
    base = set(_base_indices(env))
    return [int(i) for i in selected if int(i) not in base]


def _cost_from_selected(env, selected: Sequence[int]) -> float:
    if hasattr(env, "_compute_cost"):
        mask = np.zeros(env.n_components, dtype=np.float32)
        if len(selected) > 0:
            mask[np.asarray(selected, dtype=int)] = 1.0
        return float(env._compute_cost(mask))
    costs = _component_costs(env)
    max_cost = float(getattr(env, "max_cost", np.sum(costs)))
    if len(selected) == 0 or max_cost <= 0:
        return 0.0
    return float(np.sum(costs[list(selected)]) / max_cost)


def _evaluate_selected(env, image: np.ndarray, image_id: int, selected: Sequence[int], method: str, k_budget: int, repeat: int | None = None) -> dict:
    selected = _with_base(env, selected)
    selected_detail = _effective_selected(env, selected)
    analysis = env.representation.analyze(image)
    reconstruction = env.representation.reconstruct(
        image,
        analysis.coefficients,
        selected,
        calibrated=bool(getattr(env, "calibrated_reconstruction", True)),
    )
    k_total = len(selected)
    k_effective = len(selected_detail)
    max_effective = max(1, int(getattr(env, "max_effective_components", env.n_components)))
    cost = _cost_from_selected(env, selected)
    labels = _component_labels(env)
    mse_value = mse(image, reconstruction)
    ssim_value = ssim(image, reconstruction)
    edge_value = edge_correlation(image, reconstruction)
    return {
        "method": method,
        "image_id": int(image_id),
        "repeat": "" if repeat is None else int(repeat),
        "k_budget": int(k_budget),
        "k": int(k_effective),
        "k_total": int(k_total),
        "k_effective": int(k_effective),
        "cost": cost,
        "k_norm": float(k_effective / max_effective),
        "k_norm_effective": float(k_effective / max_effective),
        "mse": mse_value,
        "ssim": ssim_value,
        "edge_corr": edge_value,
        "obj_mse": float(1.0 - min(1.0, mse_value)),
        "obj_ssim": ssim_value,
        "obj_cost": float(1.0 - cost),
        "obj_k": float(1.0 - k_effective / max_effective),
        "selected_indices": " ".join(map(str, selected)),
        "selected_labels": " ".join(labels[i] for i in selected),
        "selected_detail_indices": " ".join(map(str, selected_detail)),
        "selected_detail_labels": " ".join(labels[i] for i in selected_detail),
    }


def _energy_order(env, image: np.ndarray) -> np.ndarray:
    analysis = env.representation.analyze(image)
    return np.argsort(analysis.energies)[::-1]


def evaluate_random_baseline(env, ks: Sequence[int], n_repeats: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    candidates = _detail_candidates(env)
    ks = [int(k) for k in ks if 1 <= int(k) <= len(candidates)]
    rows = []
    for image_id, image in enumerate(env.images):
        for k in ks:
            for repeat in range(n_repeats):
                selected = rng.choice(candidates, size=k, replace=False).astype(int).tolist()
                rows.append(_evaluate_selected(env, image, image_id, selected, "random", k, repeat=repeat))
    return pd.DataFrame(rows)


def evaluate_energy_baseline(env, ks: Sequence[int]) -> pd.DataFrame:
    candidates = set(_detail_candidates(env))
    ks = [int(k) for k in ks if 1 <= int(k) <= len(candidates)]
    rows = []
    for image_id, image in enumerate(env.images):
        order = [int(i) for i in _energy_order(env, image) if int(i) in candidates]
        for k in ks:
            selected = order[:k]
            rows.append(_evaluate_selected(env, image, image_id, selected, "top-k energy", k))
    return pd.DataFrame(rows)


def evaluate_topk_baseline(env, budgets: Sequence[int]) -> pd.DataFrame:
    return evaluate_energy_baseline(env, budgets)


def greedy_selection(env, image: np.ndarray, max_k: int, alpha: float = 0.5, beta: float = 0.5, lambda_cost: float = 0.1) -> list[int]:
    selected: list[int] = []
    current_row = _evaluate_selected(env, image, 0, selected, "greedy_internal", 0)
    current_mse = current_row["mse"]
    current_ssim = current_row["ssim"]
    current_cost = current_row["cost"]

    for _ in range(max_k):
        remaining = [i for i in _detail_candidates(env) if i not in selected]
        if not remaining:
            break
        best_component = None
        best_score = -np.inf
        best_metrics = None
        for component in remaining:
            candidate = selected + [component]
            row = _evaluate_selected(env, image, 0, candidate, "greedy_internal", len(candidate))
            score = (
                alpha * (row["ssim"] - current_ssim)
                + beta * (current_mse - row["mse"])
                - lambda_cost * (row["cost"] - current_cost)
            )
            if score > best_score:
                best_score = score
                best_component = component
                best_metrics = row
        selected.append(int(best_component))
        current_mse = float(best_metrics["mse"])
        current_ssim = float(best_metrics["ssim"])
        current_cost = float(best_metrics["cost"])
    return selected


def evaluate_greedy_baseline(env, ks: Sequence[int], alpha: float = 0.5, beta: float = 0.5, lambda_cost: float = 0.1) -> pd.DataFrame:
    max_available = len(_detail_candidates(env))
    ks = [int(k) for k in ks if 1 <= int(k) <= max_available]
    max_k = max(ks) if ks else max_available
    rows = []
    for image_id, image in enumerate(env.images):
        full_order = greedy_selection(env, image, max_k=max_k, alpha=alpha, beta=beta, lambda_cost=lambda_cost)
        for k in ks:
            rows.append(_evaluate_selected(env, image, image_id, full_order[:k], "greedy", k))
    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "k_budget"] if "k_budget" in df.columns else ["method"]
    summary = df.groupby(group_cols).agg(
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        ssim_mean=("ssim", "mean"),
        ssim_std=("ssim", "std"),
        edge_corr_mean=("edge_corr", "mean"),
        edge_corr_std=("edge_corr", "std"),
        k_mean=("k", "mean"),
        k_effective_mean=("k_effective", "mean"),
        cost_mean=("cost", "mean"),
        obj_mse_mean=("obj_mse", "mean"),
        obj_ssim_mean=("obj_ssim", "mean"),
        obj_cost_mean=("obj_cost", "mean"),
        obj_k_mean=("obj_k", "mean"),
        n=("mse", "count"),
    ).reset_index()
    return summary.sort_values(group_cols).reset_index(drop=True)


def run_all_baselines(env, config: BaselineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    dfs = [
        evaluate_random_baseline(env, config.ks, n_repeats=config.random_repeats, seed=config.seed),
        evaluate_energy_baseline(env, config.ks),
        evaluate_greedy_baseline(env, config.ks, alpha=config.greedy_alpha, beta=config.greedy_beta, lambda_cost=config.greedy_lambda),
    ]
    by_image = pd.concat(dfs, ignore_index=True)
    summary = summarize_results(by_image)
    return by_image, summary
