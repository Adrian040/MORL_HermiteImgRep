from __future__ import annotations

import ast
from typing import Optional

import numpy as np
import pandas as pd

from .days19_22_analysis import OBJECTIVE_COLUMNS, hypervolume_pymoo_or_mc


def _objective_from_record(record: dict, objective_key: str) -> np.ndarray:
    value = record.get(objective_key)
    if value is not None and not (isinstance(value, float) and np.isnan(value)):
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception:
                value = value.replace(",", " ").split()
        return np.asarray(value, dtype=float)
    if all(col in record for col in OBJECTIVE_COLUMNS):
        return np.asarray([record[col] for col in OBJECTIVE_COLUMNS], dtype=float)
    raise ValueError(f"Record lacks '{objective_key}' and objective columns {OBJECTIVE_COLUMNS}.")


def equal_budget_hypervolume(
    records: list[dict],
    method_key: str = "method",
    objective_key: str = "objective_vector",
    budget: int = 50,
    repeats: int = 20,
    seed: int = 0,
    reference_point: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=[
            "method",
            "requested_budget",
            "effective_budget",
            "repeats",
            "hv_mean",
            "hv_std",
            "num_available_solutions",
        ])

    grouped: dict[str, list[np.ndarray]] = {}
    for record in records:
        method = str(record.get(method_key, "unknown"))
        grouped.setdefault(method, []).append(_objective_from_record(record, objective_key))

    available = {method: len(points) for method, points in grouped.items()}
    effective_budget = min(int(budget), min(available.values()))
    effective_budget = max(1, effective_budget)
    rng = np.random.default_rng(seed)
    rows = []

    for method, point_list in grouped.items():
        points = np.asarray(point_list, dtype=float)
        if reference_point is not None:
            points = points - np.asarray(reference_point, dtype=float)
        hvs = []
        for repeat in range(int(repeats)):
            idx = rng.choice(len(points), size=effective_budget, replace=False)
            hv, _ = hypervolume_pymoo_or_mc(points[idx], seed=seed + repeat)
            hvs.append(hv)
        rows.append({
            "method": method,
            "requested_budget": int(budget),
            "effective_budget": int(effective_budget),
            "repeats": int(repeats),
            "hv_mean": float(np.mean(hvs)),
            "hv_std": float(np.std(hvs, ddof=1)) if len(hvs) > 1 else 0.0,
            "num_available_solutions": int(len(points)),
        })
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
