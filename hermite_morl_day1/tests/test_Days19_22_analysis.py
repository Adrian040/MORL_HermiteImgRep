from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.days19_22_analysis import (
    add_pareto_flags,
    build_hypervolume_summary,
    build_method_summary,
    ensure_objective_columns,
    hypervolume_2d_exact,
    pareto_mask,
)


def main() -> None:
    df = pd.DataFrame({
        "method": ["random", "random", "energy", "greedy", "Envelope-DQN"],
        "image_id": [0, 1, 0, 0, 0],
        "k": [1, 2, 2, 3, 2],
        "cost": [0.1, 0.2, 0.2, 0.3, 0.2],
        "k_norm": [0.1, 0.2, 0.2, 0.3, 0.2],
        "mse": [0.20, 0.16, 0.12, 0.10, 0.11],
        "ssim": [0.50, 0.55, 0.65, 0.70, 0.68],
        "selected_labels": ["H00", "H00 H10", "H00 H10", "H00 H10 H01", "H00 H01"],
        "preference_name": ["", "", "", "", "general_balance"],
    })
    df, mse_ref = ensure_objective_columns(df)
    assert mse_ref > 0
    assert all(c in df.columns for c in ["obj_mse", "obj_ssim", "obj_cost", "obj_k"])
    points = df[["obj_mse", "obj_ssim", "obj_cost", "obj_k"]].to_numpy()
    mask = pareto_mask(points)
    assert mask.dtype == bool and len(mask) == len(df)
    df = add_pareto_flags(df)
    assert "is_pareto_global" in df.columns
    hv2d = hypervolume_2d_exact(df[["obj_k", "obj_ssim"]].to_numpy())
    assert np.isfinite(hv2d) and hv2d >= 0
    hv = build_hypervolume_summary(df, hv_samples=1000)
    summary = build_method_summary(df, hv)
    assert not hv.empty and not summary.empty
    print("Pruebas Days19-22 de análisis completadas correctamente.")


if __name__ == "__main__":
    main()
