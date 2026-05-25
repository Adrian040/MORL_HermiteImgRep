from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.baselines import BaselineConfig, evaluate_energy_baseline, evaluate_greedy_baseline, evaluate_random_baseline, evaluate_topk_baseline, run_all_baselines
from src.data_utils import build_skimage_dataset
from src.days9_14_env_adapter import make_selection_env_from_images


def main() -> None:
    images = build_skimage_dataset(image_size=32, n_images=5, seed=15)
    config = {
        "project_root": ".",
        "seed": 15,
        "hermite": {"max_order": 2, "sigma": 1.5, "kernel_size": 7},
        "env": {"max_steps": 6, "calibrated_reconstruction": True, "repeated_action_penalty": 0.05, "terminate_on_repeated_action": False},
    }
    env = make_selection_env_from_images(images, config, split="test")
    ks = [1, 2, 3]

    random_df = evaluate_random_baseline(env, ks=ks, n_repeats=2, seed=15)
    energy_df = evaluate_energy_baseline(env, ks=ks)
    topk_df = evaluate_topk_baseline(env, budgets=[1, 3])
    greedy_df = evaluate_greedy_baseline(env, ks=ks)
    all_df, summary = run_all_baselines(env, BaselineConfig(ks=ks, topk_budgets=[1, 3], random_repeats=2, seed=15))

    for df in [random_df, energy_df, topk_df, greedy_df, all_df, summary]:
        assert not df.empty
        assert np.isfinite(df.select_dtypes(include=[float, int]).to_numpy()).all()
    assert set(["mse", "ssim", "k", "cost", "selected_indices"]).issubset(all_df.columns)
    assert summary["method"].nunique() >= 4
    print("Pruebas Days15-18 de baselines completadas correctamente.")


if __name__ == "__main__":
    main()
