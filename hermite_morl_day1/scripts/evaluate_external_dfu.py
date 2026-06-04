from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.baselines import BaselineConfig, run_all_baselines
from src.data_utils import IMAGE_EXTENSIONS, load_images_from_folder
from src.days9_14_env_adapter import make_selection_env_from_images
from src.env_utils import load_config
from src.evaluate import combine_baselines_and_agent, evaluate_agent_policy
from src.plots import save_image_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--raw_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/Days9-14_best_envelope_dqn.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    ext_cfg = config.get("external_eval", {})
    raw_dir = Path(args.raw_dir or ext_cfg.get("raw_dir", "data/external_dfu"))
    if not raw_dir.exists() or not any(p.suffix.lower() in IMAGE_EXTENSIONS for p in raw_dir.rglob("*")):
        print("No external DFU images found; skipping external evaluation.")
        return

    root = Path(config.get("project_root", "."))
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    images = load_images_from_folder(
        raw_dir,
        image_size=int(ext_cfg.get("image_size", config.get("dataset", {}).get("image_size", 64))),
        max_images=int(ext_cfg.get("max_images", 20)),
    )
    env = make_selection_env_from_images(images, config, split="test")
    ks = config.get("evaluation", {}).get("ks", [1, 3, 5, min(10, env.n_components)])
    baseline_df, _ = run_all_baselines(env, BaselineConfig(ks=ks, topk_budgets=ks, random_repeats=5, seed=int(config.get("seed", 0))))
    agent_df = evaluate_agent_policy(env, args.checkpoint, config, device="cpu", max_images=None)
    combined, summary = combine_baselines_and_agent(baseline_df, agent_df)
    combined.to_csv(tables_dir / "external_dfu_evaluation.csv", index=False)
    summary.to_csv(tables_dir / "external_dfu_summary.csv", index=False)

    image = env.images[0]
    analysis = env.representation.analyze(image)
    base = list(getattr(env, "base_components", [])) if getattr(env, "detail_only_h00_free", False) else []
    order = [i for i in analysis.energies.argsort()[::-1].astype(int).tolist() if i not in set(base)]
    compact = env.representation.reconstruct(image, analysis.coefficients, sorted(set(base + order[:3])), calibrated=True)
    high = env.representation.reconstruct(image, analysis.coefficients, sorted(set(base + order[:10])), calibrated=True)
    save_image_grid([image, compact, high], ["Original", "Compact", "High fidelity"], figures_dir / "external_dfu_visual_comparison.png", cols=3)
    print(f"Evaluacion DFU externa guardada en {tables_dir}")


if __name__ == "__main__":
    main()
