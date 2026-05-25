from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.baselines import BaselineConfig, run_all_baselines
from src.days15_18_plots import save_quality_cost_curve, save_visual_baseline_comparison
from src.days9_14_env_adapter import make_selection_env_from_images
from src.data_utils import prepare_dataset, save_processed_dataset
from src.env_utils import load_config
from src.evaluate import combine_baselines_and_agent, evaluate_agent_policy


def _parse_int_list(values) -> list[int]:
    return [int(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/Days15-18_baselines.yaml")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--agent-checkpoint", type=str, default=None)
    parser.add_argument("--skip-agent", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    baseline_cfg = config.get("baselines", {})
    split = args.split or baseline_cfg.get("split", "test")

    root = Path(config.get("project_root", "."))
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    recon_dir = root / "results" / "reconstructions"
    processed_dir = root / "data" / "processed"
    for d in [tables_dir, figures_dir, recon_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    splits = prepare_dataset(config)
    save_processed_dataset(splits, processed_dir / "dataset_Days15-18.npz")
    env = make_selection_env_from_images(splits[split], config, split=split)

    max_k = env.n_components
    ks = baseline_cfg.get("ks", list(range(1, max_k + 1)))
    topk_budgets = baseline_cfg.get("topk_budgets", [1, 2, 3, 5, 7, max_k])
    cfg = BaselineConfig(
        ks=_parse_int_list(ks),
        topk_budgets=_parse_int_list(topk_budgets),
        random_repeats=int(baseline_cfg.get("random_repeats", 20)),
        greedy_alpha=float(baseline_cfg.get("greedy_alpha", 0.5)),
        greedy_beta=float(baseline_cfg.get("greedy_beta", 0.5)),
        greedy_lambda=float(baseline_cfg.get("greedy_lambda", 0.1)),
        seed=int(config.get("seed", 0)),
    )

    baseline_df, baseline_summary = run_all_baselines(env, cfg)
    baseline_df.to_csv(tables_dir / "Days15-18_baselines_by_image.csv", index=False)
    baseline_summary.to_csv(tables_dir / "Days15-18_baselines_summary.csv", index=False)

    for method in ["random", "energy", "greedy"]:
        baseline_df[baseline_df["method"] == method].to_csv(tables_dir / f"Days15-18_{method}_baseline.csv", index=False)
    baseline_df[baseline_df["method"].str.startswith("top_")].to_csv(tables_dir / "Days15-18_topk_baseline.csv", index=False)

    agent_df = pd.DataFrame()
    checkpoint_path = args.agent_checkpoint or baseline_cfg.get("agent_checkpoint", "results/checkpoints/Days9-14_best_envelope_dqn.pt")
    if not args.skip_agent and bool(baseline_cfg.get("eval_agent", True)):
        agent_df = evaluate_agent_policy(
            env=env,
            checkpoint_path=checkpoint_path,
            config=config,
            device=str(baseline_cfg.get("agent_device", "cpu")),
            max_images=baseline_cfg.get("agent_max_images", None),
        )
        if not agent_df.empty:
            agent_df.to_csv(tables_dir / "Days15-18_agent_policy_by_image.csv", index=False)

    combined_df, combined_summary = combine_baselines_and_agent(baseline_df, agent_df)
    combined_df.to_csv(tables_dir / "Days15-18_all_methods_by_image.csv", index=False)
    combined_summary.to_csv(tables_dir / "Days15-18_all_methods_summary.csv", index=False)

    save_quality_cost_curve(baseline_summary, figures_dir / "Days15-18_ssim_vs_k_baselines.png", y_col="ssim_mean")
    save_quality_cost_curve(baseline_summary, figures_dir / "Days15-18_mse_vs_k_baselines.png", y_col="mse_mean")
    save_visual_baseline_comparison(
        env,
        image_id=int(baseline_cfg.get("visual_image_id", 0)),
        output_path=figures_dir / "Days15-18_visual_baseline_comparison.png",
        k=int(baseline_cfg.get("visual_k", 5)),
        seed=int(config.get("seed", 0)),
    )

    manifest = {
        "split": split,
        "n_images": int(len(env.images)),
        "n_components": int(env.n_components),
        "n_actions": int(env.n_actions),
        "random_repeats": int(cfg.random_repeats),
        "agent_evaluated": bool(not agent_df.empty),
        "outputs": {
            "baselines_by_image": str(tables_dir / "Days15-18_baselines_by_image.csv"),
            "baselines_summary": str(tables_dir / "Days15-18_baselines_summary.csv"),
            "all_methods_by_image": str(tables_dir / "Days15-18_all_methods_by_image.csv"),
            "all_methods_summary": str(tables_dir / "Days15-18_all_methods_summary.csv"),
            "ssim_curve": str(figures_dir / "Days15-18_ssim_vs_k_baselines.png"),
            "mse_curve": str(figures_dir / "Days15-18_mse_vs_k_baselines.png"),
            "visual_comparison": str(figures_dir / "Days15-18_visual_baseline_comparison.png"),
        },
    }
    with open(root / "results" / "Days15-18_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\nResumen Days15-18:")
    print(combined_summary.round(4).to_string(index=False))
    if agent_df.empty:
        print("\nNota: no se evaluó Envelope-DQN porque no se encontró checkpoint o se usó --skip-agent.")
    print(f"\nResultados guardados en: {root / 'results'}")


if __name__ == "__main__":
    main()
