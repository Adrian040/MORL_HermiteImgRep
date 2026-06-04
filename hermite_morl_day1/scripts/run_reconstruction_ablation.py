from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.baselines import BaselineConfig, run_all_baselines
from src.data_utils import prepare_dataset
from src.days9_14_env_adapter import make_selection_env_from_images
from src.env_utils import load_config
from src.evaluate import evaluate_agent_policy


def _mode_config(config: dict, calibrated: bool, project_root: Path | None = None) -> dict:
    cfg = copy.deepcopy(config)
    cfg.setdefault("env", {})
    cfg["env"]["calibrated_reconstruction"] = bool(calibrated)
    if project_root is not None:
        cfg["project_root"] = str(project_root)
        raw_dir = Path(config.get("dataset", {}).get("raw_dir", ""))
        if raw_dir and not raw_dir.is_absolute():
            cfg.setdefault("dataset", {})["raw_dir"] = str((PROJECT_ROOT / raw_dir).resolve())
    return cfg


def _train_for_mode(config: dict, mode_name: str, calibrated: bool) -> Path:
    from scripts.train_Days9_14_envelope import train

    mode_root = PROJECT_ROOT / "results" / "reconstruction_ablation_training" / mode_name
    cfg = _mode_config(config, calibrated, project_root=mode_root)
    train(cfg)
    return mode_root / "results" / "checkpoints" / "Days9-14_best_envelope_dqn.pt"


def _evaluate_mode(config: dict, mode_name: str, split: str, checkpoint: Path | None, max_images: int | None = None) -> pd.DataFrame:
    splits = prepare_dataset(config)
    env = make_selection_env_from_images(splits[split], config, split=split)
    baseline_cfg = config.get("baselines", {})
    ks = baseline_cfg.get("ks", config.get("evaluation", {}).get("ks", [1, 3, 5, 10]))
    cfg = BaselineConfig(
        ks=[int(k) for k in ks],
        topk_budgets=[],
        random_repeats=int(baseline_cfg.get("random_repeats", 20)),
        greedy_alpha=float(baseline_cfg.get("greedy_alpha", 0.5)),
        greedy_beta=float(baseline_cfg.get("greedy_beta", 0.5)),
        greedy_lambda=float(baseline_cfg.get("greedy_lambda", 0.1)),
        seed=int(config.get("seed", 0)),
    )
    by_image, _ = run_all_baselines(env, cfg)
    dfs = [by_image]
    if checkpoint is not None and checkpoint.exists():
        device = str(config.get("training", {}).get("device", "cpu"))
        if device == "auto":
            device = "cpu"
        agent_df = evaluate_agent_policy(env, checkpoint, config, device=device, max_images=max_images)
        if not agent_df.empty:
            dfs.append(agent_df)
    out = pd.concat(dfs, ignore_index=True)
    out["reconstruction_mode"] = mode_name
    return out


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["reconstruction_mode", "method"]).agg(
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        ssim_mean=("ssim", "mean"),
        ssim_std=("ssim", "std"),
        edge_corr_mean=("edge_corr", "mean"),
        edge_corr_std=("edge_corr", "std"),
        cost_mean=("cost", "mean"),
        k_effective_mean=("k_effective", "mean"),
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--agent-checkpoint", type=str, default="results/checkpoints/Days9-14_best_envelope_dqn.pt")
    parser.add_argument("--train-per-mode", action="store_true")
    parser.add_argument("--eval-max-images", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config.get("project_root", "."))
    tables_dir = root / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    modes = [("direct_sum", False), ("least_squares", True)]
    dfs = []
    for mode_name, calibrated in modes:
        cfg = _mode_config(config, calibrated)
        checkpoint = _train_for_mode(config, mode_name, calibrated) if args.train_per_mode else root / args.agent_checkpoint
        dfs.append(_evaluate_mode(cfg, mode_name, args.split, checkpoint, max_images=args.eval_max_images))

    per_image = pd.concat(dfs, ignore_index=True)
    summary = _summary(per_image)
    per_image.to_csv(tables_dir / "reconstruction_ablation_per_image.csv", index=False)
    summary.to_csv(tables_dir / "reconstruction_ablation_summary.csv", index=False)
    print(f"Reconstruction ablation saved to {tables_dir / 'reconstruction_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
