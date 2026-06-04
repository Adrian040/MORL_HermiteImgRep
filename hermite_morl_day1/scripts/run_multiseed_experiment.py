from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.train_Days9_14_envelope import train
from src.baselines import BaselineConfig, run_all_baselines
from src.data_utils import prepare_dataset
from src.days19_22_analysis import build_hypervolume_summary, ensure_objective_columns
from src.days9_14_env_adapter import make_selection_env_from_images
from src.env_utils import load_config
from src.evaluate import combine_baselines_and_agent, evaluate_agent_policy


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _seed_config(config: dict, seed: int, seed_dir: Path, episodes: int | None, eval_max_images: int | None) -> dict:
    cfg = copy.deepcopy(config)
    cfg["seed"] = int(seed)
    cfg["project_root"] = str(seed_dir)
    raw_dir = Path(config.get("dataset", {}).get("raw_dir", ""))
    if raw_dir and not raw_dir.is_absolute():
        cfg.setdefault("dataset", {})["raw_dir"] = str((PROJECT_ROOT / raw_dir).resolve())
    cfg.setdefault("training", {})
    if episodes is not None:
        cfg["training"]["episodes"] = int(episodes)
        cfg["training"]["eval_every"] = min(int(cfg["training"].get("eval_every", episodes)), int(episodes))
    if eval_max_images is not None:
        cfg["training"]["eval_max_images"] = int(eval_max_images)
    return cfg


def _seed_outputs(config: dict, seed_dir: Path, eval_max_images: int | None) -> dict:
    splits = prepare_dataset(config)
    env = make_selection_env_from_images(splits["test"], config, split="test")
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
    baseline_df, _ = run_all_baselines(env, cfg)
    checkpoint = seed_dir / "results" / "checkpoints" / "Days9-14_best_envelope_dqn.pt"
    device = str(config.get("training", {}).get("device", "cpu"))
    if device == "auto":
        device = "cpu"
    agent_df = evaluate_agent_policy(env, checkpoint, config, device=device, max_images=eval_max_images)
    combined_df, combined_summary = combine_baselines_and_agent(baseline_df, agent_df)
    combined_df, _ = ensure_objective_columns(combined_df)
    hv_summary = build_hypervolume_summary(combined_df, seed=int(config.get("seed", 0)))

    seed_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(seed_dir / "all_methods_by_image.csv", index=False)
    combined_summary.to_csv(seed_dir / "all_methods_summary.csv", index=False)
    hv_summary.to_csv(seed_dir / "hypervolume_summary.csv", index=False)
    return {
        "combined_df": combined_df,
        "combined_summary": combined_summary,
        "hv_summary": hv_summary,
        "checkpoint": str(checkpoint),
        "n_images": int(len(env.images)),
    }


def _method_seed_metrics(seed: int, combined_df: pd.DataFrame, hv_summary: pd.DataFrame) -> pd.DataFrame:
    summary = combined_df.groupby("method").agg(
        mse=("mse", "mean"),
        ssim=("ssim", "mean"),
        edge_corr=("edge_corr", "mean"),
        cost=("cost", "mean"),
        k_effective=("k_effective", "mean"),
    ).reset_index()
    hv = hv_summary[hv_summary["method"] != "GLOBAL"][["method", "hv_4d"]].rename(columns={"hv_4d": "hypervolume"})
    summary = summary.merge(hv, on="method", how="left")
    summary["seed"] = int(seed)
    return summary


def _preference_seed_metrics(seed: int, combined_df: pd.DataFrame, hv_summary: pd.DataFrame) -> pd.DataFrame:
    agent = combined_df[combined_df["method"].astype(str).eq("Envelope-DQN")].copy()
    if agent.empty:
        return pd.DataFrame()
    summary = agent.groupby(["preference_id", "preference_name"]).agg(
        mse=("mse", "mean"),
        ssim=("ssim", "mean"),
        edge_corr=("edge_corr", "mean"),
        cost=("cost", "mean"),
        k_effective=("k_effective", "mean"),
    ).reset_index()
    hv = hv_summary[hv_summary["method"].eq("Envelope-DQN")]["hv_4d"]
    summary["hypervolume"] = float(hv.iloc[0]) if not hv.empty else np.nan
    summary["seed"] = int(seed)
    return summary


def _long_aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = ["mse", "ssim", "edge_corr", "cost", "k_effective", "hypervolume"]
    rows = []
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for metric in metrics:
            rows.append({
                **base,
                "metric": metric,
                "metric_mean": float(group[metric].mean()),
                "metric_std": float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-max-images", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = args.seeds or config.get("experiment", {}).get("seeds", [int(config.get("seed", 0))])
    root = Path(config.get("project_root", "."))
    multiseed_dir = root / "results" / "multiseed"
    multiseed_dir.mkdir(parents=True, exist_ok=True)

    method_seed_frames = []
    preference_seed_frames = []
    manifest_seeds = []
    for seed in [int(s) for s in seeds]:
        _set_global_seed(seed)
        seed_dir = multiseed_dir / f"seed_{seed}"
        cfg = _seed_config(config, seed, seed_dir, args.episodes, args.eval_max_images)
        train_manifest = train(cfg)
        outputs = _seed_outputs(cfg, seed_dir, args.eval_max_images)
        method_seed_frames.append(_method_seed_metrics(seed, outputs["combined_df"], outputs["hv_summary"]))
        pref_metrics = _preference_seed_metrics(seed, outputs["combined_df"], outputs["hv_summary"])
        if not pref_metrics.empty:
            preference_seed_frames.append(pref_metrics)
        manifest_seeds.append({
            "seed": seed,
            "directory": str(seed_dir),
            "checkpoint": outputs["checkpoint"],
            "n_images": outputs["n_images"],
            "train_manifest": train_manifest,
        })

    method_seed_df = pd.concat(method_seed_frames, ignore_index=True)
    method_summary = _long_aggregate(method_seed_df, ["method"])
    method_summary.to_csv(multiseed_dir / "multiseed_method_summary.csv", index=False)

    if preference_seed_frames:
        preference_seed_df = pd.concat(preference_seed_frames, ignore_index=True)
        preference_summary = _long_aggregate(preference_seed_df, ["preference_id", "preference_name"])
    else:
        preference_summary = pd.DataFrame(columns=["preference_id", "preference_name", "metric", "metric_mean", "metric_std"])
    preference_summary.to_csv(multiseed_dir / "multiseed_agent_preference_summary.csv", index=False)

    manifest = {
        "seeds": [int(s) for s in seeds],
        "date": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "seed_runs": manifest_seeds,
        "n_preferences": int(len(config.get("training", {}).get("preferences", []))),
        "detail_only": bool(config.get("env", {}).get("detail_only", True)),
        "reward_normalization": bool(config.get("env", {}).get("reward_normalization", True)),
        "outputs": {
            "method_summary": str(multiseed_dir / "multiseed_method_summary.csv"),
            "agent_preference_summary": str(multiseed_dir / "multiseed_agent_preference_summary.csv"),
        },
    }
    with open(multiseed_dir / "multiseed_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Multi-seed experiment saved to {multiseed_dir}")


if __name__ == "__main__":
    main()
