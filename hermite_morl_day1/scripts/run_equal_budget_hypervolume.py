from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.baselines import BaselineConfig, run_all_baselines
from src.data_utils import prepare_dataset
from src.days19_22_analysis import ensure_objective_columns
from src.days9_14_env_adapter import make_selection_env_from_images
from src.env_utils import load_config
from src.pareto import equal_budget_hypervolume


def _records_from_existing(root: Path) -> list[dict]:
    candidates = [
        root / "results" / "tables" / "Days19-22_objective_space.csv",
        root / "results" / "tables" / "Days15-18_all_methods_by_image.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df, _ = ensure_objective_columns(df)
            return df.to_dict("records")
    return []


def _generate_records(config: dict, split: str) -> list[dict]:
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
    by_image, _ = ensure_objective_columns(by_image)
    return by_image.to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config.get("project_root", "."))
    tables_dir = root / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    records = _records_from_existing(root)
    if not records:
        records = _generate_records(config, args.split)

    summary = equal_budget_hypervolume(
        records,
        budget=int(args.budget),
        repeats=int(args.repeats),
        seed=int(config.get("seed", 0) if args.seed is None else args.seed),
    )
    output = tables_dir / "equal_budget_hypervolume.csv"
    summary.to_csv(output, index=False)
    print(f"Equal-budget hypervolume saved to {output}")


if __name__ == "__main__":
    main()
