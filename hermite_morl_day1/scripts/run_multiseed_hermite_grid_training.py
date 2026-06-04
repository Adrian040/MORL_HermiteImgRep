from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from scripts.train_Days9_14_envelope import train


def _apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    out = copy.deepcopy(config)
    out.setdefault("training", {})
    out.setdefault("dataset", {})
    out.setdefault("hermite_multi_config", {})
    if args.episodes is not None:
        out["training"]["episodes"] = int(args.episodes)
    if args.eval_every is not None:
        out["training"]["eval_every"] = int(args.eval_every)
    if args.device is not None:
        out["training"]["device"] = args.device
    if args.train_images is not None:
        out["dataset"]["train_images"] = int(args.train_images)
    if args.val_images is not None:
        out["dataset"]["val_images"] = int(args.val_images)
    if args.test_images is not None:
        out["dataset"]["test_images"] = int(args.test_images)
    if args.max_order is not None:
        out["hermite_multi_config"]["max_order"] = int(args.max_order)
        out.setdefault("hermite", {})
        out["hermite"]["max_order"] = int(args.max_order)
    if args.sigmas:
        out["hermite_multi_config"]["sigmas"] = [float(v) for v in args.sigmas]
    if args.kernel_sizes:
        out["hermite_multi_config"]["kernel_sizes"] = [int(v) for v in args.kernel_sizes]
    out["hermite_multi_config"]["enabled"] = True
    return out


def _mixed_slug(config: dict) -> str:
    multi = config.get("hermite_multi_config", {})
    sigmas = "_".join(str(float(v)).replace(".", "p") for v in multi.get("sigmas", []))
    kernels = "_".join(str(int(v)) for v in multi.get("kernel_sizes", []))
    max_order = int(multi.get("max_order", config.get("hermite", {}).get("max_order", 4)))
    return f"mixed_sigmas_{sigmas}__kernels_{kernels}__order_{max_order}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-images", type=int, default=None)
    parser.add_argument("--val-images", type=int, default=None)
    parser.add_argument("--test-images", type=int, default=None)
    parser.add_argument("--sigmas", type=float, nargs="*", default=None)
    parser.add_argument("--kernel-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--max-order", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    base_config = _apply_overrides(base_config, args)

    seeds = args.seeds or base_config.get("experiment", {}).get("seeds", [int(base_config.get("seed", 0))])
    base_output_root = Path(args.output_root or base_config.get("experiment", {}).get("output_root", "results"))
    slug = _mixed_slug(base_config)

    manifest_rows = []
    for run_idx, seed in enumerate(seeds, start=1):
        run_config = copy.deepcopy(base_config)
        run_config["seed"] = int(seed)
        run_config.setdefault("experiment", {})
        run_config["experiment"]["output_root"] = str(base_output_root / "mixed_hermite_multiseed" / f"seed_{seed}")
        run_config["experiment"]["active_hermite_grid_slug"] = slug
        run_config["experiment"]["active_hermite_grid_config"] = run_config.get("hermite_multi_config", {})
        print(f"[Mixed Hermite] run {run_idx}/{len(seeds)}: seed={seed}, config={slug}", flush=True)
        manifest = train(run_config)
        manifest_rows.append({
            "seed": int(seed),
            "mixed_config": slug,
            "sigmas": run_config["hermite_multi_config"].get("sigmas", []),
            "kernel_sizes": run_config["hermite_multi_config"].get("kernel_sizes", []),
            "max_order": int(run_config["hermite_multi_config"].get("max_order", 4)),
            "output_root": run_config["experiment"]["output_root"],
            "n_components": manifest.get("n_components"),
            "n_actions": manifest.get("n_actions"),
            "best_score": manifest.get("best_score"),
        })

    manifest_root = base_output_root / "mixed_hermite_multiseed"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / "mixed_hermite_training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "one_agent_per_seed_with_multi_config_bank",
                "n_seeds": len(seeds),
                "n_runs": len(manifest_rows),
                "seeds": [int(s) for s in seeds],
                "mixed_config": base_config.get("hermite_multi_config", {}),
                "runs": manifest_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Entrenamiento multi-seed con banco Hermite mixto completado. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
