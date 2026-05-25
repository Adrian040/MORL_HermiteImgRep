from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.days19_22_visual_examples import VisualConfig, run_visual_examples


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_image_ids(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/Days19-22_visual_examples.yaml")
    parser.add_argument("--output-root", type=str, default="results")
    parser.add_argument("--prefix", type=str, default="Days19-22")
    parser.add_argument("--results-csv", type=str, default="results/tables/Days15-18_all_methods_by_image.csv")
    parser.add_argument("--image-ids", type=str, default="0,1,2")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    visual_cfg = cfg.get("visual_examples", {})
    hermite_cfg = cfg.get("hermite", {})
    dataset_cfg = cfg.get("dataset", {})
    split = args.split or visual_cfg.get("split", "test")
    visual_config = VisualConfig(
        max_order=int(hermite_cfg.get("max_order", visual_cfg.get("max_order", 3))),
        sigma=float(hermite_cfg.get("sigma", visual_cfg.get("sigma", 1.5))),
        kernel_size=int(hermite_cfg.get("kernel_size", visual_cfg.get("kernel_size", 11))),
        image_size=int(dataset_cfg.get("image_size", visual_cfg.get("image_size", 64))),
        seed=int(cfg.get("seed", visual_cfg.get("seed", 0))),
        split=str(split),
        fig2_image_id=int(visual_cfg.get("fig2_image_id", 0)),
        fig6_images=int(visual_cfg.get("fig6_images", 3)),
        fig6_k=int(visual_cfg.get("fig6_k", 5)),
        compact_k=int(visual_cfg.get("compact_k", 3)),
        balanced_k=int(visual_cfg.get("balanced_k", 5)),
        high_fidelity_k=visual_cfg.get("high_fidelity_k", None),
        require_agent=bool(visual_cfg.get("require_agent", False)),
    )
    image_ids = _parse_image_ids(args.image_ids)
    manifest = run_visual_examples(
        project_config=cfg,
        output_root=args.output_root,
        prefix=args.prefix,
        config=visual_config,
        results_csv=args.results_csv,
        image_ids=image_ids,
    )
    manifest_path = Path(args.output_root) / f"{args.prefix}_visual_examples_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("Figuras visuales Days19-22 generadas.")
    print(f"Fuente de imágenes: {manifest['dataset_source']} | split={manifest['split']}")
    print(f"Figura 2: {manifest['fig2']}")
    print(f"Figura 6: {manifest['fig6']}")
    print(f"Metadata: {manifest['metadata']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
