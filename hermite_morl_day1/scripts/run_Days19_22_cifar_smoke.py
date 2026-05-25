from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.days19_22_cifar_utils import load_cifar10_or_fallback, save_images_to_raw_dir


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/Days19-22_cifar_smoke.yaml")
    parser.add_argument("--n-images", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--raw-dir", type=str, default="data/raw/Days19-22_cifar_smoke")
    parser.add_argument("--skip-agent", action="store_true", default=True)
    args = parser.parse_args()

    images, source = load_cifar10_or_fallback(n_images=args.n_images, size=args.image_size, seed=0)
    save_images_to_raw_dir(images, args.raw_dir, prefix="cifar_smoke")

    notes = {
        "source": source,
        "n_images": int(len(images)),
        "image_size": int(args.image_size),
        "raw_dir": args.raw_dir,
        "note": "Si CIFAR-10 no está disponible localmente, se usa un fallback tipo CIFAR con skimage sin descargar datos.",
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/Days19-22_cifar_smoke_dataset.json", "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)

    run([sys.executable, "scripts/run_Days15_18_baselines.py", "--config", args.config, "--skip-agent"])
    run([
        sys.executable,
        "scripts/run_Days19_22_analysis.py",
        "--input-csv", "results/tables/Days15-18_all_methods_by_image.csv",
        "--prefix", "Days19-22_cifar_smoke",
        "--hv-samples", "20000",
    ])
    print("Smoke CIFAR/compatible completado.")


if __name__ == "__main__":
    main()
