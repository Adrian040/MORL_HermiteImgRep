from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    seeds = args.seeds or config.get("experiment", {}).get("seeds", [int(config.get("seed", 0))])
    cmd = [sys.executable, "scripts/train_Days9_14_envelope.py", "--config", args.config, "--seeds", *map(str, seeds)]
    subprocess.run(cmd, check=True)
    print("Entrenamiento multi-seed completado.")


if __name__ == "__main__":
    main()
