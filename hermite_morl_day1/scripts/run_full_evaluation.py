from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--skip-agent", action="store_true")
    args = parser.parse_args()
    cmd = [sys.executable, "scripts/run_Days15_18_baselines.py", "--config", args.config]
    if args.skip_agent:
        cmd.append("--skip-agent")
    subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, "scripts/run_Days19_22_analysis.py"], check=True)
    print("Evaluacion principal completada.")


if __name__ == "__main__":
    main()
