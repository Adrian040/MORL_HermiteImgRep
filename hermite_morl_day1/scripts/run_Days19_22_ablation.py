from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.days19_22_plots import save_ablation_plot


def n_components_for_order(max_order: int) -> int:
    return int((max_order + 1) * (max_order + 2) // 2)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def run(cmd: list[str], enabled: bool = True) -> None:
    if not enabled:
        return
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def make_order_config(base: dict[str, Any], order: int, workspace: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out["project_root"] = str(workspace)
    out.setdefault("hermite", {})
    out["hermite"]["max_order"] = int(order)
    out.setdefault("env", {})
    out["env"]["max_steps"] = n_components_for_order(order)
    out["env"]["random_episode_steps"] = n_components_for_order(order) + 1
    out.setdefault("dataset", {})
    if "dataset" in cfg.get("ablation", {}):
        out["dataset"].update(cfg["ablation"]["dataset"])
    out.setdefault("training", {})
    out["training"].update(cfg.get("ablation", {}).get("training", {}))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/Days19-22_ablation.yaml")
    parser.add_argument("--base-config", type=str, default="configs/Days9-14_train.yaml")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        fallback = Path("configs/default.yaml")
        if not fallback.exists():
            raise FileNotFoundError("No encontré configs/Days9-14_train.yaml ni configs/default.yaml.")
        base_config_path = fallback
    base = load_yaml(base_config_path)

    orders = cfg.get("ablation", {}).get("orders", [2, 3])
    workspace_root = Path(cfg.get("ablation", {}).get("workspace_root", "results/Days19-22_ablation"))
    workspace_root.mkdir(parents=True, exist_ok=True)
    run_training = bool(cfg.get("ablation", {}).get("train_agent", True)) and not args.skip_training
    skip_agent_eval = bool(cfg.get("ablation", {}).get("skip_agent_eval", False)) or args.skip_agent

    rows = []
    for order in orders:
        order = int(order)
        workspace = workspace_root / f"N{order}"
        if workspace.exists() and cfg.get("ablation", {}).get("clean_workspaces", False):
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        order_config = make_order_config(base, order, workspace, cfg)
        cfg_path = workspace_root / f"Days19-22_ablation_N{order}.yaml"
        save_yaml(order_config, cfg_path)

        if run_training:
            run([sys.executable, "scripts/train_Days9_14_envelope.py", "--config", str(cfg_path)])

        baseline_cmd = [sys.executable, "scripts/run_Days15_18_baselines.py", "--config", str(cfg_path)]
        if skip_agent_eval or not run_training:
            baseline_cmd.append("--skip-agent")
        run(baseline_cmd)

        input_csv = workspace / "results" / "tables" / "Days15-18_all_methods_by_image.csv"
        if not input_csv.exists():
            raise FileNotFoundError(f"No se generó {input_csv}")
        prefix = f"Days19-22_ablation_N{order}"
        run([
            sys.executable,
            "scripts/run_Days19_22_analysis.py",
            "--input-csv", str(input_csv),
            "--output-root", str(workspace / "results"),
            "--prefix", prefix,
            "--hv-samples", str(cfg.get("ablation", {}).get("hv_samples", 50000)),
        ])

        method_summary = pd.read_csv(workspace / "results" / "tables" / f"{prefix}_method_summary.csv")
        method_summary.insert(0, "N", order)
        rows.append(method_summary)

    ablation_summary = pd.concat(rows, ignore_index=True)
    out_tables = workspace_root / "tables"
    out_figures = workspace_root / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)
    ablation_summary.to_csv(out_tables / "Days19-22_ablation_summary.csv", index=False)
    ablation_summary.round(4).to_csv(out_tables / "Days19-22_ablation_summary_report.csv", index=False)
    try:
        ablation_summary.round(4).to_latex(out_tables / "Days19-22_ablation_summary_report.tex", index=False, escape=False)
    except Exception:
        pass
    save_ablation_plot(ablation_summary, out_figures / "Days19-22_ablation_N2_vs_N3.png")
    with open(workspace_root / "Days19-22_ablation_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"orders": orders, "workspace_root": str(workspace_root), "summary": str(out_tables / "Days19-22_ablation_summary.csv")}, f, indent=2, ensure_ascii=False)
    print("Ablación Days19-22 completada.")
    print(ablation_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
