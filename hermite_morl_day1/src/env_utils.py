from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from .data_utils import prepare_dataset
from .env_hermite_momdp import HermiteSelectionEnv
from .hermite_filters import build_hermite_filter_bank
from .hermite_representation import HermiteRepresentation


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_representation_from_config(config: dict) -> HermiteRepresentation:
    hermite_cfg = config["hermite"]
    bank = build_hermite_filter_bank(
        max_order=int(hermite_cfg.get("max_order", 3)),
        sigma=float(hermite_cfg.get("sigma", 1.5)),
        kernel_size=int(hermite_cfg.get("kernel_size", 11)),
    )
    return HermiteRepresentation(bank)


def make_env_from_config(config: dict, split: str = "train") -> tuple[HermiteSelectionEnv, dict[str, np.ndarray]]:
    splits = prepare_dataset(config)
    representation = build_representation_from_config(config)
    env_cfg = config.get("env", {})
    env = HermiteSelectionEnv(
        images=splits[split],
        representation=representation,
        max_steps=int(env_cfg.get("max_steps", representation.filter_bank.n_components)),
        calibrated_reconstruction=bool(env_cfg.get("calibrated_reconstruction", True)),
        repeated_action_penalty=float(env_cfg.get("repeated_action_penalty", 0.05)),
        terminate_on_repeated_action=bool(env_cfg.get("terminate_on_repeated_action", False)),
        seed=int(config.get("seed", 0)),
    )
    return env, splits


def run_random_episode(env: HermiteSelectionEnv, max_interactions: int | None = None, avoid_repeated: bool = True) -> pd.DataFrame:
    obs, info = env.reset()
    rows = []
    max_interactions = max_interactions or env.max_steps + 1
    done = False

    for step_id in range(max_interactions):
        if avoid_repeated:
            action = env.sample_valid_action(include_stop=True)
        else:
            action = int(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append({
            "step": step_id,
            "action": action,
            "action_label": info["action_label"],
            "event": info["event"],
            "reward_mse": float(reward[0]),
            "reward_ssim": float(reward[1]),
            "reward_cost": float(reward[2]),
            "reward_k": float(reward[3]),
            "mse": info["mse"],
            "ssim": info["ssim"],
            "k": info["k"],
            "cost": info["cost"],
            "selected_indices": " ".join(map(str, info["selected_indices"])),
            "selected_labels": " ".join(info["selected_labels"]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
        done = bool(terminated or truncated)
        if done:
            break

    if not done:
        obs, reward, terminated, truncated, info = env.step(env.stop_action)
        rows.append({
            "step": len(rows),
            "action": env.stop_action,
            "action_label": "STOP",
            "event": info["event"],
            "reward_mse": float(reward[0]),
            "reward_ssim": float(reward[1]),
            "reward_cost": float(reward[2]),
            "reward_k": float(reward[3]),
            "mse": info["mse"],
            "ssim": info["ssim"],
            "k": info["k"],
            "cost": info["cost"],
            "selected_indices": " ".join(map(str, info["selected_indices"])),
            "selected_labels": " ".join(info["selected_labels"]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })

    return pd.DataFrame(rows)
