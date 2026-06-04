from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from .data_utils import prepare_dataset
from .costs import build_component_costs, component_kernel_sizes
from .env_hermite_momdp import HermiteSelectionEnv
from .hermite_filters import build_hermite_filter_bank
from .hermite_representation import HermiteRepresentation, MultiConfigHermiteRepresentation


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_representation_from_config(config: dict) -> HermiteRepresentation:
    hermite_cfg = config["hermite"]
    multi_cfg = config.get("hermite_multi_config", {})
    reconstruction_cfg = config.get("reconstruction", {})
    if bool(multi_cfg.get("enabled", False)):
        return MultiConfigHermiteRepresentation(
            sigmas=multi_cfg.get("sigmas", hermite_cfg.get("sigmas", [hermite_cfg.get("sigma", 1.5)])),
            kernel_sizes=multi_cfg.get("kernel_sizes", hermite_cfg.get("kernel_sizes", [hermite_cfg.get("kernel_size", 13)])),
            max_order=int(multi_cfg.get("max_order", hermite_cfg.get("max_order", 4))),
            reconstruction_mode=str(reconstruction_cfg.get("mode", "least_squares")),
            ridge=float(reconstruction_cfg.get("ridge", 1e-6)),
            clip_output=bool(reconstruction_cfg.get("clip_output", True)),
            normalize_output=bool(reconstruction_cfg.get("normalize_output", True)),
        )
    bank = build_hermite_filter_bank(
        max_order=int(hermite_cfg.get("max_order", 3)),
        sigma=float(hermite_cfg.get("sigma", 1.5)),
        kernel_size=int(hermite_cfg.get("kernel_size", 13)),
    )
    return HermiteRepresentation(
        bank,
        reconstruction_mode=str(reconstruction_cfg.get("mode", "least_squares")),
        ridge=float(reconstruction_cfg.get("ridge", 1e-6)),
        clip_output=bool(reconstruction_cfg.get("clip_output", True)),
        normalize_output=bool(reconstruction_cfg.get("normalize_output", True)),
    )


def build_env_kwargs_from_config(config: dict, representation: HermiteRepresentation, split: str = "train") -> dict:
    env_cfg = config.get("env", {})
    reward_cfg = config.get("reward", {})
    cost_cfg = config.get("cost", {})
    reconstruction_cfg = config.get("reconstruction", {})
    n_components = representation.filter_bank.n_components
    if hasattr(representation, "kernel_sizes"):
        default_kernel_size = max(int(k) for k in getattr(representation, "kernel_sizes"))
    else:
        default_kernel_size = int(config.get("hermite", {}).get("kernel_size", representation.filter_bank.kernel_size))
    kernels = component_kernel_sizes(
        n_components,
        default_kernel_size=default_kernel_size,
        values=config.get("hermite", {}).get("component_kernel_sizes"),
    )
    cost_mode = str(cost_cfg.get("mode", "extra_kernel_flops"))
    free_kernel_size = int(cost_cfg.get("free_kernel_size", 13))
    component_costs = build_component_costs(
        n_components,
        kernels,
        mode=cost_mode,
        free_kernel_size=free_kernel_size,
        eps=float(cost_cfg.get("eps", 1e-8)),
    )
    seed_offset = 0 if split == "train" else 1000 if split == "val" else 2000
    return {
        "max_steps": int(env_cfg.get("max_steps", n_components)),
        "component_costs": component_costs,
        "calibrated_reconstruction": str(reconstruction_cfg.get("mode", "least_squares")) == "least_squares"
        if "calibrated_reconstruction" not in env_cfg
        else bool(env_cfg.get("calibrated_reconstruction")),
        "repeated_action_penalty": abs(float(env_cfg.get("repeated_action_penalty", -0.05))),
        "terminate_on_repeated_action": bool(env_cfg.get("terminate_on_repeated_action", False)),
        "detail_only_h00_free": bool(env_cfg.get("detail_only_h00_free", True)),
        "base_components": env_cfg.get("base_components", [0]),
        "count_base_components_in_k": bool(env_cfg.get("count_base_components_in_k", False)),
        "count_base_components_in_cost": bool(env_cfg.get("count_base_components_in_cost", False)),
        "reward_normalize": bool(reward_cfg.get("normalize", True)),
        "reward_mode": str(reward_cfg.get("mode", "relative")),
        "reward_eps": float(reward_cfg.get("eps", 1e-8)),
        "cost_mode": cost_mode,
        "free_kernel_size": free_kernel_size,
        "component_kernel_sizes": kernels,
        "seed": int(config.get("seed", 0)) + seed_offset,
    }


def make_env_from_config(config: dict, split: str = "train") -> tuple[HermiteSelectionEnv, dict[str, np.ndarray]]:
    splits = prepare_dataset(config)
    representation = build_representation_from_config(config)
    env = HermiteSelectionEnv(
        images=splits[split],
        representation=representation,
        **build_env_kwargs_from_config(config, representation, split=split),
    )
    return env, splits


def run_random_episode(env: HermiteSelectionEnv, max_interactions: int | None = None, avoid_repeated: bool = True) -> pd.DataFrame:
    import pandas as pd

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
