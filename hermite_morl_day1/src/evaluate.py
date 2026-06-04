from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .baselines import _component_labels, _cost_from_selected, summarize_results
from .metrics import reconstruction_metrics


def _checkpoint_exists(path: str | Path | None) -> bool:
    return bool(path) and Path(path).exists()


def evaluate_agent_policy(env, checkpoint_path: str | Path | None, config: dict, device: str = "cpu", max_images: int | None = None) -> pd.DataFrame:
    """Evalúa Envelope-DQN si existe checkpoint; si no, devuelve un DataFrame vacío."""
    if not _checkpoint_exists(checkpoint_path):
        return pd.DataFrame()

    import torch

    from .days9_14_env_adapter import env_state_dim, env_valid_action_mask
    from .morl_envelope import VectorQNetwork, default_preferences, preference_names, select_action

    device_t = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device_t, weights_only=False)
    preferences = np.asarray(checkpoint.get("preferences", default_preferences()), dtype=np.float32)
    hidden_dim = int(checkpoint.get("hidden_dim", config.get("training", {}).get("hidden_dim", 128)))
    state_dim = int(checkpoint.get("state_dim", env_state_dim(env)))

    net = VectorQNetwork(state_dim, env.reward_dim, env.n_actions, hidden_dim=hidden_dim).to(device_t)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    rows = []
    rng = np.random.default_rng(int(config.get("seed", 0)) + 1518)
    n_images = len(env.images) if max_images is None else min(int(max_images), len(env.images))
    names = preference_names()

    for pref_id, pref in enumerate(preferences):
        for image_id in range(n_images):
            state, info = env.reset(options={"image_index": image_id})
            done = False
            truncated = False
            actions = []
            while not (done or truncated):
                action = select_action(net, state, pref, epsilon=0.0, valid_action_mask=env_valid_action_mask(env), device=device_t, rng=rng)
                state, reward, done, truncated, info = env.step(action)
                actions.append(action)
            selected = info["selected_indices"]
            paid_k = int(info.get("paid_k", info.get("k", 0)))
            total_k = int(info.get("total_k", paid_k))
            cost = float(info["cost"])
            metrics = reconstruction_metrics(info.get("image", env.images[image_id]), info["reconstruction"]) if "reconstruction" in info else {
                "mse": float(info["mse"]),
                "ssim": float(info["ssim"]),
                "psnr": float("nan"),
                "gradient_mse": float("nan"),
                "edge_corr": float("nan"),
            }
            rows.append({
                "method": "Envelope-DQN",
                "preference_id": int(pref_id),
                "preference_name": names[pref_id] if pref_id < len(names) else f"w{pref_id}",
                "preference": " ".join(f"{v:.3f}" for v in pref),
                "image_id": int(image_id),
                "k_budget": paid_k,
                "k": paid_k,
                "paid_k": paid_k,
                "total_k": total_k,
                "cost": cost,
                "k_norm": float(info["k_norm"]),
                "mse": metrics["mse"],
                "ssim": metrics["ssim"],
                "psnr": metrics["psnr"],
                "gradient_mse": metrics["gradient_mse"],
                "edge_corr": metrics["edge_corr"],
                "obj_mse": float(1.0 - min(1.0, metrics["mse"])),
                "obj_ssim": metrics["ssim"],
                "obj_cost": float(1.0 - cost),
                "obj_k": float(1.0 - float(info["k_norm"])),
                "selected_indices": " ".join(map(str, selected)),
                "paid_selected_indices": " ".join(map(str, info.get("paid_selected_indices", selected))),
                "selected_labels": " ".join(info["selected_labels"]),
                "paid_selected_labels": " ".join(info.get("paid_selected_labels", info["selected_labels"])),
                "actions": " ".join(map(str, actions)),
                "cost_mode": info.get("cost_mode", ""),
                "free_kernel_size": info.get("free_kernel_size", ""),
            })
    return pd.DataFrame(rows)


def combine_baselines_and_agent(baseline_df: pd.DataFrame, agent_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if agent_df is not None and not agent_df.empty:
        combined = pd.concat([baseline_df, agent_df], ignore_index=True)
    else:
        combined = baseline_df.copy()
    return combined, summarize_results(combined)
