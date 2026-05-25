from __future__ import annotations

from typing import Dict

import numpy as np

from .env_hermite_momdp import HermiteSelectionEnv
from .env_utils import build_representation_from_config


def make_selection_env_from_images(images: np.ndarray, config: Dict, split: str = "train") -> HermiteSelectionEnv:
    """Construye el ambiente existente sin reemplazar su API.

    Se conserva `repeated_action_penalty` como magnitud positiva, consistente
    con la implementación de Days5-8.
    """
    representation = build_representation_from_config(config)
    env_cfg = config.get("env", {})
    seed_offset = 0 if split == "train" else 1000 if split == "val" else 2000
    return HermiteSelectionEnv(
        images=images,
        representation=representation,
        max_steps=int(env_cfg.get("max_steps", representation.filter_bank.n_components)),
        calibrated_reconstruction=bool(env_cfg.get("calibrated_reconstruction", True)),
        repeated_action_penalty=abs(float(env_cfg.get("repeated_action_penalty", 0.05))),
        terminate_on_repeated_action=bool(env_cfg.get("terminate_on_repeated_action", False)),
        seed=int(config.get("seed", 0)) + seed_offset,
    )


def env_state_dim(env: HermiteSelectionEnv) -> int:
    if hasattr(env, "state_dim"):
        return int(env.state_dim)
    if hasattr(env, "observation_dim"):
        return int(env.observation_dim)
    return int(env.observation_space.shape[0])


def env_valid_action_mask(env: HermiteSelectionEnv) -> np.ndarray:
    if hasattr(env, "valid_action_mask"):
        return np.asarray(env.valid_action_mask(), dtype=bool)
    if hasattr(env, "get_valid_actions"):
        valid = np.zeros(env.n_actions, dtype=bool)
        valid[np.asarray(env.get_valid_actions(include_stop=True), dtype=int)] = True
        return valid
    if getattr(env, "state", None) is not None and hasattr(env.state, "mask"):
        valid = np.ones(env.n_actions, dtype=bool)
        valid[: env.n_components] = np.asarray(env.state.mask) < 0.5
        valid[env.stop_action] = True
        return valid
    return np.ones(env.n_actions, dtype=bool)
