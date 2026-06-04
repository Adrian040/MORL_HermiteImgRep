from __future__ import annotations

from typing import Dict

import numpy as np

from .env_hermite_momdp import HermiteSelectionEnv
from .env_utils import build_env_kwargs_from_config, build_representation_from_config


def make_selection_env_from_images(images: np.ndarray, config: Dict, split: str = "train") -> HermiteSelectionEnv:
    """Construye el ambiente existente sin reemplazar su API.

    Se conserva `repeated_action_penalty` como magnitud positiva, consistente
    con la implementación de Days5-8.
    """
    representation = build_representation_from_config(config)
    return HermiteSelectionEnv(
        images=images,
        representation=representation,
        **build_env_kwargs_from_config(config, representation, split=split),
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
