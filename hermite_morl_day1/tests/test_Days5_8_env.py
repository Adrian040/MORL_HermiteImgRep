from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.env_utils import load_config, make_env_from_config


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    env, _ = make_env_from_config(config, split="train")

    obs, info = env.reset(options={"image_index": 0})
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))
    assert env.reward_dim == 4
    assert env.n_actions == env.n_components + 1
    assert env.stop_action == env.n_components
    assert info["k"] == 0

    obs, reward, terminated, truncated, info = env.step(0)
    assert reward.shape == (4,)
    assert np.all(np.isfinite(reward))
    assert info["event"] == "select_component"
    assert info["k"] == 1
    assert np.isfinite(info["mse"])
    assert np.isfinite(info["ssim"])

    obs, reward, terminated, truncated, info = env.step(0)
    assert info["event"] == "repeated_action"
    assert reward.shape == (4,)

    obs, reward, terminated, truncated, info = env.step(env.stop_action)
    assert terminated is True
    assert truncated is False
    assert info["event"] == "stop"

    obs, info = env.reset(options={"image_index": 0})
    for _ in range(env.max_steps):
        action = env.sample_valid_action(include_stop=False)
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward.shape == (4,)
        assert np.all(np.isfinite(reward))
        assert np.all(np.isfinite(obs))
        if terminated or truncated:
            break

    print("Pruebas Days5-8 del ambiente completadas correctamente.")


if __name__ == "__main__":
    main()
