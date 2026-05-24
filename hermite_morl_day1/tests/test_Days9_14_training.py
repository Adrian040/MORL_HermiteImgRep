from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data_utils import build_skimage_dataset
from src.days9_14_env_adapter import env_state_dim, env_valid_action_mask, make_selection_env_from_images
from src.morl_envelope import ReplayBuffer, VectorQNetwork, default_preferences, envelope_dqn_loss, select_action


def main() -> None:
    images = build_skimage_dataset(image_size=32, n_images=6, seed=7)
    config = {
        "seed": 7,
        "hermite": {"max_order": 2, "sigma": 1.5, "kernel_size": 7},
        "env": {"max_steps": 6, "repeated_action_penalty": 0.05, "calibrated_reconstruction": True},
    }
    env = make_selection_env_from_images(images, config, split="train")
    state, info = env.reset(options={"image_index": 0})
    preferences = default_preferences()
    pref = preferences[0]
    device = torch.device("cpu")
    state_dim = env_state_dim(env)
    net = VectorQNetwork(state_dim, env.reward_dim, env.n_actions, hidden_dim=32).to(device)
    target = VectorQNetwork(state_dim, env.reward_dim, env.n_actions, hidden_dim=32).to(device)
    target.load_state_dict(net.state_dict())
    rng = np.random.default_rng(0)
    action = select_action(net, state, pref, epsilon=0.0, valid_action_mask=env_valid_action_mask(env), device=device, rng=rng)
    next_state, reward, terminated, truncated, info = env.step(action)

    assert state.shape == (state_dim,)
    assert reward.shape == (env.reward_dim,)
    assert 0 <= action < env.n_actions
    assert np.isfinite(reward).all()

    buffer = ReplayBuffer(capacity=10, state_dim=state_dim, reward_dim=env.reward_dim, seed=0)
    buffer.add(state, pref, action, reward, next_state, terminated or truncated)
    for _ in range(3):
        buffer.add(next_state, pref, env.stop_action, np.zeros(env.reward_dim, dtype=np.float32), next_state, True)
    batch = buffer.sample(batch_size=4, device=device)
    loss = envelope_dqn_loss(net, target, batch, gamma=0.95, preference_bank=torch.as_tensor(preferences, dtype=torch.float32), n_components=env.n_components)
    assert torch.isfinite(loss)
    print("Pruebas Days9-14 del entrenamiento completadas correctamente.")


if __name__ == "__main__":
    main()
