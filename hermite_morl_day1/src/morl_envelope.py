from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch import nn


def default_preferences() -> np.ndarray:
    return np.asarray([
        [0.55, 0.25, 0.10, 0.10],
        [0.35, 0.35, 0.15, 0.15],
        [0.25, 0.25, 0.25, 0.25],
        [0.20, 0.20, 0.30, 0.30],
        [0.15, 0.15, 0.35, 0.35],
    ], dtype=np.float32)


def preference_names() -> List[str]:
    return ["high_fidelity", "mse_ssim_balance", "general_balance", "compact", "very_compact"]


def normalize_preferences(preferences: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    preferences = np.asarray(preferences, dtype=np.float32)
    return preferences / np.maximum(preferences.sum(axis=-1, keepdims=True), eps)


class PreferenceSampler:
    def __init__(self, preferences: np.ndarray, seed: int = 0) -> None:
        self.preferences = normalize_preferences(preferences)
        self.rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        idx = int(self.rng.integers(0, len(self.preferences)))
        return self.preferences[idx].copy()


class VectorQNetwork(nn.Module):
    """Q vectorial condicionado por preferencia. La política usa argmax_a w^T Q(s,a,w)."""

    def __init__(self, state_dim: int, reward_dim: int, n_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.reward_dim = int(reward_dim)
        self.n_actions = int(n_actions)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.reward_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_actions * self.reward_dim),
        )

    def forward(self, states: torch.Tensor, preferences: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, preferences], dim=-1)
        q = self.net(x)
        return q.view(-1, self.n_actions, self.reward_dim)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, reward_dim: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.reward_dim = int(reward_dim)
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.preferences = np.zeros((capacity, reward_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity, reward_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, state, preference, action, reward, next_state, done) -> None:
        self.states[self.pos] = np.asarray(state, dtype=np.float32)
        self.preferences[self.pos] = np.asarray(preference, dtype=np.float32)
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = np.asarray(reward, dtype=np.float32)
        self.next_states[self.pos] = np.asarray(next_state, dtype=np.float32)
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "states": torch.as_tensor(self.states[idx], dtype=torch.float32, device=device),
            "preferences": torch.as_tensor(self.preferences[idx], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.long, device=device),
            "rewards": torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            "next_states": torch.as_tensor(self.next_states[idx], dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device),
        }


def epsilon_by_step(step: int, epsilon_start: float, epsilon_end: float, decay_steps: int) -> float:
    frac = min(1.0, max(0.0, step / max(1, decay_steps)))
    return float(epsilon_start + frac * (epsilon_end - epsilon_start))


def invalid_action_mask_from_states(states: torch.Tensor, n_components: int, n_actions: int) -> torch.Tensor:
    mask_section = states[:, n_components:2 * n_components]
    invalid = torch.zeros((states.shape[0], n_actions), dtype=torch.bool, device=states.device)
    invalid[:, :n_components] = mask_section > 0.5
    return invalid


@torch.no_grad()
def select_action(
    network: VectorQNetwork,
    state: np.ndarray,
    preference: np.ndarray,
    epsilon: float,
    valid_action_mask: np.ndarray | None,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    if valid_action_mask is not None:
        valid_actions = np.flatnonzero(valid_action_mask)
    else:
        valid_actions = np.arange(network.n_actions)
    if len(valid_actions) == 0:
        return network.n_actions - 1
    if rng.random() < epsilon:
        return int(rng.choice(valid_actions))

    state_t = torch.as_tensor(state[None, :], dtype=torch.float32, device=device)
    pref_t = torch.as_tensor(preference[None, :], dtype=torch.float32, device=device)
    q_vec = network(state_t, pref_t)[0]
    scores = torch.mv(q_vec, pref_t[0])
    if valid_action_mask is not None:
        invalid = torch.as_tensor(~valid_action_mask, dtype=torch.bool, device=device)
        scores = scores.masked_fill(invalid, -1e9)
    return int(torch.argmax(scores).item())


def scalarized_score(metrics: Dict[str, float], mse_reference: float = 0.25) -> float:
    mse_norm = min(1.0, float(metrics["mse"]) / max(1e-8, mse_reference))
    return float(0.5 * metrics["ssim"] + 0.5 * (1.0 - mse_norm) - 0.25 * metrics["cost"] - 0.25 * metrics["k_norm"])


def envelope_dqn_loss(
    policy_net: VectorQNetwork,
    target_net: VectorQNetwork,
    batch: Dict[str, torch.Tensor],
    gamma: float,
    preference_bank: torch.Tensor,
    n_components: int,
    use_envelope_target: bool = True,
) -> torch.Tensor:
    states = batch["states"]
    preferences = batch["preferences"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_states = batch["next_states"]
    dones = batch["dones"]

    q_pred_all = policy_net(states, preferences)
    q_pred = q_pred_all[torch.arange(states.shape[0], device=states.device), actions]

    with torch.no_grad():
        if use_envelope_target:
            b = next_states.shape[0]
            m = preference_bank.shape[0]
            expanded_states = next_states[:, None, :].expand(b, m, next_states.shape[-1]).reshape(b * m, -1)
            expanded_preferences = preference_bank[None, :, :].expand(b, m, preference_bank.shape[-1]).reshape(b * m, -1)
            q_next = target_net(expanded_states, expanded_preferences).view(b, m, policy_net.n_actions, policy_net.reward_dim)
            invalid = invalid_action_mask_from_states(next_states, n_components, policy_net.n_actions)[:, None, :]
            scores = torch.einsum("bmar,br->bma", q_next, preferences)
            scores = scores.masked_fill(invalid, -1e9)
            flat_idx = torch.argmax(scores.view(b, -1), dim=1)
            pref_idx = flat_idx // policy_net.n_actions
            action_idx = flat_idx % policy_net.n_actions
            q_next_best = q_next[torch.arange(b, device=states.device), pref_idx, action_idx]
        else:
            q_next = target_net(next_states, preferences)
            invalid = invalid_action_mask_from_states(next_states, n_components, policy_net.n_actions)
            scores = torch.einsum("bar,br->ba", q_next, preferences)
            scores = scores.masked_fill(invalid, -1e9)
            action_idx = torch.argmax(scores, dim=1)
            q_next_best = q_next[torch.arange(states.shape[0], device=states.device), action_idx]
        target = rewards + gamma * (1.0 - dones[:, None]) * q_next_best
    return nn.functional.smooth_l1_loss(q_pred, target)


def load_checkpoint(path: str, device: torch.device) -> Dict:
    return torch.load(path, map_location=device, weights_only=False)
