from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .hermite_filters import component_labels
from .hermite_representation import HermiteAnalysis, HermiteRepresentation
from .metrics import mse, ssim

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - fallback para pruebas sin gymnasium instalado.
    class _Env:
        metadata: Dict[str, Any] = {}

    class _Discrete:
        def __init__(self, n: int) -> None:
            self.n = int(n)

        def sample(self) -> int:
            return int(np.random.randint(self.n))

    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Gym:
        Env = _Env

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    gym = _Gym()
    spaces = _Spaces()


@dataclass
class HermiteEnvState:
    image_index: int
    image: np.ndarray
    analysis: HermiteAnalysis
    mask: np.ndarray
    reconstruction: np.ndarray
    current_mse: float
    current_ssim: float
    cost: float
    k_norm: float
    steps: int


class HermiteSelectionEnv(gym.Env):
    """MOMDP para selección secuencial de componentes Hermite-Gauss."""

    metadata = {"render_modes": []}
    reward_dim = 4

    def __init__(
        self,
        images: np.ndarray,
        representation: HermiteRepresentation,
        max_steps: Optional[int] = None,
        component_costs: Optional[Sequence[float]] = None,
        calibrated_reconstruction: bool = True,
        repeated_action_penalty: float = 0.05,
        terminate_on_repeated_action: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        images = np.asarray(images, dtype=np.float32)
        if images.ndim != 3:
            raise ValueError("images debe tener forma [n_images, height, width].")
        if len(images) == 0:
            raise ValueError("El ambiente requiere al menos una imagen.")

        self.images = images
        self.representation = representation
        self.n_components = representation.filter_bank.n_components
        self.stop_action = self.n_components
        self.n_actions = self.n_components + 1
        self.max_steps = int(max_steps) if max_steps is not None else self.n_components
        self.max_steps = max(1, min(self.max_steps, self.n_components))
        self.calibrated_reconstruction = bool(calibrated_reconstruction)
        self.repeated_action_penalty = float(repeated_action_penalty)
        self.terminate_on_repeated_action = bool(terminate_on_repeated_action)

        if component_costs is None:
            component_costs = np.ones(self.n_components, dtype=np.float32)
        component_costs = np.asarray(component_costs, dtype=np.float32)
        if component_costs.shape != (self.n_components,):
            raise ValueError("component_costs debe tener longitud n_components.")
        if np.any(component_costs <= 0):
            raise ValueError("Todos los costos deben ser positivos.")
        self.component_costs = component_costs
        self.max_cost = float(np.sum(component_costs))

        self.observation_dim = 2 * self.n_components + 4
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.n_actions)

        self.rng = np.random.default_rng(seed)
        self.state: Optional[HermiteEnvState] = None
        self.components = self.representation.filter_bank.components
        self.action_labels = component_labels(self.components) + ["STOP"]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        image_index = options.get("image_index")
        if image_index is None:
            image_index = int(self.rng.integers(0, len(self.images)))
        image_index = int(image_index) % len(self.images)

        image = self.images[image_index]
        analysis = self.representation.analyze(image)
        mask = np.zeros(self.n_components, dtype=np.float32)
        reconstruction = np.zeros_like(image, dtype=np.float32)

        self.state = HermiteEnvState(
            image_index=image_index,
            image=image,
            analysis=analysis,
            mask=mask,
            reconstruction=reconstruction,
            current_mse=mse(image, reconstruction),
            current_ssim=ssim(image, reconstruction),
            cost=0.0,
            k_norm=0.0,
            steps=0,
        )
        return self._get_obs(), self._get_info(event="reset")

    def step(self, action: int):
        if self.state is None:
            raise RuntimeError("Debes llamar reset() antes de step().")
        action = int(action)
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"Acción inválida {action}. Debe estar en [0, {self.n_actions - 1}].")

        if action == self.stop_action:
            reward = np.zeros(self.reward_dim, dtype=np.float32)
            return self._get_obs(), reward, True, False, self._get_info(action, event="stop")

        repeated = bool(self.state.mask[action] > 0.5)
        if repeated:
            reward = np.asarray([0.0, 0.0, -self.repeated_action_penalty, -self.repeated_action_penalty], dtype=np.float32)
            self.state.steps += 1
            terminated = self.terminate_on_repeated_action or self.state.steps >= self.max_steps
            return self._get_obs(), reward, terminated, False, self._get_info(action, event="repeated_action")

        old_mse = self.state.current_mse
        old_ssim = self.state.current_ssim
        old_cost = self.state.cost
        old_k_norm = self.state.k_norm

        self.state.mask[action] = 1.0
        selected = np.where(self.state.mask.astype(bool))[0].tolist()
        reconstruction = self.representation.reconstruct(
            self.state.image,
            self.state.analysis.coefficients,
            selected,
            calibrated=self.calibrated_reconstruction,
        )
        new_mse = mse(self.state.image, reconstruction)
        new_ssim = ssim(self.state.image, reconstruction)
        new_cost = self._compute_cost(self.state.mask)
        new_k_norm = float(np.sum(self.state.mask) / self.n_components)

        self.state.reconstruction = reconstruction
        self.state.current_mse = new_mse
        self.state.current_ssim = new_ssim
        self.state.cost = new_cost
        self.state.k_norm = new_k_norm
        self.state.steps += 1

        reward = np.asarray([
            old_mse - new_mse,
            new_ssim - old_ssim,
            -(new_cost - old_cost),
            -(new_k_norm - old_k_norm),
        ], dtype=np.float32)

        terminated = bool(self.state.steps >= self.max_steps or np.all(self.state.mask > 0.5))
        return self._get_obs(), reward, terminated, False, self._get_info(action, event="select_component")

    def _compute_cost(self, mask: np.ndarray) -> float:
        used_cost = float(np.sum(self.component_costs * mask))
        return used_cost / self.max_cost if self.max_cost > 0 else 0.0

    def _get_obs(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Estado no inicializado.")
        return self.representation.momdp_state_features(
            normalized_energies=self.state.analysis.normalized_energies,
            mask=self.state.mask,
            current_mse=np.clip(self.state.current_mse, 0.0, 1.0),
            current_ssim=np.clip(self.state.current_ssim, 0.0, 1.0),
            cost=np.clip(self.state.cost, 0.0, 1.0),
            k_norm=np.clip(self.state.k_norm, 0.0, 1.0),
        )

    def _get_info(self, action: Optional[int] = None, event: str = "") -> Dict[str, Any]:
        if self.state is None:
            return {}
        selected = np.where(self.state.mask.astype(bool))[0].tolist()
        return {
            "event": event,
            "image_index": self.state.image_index,
            "action": None if action is None else int(action),
            "action_label": None if action is None else self.action_labels[int(action)],
            "mse": float(self.state.current_mse),
            "ssim": float(self.state.current_ssim),
            "cost": float(self.state.cost),
            "k": int(np.sum(self.state.mask)),
            "k_norm": float(self.state.k_norm),
            "selected_indices": selected,
            "selected_labels": [self.action_labels[i] for i in selected],
            "mask": self.state.mask.copy(),
            "reconstruction": self.state.reconstruction.copy(),
            "objective_vector": np.asarray([
                -self.state.current_mse,
                self.state.current_ssim,
                -self.state.cost,
                -self.state.k_norm,
            ], dtype=np.float32),
        }

    def get_valid_actions(self, include_stop: bool = True) -> list[int]:
        if self.state is None:
            return list(range(self.n_actions)) if include_stop else list(range(self.n_components))
        unused = np.where(self.state.mask < 0.5)[0].astype(int).tolist()
        if include_stop:
            unused.append(self.stop_action)
        return unused

    def sample_valid_action(self, include_stop: bool = True) -> int:
        valid_actions = self.get_valid_actions(include_stop=include_stop)
        return int(self.rng.choice(valid_actions))
