from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def to_float01(image: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val - min_val < eps:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - min_val) / (max_val - min_val)).astype(np.float32)


def clip01(image: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)


def mse(image: np.ndarray, reconstruction: np.ndarray) -> float:
    image = np.asarray(image, dtype=np.float32)
    reconstruction = np.asarray(reconstruction, dtype=np.float32)
    return float(np.mean((image - reconstruction) ** 2))


def ssim(image: np.ndarray, reconstruction: np.ndarray) -> float:
    return float(structural_similarity(
        np.asarray(image, dtype=np.float32),
        np.asarray(reconstruction, dtype=np.float32),
        data_range=1.0,
    ))
