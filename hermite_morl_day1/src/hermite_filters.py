from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


Component = Tuple[int, int]


@dataclass(frozen=True)
class HermiteFilterBank:
    filters: np.ndarray
    components: List[Component]
    sigma: float
    kernel_size: int
    max_order: int

    @property
    def n_components(self) -> int:
        return len(self.components)


def hermite_1d(n: int, x: np.ndarray) -> np.ndarray:
    """Polinomio de Hermite físico H_n(x) por recurrencia."""
    if n == 0:
        return np.ones_like(x, dtype=np.float64)
    if n == 1:
        return 2.0 * x

    h_nm2 = np.ones_like(x, dtype=np.float64)
    h_nm1 = 2.0 * x
    for k in range(1, n):
        h_n = 2.0 * x * h_nm1 - 2.0 * k * h_nm2
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1


def hermite_components(max_order: int) -> List[Component]:
    return [(m, n) for total in range(max_order + 1) for m in range(total + 1) for n in [total - m]]


def _normalize_filter(kernel: np.ndarray, order_sum: int, eps: float = 1e-12) -> np.ndarray:
    kernel = kernel.astype(np.float64)
    if order_sum > 0:
        kernel = kernel - kernel.mean()
    norm = np.sqrt(np.sum(kernel**2))
    if norm < eps:
        return kernel
    return kernel / norm


def build_hermite_filter_bank(max_order: int = 3, sigma: float = 1.5, kernel_size: int = 11) -> HermiteFilterBank:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size debe ser impar.")
    if max_order < 0:
        raise ValueError("max_order debe ser no negativo.")
    if sigma <= 0:
        raise ValueError("sigma debe ser positivo.")

    radius = kernel_size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    x = xx / sigma
    y = yy / sigma
    gaussian = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

    components = hermite_components(max_order)
    filters = []
    for m, n in components:
        kernel = hermite_1d(m, x) * hermite_1d(n, y) * gaussian
        filters.append(_normalize_filter(kernel, m + n))

    return HermiteFilterBank(
        filters=np.stack(filters, axis=0).astype(np.float32),
        components=components,
        sigma=float(sigma),
        kernel_size=int(kernel_size),
        max_order=int(max_order),
    )


def component_labels(components: Sequence[Component]) -> List[str]:
    return [f"H{m}{n}" for m, n in components]
