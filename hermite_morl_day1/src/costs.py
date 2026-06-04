from __future__ import annotations

from typing import Sequence

import numpy as np


def extra_kernel_flops_cost(
    kernel_size: int | float,
    free_kernel_size: int | float = 13,
    max_kernel_size: int | float | None = None,
    eps: float = 1e-8,
) -> float:
    if max_kernel_size is None:
        max_kernel_size = kernel_size
    kernel_size = float(kernel_size)
    free_kernel_size = float(free_kernel_size)
    max_kernel_size = float(max_kernel_size)
    if max_kernel_size <= free_kernel_size:
        return 0.0
    extra = max(0.0, kernel_size**2 - free_kernel_size**2)
    denom = max(max_kernel_size**2 - free_kernel_size**2, eps)
    return float(extra / denom)


def component_kernel_sizes(n_components: int, default_kernel_size: int, values: Sequence[int] | None = None) -> np.ndarray:
    if values is None:
        return np.full(int(n_components), int(default_kernel_size), dtype=np.int32)
    arr = np.asarray(values, dtype=np.int32)
    if arr.shape != (int(n_components),):
        raise ValueError("component_kernel_sizes debe tener longitud n_components.")
    return arr


def build_component_costs(
    n_components: int,
    kernel_sizes: Sequence[int],
    mode: str = "extra_kernel_flops",
    free_kernel_size: int = 13,
    eps: float = 1e-8,
) -> np.ndarray:
    if mode == "count":
        return np.ones(int(n_components), dtype=np.float32)
    if mode != "extra_kernel_flops":
        raise ValueError(f"cost.mode desconocido: {mode}")
    kernel_sizes_arr = np.asarray(kernel_sizes, dtype=np.int32)
    max_kernel_size = int(np.max(kernel_sizes_arr)) if len(kernel_sizes_arr) else int(free_kernel_size)
    return np.asarray(
        [
            extra_kernel_flops_cost(k, free_kernel_size=free_kernel_size, max_kernel_size=max_kernel_size, eps=eps)
            for k in kernel_sizes_arr
        ],
        dtype=np.float32,
    )
