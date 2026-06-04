from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

try:
    if int(np.__version__.split(".")[0]) >= 2:
        raise ImportError("Skipping scipy with NumPy 2.x binary-incompatible environment.")
    from scipy.signal import convolve2d as _scipy_convolve2d
except Exception:
    _scipy_convolve2d = None

from .hermite_filters import HermiteFilterBank, build_hermite_filter_bank
from .metrics import clip01, to_float01


def convolve2d(image: np.ndarray, kernel: np.ndarray, mode: str = "same", boundary: str = "symm") -> np.ndarray:
    if _scipy_convolve2d is not None:
        return _scipy_convolve2d(image, kernel, mode=mode, boundary=boundary)
    if mode != "same":
        raise ValueError("Fallback convolve2d solo soporta mode='same'.")
    image = np.asarray(image, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    pad_mode = "symmetric" if boundary == "symm" else "constant"
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode=pad_mode)
    flipped = np.flip(kernel, axis=(0, 1))
    out = np.zeros_like(image, dtype=np.float32)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            out[y, x] = float(np.sum(padded[y:y + kernel.shape[0], x:x + kernel.shape[1]] * flipped))
    return out


@dataclass
class HermiteAnalysis:
    coefficients: np.ndarray
    energies: np.ndarray
    normalized_energies: np.ndarray


class HermiteRepresentation:
    def __init__(
        self,
        filter_bank: HermiteFilterBank,
        boundary: str = "symm",
        reconstruction_mode: Literal["direct_sum", "least_squares"] = "least_squares",
        ridge: float = 1e-6,
        clip_output: bool = True,
        normalize_output: bool = True,
    ) -> None:
        self.filter_bank = filter_bank
        self.boundary = boundary
        self.reconstruction_mode = reconstruction_mode
        self.ridge = float(ridge)
        self.clip_output = bool(clip_output)
        self.normalize_output = bool(normalize_output)

    def analyze(self, image: np.ndarray) -> HermiteAnalysis:
        image = np.asarray(image, dtype=np.float32)
        coeffs = []
        for kernel in self.filter_bank.filters:
            coeff = convolve2d(image, kernel, mode="same", boundary=self.boundary)
            coeffs.append(coeff.astype(np.float32))
        coeffs_arr = np.stack(coeffs, axis=0)
        energies = np.sum(coeffs_arr**2, axis=(1, 2)).astype(np.float32)
        total = float(np.sum(energies))
        normalized = energies / total if total > 1e-12 else np.zeros_like(energies)
        return HermiteAnalysis(coeffs_arr, energies, normalized.astype(np.float32))

    def component_reconstructions(self, coefficients: np.ndarray) -> np.ndarray:
        reconstructions = []
        for coeff, kernel in zip(coefficients, self.filter_bank.filters):
            synth_kernel = np.flip(kernel, axis=(0, 1))
            rec = convolve2d(coeff, synth_kernel, mode="same", boundary=self.boundary)
            reconstructions.append(rec.astype(np.float32))
        return np.stack(reconstructions, axis=0)

    def reconstruct(
        self,
        image: np.ndarray,
        coefficients: np.ndarray,
        selected: Sequence[int],
        calibrated: bool = True,
        mode: Literal["direct_sum", "least_squares"] | None = None,
        return_weights: bool = False,
    ) -> np.ndarray:
        if len(selected) == 0:
            reconstruction = np.zeros_like(image, dtype=np.float32)
            if return_weights:
                return reconstruction, np.zeros(0, dtype=np.float32)
            return reconstruction

        component_recs = self.component_reconstructions(coefficients)[list(selected)]
        reconstruction_mode = mode or ("least_squares" if calibrated else "direct_sum")
        weights = np.ones(len(selected), dtype=np.float32)
        if reconstruction_mode == "least_squares":
            basis = component_recs.reshape(len(selected), -1).T
            target = np.asarray(image, dtype=np.float32).reshape(-1)
            if self.ridge > 0:
                lhs = basis.T @ basis + self.ridge * np.eye(len(selected), dtype=np.float32)
                rhs = basis.T @ target
                weights = np.linalg.solve(lhs, rhs)
            else:
                weights, *_ = np.linalg.lstsq(basis, target, rcond=None)
            reconstruction = basis @ weights
            reconstruction = reconstruction.reshape(image.shape)
        elif reconstruction_mode == "direct_sum":
            reconstruction = np.sum(component_recs, axis=0)
        else:
            raise ValueError(f"reconstruction_mode desconocido: {reconstruction_mode}")

        if self.clip_output:
            reconstruction = clip01(reconstruction)
        elif self.normalize_output:
            reconstruction = to_float01(reconstruction)
        reconstruction = np.asarray(reconstruction, dtype=np.float32)
        if return_weights:
            return reconstruction, np.asarray(weights, dtype=np.float32)
        return reconstruction

    def reconstruct_from_mask(
        self,
        image: np.ndarray,
        coefficients: np.ndarray,
        mask: np.ndarray,
        calibrated: bool = True,
    ) -> np.ndarray:
        selected = np.where(np.asarray(mask).astype(bool))[0].tolist()
        return self.reconstruct(image, coefficients, selected, calibrated=calibrated)

    @staticmethod
    def order_by_energy(energies: np.ndarray, descending: bool = True) -> np.ndarray:
        order = np.argsort(np.asarray(energies))
        return order[::-1] if descending else order

    @staticmethod
    def momdp_state_features(
        normalized_energies: np.ndarray,
        mask: np.ndarray,
        current_mse: float,
        current_ssim: float,
        cost: float,
        k_norm: float,
    ) -> np.ndarray:
        return np.concatenate([
            np.asarray(normalized_energies, dtype=np.float32),
            np.asarray(mask, dtype=np.float32),
            np.asarray([current_mse, current_ssim, cost, k_norm], dtype=np.float32),
        ])


def _pad_kernel(kernel: np.ndarray, target_size: int) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float32)
    pad_total_y = target_size - kernel.shape[0]
    pad_total_x = target_size - kernel.shape[1]
    if pad_total_y < 0 or pad_total_x < 0:
        raise ValueError("target_size debe ser mayor o igual al tamano del kernel.")
    return np.pad(
        kernel,
        (
            (pad_total_y // 2, pad_total_y - pad_total_y // 2),
            (pad_total_x // 2, pad_total_x - pad_total_x // 2),
        ),
        mode="constant",
    )


def _aggregate_filter_bank(filter_banks: Sequence[HermiteFilterBank]) -> HermiteFilterBank:
    components = filter_banks[0].components
    max_kernel_size = max(bank.kernel_size for bank in filter_banks)
    filters = []
    for component_idx in range(len(components)):
        variants = [_pad_kernel(bank.filters[component_idx], max_kernel_size) for bank in filter_banks]
        filters.append(np.mean(np.stack(variants, axis=0), axis=0))
    return HermiteFilterBank(
        filters=np.stack(filters, axis=0).astype(np.float32),
        components=list(components),
        sigma=float("nan"),
        kernel_size=int(max_kernel_size),
        max_order=int(filter_banks[0].max_order),
    )


class MultiConfigHermiteRepresentation(HermiteRepresentation):
    """Representacion multi-escala/soporte con una accion por componente Hermite unico."""

    def __init__(
        self,
        sigmas: Sequence[float],
        kernel_sizes: Sequence[int],
        max_order: int = 4,
        boundary: str = "symm",
        reconstruction_mode: Literal["direct_sum", "least_squares"] = "least_squares",
        ridge: float = 1e-6,
        clip_output: bool = True,
        normalize_output: bool = True,
    ) -> None:
        self.variant_filter_banks = [
            build_hermite_filter_bank(max_order=max_order, sigma=float(sigma), kernel_size=int(kernel_size))
            for sigma in sigmas
            for kernel_size in kernel_sizes
        ]
        if not self.variant_filter_banks:
            raise ValueError("MultiConfigHermiteRepresentation requiere al menos una combinacion sigma/kernel_size.")
        first_components = self.variant_filter_banks[0].components
        for bank in self.variant_filter_banks[1:]:
            if bank.components != first_components:
                raise ValueError("Todas las variantes deben compartir los mismos componentes Hermite.")
        self.sigmas = [float(s) for s in sigmas]
        self.kernel_sizes = [int(k) for k in kernel_sizes]
        super().__init__(
            filter_bank=_aggregate_filter_bank(self.variant_filter_banks),
            boundary=boundary,
            reconstruction_mode=reconstruction_mode,
            ridge=ridge,
            clip_output=clip_output,
            normalize_output=normalize_output,
        )

    def analyze(self, image: np.ndarray) -> HermiteAnalysis:
        image = np.asarray(image, dtype=np.float32)
        variant_coeffs = []
        variant_energies = []
        for bank in self.variant_filter_banks:
            coeffs = []
            for kernel in bank.filters:
                coeff = convolve2d(image, kernel, mode="same", boundary=self.boundary)
                coeffs.append(coeff.astype(np.float32))
            coeffs_arr = np.stack(coeffs, axis=0)
            variant_coeffs.append(coeffs_arr)
            variant_energies.append(np.sum(coeffs_arr**2, axis=(1, 2)).astype(np.float32))
        coeffs_4d = np.stack(variant_coeffs, axis=0)
        energies = np.mean(np.stack(variant_energies, axis=0), axis=0).astype(np.float32)
        total = float(np.sum(energies))
        normalized = energies / total if total > 1e-12 else np.zeros_like(energies)
        return HermiteAnalysis(coeffs_4d, energies, normalized.astype(np.float32))

    def component_reconstructions(self, coefficients: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(coefficients, dtype=np.float32)
        if coefficients.ndim != 4:
            raise ValueError("MultiConfigHermiteRepresentation espera coefficients con forma [variantes, componentes, H, W].")
        variant_recs = []
        for variant_idx, bank in enumerate(self.variant_filter_banks):
            reconstructions = []
            for coeff, kernel in zip(coefficients[variant_idx], bank.filters):
                synth_kernel = np.flip(kernel, axis=(0, 1))
                rec = convolve2d(coeff, synth_kernel, mode="same", boundary=self.boundary)
                reconstructions.append(rec.astype(np.float32))
            variant_recs.append(np.stack(reconstructions, axis=0))
        return np.mean(np.stack(variant_recs, axis=0), axis=0).astype(np.float32)
