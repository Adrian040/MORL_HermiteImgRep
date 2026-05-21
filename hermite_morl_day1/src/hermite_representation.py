from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.signal import convolve2d

from .hermite_filters import HermiteFilterBank
from .metrics import clip01, to_float01


@dataclass
class HermiteAnalysis:
    coefficients: np.ndarray
    energies: np.ndarray
    normalized_energies: np.ndarray


class HermiteRepresentation:
    def __init__(self, filter_bank: HermiteFilterBank, boundary: str = "symm") -> None:
        self.filter_bank = filter_bank
        self.boundary = boundary

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
    ) -> np.ndarray:
        if len(selected) == 0:
            return np.zeros_like(image, dtype=np.float32)

        component_recs = self.component_reconstructions(coefficients)[list(selected)]
        if calibrated:
            basis = component_recs.reshape(len(selected), -1).T
            target = np.asarray(image, dtype=np.float32).reshape(-1)
            weights, *_ = np.linalg.lstsq(basis, target, rcond=None)
            reconstruction = basis @ weights
            reconstruction = reconstruction.reshape(image.shape)
            return clip01(reconstruction)

        reconstruction = np.sum(component_recs, axis=0)
        return to_float01(reconstruction)

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
