from __future__ import annotations

import numpy as np

_NUMPY_MAJOR = int(np.__version__.split(".")[0])

try:
    if _NUMPY_MAJOR >= 2:
        raise ImportError("Skipping skimage with NumPy 2.x binary-incompatible environment.")
    from skimage.filters import sobel as _skimage_sobel
    _ = _skimage_sobel(np.zeros((3, 3), dtype=np.float32))
except Exception:
    _skimage_sobel = None

try:
    if _NUMPY_MAJOR >= 2:
        raise ImportError("Skipping skimage with NumPy 2.x binary-incompatible environment.")
    from skimage.metrics import peak_signal_noise_ratio as _skimage_psnr
    from skimage.metrics import structural_similarity as _skimage_ssim
    _ = _skimage_psnr(np.zeros((8, 8), dtype=np.float32), np.zeros((8, 8), dtype=np.float32), data_range=1.0)
    _ = _skimage_ssim(np.zeros((8, 8), dtype=np.float32), np.zeros((8, 8), dtype=np.float32), data_range=1.0)
except Exception:
    _skimage_psnr = None
    _skimage_ssim = None


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
    image = np.asarray(image, dtype=np.float32)
    reconstruction = np.asarray(reconstruction, dtype=np.float32)
    if _skimage_ssim is not None:
        return float(_skimage_ssim(image, reconstruction, data_range=1.0))
    err = mse(image, reconstruction)
    return float(max(0.0, 1.0 - err / max(float(np.var(image)), 1e-8)))


def psnr(image: np.ndarray, reconstruction: np.ndarray, data_range: float = 1.0) -> float:
    image = np.asarray(image, dtype=np.float32)
    reconstruction = np.asarray(reconstruction, dtype=np.float32)
    if _skimage_psnr is not None:
        return float(_skimage_psnr(image, reconstruction, data_range=data_range))
    err = mse(image, reconstruction)
    if err <= 0.0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(err))


def _sobel(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if _skimage_sobel is not None:
        return np.asarray(_skimage_sobel(image), dtype=np.float32)
    gy, gx = np.gradient(image)
    return np.sqrt(gx**2 + gy**2).astype(np.float32)


def gradient_mse(image: np.ndarray, reconstruction: np.ndarray) -> float:
    grad_image = _sobel(image)
    grad_reconstruction = _sobel(reconstruction)
    return mse(grad_image, grad_reconstruction)


def edge_corr(image: np.ndarray, reconstruction: np.ndarray, eps: float = 1e-8) -> float:
    grad_image = _sobel(image).reshape(-1)
    grad_reconstruction = _sobel(reconstruction).reshape(-1)
    grad_image = grad_image - float(np.mean(grad_image))
    grad_reconstruction = grad_reconstruction - float(np.mean(grad_reconstruction))
    denom = float(np.linalg.norm(grad_image) * np.linalg.norm(grad_reconstruction))
    if denom < eps:
        return 0.0
    return float(np.dot(grad_image, grad_reconstruction) / denom)


def reconstruction_metrics(image: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse(image, reconstruction),
        "ssim": ssim(image, reconstruction),
        "psnr": psnr(image, reconstruction),
        "gradient_mse": gradient_mse(image, reconstruction),
        "edge_corr": edge_corr(image, reconstruction),
    }
