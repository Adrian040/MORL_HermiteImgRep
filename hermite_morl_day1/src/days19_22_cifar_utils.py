from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from skimage import color, data, transform


def _to_uint8_rgb(image: np.ndarray, size: int = 32) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image.astype(np.float32)
    image = image - image.min()
    image = image / max(float(image.max()), 1e-8)
    image = transform.resize(image, (size, size, 3), preserve_range=True, anti_aliasing=True)
    return np.clip(image * 255, 0, 255).astype(np.uint8)


def _fallback_cifar_like_images(n_images: int, size: int, seed: int) -> np.ndarray:
    base = [
        data.astronaut(), data.chelsea(), data.coffee(), data.rocket(), data.camera(),
        data.coins(), data.moon(), data.page(), data.immunohistochemistry(), data.clock(),
    ]
    rng = np.random.default_rng(seed)
    images = []
    for i in range(n_images):
        img = _to_uint8_rgb(base[i % len(base)], size=max(size * 2, size))
        h, w = img.shape[:2]
        if h > size and w > size:
            y = int(rng.integers(0, h - size + 1))
            x = int(rng.integers(0, w - size + 1))
            img = img[y:y + size, x:x + size]
        img = _to_uint8_rgb(img, size=size)
        if rng.random() < 0.5:
            img = np.fliplr(img)
        images.append(img)
    return np.stack(images, axis=0)


def load_cifar10_or_fallback(n_images: int = 12, size: int = 32, seed: int = 0, data_root: str | Path = "data/cifar10") -> Tuple[np.ndarray, str]:
    """Carga CIFAR-10 local si existe; si no, usa fallback tipo CIFAR sin descargar."""
    try:
        from torchvision.datasets import CIFAR10  # type: ignore
        ds = CIFAR10(root=str(data_root), train=False, download=False)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(ds))[:n_images]
        images = []
        for i in idx:
            img, _ = ds[int(i)]
            images.append(_to_uint8_rgb(np.asarray(img), size=size))
        return np.stack(images, axis=0), "cifar10_local"
    except Exception:
        return _fallback_cifar_like_images(n_images=n_images, size=size, seed=seed), "skimage_cifar_like_fallback"


def save_images_to_raw_dir(images: np.ndarray, raw_dir: str | Path, prefix: str = "cifar_smoke") -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for old in raw_dir.glob(f"{prefix}_*.png"):
        old.unlink()
    for i, image in enumerate(images):
        Image.fromarray(image.astype(np.uint8)).save(raw_dir / f"{prefix}_{i:03d}.png")
