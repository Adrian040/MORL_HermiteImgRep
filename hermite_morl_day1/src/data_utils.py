from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage import color, data, transform

from .metrics import to_float01


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _resize_gray(image: np.ndarray, image_size: int) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = color.rgb2gray(image)
    image = to_float01(image)
    resized = transform.resize(
        image,
        (image_size, image_size),
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.asarray(resized, dtype=np.float32)


def load_images_from_folder(raw_dir: str | Path, image_size: int, max_images: Optional[int] = None) -> np.ndarray:
    raw_dir = Path(raw_dir)
    paths = sorted([p for p in raw_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
    if max_images is not None:
        paths = paths[:max_images]
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(_resize_gray(np.asarray(img), image_size))
    if not images:
        raise FileNotFoundError(f"No se encontraron imágenes en {raw_dir}.")
    return np.stack(images, axis=0)


def _extract_patches(image: np.ndarray, image_size: int, stride: Optional[int] = None) -> List[np.ndarray]:
    gray = _resize_gray(image, max(image_size * 2, image_size))
    if gray.shape[0] == image_size and gray.shape[1] == image_size:
        return [gray]
    stride = stride or image_size
    patches = []
    for y in range(0, max(1, gray.shape[0] - image_size + 1), stride):
        for x in range(0, max(1, gray.shape[1] - image_size + 1), stride):
            patch = gray[y:y + image_size, x:x + image_size]
            if patch.shape == (image_size, image_size):
                patches.append(patch.astype(np.float32))
    return patches


def build_skimage_dataset(image_size: int = 64, n_images: int = 60, seed: int = 0) -> np.ndarray:
    base_images = [
        data.camera(),
        data.coins(),
        data.moon(),
        data.page(),
        data.text(),
        data.clock(),
        data.astronaut(),
        data.chelsea(),
        data.coffee(),
        data.rocket(),
        data.immunohistochemistry(),
    ]
    patches: List[np.ndarray] = []
    for image in base_images:
        patches.extend(_extract_patches(image, image_size=image_size, stride=image_size // 2))

    rng = np.random.default_rng(seed)
    rng.shuffle(patches)
    if len(patches) < n_images:
        augmented = []
        for patch in patches:
            augmented.extend([patch, np.fliplr(patch), np.flipud(patch), np.rot90(patch)])
        rng.shuffle(augmented)
        patches = augmented
    return np.stack(patches[:n_images], axis=0).astype(np.float32)


def split_dataset(
    images: np.ndarray,
    train: int,
    val: int,
    test: int,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    required = train + val + test
    if len(images) < required:
        raise ValueError(f"Se requieren {required} imágenes y solo hay {len(images)}.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(images))[:required]
    images = images[indices]
    return {
        "train": images[:train],
        "val": images[train:train + val],
        "test": images[train + val:train + val + test],
    }


def prepare_dataset(config: dict) -> Dict[str, np.ndarray]:
    dataset_cfg = config["dataset"]
    image_size = int(dataset_cfg.get("image_size", 64))
    seed = int(config.get("seed", 0))
    total = int(dataset_cfg.get("train_images", 40)) + int(dataset_cfg.get("val_images", 10)) + int(dataset_cfg.get("test_images", 10))

    raw_dir = dataset_cfg.get("raw_dir", "")
    use_skimage = bool(dataset_cfg.get("use_skimage_if_empty", True))

    if raw_dir and Path(raw_dir).exists() and any(Path(raw_dir).rglob("*")):
        images = load_images_from_folder(raw_dir, image_size=image_size, max_images=None)
    elif use_skimage:
        images = build_skimage_dataset(image_size=image_size, n_images=total, seed=seed)
    else:
        raise FileNotFoundError("No hay imágenes locales y use_skimage_if_empty=False.")

    return split_dataset(
        images,
        train=int(dataset_cfg.get("train_images", 40)),
        val=int(dataset_cfg.get("val_images", 10)),
        test=int(dataset_cfg.get("test_images", 10)),
        seed=seed,
    )


def save_processed_dataset(splits: Dict[str, np.ndarray], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **splits)
