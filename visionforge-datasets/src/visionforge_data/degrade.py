"""Degradation operators producing low-resolution inputs from high-resolution images.

Images are float32 arrays of shape ``(H, W, C)`` with values in [0, 1]. ``bicubic`` is the
clean benchmark protocol; ``blur_noise`` and ``jpeg`` produce out-of-distribution inputs for
robustness tests.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

DEGRADATIONS = ("bicubic", "blur_noise", "jpeg")


def crop_to_multiple(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - h % scale, : w - w % scale]


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bicubic resize (PIL works on 8-bit, so go through float 'F' mode per channel)."""
    chans = [np.asarray(Image.fromarray(img[..., c], mode="F").resize(size, Image.BICUBIC))
             for c in range(img.shape[2])]
    return np.clip(np.stack(chans, axis=-1), 0.0, 1.0).astype(np.float32)


def bicubic_down(hr: np.ndarray, scale: int) -> np.ndarray:
    hr = crop_to_multiple(hr, scale)
    h, w = hr.shape[:2]
    return _resize(hr, (w // scale, h // scale))


def bicubic_up(lr: np.ndarray, scale: int) -> np.ndarray:
    h, w = lr.shape[:2]
    return _resize(lr, (w * scale, h * scale))


def blur_noise(hr: np.ndarray, scale: int, sigma: float = 1.2, noise: float = 0.02,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Gaussian blur, bicubic downscale, then additive Gaussian noise."""
    rng = rng or np.random.default_rng(0)
    blurred = np.stack([gaussian_filter(hr[..., c], sigma) for c in range(hr.shape[2])], axis=-1)
    lr = bicubic_down(blurred, scale)
    return np.clip(lr + rng.normal(scale=noise, size=lr.shape), 0, 1).astype(np.float32)


def jpeg(hr: np.ndarray, scale: int, quality: int = 30) -> np.ndarray:
    """Bicubic downscale followed by JPEG compression."""
    lr = bicubic_down(hr, scale)
    buf = io.BytesIO()
    Image.fromarray((lr * 255).round().astype(np.uint8)).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0


def degrade(hr: np.ndarray, scale: int, kind: str = "bicubic", seed: int = 0) -> np.ndarray:
    if kind == "bicubic":
        return bicubic_down(hr, scale)
    if kind == "blur_noise":
        return blur_noise(hr, scale, rng=np.random.default_rng(seed))
    if kind == "jpeg":
        return jpeg(hr, scale)
    raise ValueError(f"unknown degradation {kind!r}; choose from {DEGRADATIONS}")
