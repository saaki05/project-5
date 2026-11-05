"""PSNR and SSIM, implemented from their definitions.

Super-resolution convention: images in [0, 1], scores computed on the luma (Y) channel of
BT.601 after shaving ``scale`` border pixels.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    """BT.601 luma of an (H, W, 3) image in [0, 1] (output in [0, 1], without the 16 offset)."""
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def _shave(x: np.ndarray, border: int) -> np.ndarray:
    return x[border:x.shape[0] - border, border:x.shape[1] - border] if border else x


def psnr(a: np.ndarray, b: np.ndarray, border: int = 0, y_channel: bool = True) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if y_channel and a.ndim == 3:
        a, b = rgb_to_y(a), rgb_to_y(b)
    mse = float(np.mean((_shave(a, border) - _shave(b, border)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(a: np.ndarray, b: np.ndarray, border: int = 0, y_channel: bool = True) -> float:
    """Mean structural similarity (Wang et al., 2004): 11x11 Gaussian window, sigma 1.5."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if y_channel and a.ndim == 3:
        a, b = rgb_to_y(a), rgb_to_y(b)
    a, b = _shave(a, border), _shave(b, border)
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    if min(a.shape[:2]) < 11:
        raise ValueError("image too small for an 11x11 SSIM window")
    c1, c2 = 0.01**2, 0.03**2
    w = _gaussian_window()
    scores = []
    for ch in range(a.shape[2]):
        x = torch.from_numpy(np.ascontiguousarray(a[..., ch]))[None, None].double()
        y = torch.from_numpy(np.ascontiguousarray(b[..., ch]))[None, None].double()
        mu_x, mu_y = F.conv2d(x, w), F.conv2d(y, w)
        sxx = F.conv2d(x * x, w) - mu_x**2
        syy = F.conv2d(y * y, w) - mu_y**2
        sxy = F.conv2d(x * y, w) - mu_x * mu_y
        num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
        den = (mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2)
        scores.append((num / den).mean().item())
    return float(np.mean(scores))
