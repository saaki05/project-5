"""Paired LR/HR patch sampling and image-level splits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from visionforge_data.degrade import bicubic_down, crop_to_multiple, degrade


def split_images(n: int, fractions=(0.8, 0.1, 0.1), seed: int = 0):
    """Split image *indices* (never patches) so no image contributes to two splits."""
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    if n < 3:
        raise ValueError("need at least 3 images")
    idx = np.random.default_rng(seed).permutation(n)
    n_tr = max(1, round(fractions[0] * n))
    n_va = max(1, round(fractions[1] * n))
    parts = (np.sort(idx[:n_tr]), np.sort(idx[n_tr:n_tr + n_va]), np.sort(idx[n_tr + n_va:]))
    if any(len(p) == 0 for p in parts):
        raise ValueError("a split is empty; use more images or different fractions")
    return parts


def audit_image_split(train: Sequence[int], val: Sequence[int], test: Sequence[int]) -> bool:
    a, b, c = set(map(int, train)), set(map(int, val)), set(map(int, test))
    return not (a & b or a & c or b & c)


def sample_pairs(images: Sequence[np.ndarray], n: int, patch: int, scale: int,
                 degradation: str = "bicubic", seed: int = 0):
    """Random aligned patches: HR ``(n, patch, patch, 3)`` and LR ``(n, patch/s, patch/s, 3)``.

    The degradation is applied to the whole image before cropping (as in standard SR
    pipelines), so LR patches have no border artefacts.
    """
    if patch % scale:
        raise ValueError("patch must be a multiple of scale")
    rng = np.random.default_rng(seed)
    hrs, lrs = [], []
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for _ in range(n):
        i = int(rng.integers(0, len(images)))
        if i not in cache:
            hr = crop_to_multiple(images[i], scale)
            cache[i] = (hr, degrade(hr, scale, degradation, seed + i))
        hr, lr = cache[i]
        h, w = hr.shape[:2]
        if h < patch or w < patch:
            raise ValueError(f"image {i} ({h}x{w}) smaller than patch {patch}")
        y = int(rng.integers(0, (h - patch) // scale + 1)) * scale
        x = int(rng.integers(0, (w - patch) // scale + 1)) * scale
        hrs.append(hr[y:y + patch, x:x + patch])
        lrs.append(lr[y // scale:(y + patch) // scale, x // scale:(x + patch) // scale])
    return np.stack(hrs), np.stack(lrs)


def augment_pairs(hr: np.ndarray, lr: np.ndarray, rng: np.random.Generator):
    """Identical random flips / 90-degree rotations applied to a patch batch pair."""
    if rng.random() < 0.5:
        hr, lr = hr[:, :, ::-1], lr[:, :, ::-1]
    if rng.random() < 0.5:
        hr, lr = hr[:, ::-1], lr[:, ::-1]
    k = int(rng.integers(0, 4))
    return np.ascontiguousarray(np.rot90(hr, k, (1, 2))), np.ascontiguousarray(
        np.rot90(lr, k, (1, 2)))


__all__ = ["audit_image_split", "augment_pairs", "bicubic_down", "sample_pairs", "split_images"]
