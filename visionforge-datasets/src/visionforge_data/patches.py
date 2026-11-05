"""Paired LR/HR patch sampling and image-level splits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from visionforge_data.degrade import bicubic_down, crop_to_multiple, degrade


def split_images(n: int, fractions=(0.8, 0.1, 0.1), seed: int = 0, groups=None):
    """Split image *indices* (never patches) so no image contributes to two splits.

    If ``groups`` (one label per image, for example the scene or city an image belongs to) is
    given, whole groups are assigned to a split, so different acquisitions of the same
    location can never straddle the train/test boundary.
    """
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    if n < 3:
        raise ValueError("need at least 3 images")
    if groups is None:
        units = np.arange(n)
    else:
        groups = np.asarray(groups)
        if groups.shape != (n,):
            raise ValueError("groups must have one entry per image")
        units = np.unique(groups)
    if len(units) < 3:
        raise ValueError("need at least 3 images (or groups)")
    perm = np.random.default_rng(seed).permutation(len(units))
    count = len(units)
    n_tr = min(max(1, round(fractions[0] * count)), count - 2)  # leave room for val and test
    n_va = min(max(1, round(fractions[1] * count)), count - n_tr - 1)
    chosen = (units[perm[:n_tr]], units[perm[n_tr:n_tr + n_va]], units[perm[n_tr + n_va:]])
    if groups is None:
        return tuple(np.sort(c) for c in chosen)
    return tuple(np.flatnonzero(np.isin(groups, c)) for c in chosen)


def audit_image_split(train: Sequence[int], val: Sequence[int], test: Sequence[int],
                      groups=None) -> bool:
    """True if no image (and, when ``groups`` is given, no group) is in two splits."""
    parts = [set(map(int, p)) for p in (train, val, test)]
    if any(parts[a] & parts[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        return False
    if groups is not None:
        gs = [{groups[i] for i in p} for p in parts]
        return not any(gs[a] & gs[b] for a, b in ((0, 1), (0, 2), (1, 2)))
    return True


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
