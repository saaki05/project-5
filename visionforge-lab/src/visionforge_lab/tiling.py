"""Overlapping-tile application of a super-resolution function to large images."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _origins(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tile_apply(fn: Callable[[np.ndarray], np.ndarray], lr: np.ndarray, scale: int,
               tile: int = 64, overlap: int = 8) -> np.ndarray:
    """Apply ``fn`` (LR tile -> HR tile) over overlapping tiles and average the overlaps.

    ``lr`` is (h, w, C); the result is (h*scale, w*scale, C). Small images are processed in
    one piece. Averaging uniformly keeps the operation exact for translation-invariant ``fn``.
    """
    h, w, c = lr.shape
    if max(h, w) <= tile:
        return fn(lr)
    stride = max(1, tile - overlap)
    acc = np.zeros((h * scale, w * scale, c), dtype=np.float64)
    cnt = np.zeros((h * scale, w * scale, 1), dtype=np.float64)
    for y in _origins(h, tile, stride):
        for x in _origins(w, tile, stride):
            patch = lr[y:y + tile, x:x + tile]
            out = fn(patch)
            ys, xs = y * scale, x * scale
            acc[ys:ys + out.shape[0], xs:xs + out.shape[1]] += out
            cnt[ys:ys + out.shape[0], xs:xs + out.shape[1]] += 1
    return (acc / cnt).astype(np.float32)
