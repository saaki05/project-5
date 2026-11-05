"""Systems, per-image evaluation, latency/memory measurement and paired statistics."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from visionforge_data import bicubic_up, crop_to_multiple, degrade
from visionforge_models.edsr import count_parameters
from visionforge_models.metrics import psnr, ssim

from visionforge_lab.tiling import tile_apply

SRFn = Callable[[np.ndarray], np.ndarray]


def bicubic_system(scale: int) -> SRFn:
    return lambda lr: bicubic_up(lr, scale)


def _to_tensor(img: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].float().to(device)


def _to_image(t: torch.Tensor) -> np.ndarray:
    return t[0].detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)


def net_system(model: torch.nn.Module, scale: int, device, tile: int = 64) -> SRFn:
    model.to(device).eval()

    @torch.no_grad()
    def run(lr: np.ndarray) -> np.ndarray:
        return tile_apply(lambda p: _to_image(model(_to_tensor(p, device))), lr, scale, tile)

    return run


def diffusion_system(diffusion, scale: int, device, steps: int = 10, tile: int = 64,
                     seed: int = 0) -> SRFn:
    diffusion.to(device).eval()

    @torch.no_grad()
    def one(p: np.ndarray) -> np.ndarray:
        cond = F.interpolate(_to_tensor(p, device), scale_factor=scale, mode="bicubic",
                             align_corners=False)
        h, w = cond.shape[-2:]
        ph, pw = (-h) % diffusion.net.stride, (-w) % diffusion.net.stride
        cond = F.pad(cond, (0, pw, 0, ph), mode="replicate")
        out = diffusion.sample(cond, steps=steps, seed=seed)[..., :h, :w]
        return _to_image(out)

    return lambda lr: tile_apply(one, lr, scale, tile)


def evaluate(system: SRFn, hr_images: Sequence[np.ndarray], scale: int,
             degradation: str = "bicubic", seed: int = 0) -> dict[str, np.ndarray]:
    """Per-image PSNR / SSIM (Y channel, ``scale`` pixels shaved)."""
    ps, ss = [], []
    for i, hr in enumerate(hr_images):
        hr = crop_to_multiple(hr, scale)
        sr = system(degrade(hr, scale, degradation, seed + i))
        ps.append(psnr(sr, hr, border=scale))
        ss.append(ssim(sr, hr, border=scale))
    return {"psnr": np.asarray(ps), "ssim": np.asarray(ss)}


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    d = a - b
    boots = d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))}


def measure_cost(system: SRFn, lr: np.ndarray, device, repeats: int = 3) -> dict:
    """Median latency per image (after one warm-up) and peak CUDA memory when available."""
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    system(lr)  # warm-up
    times = []
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        system(lr)
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    out = {"latency_ms": 1000 * float(np.median(times)),
           "input_pixels": int(lr.shape[0] * lr.shape[1])}
    out["throughput_mpix_per_s"] = out["input_pixels"] / 1e6 / (out["latency_ms"] / 1000)
    if cuda:
        out["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return out


def parameters(model: torch.nn.Module | None) -> int | None:
    return None if model is None else count_parameters(model)
