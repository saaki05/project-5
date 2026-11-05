"""Budget-matched comparison of bicubic, regression, adversarial and diffusion SR.

Image-level splits keep test images unseen. All learned models see the same training patch
bank, patch size and batch size, and share one budget (optimiser steps and/or wall-clock
seconds), so differences cannot be explained by unequal training effort. Wall-clock and
parameter counts are reported beside quality because the families cost different amounts per
step.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from visionforge_data import (
    DEGRADATIONS,
    audit_image_split,
    augment_pairs,
    list_images,
    load_image,
    sample_pairs,
    split_images,
)
from visionforge_models.diffusion import CondUNet, ResidualDiffusion
from visionforge_models.edsr import EDSR
from visionforge_models.train import pick_device, train_diffusion, train_gan, train_regression

from visionforge_lab import benchmark as B


def _nchw(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a.transpose(0, 3, 1, 2))).float()


def _make_batch_fn(hr_bank: np.ndarray, lr_bank: np.ndarray, batch: int, seed: int, device):
    def fn(step: int):
        rng = np.random.default_rng((seed * 100003 + step) % (2**63))
        idx = rng.integers(0, len(hr_bank), batch)
        hr, lr = augment_pairs(hr_bank[idx], lr_bank[idx], rng)
        return _nchw(hr).to(device), _nchw(lr).to(device)
    return fn


def run_study(images: list[np.ndarray], scale: int = 4, steps: int = 2000,
              max_seconds: float | None = None, patch: int = 64, batch: int = 16,
              bank_size: int = 2000, gan_steps: int | None = None, diffusion_steps_sample: int = 10,
              seed: int = 0, device=None, out_path=None, features: int = 48, blocks: int = 8,
              diffusion_timesteps: int = 1000, diffusion_base: int = 32,
              degradations=DEGRADATIONS, tile: int = 64, groups=None,
              test_max_side: int | None = None, diffusion_train_steps: int | None = None) -> dict:
    device = device or pick_device()
    train_i, val_i, test_i = split_images(len(images), seed=seed, groups=groups)
    if not audit_image_split(train_i, val_i, test_i, groups):
        raise RuntimeError("image split leaks")
    train_imgs = [images[i] for i in train_i]
    test_imgs = [images[i] for i in test_i]
    if test_max_side:  # bound evaluation cost: centre-crop test images
        def centre(img):
            h, w = img.shape[:2]
            y, x = max(0, (h - test_max_side) // 2), max(0, (w - test_max_side) // 2)
            return np.ascontiguousarray(img[y:y + test_max_side, x:x + test_max_side])

        test_imgs = [centre(im) for im in test_imgs]
    hr_bank, lr_bank = sample_pairs(train_imgs, bank_size, patch, scale, "bicubic", seed)
    batch_fn = _make_batch_fn(hr_bank, lr_bank, batch, seed, device)

    result: dict = {"scale": scale, "device": str(device), "seed": seed,
                    "n_train_images": len(train_i), "n_val_images": len(val_i),
                    "n_test_images": len(test_i), "test_max_side": test_max_side,
                    "diffusion_sampling_steps": diffusion_steps_sample, "budget": {"steps": steps,
                                                             "max_seconds": max_seconds},
                    "patch": patch, "batch": batch, "systems": {}}
    probe_lr = B.degrade(B.crop_to_multiple(test_imgs[0], scale), scale, "bicubic")

    systems: dict[str, tuple] = {"bicubic": (B.bicubic_system(scale), None, None)}

    torch.manual_seed(seed)
    edsr = EDSR(scale, 3, features, blocks)
    hist = train_regression(edsr, batch_fn, steps, device=device, max_seconds=max_seconds,
                            seed=seed)
    systems["EDSR (L1)"] = (B.net_system(edsr, scale, device, tile), edsr, hist)

    gan = copy.deepcopy(edsr)
    hist_g = train_gan(gan, batch_fn, gan_steps or max(1, steps // 2), device=device,
                       max_seconds=max_seconds, seed=seed)
    systems["EDSR + adversarial"] = (B.net_system(gan, scale, device, tile), gan, hist_g)

    net = CondUNet(3, diffusion_base, levels=2)
    diff = ResidualDiffusion(net, diffusion_timesteps)
    hist_d, ema = train_diffusion(diff, batch_fn, diffusion_train_steps or steps, device=device,
                                  max_seconds=max_seconds, seed=seed)
    result["diffusion_gain"] = float(diff.gain)
    systems[f"residual diffusion (DDIM-{diffusion_steps_sample})"] = (
        B.diffusion_system(ema, scale, device, diffusion_steps_sample, tile, seed), ema, hist_d)

    evals: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name, (fn, model, hist) in systems.items():
        entry = {"parameters": B.parameters(model),
                 "train_steps": None if hist is None else hist["steps"],
                 "train_seconds": None if hist is None else hist["seconds"],
                 "cost": B.measure_cost(fn, probe_lr, device), "quality": {}}
        evals[name] = {}
        for deg in degradations:
            per = B.evaluate(fn, test_imgs, scale, deg, seed)
            evals[name][deg] = per
            entry["quality"][deg] = {m: dict(zip(("mean", "ci_low", "ci_high"),
                                                 B.bootstrap_ci(v), strict=True))
                                     for m, v in per.items()}
        result["systems"][name] = entry

    for name in systems:
        if name == "bicubic":
            continue
        for deg in degradations:
            result["systems"][name].setdefault("vs_bicubic", {})[deg] = {
                m: B.paired_bootstrap(evals[name][deg][m], evals["bicubic"][deg][m])
                for m in ("psnr", "ssim")}
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def load_images(folder: str | Path, limit: int | None = None) -> list[np.ndarray]:
    paths = list_images(folder)[:limit]
    return [load_image(p) for p in paths]


def load_images_grouped(folder: str | Path, group_level: int = 2, min_valid: float = 0.98,
                        max_side: int | None = None, limit: int | None = None):
    """Load images with a group label taken from the directory ``group_level`` levels up.

    For ``cache/<scene>/images_masked/<file>.tif`` use ``group_level=2`` to group by scene.
    Images whose alpha channel marks more than ``1 - min_valid`` of the pixels invalid are
    skipped; ``max_side`` centre-crops large images to bound compute.
    """
    from PIL import Image

    images, groups = [], []
    for path in list_images(folder):
        arr = np.asarray(Image.open(path).convert("RGBA"))
        if (arr[..., 3] > 0).mean() < min_valid:
            continue
        img = arr[..., :3].astype(np.float32) / 255.0
        if max_side:
            h, w = img.shape[:2]
            y, x = max(0, (h - max_side) // 2), max(0, (w - max_side) // 2)
            img = img[y:y + max_side, x:x + max_side]
        images.append(np.ascontiguousarray(img))
        groups.append(path.parents[group_level - 1].name)
        if limit and len(images) >= limit:
            break
    return images, groups
