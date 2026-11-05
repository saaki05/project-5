"""Budgeted training loops for the three model families.

Every loop stops at ``steps`` optimiser updates or ``max_seconds`` of wall-clock time,
whichever comes first, so comparisons can be matched either by updates or by compute time.
``batch_fn(step)`` returns ``(hr, lr)`` float tensors in NCHW layout on any device.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from visionforge_models.diffusion import ResidualDiffusion
from visionforge_models.gan import PatchDiscriminator, discriminator_loss, generator_loss

BatchFn = Callable[[int], tuple[Tensor, Tensor]]


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EMA:
    """Exponential moving average of model weights (used when sampling diffusion models)."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for e, p in zip(self.model.parameters(), model.parameters(), strict=True):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers(), strict=True):
            e.copy_(b)


def _out_of_budget(step: int, steps: int, start: float, max_seconds: float | None) -> bool:
    return step >= steps or (max_seconds is not None and time.perf_counter() - start > max_seconds)


def train_regression(model: nn.Module, batch_fn: BatchFn, steps: int, lr: float = 2e-4,
                     device: torch.device | None = None, max_seconds: float | None = None,
                     amp: bool = True, seed: int = 0) -> dict:
    """L1 regression (the standard EDSR objective)."""
    device = device or pick_device()
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    losses, start, step = [], time.perf_counter(), 0
    while not _out_of_budget(step, steps, start, max_seconds):
        hr, lr_img = (t.to(device) for t in batch_fn(step))
        with torch.autocast(device.type, enabled=use_amp):
            loss = F.l1_loss(model(lr_img).float(), hr)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        step += 1
    return {"losses": losses, "steps": step, "seconds": time.perf_counter() - start}


def train_gan(generator: nn.Module, batch_fn: BatchFn, steps: int, lr: float = 1e-4,
              adv_weight: float = 5e-3, device: torch.device | None = None,
              max_seconds: float | None = None, seed: int = 0,
              discriminator: PatchDiscriminator | None = None) -> dict:
    """Adversarial fine-tuning of a (pre-trained) generator."""
    device = device or pick_device()
    torch.manual_seed(seed)
    generator.to(device).train()
    disc = (discriminator or PatchDiscriminator()).to(device).train()
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.99))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.9, 0.99))
    hist = {"g": [], "d": []}
    start, step = time.perf_counter(), 0
    while not _out_of_budget(step, steps, start, max_seconds):
        hr, lr_img = (t.to(device) for t in batch_fn(step))
        sr = generator(lr_img)
        d_loss = discriminator_loss(disc(hr), disc(sr.detach()))
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        g_loss = generator_loss(sr, hr, disc(sr), adv_weight)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        hist["g"].append(g_loss.item())
        hist["d"].append(d_loss.item())
        step += 1
    return {**hist, "steps": step, "seconds": time.perf_counter() - start}


def train_diffusion(diffusion: ResidualDiffusion, batch_fn: BatchFn, steps: int,
                    lr: float = 2e-4, device: torch.device | None = None,
                    max_seconds: float | None = None, seed: int = 0, ema_decay: float = 0.995):
    """Noise-prediction training; returns ``(history, ema_model)`` for sampling."""
    device = device or pick_device()
    torch.manual_seed(seed)
    diffusion.to(device).train()
    ema = EMA(diffusion, ema_decay)
    opt = torch.optim.Adam(diffusion.parameters(), lr=lr)
    losses, start, step = [], time.perf_counter(), 0
    while not _out_of_budget(step, steps, start, max_seconds):
        hr, lr_img = (t.to(device) for t in batch_fn(step))
        cond = F.interpolate(lr_img, size=hr.shape[-2:], mode="bicubic", align_corners=False)
        loss = diffusion.loss(hr, cond)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
        opt.step()
        ema.update(diffusion)
        losses.append(loss.item())
        step += 1
    return {"losses": losses, "steps": step, "seconds": time.perf_counter() - start}, ema.model
