"""Conditional residual diffusion for super-resolution (DDPM training, DDIM sampling).

Instead of generating the HR image from noise, the model generates the *residual*
``r = HR - bicubic(LR)`` conditioned on the bicubic upsampling. The residual is small and
sparse (edges and texture), so a small network and few sampling steps are enough.

Training (Ho et al., 2020): ``x_t = sqrt(abar_t) r0 + sqrt(1 - abar_t) eps``; the network
predicts ``eps``; loss is MSE. Sampling (Song et al., 2021, eta = 0) is deterministic given
the initial noise, so a fixed seed makes results reproducible.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def make_schedule(timesteps: int = 1000, kind: str = "cosine") -> dict[str, Tensor]:
    """Noise schedule; ``abar`` is the cumulative product of (1 - beta)."""
    if kind == "linear":
        scale = 1000 / timesteps  # keeps the total noise level independent of the step count
        betas = torch.linspace(scale * 1e-4, scale * 0.02, timesteps, dtype=torch.float64)
        betas = betas.clamp(max=0.999)
    elif kind == "cosine":
        s = 0.008
        t = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
    else:
        raise ValueError("kind must be 'linear' or 'cosine'")
    abar = torch.cumprod(1 - betas, dim=0)
    return {"betas": betas.float(), "abar": abar.float()}


def timestep_embedding(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class _Block(nn.Module):
    def __init__(self, c_in: int, c_out: int, t_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, c_in), c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, c_out)
        self.norm2 = nn.GroupNorm(min(8, c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class CondUNet(nn.Module):
    """Small time-conditioned U-Net; input is [noisy residual, bicubic condition]."""

    def __init__(self, channels: int = 3, base: int = 32, levels: int = 2) -> None:
        super().__init__()
        t_dim = base * 4
        self.base = base
        self.t_mlp = nn.Sequential(nn.Linear(base, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        widths = [base * 2**i for i in range(levels + 1)]
        self.stem = nn.Conv2d(2 * channels, widths[0], 3, padding=1)
        self.down = nn.ModuleList(_Block(widths[i], widths[i + 1], t_dim) for i in range(levels))
        self.mid = _Block(widths[-1], widths[-1], t_dim)
        self.up = nn.ModuleList(_Block(widths[i + 1] + widths[i], widths[i], t_dim)
                                for i in reversed(range(levels)))
        self.out = nn.Sequential(nn.GroupNorm(min(8, widths[0]), widths[0]), nn.SiLU(),
                                 nn.Conv2d(widths[0], channels, 3, padding=1))
        self.stride = 2**levels

    def forward(self, x_t: Tensor, cond: Tensor, t: Tensor) -> Tensor:
        if x_t.shape[-1] % self.stride or x_t.shape[-2] % self.stride:
            raise ValueError(f"spatial sides must be divisible by {self.stride}")
        temb = self.t_mlp(timestep_embedding(t, self.base))
        h = self.stem(torch.cat([x_t, cond], dim=1))
        skips = []
        for block in self.down:
            skips.append(h)
            h = block(F.avg_pool2d(h, 2), temb)
        h = self.mid(h, temb)
        for block in self.up:
            skip = skips.pop()
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), temb)
        return self.out(h)


class ResidualDiffusion(nn.Module):
    def __init__(self, net: CondUNet | None = None, timesteps: int = 1000, schedule="cosine",
                 residual_gain: float = 4.0) -> None:
        super().__init__()
        self.net = net or CondUNet()
        self.timesteps = timesteps
        self.gain = residual_gain  # lifts the small residual towards unit variance
        for k, v in make_schedule(timesteps, schedule).items():
            self.register_buffer(k, v)

    def loss(self, hr: Tensor, cond: Tensor) -> Tensor:
        """Noise-prediction MSE. ``cond`` is the bicubic upsampling of the LR image."""
        x0 = (hr - cond) * self.gain
        t = torch.randint(0, self.timesteps, (hr.shape[0],), device=hr.device)
        eps = torch.randn_like(x0)
        ab = self.abar[t][:, None, None, None]
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        return F.mse_loss(self.net(x_t, cond, t), eps)

    @torch.no_grad()
    def sample(self, cond: Tensor, steps: int = 10, seed: int | None = 0) -> Tensor:
        """DDIM (eta=0) sampling; returns the HR estimate ``cond + residual``."""
        gen = None
        if seed is not None:
            gen = torch.Generator(device=cond.device).manual_seed(seed)
        x = torch.randn(cond.shape, generator=gen, device=cond.device)
        ts = torch.linspace(self.timesteps - 1, 0, steps, device=cond.device).round().long()
        for i, t in enumerate(ts):
            ab = self.abar[t]
            ab_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.ones((), device=x.device)
            tt = torch.full((cond.shape[0],), int(t), device=cond.device, dtype=torch.long)
            eps = self.net(x, cond, tt)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-3.0, 3.0)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return (cond + x / self.gain).clamp(0.0, 1.0)
