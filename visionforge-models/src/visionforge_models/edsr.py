"""EDSR-style residual network (Lim et al., 2017) predicting a residual over bicubic upsampling."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int, res_scale: float = 0.1) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(ch, ch, 3, padding=1))
        self.res_scale = res_scale

    def forward(self, x: Tensor) -> Tensor:
        return x + self.res_scale * self.body(x)


class Upsampler(nn.Sequential):
    """Sub-pixel convolution stages; scale 2^k uses k doubling stages, scale 3 one stage."""

    def __init__(self, scale: int, ch: int) -> None:
        layers: list[nn.Module] = []
        if scale in (2, 4, 8):
            for _ in range(scale.bit_length() - 1):
                layers += [nn.Conv2d(ch, 4 * ch, 3, padding=1), nn.PixelShuffle(2)]
        elif scale == 3:
            layers += [nn.Conv2d(ch, 9 * ch, 3, padding=1), nn.PixelShuffle(3)]
        else:
            raise ValueError("scale must be 2, 3, 4 or 8")
        super().__init__(*layers)


class EDSR(nn.Module):
    def __init__(self, scale: int = 4, channels: int = 3, features: int = 48, blocks: int = 8):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(channels, features, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(features) for _ in range(blocks)],
                                  nn.Conv2d(features, features, 3, padding=1))
        self.up = Upsampler(scale, features)
        self.tail = nn.Conv2d(features, channels, 3, padding=1)

    def forward(self, lr: Tensor) -> Tensor:
        """``lr``: (N, C, h, w) in [0, 1]; returns (N, C, h*scale, w*scale)."""
        base = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        x = self.head(lr)
        x = x + self.body(x)
        return base + self.tail(self.up(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = ["EDSR", "ResBlock", "Upsampler", "count_parameters", "torch"]
