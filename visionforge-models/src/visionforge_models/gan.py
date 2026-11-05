"""Adversarial training components for super-resolution.

The generator is any LR->HR network (here the EDSR above). The discriminator scores local
patches (PatchGAN style). Generator loss = pixel L1 + ``adv_weight`` * non-saturating
adversarial loss. A VGG perceptual term is intentionally absent: it needs pretrained weights,
and this implementation must run offline and reproducibly.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    def __init__(self, channels: int = 3, base: int = 32, levels: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = channels
        for i in range(levels):
            out = base * 2**i
            layers += [spectral_norm(nn.Conv2d(c, out, 3, stride=1, padding=1)),
                       nn.LeakyReLU(0.2, True),
                       spectral_norm(nn.Conv2d(out, out, 4, stride=2, padding=1)),
                       nn.LeakyReLU(0.2, True)]
            c = out
        layers.append(spectral_norm(nn.Conv2d(c, 1, 3, padding=1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def discriminator_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    return (F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()) / 2


def generator_loss(sr: Tensor, hr: Tensor, fake_logits: Tensor, adv_weight: float = 5e-3) -> Tensor:
    """L1 reconstruction plus the non-saturating adversarial term -log D(G(x))."""
    return F.l1_loss(sr, hr) + adv_weight * F.softplus(-fake_logits).mean()
