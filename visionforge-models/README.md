# visionforge-models

Super-resolution models written from scratch in PyTorch, with budgeted training loops.

| Family | Implementation |
|---|---|
| Regression CNN | EDSR-style residual network (Lim et al., 2017): residual blocks with scaling, sub-pixel upsampling, and a **residual over bicubic upsampling** so a randomly initialised network starts at the bicubic baseline |
| Adversarial | PatchGAN discriminator with spectral normalisation, non-saturating GAN loss, generator loss = L1 + weighted adversarial term. There is deliberately no VGG perceptual loss: it needs pretrained weights, and this code must run offline |
| Diffusion | conditional **residual** diffusion: a small time-conditioned U-Net predicts the noise added to `HR − bicubic(LR)`; DDPM training (cosine or linear schedule), **DDIM** (η = 0) few-step sampling, EMA weights; sampling is deterministic given a seed |
| Metrics | PSNR and SSIM from their definitions (Y channel, border shave, float64 arithmetic) |
| Training | `train_regression`, `train_gan`, `train_diffusion`, each stopping at a step budget or a wall-clock budget; mixed precision on CUDA |

## Install and test

```bash
git clone https://github.com/saaki05/project-5 && cd project-5/visionforge-models
pip install -e ".[dev]" && pytest
```

## What the tests establish

Output shapes for scales 2/3/4; the EDSR output equals bicubic upsampling when its tail is
zeroed; losses fall on a toy task; wall-clock budgets stop training; SSIM/PSNR equal their
closed forms on constant images and fall with noise; noise schedules are monotone and reach
near-pure noise for any step count (a real bug caught here: a linear schedule with fewer than
1000 steps did not); diffusion sampling is reproducible for a fixed seed and differs across
seeds; EMA converges to the live weights.

## Limitations

- LPIPS and other learned perceptual metrics are not included (pretrained weights).
- Model sizes are chosen for a single consumer GPU, not to reproduce published numbers.
- The diffusion model works on the residual with a fixed gain; it has not been tuned.

## License

MIT.
