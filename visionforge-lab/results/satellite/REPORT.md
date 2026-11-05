# VisionForge ×4 super-resolution (generated from result JSON)

Device: cpu. Images: train 64, val 8, test 8. Budget: 1500 steps; patch 64, batch 16.

## Degradation: bicubic

| system | PSNR [95% CI] | SSIM [95% CI] | Δ PSNR vs bicubic (p) | params | train steps | train s | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 29.29 [28.32, 30.23] | 0.8263 [0.8037, 0.8465] | — | — | — | — | 1 |
| EDSR (L1) | 30.82 [29.77, 31.78] | 0.8685 [0.8505, 0.8846] | +1.53 (p=0.000) | 195971 | 1500 | 270 | 49 |
| EDSR + adversarial | 30.39 [29.39, 31.33] | 0.8522 [0.8340, 0.8686] | +1.10 (p=0.000) | 195971 | 750 | 567 | 48 |
| residual diffusion (DDIM-10) | 29.97 [29.06, 30.84] | 0.8350 [0.8185, 0.8500] | +0.69 (p=0.000) | 485715 | 4000 | 3104 | 2104 |

## Degradation: blur_noise

| system | PSNR [95% CI] | SSIM [95% CI] | Δ PSNR vs bicubic (p) | params | train steps | train s | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 27.97 [27.08, 28.86] | 0.7701 [0.7468, 0.7918] | — | — | — | — | 1 |
| EDSR (L1) | 28.96 [28.16, 29.69] | 0.7888 [0.7743, 0.8040] | +0.98 (p=0.000) | 195971 | 1500 | 270 | 49 |
| EDSR + adversarial | 28.72 [27.96, 29.43] | 0.7719 [0.7587, 0.7853] | +0.74 (p=0.000) | 195971 | 750 | 567 | 48 |
| residual diffusion (DDIM-10) | 28.59 [27.83, 29.32] | 0.7679 [0.7539, 0.7816] | +0.61 (p=0.000) | 485715 | 4000 | 3104 | 2104 |

## Degradation: jpeg

| system | PSNR [95% CI] | SSIM [95% CI] | Δ PSNR vs bicubic (p) | params | train steps | train s | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 25.68 [24.84, 26.57] | 0.6587 [0.6230, 0.6954] | — | — | — | — | 1 |
| EDSR (L1) | 25.36 [24.51, 26.27] | 0.6409 [0.6050, 0.6782] | -0.32 (p=0.000) | 195971 | 1500 | 270 | 49 |
| EDSR + adversarial | 25.21 [24.36, 26.12] | 0.6291 [0.5920, 0.6674] | -0.47 (p=0.000) | 195971 | 750 | 567 | 48 |
| residual diffusion (DDIM-10) | 25.11 [24.27, 26.03] | 0.6212 [0.5853, 0.6585] | -0.57 (p=0.000) | 485715 | 4000 | 3104 | 2104 |

