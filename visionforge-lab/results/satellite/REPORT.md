# VisionForge ×4 super-resolution (generated from result JSON)

Device: cpu. Images: train 64, val 8, test 8. Budget: 3000 steps; patch 64, batch 16.

## Degradation: bicubic

| system | PSNR [95% CI] | SSIM [95% CI] | Δ PSNR vs bicubic (p) | params | train steps | train s | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 29.29 [28.32, 30.23] | 0.8263 [0.8037, 0.8465] | — | — | — | — | 1 |
| EDSR (L1) | 31.16 [30.12, 32.10] | 0.8771 [0.8614, 0.8910] | +1.87 (p=0.000) | 522243 | 3000 | 1291 | 61 |
| EDSR + adversarial | 30.16 [29.25, 31.01] | 0.8415 [0.8247, 0.8571] | +0.87 (p=0.000) | 522243 | 1500 | 747 | 97 |
| residual diffusion (DDIM-10) | 30.00 [29.10, 30.86] | 0.8367 [0.8205, 0.8515] | +0.71 (p=0.000) | 485715 | 5000 | 2249 | 1436 |

## Degradation: blur_noise

| system | PSNR [95% CI] | SSIM [95% CI] | Δ PSNR vs bicubic (p) | params | train steps | train s | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 27.97 [27.08, 28.86] | 0.7701 [0.7468, 0.7918] | — | — | — | — | 1 |
| EDSR (L1) | 29.10 [28.33, 29.82] | 0.7917 [0.7785, 0.8050] | +1.13 (p=0.000) | 522243 | 3000 | 1291 | 61 |
| EDSR + adversarial | 28.41 [27.70, 29.11] | 0.7512 [0.7399, 0.7623] | +0.44 (p=0.000) | 522243 | 1500 | 747 | 97 |
| residual diffusion (DDIM-10) | 28.57 [27.82, 29.29] | 0.7666 [0.7533, 0.7799] | +0.60 (p=0.000) | 485715 | 5000 | 2249 | 1436 |

## Degradation: jpeg

| system | PSNR [95% CI] | SSIM [95% CI] | Δ PSNR vs bicubic (p) | params | train steps | train s | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 25.68 [24.84, 26.57] | 0.6587 [0.6230, 0.6954] | — | — | — | — | 1 |
| EDSR (L1) | 25.30 [24.46, 26.21] | 0.6403 [0.6043, 0.6779] | -0.38 (p=0.000) | 522243 | 3000 | 1291 | 61 |
| EDSR + adversarial | 25.06 [24.23, 25.97] | 0.6204 [0.5820, 0.6606] | -0.62 (p=0.000) | 522243 | 1500 | 747 | 97 |
| residual diffusion (DDIM-10) | 25.11 [24.26, 26.02] | 0.6215 [0.5854, 0.6588] | -0.57 (p=0.000) | 485715 | 5000 | 2249 | 1436 |

