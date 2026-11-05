# VisionForge

Image **super-resolution** research code: a compute-budgeted comparison of regression,
adversarial and diffusion models, with proper statistics and cost reporting.

Three independently installable packages with one-way dependencies:

```
visionforge-datasets  ──▶  visionforge-models  ──▶  visionforge-lab
 degradations, paired       EDSR, GAN, residual        budget-matched study,
 patches, image-level       diffusion (DDPM/DDIM),     benchmark harness, bootstrap
 splits (numpy, scipy)      PSNR/SSIM (torch)          stats, report (torch)
```

| Package | Responsibility |
|---|---|
| [`visionforge-datasets`](visionforge-datasets) | bicubic / blur+noise / JPEG degradations, aligned patch sampling, image-level splits with audit, checksum manifests |
| [`visionforge-models`](visionforge-models) | models written from scratch, noise schedules, EMA, budgeted training loops, metrics |
| [`visionforge-lab`](visionforge-lab) | study runner, tiled inference, per-image metrics with CIs, paired tests, latency / memory, report |

## Research question

At a fixed training budget on one consumer GPU, where do an L1-trained EDSR, an
adversarially fine-tuned EDSR and a few-step conditional residual diffusion model land in
quality (PSNR/SSIM), cost (parameters, training time, latency, memory) and robustness to
degradations unseen in training? The contribution would be a reproducible, compute-matched
comparison; the field is crowded and no state-of-the-art claim is made.

## Status (honest summary)

| Item | State |
|---|---|
| All packages implemented; 29 unit tests pass (including closed-form metric checks, reproducible sampling, budget stopping, end-to-end study on tiny synthetic images) | yes |
| Experiments on real photographs | **not run**: DIV2K and the standard test sets were unreachable from the development container, which also has no GPU |
| LPIPS / perceptual metrics, transformer-based SR (SwinIR-style) | not implemented |
| Published numbers | none |

The study is designed to be run on a GPU machine:

```bash
git clone https://github.com/saaki05/project-5 && cd project-5
pip install -e visionforge-datasets -e visionforge-models -e visionforge-lab
visionforge-lab study --images /path/to/hr_images --scale 4 --steps 20000 --patch 96 \
    --out results/study.json
visionforge-lab report --results results/study.json
```

## License

MIT. Image datasets carry their own licenses.
