# visionforge-lab

Benchmark harness for the VisionForge study: **where do regression, adversarial and
diffusion super-resolution land on the quality / compute plane when trained with the same
budget, and how do they behave on degradations they were not trained on?**

## What it does

- `run_study` trains bicubic-initialised **EDSR (L1)**, **EDSR + adversarial fine-tuning** and
  **residual diffusion** from the same patch bank, with identical patch size, batch size and
  step (and optional wall-clock) budget, then evaluates every system on held-out **test
  images** under `bicubic`, `blur_noise` and `jpeg` degradations.
- Per-image PSNR / SSIM (Y channel, shaved border) with bootstrap confidence intervals and
  paired bootstrap tests against bicubic.
- Cost: parameters, training seconds, median per-image latency, throughput and peak CUDA
  memory when a GPU is used.
- Overlapping-tile inference (`tile_apply`) so large images fit in memory.
- A Markdown report renderer: every number in the report comes from the result JSON.

## Install, test, run

```bash
git clone https://github.com/saaki05/project-5 && cd project-5
pip install -e visionforge-datasets -e visionforge-models
cd visionforge-lab && pip install -e ".[dev]" && pytest

visionforge-lab study --images /path/to/hr_images --scale 4 --steps 20000 \
    --patch 96 --batch 16 --out results/study.json
visionforge-lab report --results results/study.json
```

A CUDA GPU is used automatically (mixed precision for the regression loop).

## Status

The code is complete and unit-tested end to end on tiny synthetic images (training, GAN
fine-tuning, diffusion sampling, evaluation, statistics, report). **No experiment on real
photographs has been run**: DIV2K and the standard test sets were not reachable from the
development container, which also has no GPU. This repository therefore contains no
benchmark numbers; the study is meant to be run on a GPU machine with a dataset you have
downloaded and have the right to use.

## Reading the results responsibly

- Differences between systems should be judged against the confidence intervals; with few
  test images they are wide.
- Without a learned perceptual metric, PSNR/SSIM favour smooth regression outputs; a GAN or
  diffusion model can look sharper to a human while scoring lower. Add LPIPS (needs
  pretrained weights) or a user study before drawing perceptual conclusions.
- Budget-matching by steps does not match compute (a diffusion step is not an EDSR step);
  use `--max-seconds` and report the printed training times.

## License

MIT.
