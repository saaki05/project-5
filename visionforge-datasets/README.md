# visionforge-datasets

Data layer for VisionForge (image super-resolution and restoration).

## What it does

- **Degradation operators** producing low-resolution inputs from HR images: `bicubic`
  (the clean benchmark protocol), `blur_noise` (Gaussian blur + bicubic + additive noise) and
  `jpeg` (bicubic + JPEG compression). The last two give out-of-distribution inputs for
  robustness tests. All are seeded and deterministic.
- **Paired patch sampling**: the degradation is applied to the whole image before cropping so
  LR patches have no border artefacts; HR crops are aligned to the scale factor.
- **Image-level splits** (`split_images`, `audit_image_split`): whole images go to
  train / validation / test, never patches, so no test content leaks into training.
- Joint flip/rotation augmentation that keeps LR/HR pairs aligned.
- Folder listing, loading and a SHA-256 manifest (`visionforge-data manifest`) to pin a
  dataset version.
- A procedural image generator for tests and demos.

## Install and test

```bash
git clone https://github.com/saaki05/project-5 && cd project-5/visionforge-datasets
pip install -e ".[dev]" && pytest
visionforge-data synth --out data/synth --count 12     # smoke-test images only
visionforge-data manifest --root /path/to/DIV2K_train_HR --out manifest.json
```

## Data

Benchmarks commonly used for super-resolution include DIV2K / Flickr2K (training) and Set5,
Set14, BSD100, Urban100 (testing). Each has its own license and download page; check the
terms and download them yourself. They were not reachable from the development environment,
so no experiment in this repository used them.

## License

MIT.
