import json

import numpy as np
import pytest
import torch
from visionforge_data import bicubic_down, synthetic_image

from visionforge_lab import benchmark as B
from visionforge_lab.cli import main, render
from visionforge_lab.study import run_study
from visionforge_lab.tiling import tile_apply


def nearest_up(scale):
    return lambda lr: np.kron(lr, np.ones((scale, scale, 1), dtype=lr.dtype))


def test_tile_apply_is_exact_for_translation_invariant_fn():
    rng = np.random.default_rng(0)
    lr = rng.random((50, 37, 3)).astype(np.float32)
    fn = nearest_up(3)
    assert np.allclose(tile_apply(fn, lr, 3, tile=16, overlap=4), fn(lr), atol=1e-6)
    small = rng.random((10, 12, 3)).astype(np.float32)
    assert np.array_equal(tile_apply(fn, small, 3, tile=16), fn(small))


def test_oracle_system_scores_perfectly_and_bicubic_does_not():
    imgs = [synthetic_image(64, s) for s in range(4)]
    oracle_queue = iter(imgs)
    # an oracle that returns the true HR image regardless of its input
    oracle = lambda lr: next(oracle_queue)  # noqa: E731
    per = B.evaluate(oracle, imgs, 4)
    assert (per["ssim"] > 0.9999).all() and (per["psnr"] > 80).all()
    bic = B.evaluate(B.bicubic_system(4), imgs, 4)
    assert (bic["psnr"] < 60).all() and bic["psnr"].mean() > 15


def test_blur_noise_degradation_lowers_quality():
    imgs = [synthetic_image(64, s) for s in range(4)]
    clean = B.evaluate(B.bicubic_system(4), imgs, 4, "bicubic")["psnr"].mean()
    noisy = B.evaluate(B.bicubic_system(4), imgs, 4, "blur_noise")["psnr"].mean()
    assert noisy < clean


def test_statistics_helpers():
    rng = np.random.default_rng(0)
    a = rng.normal(30, 1, 40)
    b = a - 0.5 + rng.normal(0, 0.1, 40)
    res = B.paired_bootstrap(a, b)
    assert res["diff"] > 0.3 and res["ci_low"] > 0 and res["p_value"] < 0.01
    mean, lo, hi = B.bootstrap_ci(a)
    assert lo <= mean <= hi
    with pytest.raises(ValueError):
        B.paired_bootstrap(a, b[:-1])


def test_measure_cost_reports_latency():
    lr = bicubic_down(synthetic_image(64, 0), 4)
    cost = B.measure_cost(B.bicubic_system(4), lr, torch.device("cpu"), repeats=2)
    assert cost["latency_ms"] > 0 and cost["throughput_mpix_per_s"] > 0


def test_study_end_to_end_and_report(tmp_path, capsys):
    images = [synthetic_image(96, s) for s in range(8)]
    out = tmp_path / "res.json"
    res = run_study(images, scale=2, steps=3, patch=32, batch=2, bank_size=16, features=8,
                    blocks=1, diffusion_timesteps=20, diffusion_base=8,
                    diffusion_steps_sample=2, device=torch.device("cpu"), out_path=out, tile=48)
    names = set(res["systems"])
    assert {"bicubic", "EDSR (L1)", "EDSR + adversarial"} <= names
    assert any(n.startswith("residual diffusion") for n in names)
    edsr = res["systems"]["EDSR (L1)"]
    assert edsr["train_steps"] == 3 and edsr["parameters"] > 0
    assert "vs_bicubic" in edsr and set(edsr["quality"]) == {"bicubic", "blur_noise", "jpeg"}
    assert res["n_train_images"] + res["n_val_images"] + res["n_test_images"] == 8
    text = render(json.loads(out.read_text()))
    assert "Degradation: jpeg" in text and "| bicubic |" in text
    assert main(["report", "--results", str(out)]) == 0
    assert "super-resolution" in capsys.readouterr().out
