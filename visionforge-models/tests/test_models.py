import numpy as np
import pytest
import torch
from torch.nn import functional as F

from visionforge_models.diffusion import (
    CondUNet,
    ResidualDiffusion,
    make_schedule,
    timestep_embedding,
)
from visionforge_models.edsr import EDSR, Upsampler, count_parameters
from visionforge_models.gan import PatchDiscriminator, discriminator_loss, generator_loss
from visionforge_models.metrics import psnr, rgb_to_y, ssim
from visionforge_models.train import EMA, train_diffusion, train_gan, train_regression

CPU = torch.device("cpu")


def toy_batch(step, n=4, size=16, scale=2):
    g = torch.Generator().manual_seed(step)
    xx = torch.linspace(0, 1, size * scale)
    base = (xx[None, :] + xx[:, None]) / 2
    hr = base[None, None].repeat(n, 3, 1, 1) * (0.5 + 0.5 * torch.rand(n, 1, 1, 1, generator=g))
    lr = F.interpolate(hr, scale_factor=1 / scale, mode="bicubic", align_corners=False)
    return hr.clamp(0, 1), lr.clamp(0, 1)


def test_psnr_known_values_and_ssim_properties():
    a = np.full((32, 32, 3), 0.5, dtype=np.float32)
    b = a + 0.1
    assert psnr(a, b, y_channel=False) == pytest.approx(20.0, abs=1e-4)  # mse = 0.01
    assert psnr(a, a) == float("inf")
    assert ssim(a, a) == pytest.approx(1.0, abs=1e-6)
    # constant images: ssim = (2ab + C1) / (a^2 + b^2 + C1)
    c1 = 0.01**2
    expect = (2 * 0.5 * 0.6 + c1) / (0.5**2 + 0.6**2 + c1)
    assert ssim(a, b, y_channel=False) == pytest.approx(expect, abs=1e-5)
    rng = np.random.default_rng(0)
    img = rng.random((48, 48, 3)).astype(np.float32)
    noisy = np.clip(img + rng.normal(scale=0.1, size=img.shape), 0, 1).astype(np.float32)
    noisier = np.clip(img + rng.normal(scale=0.3, size=img.shape), 0, 1).astype(np.float32)
    assert ssim(img, noisy) > ssim(img, noisier)
    assert psnr(img, noisy) > psnr(img, noisier)
    assert psnr(img, noisy, border=4) != psnr(img, noisy)
    assert rgb_to_y(np.ones((2, 2, 3)))[0, 0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        psnr(a, a[:16])


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_edsr_output_shape_and_residual_over_bicubic(scale):
    net = EDSR(scale=scale, features=8, blocks=2)
    lr = torch.rand(2, 3, 10, 12)
    out = net(lr)
    assert out.shape == (2, 3, 10 * scale, 12 * scale)
    out.mean().backward()
    assert all(p.grad is not None for p in net.parameters())
    # zeroing the tail makes the network exactly bicubic upsampling
    with torch.no_grad():
        net.tail.weight.zero_()
        net.tail.bias.zero_()
    base = F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
    assert torch.allclose(net(lr), base, atol=1e-6)
    with pytest.raises(ValueError):
        Upsampler(5, 8)
    assert count_parameters(net) > 0


def test_regression_training_reduces_loss():
    torch.manual_seed(0)
    net = EDSR(scale=2, features=8, blocks=2)
    hist = train_regression(net, toy_batch, steps=60, lr=2e-3, device=CPU)
    assert hist["steps"] == 60
    assert np.mean(hist["losses"][-10:]) < np.mean(hist["losses"][:10])


def test_time_budget_stops_training():
    net = EDSR(scale=2, features=8, blocks=2)
    hist = train_regression(net, toy_batch, steps=10_000, device=CPU, max_seconds=0.5)
    assert 1 <= hist["steps"] < 10_000 and hist["seconds"] < 5


def test_gan_losses_and_training_step():
    d = PatchDiscriminator(base=8, levels=2)
    real, fake = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
    out = d(real)
    assert out.shape[1] == 1 and out.shape[0] == 2
    # a perfect discriminator has lower loss than a confused one
    sharp = discriminator_loss(torch.full((4,), 8.0), torch.full((4,), -8.0))
    confused = discriminator_loss(torch.zeros(4), torch.zeros(4))
    assert sharp < confused
    assert generator_loss(real, real, torch.zeros(4), 0.0).item() == 0.0
    g = EDSR(scale=2, features=8, blocks=1)
    hist = train_gan(g, toy_batch, steps=5, device=CPU, discriminator=d)
    assert hist["steps"] == 5 and all(np.isfinite(hist["g"])) and all(np.isfinite(hist["d"]))
    del fake


def test_schedule_properties():
    for kind in ("cosine", "linear"):
        s = make_schedule(200, kind)
        assert (s["abar"][1:] < s["abar"][:-1]).all()
        assert 0 < s["abar"][-1] < 0.05 and s["abar"][0] > 0.9
        assert (s["betas"] > 0).all() and (s["betas"] < 1).all()
    with pytest.raises(ValueError):
        make_schedule(10, "nope")
    assert timestep_embedding(torch.arange(3), 16).shape == (3, 16)


def test_unet_shapes_and_divisibility():
    net = CondUNet(base=8, levels=2)
    x, c = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)
    assert net(x, c, torch.tensor([3, 40])).shape == x.shape
    with pytest.raises(ValueError, match="divisible"):
        net(torch.randn(1, 3, 18, 16), torch.randn(1, 3, 18, 16), torch.tensor([1]))


def test_forward_process_matches_closed_form():
    diff = ResidualDiffusion(CondUNet(base=8, levels=1), timesteps=50)
    hr, cond = torch.rand(8, 3, 8, 8), torch.rand(8, 3, 8, 8)
    torch.manual_seed(0)
    assert diff.loss(hr, cond).item() > 0
    # at t = T-1 the signal is almost gone: x_t ~ pure noise
    ab = diff.abar[-1].item()
    assert ab < 0.05


def test_diffusion_trains_samples_and_is_reproducible():
    torch.manual_seed(0)
    diff = ResidualDiffusion(CondUNet(base=8, levels=1), timesteps=50)
    hist, ema = train_diffusion(diff, toy_batch, steps=40, lr=2e-3, device=CPU)
    assert np.mean(hist["losses"][-10:]) < np.mean(hist["losses"][:10])
    hr, lr = toy_batch(0)
    cond = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
    a = ema.sample(cond, steps=4, seed=1)
    b = ema.sample(cond, steps=4, seed=1)
    c = ema.sample(cond, steps=4, seed=2)
    assert a.shape == hr.shape and a.min() >= 0 and a.max() <= 1
    assert torch.allclose(a, b) and not torch.allclose(a, c)


def test_ema_tracks_weights():
    net = torch.nn.Linear(2, 2)
    ema = EMA(net, 0.5)
    with torch.no_grad():
        net.weight.add_(1.0)
    ema.update(net)
    assert not torch.allclose(ema.model.weight, net.weight)
    for _ in range(40):
        ema.update(net)
    assert torch.allclose(ema.model.weight, net.weight, atol=1e-4)


def test_gain_calibration_matches_residual_scale_and_helps_small_residuals():
    torch.manual_seed(0)
    diff = ResidualDiffusion(CondUNet(base=8, levels=1), timesteps=30, residual_gain=4.0)
    hr = torch.rand(16, 3, 8, 8)
    cond = hr + 0.02 * torch.randn_like(hr)  # a small residual, like real imagery
    gain = diff.calibrate_gain(hr, cond)
    assert gain == pytest.approx(1 / (hr - cond).std().item(), rel=1e-5) and gain > 30
    assert ResidualDiffusion(CondUNet(base=8, levels=1)).calibrate_gain(hr, hr) == 200.0  # capped

    def batch(step):
        g = torch.Generator().manual_seed(abs(step))
        h = torch.rand(8, 3, 16, 16, generator=g)
        lr = F.interpolate(h, scale_factor=0.5, mode="bicubic", align_corners=False)
        return h, lr

    d2 = ResidualDiffusion(CondUNet(base=8, levels=1), timesteps=30)
    _, ema = train_diffusion(d2, batch, steps=5, device=CPU)
    assert ema.gain == d2.gain and d2.gain != 4.0  # calibrated, and the EMA copy shares it


def test_v_and_eps_parameterisations_are_consistent():
    diff_v = ResidualDiffusion(CondUNet(base=8, levels=1), timesteps=40, prediction="v")
    diff_e = ResidualDiffusion(CondUNet(base=8, levels=1), timesteps=40, prediction="eps")
    x0, eps = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8)
    for t in (0, 20, 39):
        ab = diff_v.abar[t]
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        v = ab.sqrt() * eps - (1 - ab).sqrt() * x0
        rx0, reps = diff_v._x0_eps(x_t, v, ab)  # exact inversion for the true v
        assert torch.allclose(rx0, x0, atol=1e-4) and torch.allclose(reps, eps, atol=1e-4)
        ex0, eeps = diff_e._x0_eps(x_t, eps, ab)
        assert torch.allclose(eeps, eps) and torch.allclose(ex0, x0, atol=1e-2 / ab.sqrt().item())
    with pytest.raises(ValueError):
        ResidualDiffusion(prediction="x0")


def test_v_prediction_error_is_not_amplified_at_the_last_step():
    ab = make_schedule(1000, "cosine")["abar"][-1]
    assert ab.sqrt() < 0.01  # near-zero terminal signal: eps -> x0 divides by this
    x_t = torch.randn(1000)
    err = 0.01 * torch.randn(1000)  # a small error in the network output
    diff = ResidualDiffusion(CondUNet(base=8, levels=1), timesteps=1000, prediction="eps")
    x0_eps = diff._x0_eps(x_t, x_t * (1 - ab).sqrt().item() + err, ab)[0]
    diff_v = ResidualDiffusion(CondUNet(base=8, levels=1), timesteps=1000, prediction="v")
    v_true = ab.sqrt() * (x_t - ab.sqrt() * 0) - (1 - ab).sqrt() * 0  # x0 = 0 case
    x0_v = diff_v._x0_eps(x_t, v_true + err, ab)[0]
    assert (x0_v - 0).abs().mean() < 0.1 * (x0_eps - 0).abs().mean()
