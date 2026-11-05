import numpy as np
import pytest

from visionforge_data import (
    DEGRADATIONS,
    audit_image_split,
    augment_pairs,
    bicubic_down,
    bicubic_up,
    crop_to_multiple,
    degrade,
    list_images,
    load_image,
    sample_pairs,
    save_image,
    split_images,
    synthetic_image,
    verify_manifest,
)
from visionforge_data.cli import main


def test_synthetic_image_deterministic_and_in_range():
    a, b = synthetic_image(64, 3), synthetic_image(64, 3)
    assert a.shape == (64, 64, 3) and a.dtype == np.float32
    assert np.array_equal(a, b) and a.min() >= 0 and a.max() <= 1
    assert not np.array_equal(a, synthetic_image(64, 4))


def test_down_up_shapes_and_constant_image_is_preserved():
    flat = np.full((32, 48, 3), 0.4, dtype=np.float32)
    lr = bicubic_down(flat, 4)
    assert lr.shape == (8, 12, 3) and np.allclose(lr, 0.4, atol=1e-5)
    assert bicubic_up(lr, 4).shape == (32, 48, 3)
    assert crop_to_multiple(np.zeros((33, 50, 3)), 4).shape == (32, 48, 3)


def test_bicubic_up_down_is_close_for_smooth_images():
    img = synthetic_image(64, 1)
    smooth = np.stack([np.linspace(0, 1, 64)[None, :].repeat(64, 0)] * 3, axis=-1).astype(
        np.float32)
    rec = bicubic_up(bicubic_down(smooth, 4), 4)
    assert np.abs(rec - smooth).mean() < 0.01
    assert np.abs(bicubic_up(bicubic_down(img, 4), 4) - img).mean() > 0.0  # detail is lost


@pytest.mark.parametrize("kind", DEGRADATIONS)
def test_every_degradation_gives_right_shape_range(kind):
    hr = synthetic_image(64, 2)
    lr = degrade(hr, 4, kind)
    assert lr.shape == (16, 16, 3) and lr.min() >= 0 and lr.max() <= 1
    assert np.array_equal(lr, degrade(hr, 4, kind))  # seeded
    with pytest.raises(ValueError):
        degrade(hr, 4, "nope")


def test_noise_degradation_differs_from_clean():
    hr = synthetic_image(64, 2)
    assert np.abs(degrade(hr, 4, "blur_noise") - degrade(hr, 4, "bicubic")).mean() > 0.01


def test_image_split_is_disjoint_and_audited():
    tr, va, te = split_images(20, seed=1)
    assert audit_image_split(tr, va, te) and len(tr) + len(va) + len(te) == 20
    assert not audit_image_split(tr, va, np.concatenate([te, tr[:1]]))
    with pytest.raises(ValueError):
        split_images(2)


def test_sample_pairs_alignment():
    imgs = [synthetic_image(96, s) for s in range(3)]
    hr, lr = sample_pairs(imgs, n=8, patch=32, scale=4, seed=0)
    assert hr.shape == (8, 32, 32, 3) and lr.shape == (8, 8, 8, 3)
    # LR patch must equal the bicubic downscale of the HR patch up to border effects
    inner = np.abs(bicubic_down(hr[0], 4)[2:-2, 2:-2] - lr[0][2:-2, 2:-2]).mean()
    assert inner < 0.05
    with pytest.raises(ValueError):
        sample_pairs(imgs, 1, patch=30, scale=4)
    with pytest.raises(ValueError):
        sample_pairs(imgs, 1, patch=128, scale=4)


def test_augment_keeps_pairs_aligned():
    imgs = [synthetic_image(96, 5)]
    hr, lr = sample_pairs(imgs, 4, 32, 4, seed=1)
    for s in range(6):
        h2, l2 = augment_pairs(hr, lr, np.random.default_rng(s))
        assert h2.shape == hr.shape and l2.shape == lr.shape
        inner = np.abs(bicubic_down(h2[0], 4)[2:-2, 2:-2] - l2[0][2:-2, 2:-2]).mean()
        assert inner < 0.05


def test_manifest_and_cli(tmp_path, capsys):
    folder = tmp_path / "imgs"
    assert main(["synth", "--out", str(folder), "--count", "4", "--size", "64"]) == 0
    assert len(list_images(folder)) == 4
    img = load_image(list_images(folder)[0])
    assert img.shape == (64, 64, 3)
    mf = tmp_path / "m.json"
    assert main(["manifest", "--root", str(folder), "--out", str(mf)]) == 0
    verify_manifest(folder, mf)
    save_image(np.zeros((64, 64, 3)), list_images(folder)[0])
    with pytest.raises(ValueError, match="checksum"):
        verify_manifest(folder, mf)
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / "empty_dir_that_does_not_exist")


def test_grouped_split_keeps_groups_together():
    groups = np.repeat(np.arange(10), 3)  # 10 scenes x 3 acquisitions
    tr, va, te = split_images(30, seed=2, groups=groups)
    assert audit_image_split(tr, va, te, groups) and len(tr) + len(va) + len(te) == 30
    for part in (tr, va, te):
        assert len(part) % 3 == 0  # whole scenes only
    # an image-level split of the same data would typically break scenes apart
    t2, v2, e2 = split_images(30, seed=2)
    assert not audit_image_split(t2, v2, e2, groups)
    with pytest.raises(ValueError):
        split_images(30, groups=np.zeros(30))
    with pytest.raises(ValueError):
        split_images(30, groups=np.arange(5))


def test_split_always_leaves_every_part_non_empty():
    for n in (3, 4, 5, 7, 10):
        parts = split_images(n, seed=n)
        assert all(len(p) >= 1 for p in parts) and sum(len(p) for p in parts) == n
