"""Image listing, loading, checksum manifest and a procedural test-image generator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(root: str | Path) -> list[Path]:
    paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"no images under {root}")
    return paths


def load_image(path: str | Path) -> np.ndarray:
    """RGB float32 array in [0, 1], shape (H, W, 3)."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_image(img: np.ndarray, path: str | Path) -> None:
    Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(path)


def write_manifest(root: str | Path, out_path: str | Path) -> dict:
    """Record every image's relative path, size and SHA-256 so a dataset can be re-verified."""
    root = Path(root)
    files = []
    for p in list_images(root):
        w, h = Image.open(p).size
        files.append({"path": str(p.relative_to(root)), "width": w, "height": h,
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest = {"schema_version": 1, "num_images": len(files), "files": files}
    Path(out_path).write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_manifest(root: str | Path, manifest_path: str | Path) -> None:
    manifest = json.loads(Path(manifest_path).read_text())
    for f in manifest["files"]:
        digest = hashlib.sha256((Path(root) / f["path"]).read_bytes()).hexdigest()
        if digest != f["sha256"]:
            raise ValueError(f"checksum mismatch for {f['path']}")


def synthetic_image(size: int = 128, seed: int = 0) -> np.ndarray:
    """Smooth colour field with sharp-edged shapes and fine texture (for tests and demos)."""
    rng = np.random.default_rng(seed)
    base = np.stack([gaussian_filter(rng.normal(size=(size, size)), size / 16) for _ in range(3)],
                    axis=-1)
    base = (base - base.min()) / (np.ptp(base) + 1e-8)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(6):
        cy, cx, r = rng.integers(0, size, 2).tolist() + [int(rng.integers(size // 12, size // 4))]
        shape = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        base[shape] = rng.random(3)
    stripes = 0.08 * np.sin(xx / 2.0 + rng.random() * 6)[..., None]
    return np.clip(base + stripes + rng.normal(scale=0.01, size=base.shape), 0, 1).astype(
        np.float32)
