"""``visionforge-data manifest --root DIR --out manifest.json`` and ``synth --out DIR``."""

from __future__ import annotations

import argparse
from pathlib import Path

from visionforge_data.images import save_image, synthetic_image, write_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="visionforge-data")
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser("manifest", help="checksum manifest for an image folder")
    m.add_argument("--root", required=True)
    m.add_argument("--out", required=True)
    s = sub.add_parser("synth", help="write procedural test images (smoke tests only)")
    s.add_argument("--out", required=True)
    s.add_argument("--count", type=int, default=12)
    s.add_argument("--size", type=int, default=128)
    a = ap.parse_args(argv)
    if a.command == "manifest":
        info = write_manifest(a.root, a.out)
        print(f"{info['num_images']} images -> {a.out}")
    else:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for i in range(a.count):
            save_image(synthetic_image(a.size, seed=i), out / f"synth_{i:03d}.png")
        print(f"wrote {a.count} images to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
