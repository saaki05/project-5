"""``visionforge-lab study --images DIR`` and ``visionforge-lab report --results FILE``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visionforge_lab.study import load_images, load_images_grouped, run_study


def render(res: dict) -> str:
    lines = [f"# VisionForge ×{res['scale']} super-resolution (generated from result JSON)", "",
             f"Device: {res['device']}. Images: train {res['n_train_images']}, "
             f"val {res['n_val_images']}, test {res['n_test_images']}. Budget: "
             f"{res['budget']['steps']} steps"
             + (f" / {res['budget']['max_seconds']}s" if res["budget"]["max_seconds"] else "")
             + f"; patch {res['patch']}, batch {res['batch']}.", ""]
    degs = next(iter(res["systems"].values()))["quality"].keys()
    for deg in degs:
        lines += [f"## Degradation: {deg}", "",
                  "| system | PSNR [95% CI] | SSIM [95% CI] | Δ PSNR vs bicubic (p) | params | "
                  "train steps | train s | latency ms |", "|---|---|---|---|---|---|---|---|"]
        for name, s in res["systems"].items():
            q = s["quality"][deg]
            vs = s.get("vs_bicubic", {}).get(deg)
            delta = "—"
            if vs:
                delta = f"{vs['psnr']['diff']:+.2f} (p={vs['psnr']['p_value']:.3f})"
            params = "—" if s["parameters"] is None else str(s["parameters"])
            train_s = "—" if s["train_seconds"] is None else f"{s['train_seconds']:.0f}"
            steps_s = "—" if s["train_steps"] is None else str(s["train_steps"])
            lines.append(
                f"| {name} | {q['psnr']['mean']:.2f} [{q['psnr']['ci_low']:.2f}, "
                f"{q['psnr']['ci_high']:.2f}] | {q['ssim']['mean']:.4f} "
                f"[{q['ssim']['ci_low']:.4f}, {q['ssim']['ci_high']:.4f}] | {delta} | "
                f"{params} | {steps_s} | {train_s} | {s['cost']['latency_ms']:.0f} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="visionforge-lab")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("study")
    s.add_argument("--images", required=True, help="folder of high-resolution images")
    s.add_argument("--out", required=True)
    s.add_argument("--scale", type=int, default=4)
    s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--max-seconds", type=float, default=None)
    s.add_argument("--patch", type=int, default=64)
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--limit", type=int, default=None, help="use only the first N images")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--group-level", type=int, default=None,
                   help="group images by the directory this many levels up (e.g. 2 for "
                        "<scene>/images/<file>) and split by group")
    s.add_argument("--min-valid", type=float, default=0.98)
    s.add_argument("--max-side", type=int, default=None, help="centre-crop images to this size")
    s.add_argument("--features", type=int, default=48)
    s.add_argument("--blocks", type=int, default=8)
    s.add_argument("--bank", type=int, default=2000, help="size of the training patch bank")
    s.add_argument("--diffusion-base", type=int, default=32)
    s.add_argument("--diffusion-steps", type=int, default=None,
                   help="training steps for the diffusion model (default: same as --steps)")
    s.add_argument("--examples-dir", default=None, help="save example output crops here")
    s.add_argument("--n-examples", type=int, default=3)
    s.add_argument("--save-models-dir", default=None)
    s.add_argument("--sample-steps", type=int, default=10, help="DDIM sampling steps")
    s.add_argument("--test-max-side", type=int, default=None,
                   help="centre-crop test images to this size (bounds diffusion sampling cost)")
    r = sub.add_parser("report")
    r.add_argument("--results", required=True)
    a = ap.parse_args(argv)
    if a.command == "study":
        groups = None
        if a.group_level:
            images, groups = load_images_grouped(a.images, a.group_level, a.min_valid, a.max_side,
                                                 a.limit)
        else:
            images = load_images(a.images, a.limit)
        res = run_study(images, a.scale, a.steps, a.max_seconds, a.patch, a.batch,
                        bank_size=a.bank, seed=a.seed, out_path=a.out, features=a.features,
                        blocks=a.blocks, groups=groups, diffusion_base=a.diffusion_base,
                        diffusion_steps_sample=a.sample_steps, test_max_side=a.test_max_side,
                        diffusion_train_steps=a.diffusion_steps, examples_dir=a.examples_dir,
                        n_examples=a.n_examples, save_models_dir=a.save_models_dir)
        print(render(res))
    else:
        print(render(json.loads(Path(a.results).read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
