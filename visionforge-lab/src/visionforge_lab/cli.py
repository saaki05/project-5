"""``visionforge-lab study --images DIR`` and ``visionforge-lab report --results FILE``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visionforge_lab.study import load_images, run_study


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
                  "train s | latency ms |", "|---|---|---|---|---|---|---|"]
        for name, s in res["systems"].items():
            q = s["quality"][deg]
            vs = s.get("vs_bicubic", {}).get(deg)
            delta = "—"
            if vs:
                delta = f"{vs['psnr']['diff']:+.2f} (p={vs['psnr']['p_value']:.3f})"
            params = "—" if s["parameters"] is None else str(s["parameters"])
            train_s = "—" if s["train_seconds"] is None else f"{s['train_seconds']:.0f}"
            lines.append(
                f"| {name} | {q['psnr']['mean']:.2f} [{q['psnr']['ci_low']:.2f}, "
                f"{q['psnr']['ci_high']:.2f}] | {q['ssim']['mean']:.4f} "
                f"[{q['ssim']['ci_low']:.4f}, {q['ssim']['ci_high']:.4f}] | {delta} | "
                f"{params} | {train_s} | {s['cost']['latency_ms']:.0f} |")
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
    r = sub.add_parser("report")
    r.add_argument("--results", required=True)
    a = ap.parse_args(argv)
    if a.command == "study":
        res = run_study(load_images(a.images, a.limit), a.scale, a.steps, a.max_seconds, a.patch,
                        a.batch, seed=a.seed, out_path=a.out)
        print(render(res))
    else:
        print(render(json.loads(Path(a.results).read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
