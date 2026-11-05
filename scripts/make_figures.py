"""Generate every VisionForge figure from visionforge-lab/results/satellite/.

Example crops (``examples/*.png``) are written by ``visionforge-lab study --examples-dir``.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "visionforge-lab" / "results" / "satellite"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
fs.setup()
D = json.load(open(RES / "study.json"))
DEGS = ["bicubic", "blur_noise", "jpeg"]
DEG_LABEL = {"bicubic": "clean bicubic ×4", "blur_noise": "blur + noise (unseen)",
             "jpeg": "JPEG q30 (unseen)"}


def short(name):
    return {"bicubic": "bicubic", "EDSR (L1)": "EDSR (L1)", "EDSR + adversarial": "EDSR + GAN"}.get(
        name, "residual diffusion")


COL = {"bicubic": fs.GRAY, "EDSR (L1)": fs.BLUE, "EDSR + adversarial": fs.VIOLET,
       "residual diffusion": fs.ORANGE}


def key(name):
    return "residual diffusion" if name.startswith("residual") else name


def quality():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    systems = list(D["systems"])
    for ax, deg in zip(axes, DEGS, strict=True):
        for i, n in enumerate(systems):
            q = D["systems"][n]["quality"][deg]["psnr"]
            ax.plot([i, i], [q["ci_low"], q["ci_high"]], color=COL[key(n)], lw=2.6)
            ax.scatter([i], [q["mean"]], color=COL[key(n)], s=60, zorder=3, edgecolor=fs.SURFACE)
            ax.text(i + 0.12, q["mean"], f"{q['mean']:.2f}", ha="left", va="center", fontsize=8.5,
                    color=fs.INK2, bbox={"fc": fs.SURFACE, "ec": "none", "pad": 1.2})
        base = D["systems"]["bicubic"]["quality"][deg]["psnr"]["mean"]
        ax.axhline(base, color=fs.INK2, ls=":", lw=1.1)
        lo = min(D["systems"][n]["quality"][deg]["psnr"]["ci_low"] for n in systems)
        ax.set_ylim(lo - 0.8, max(D["systems"][n]["quality"][deg]["psnr"]["ci_high"]
                                  for n in systems) + 0.8)
        ax.set_xlim(-0.6, len(systems) - 0.2)
        ax.set_xticks(range(len(systems)), [short(n).replace(" ", "\n", 1) for n in systems],
                      fontsize=8.5)
        ax.set_title(DEG_LABEL[deg])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("PSNR (dB), 95% interval over test images")
    fig.suptitle("Learned models win on the clean protocol and lose to bicubic on JPEG inputs",
                 x=0.07, ha="left", y=1.03, fontweight="bold", fontsize=11)
    fs.save(fig, OUT / "quality_by_degradation.png")


def cost_quality():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for n, s in D["systems"].items():
        if s["train_seconds"] is None:
            continue
        d = s["vs_bicubic"]["bicubic"]["psnr"]
        ax.errorbar([s["train_seconds"]], [d["diff"]],
                    yerr=[[d["diff"] - d["ci_low"]], [d["ci_high"] - d["diff"]]],
                    color=COL[key(n)], marker="o", ms=8, capsize=3, lw=1.8)
        ax.annotate(f"{short(n)}\n{s['latency_ms'] if 'latency_ms' in s else s['cost']['latency_ms']:.0f} ms/image",
                    (s["train_seconds"], d["diff"]), xytext=(9, -4), textcoords="offset points",
                    fontsize=8.5, color=fs.INK2)
    ax.set_xscale("log")
    ax.set_xlabel("training time on one CPU thread (seconds, log scale)")
    ax.set_ylabel("PSNR gain over bicubic (dB)")
    ax.set_title("Quality against training cost (clean ×4 protocol)")
    fs.save(fig, OUT / "cost_vs_quality.png")


def ablation():
    base = RES / "ablation_diffusion_variants"
    files = [("noise prediction,\nfixed gain 4", base / "eps_gain4.json"),
             ("noise prediction,\ngain from data", base / "eps_calibrated_gain.json"),
             ("v-prediction,\ngain from data", RES / "earlier_runs" / "small_budget_study.json")]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ref = None
    for i, (lab, f) in enumerate(files):
        d = json.load(open(f))
        sysn = next(n for n in d["systems"] if n.startswith("residual"))
        v = d["systems"][sysn]["quality"]["bicubic"]["psnr"]["mean"]
        ref = d["systems"]["bicubic"]["quality"]["bicubic"]["psnr"]["mean"]
        colr = fs.ORANGE if i < 2 else fs.BLUE
        ax.vlines(i, ref, v, color=colr, lw=2.2)
        ax.scatter([i], [v], color=colr, s=90, zorder=3, edgecolor=fs.SURFACE)
        ax.text(i + 0.1, v, f"{v:.2f} dB", va="center", color=fs.INK2, fontsize=9,
                bbox={"fc": fs.SURFACE, "ec": "none", "pad": 1.2})
    ax.axhline(ref, color=fs.INK2, ls=":", lw=1.2)
    ax.text(-0.45, ref + 0.35, f"bicubic {ref:.2f} dB", ha="left", fontsize=8.5, color=fs.INK2)
    ax.set_xticks(range(3), [x[0] for x in files], fontsize=9)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(18, ref + 3)
    ax.set_ylabel("clean-protocol PSNR (dB)")
    ax.set_title("Diffusion variants on the same split and 4,000-step budget (clean PSNR)")
    ax.grid(axis="x", visible=False)
    fs.save(fig, OUT / "diffusion_ablation.png")


def examples():
    ex = RES / "examples"
    if not ex.exists():
        return
    cols = [("hr", "ground truth"), ("bicubic", "bicubic"), ("edsr_l1", "EDSR (L1)"),
            ("edsr_adversarial", "EDSR + GAN"), ("residual_diffusion", "residual diffusion")]
    rows = [(i, d) for i in range(3) for d in ("bicubic",)]
    rows += [(0, "jpeg")]
    fig, axes = plt.subplots(len(rows), 6, figsize=(12.6, 2.2 * len(rows)))
    for r, (i, deg) in enumerate(rows):
        lr = ex / f"ex{i}_{deg}_lr.png"
        axes[r, 0].imshow(np.asarray(Image.open(lr)), interpolation="nearest")
        axes[r, 0].set_title("input (×4 smaller)" if r == 0 else "input, JPEG q30" if deg == "jpeg"
                             else "", fontsize=9, loc="center", fontweight="normal")
        for c, (stem, ttl) in enumerate(cols, start=1):
            f = ex / (f"ex{i}_hr.png" if stem == "hr" else
                      next((p.name for p in ex.glob(f"ex{i}_{deg}_{stem}*.png")), "missing"))
            axes[r, c].imshow(np.asarray(Image.open(f)), interpolation="nearest") if f.exists() else None
            if r == 0:
                axes[r, c].set_title(ttl, fontsize=9, loc="center", fontweight="normal")
        for c in range(6):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([]); axes[r, c].grid(False)
    fig.suptitle("Three held-out test images (128×128 px crops): clean protocol (top three rows) and a "
                 "JPEG input (bottom row)", x=0.01, ha="left", fontweight="bold", fontsize=11, y=1.0)
    fs.save(fig, OUT / "examples.png")


def architecture():
    fs.pipeline(OUT / "architecture.png", [
        ("datasets", ["bicubic / blur+noise / JPEG", "aligned patch sampling", "image- and area-level splits",
                      "checksum manifests"]),
        ("models", ["EDSR (residual over bicubic)", "PatchGAN fine-tuning", "residual diffusion",
                    "v-prediction, DDIM sampling", "PSNR · SSIM from scratch"]),
        ("lab", ["explicit step / time budgets", "tiled inference", "bootstrap + paired tests",
                 "latency, params, reports"]),
    ], "VisionForge: three packages, one-way dependencies", accent=fs.BLUE, width=10.5)


if __name__ == "__main__":
    quality(); cost_quality(); ablation(); examples(); architecture()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
