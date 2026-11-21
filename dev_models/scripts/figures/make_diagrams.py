"""Architecture diagrams for Chapter 3 (BiFPN-SIRAN paper subset), drawn from the code.

Every block, size and connection below is read off the implementation named in
the figure's function, so the diagram shows what was actually run:
  fig3_2_siran       BiFPN-SIRAN (src/models/shadow.py: BiFPN_SIRAN_ShadowDetector)
  fig3_obj*          architecture of the shadow-detection study (Objective 2)

Usage:
    python scripts/figures/make_diagrams.py [name ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"

INK, INK2 = "#0b0b0b", "#52514e"
# entity colours shared with make_data_figures.py: pose blue, gait orange, shadow aqua
POSE, GAIT, SHADOW, TRACK, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#52514e"
FILL = {POSE: "#e3eefb", GAIT: "#fde9e0", SHADOW: "#ddf4ea", TRACK: "#e9e6fa", NEUTRAL: "#f1f0ec"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white"})


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, text, col=NEUTRAL, dashed=False, bold=False, fs=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.6",
                                facecolor=FILL[col], edgecolor=col, linewidth=1.3,
                                linestyle=(0, (4, 2)) if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)
    return (x, y, w, h)


box = _box


def arrow(ax, p, q, col=INK2, dashed=False, text=None, toff=(0, 1.2), fs=7, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=col, linewidth=1.1,
                                 linestyle=(0, (4, 2)) if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + toff[0], (p[1] + q[1]) / 2 + toff[1], text, ha="center", va="bottom",
                fontsize=fs, color=INK2)


def R(b):  # right-middle
    return (b[0] + b[2], b[1] + b[3] / 2)


def L(b):
    return (b[0], b[1] + b[3] / 2)


def T(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def B(b):
    return (b[0] + b[2] / 2, b[1])


def tag(ax, b, t):
    ax.text(b[0] + b[2], b[1] + b[3] + 0.8, t, ha="right", va="bottom", fontsize=6.5, fontweight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.25", facecolor=INK, edgecolor="none"))


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png"); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig)
    print("written:", OUT / f"{name}.png")


# ------------------------------------------------------------------ Figure 3.1


# ------------------------------------------------------------------ Figure 3.4

def fig3_2_siran():
    fig, ax = canvas(6.3, 4.4, 100, 68)
    ax.text(1, 66.5, "Input: 256 × 256 RGB, ImageNet-normalised", fontsize=6.5, color=INK2, va="top")
    Y = [52 - 11 * i for i in range(5)]
    enc = [box(ax, 1, Y[i], 16, 8, f"EfficientNet-B5\nstage {i + 1} → e{i + 1}") for i in range(5)]
    for i in range(4):
        arrow(ax, B(enc[i]), T(enc[i + 1]))
    lat = [box(ax, 21, Y[i], 12, 8, f"1×1 conv\np{i + 1} (512)") for i in range(5)]
    for i in range(5):
        arrow(ax, R(enc[i]), L(lat[i]))
    names = {4: "BiFPN layer 1", 3: "BiFPN layer 2", 2: "BiFPN layer 3", 1: "BiFPN layer 1\n(weights reused)"}
    bif = {}
    for lvl in (4, 3, 2, 1):
        bif[lvl] = box(ax, 37, Y[lvl - 1], 17, 8, f"{names[lvl]}\n→ p{lvl}_td")
        arrow(ax, R(lat[lvl - 1]), L(bif[lvl]))
    arrow(ax, R(lat[4]), B(bif[4]), rad=0.3)
    ax.text(47, 16.2, "p5 ↑2×", fontsize=6, color=INK2)
    for lvl in (3, 2, 1):
        arrow(ax, T(bif[lvl + 1]), B(bif[lvl]))
        ax.text(46.5, Y[lvl - 1] - 2.0, "↑2×", fontsize=6, color=INK2, va="center")
    glob = box(ax, 37, 2, 17, 8, "3×3 conv on p5\n→ global map g", SHADOW)
    arrow(ax, (33, 10), L(glob))
    sir = {}
    for lvl in (4, 3, 2, 1):
        sir[lvl] = box(ax, 59, Y[lvl - 1], 23, 8, f"SIRAN block {lvl}\nconv[p{lvl}_td ; proj(e{lvl})]\n× (1 − σ(m_prev))", SHADOW)
        arrow(ax, R(bif[lvl]), L(sir[lvl]))
    arrow(ax, R(glob), B(sir[4]), col=SHADOW, rad=0.3)
    ax.text(83, 9, "m_prev = g", fontsize=6, color=INK2)
    for lvl in (4, 3, 2):
        arrow(ax, T(sir[lvl]), B(sir[lvl - 1]), col=SHADOW)
        ax.text(71.5, Y[lvl - 2] - 2.0, f"m{lvl}", fontsize=6, color=INK2, va="center")
    out = box(ax, 86, Y[0], 13, 8, "1×1 conv\n→ mask\nlogits", SHADOW)
    arrow(ax, R(sir[1]), L(out), col=SHADOW)
    ax.text(59, 63, "proj(e_k): 1×1 conv of the raw backbone feature e_k, injected before reverse-attention gating",
            fontsize=6, color=INK2)
    save(fig, "fig3_2_siran")


# ------------------------------------------------------------------ Figure 3.5


# ------------------------------------------------------------------ Figure 3.7


# ------------------------------------------------------- Objective architectures (Figures 3.2, 3.3, 3.6, 3.8)
# One figure per objective, each drawn in the form that suits it: Objective 1 a pipeline with the association
# geometry and the location partition, Objective 2 a training lane above an inference lane, Objective 4 the tracker
# comparison on shared detections, Objective 5 an evaluation matrix. Objective 3 is covered by Figure 3.5 (fusion
# network). Every entry restates Chapter 3 (thesis/ch3_revised.txt); no Chapter 4 result appears. Small drawings
# (masks, profiles, stripe maps) are schematic.

import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402


def _title(ax, x, y, text, fs=6.6):
    ax.text(x, y, text, fontsize=fs, fontweight="bold", color=INK, va="bottom")


def _panel(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                facecolor="#fafaf8", edgecolor="#c9c8c3", linewidth=0.8))


def fig3_obj2_shadow():
    sbox = lambda *a, **k: _box(*a, fs=k.pop("fs", 5.6), **k)
    fig, ax = canvas(6.3, 4.6, 100, 72)
    _panel(ax, 0.5, 37, 99, 34)
    _title(ax, 2, 67.3, "Training and checkpoint selection (Sec 3.3.7)")
    names = ["UNet++ (baseline)", "BiFPN-RARM", "BiFPN-RAN (iteration 2)", "BiFPN-SIRAN"]
    for i, n in enumerate(names):
        box(ax, 2, 60 - i * 5.6, 19, 4.6, n, SHADOW if "BiFPN" in n else NEUTRAL, fs=5.2)
    ax.text(11.5, 39.2, "4 candidates × 5 seeds", fontsize=5.2, ha="center", color=INK2)
    p1 = sbox(ax, 26, 55, 17, 10, "Phase 1: SBU\n4,085 images\n50 epochs", SHADOW)
    p2 = sbox(ax, 26, 41, 17, 10, "Phase 2: ISTD\n1,210 images\n≤ 30 epochs", SHADOW)
    ax.plot([22.3, 22.3], [43.2, 64.6], color=INK2, linewidth=1.1)
    arrow(ax, (22.3, 60), L(p1)); arrow(ax, B(p1), T(p2))
    held = sbox(ax, 48, 55, 18, 10, "ISTD held-out\n9 training scenes\n120 images")
    sel = sbox(ax, 48, 41, 18, 10, "Selected weights:\nbest validation IoU,\nearly stopping", SHADOW)
    arrow(ax, R(p2), L(sel)); arrow(ax, B(held), T(sel), dashed=True)
    test = sbox(ax, 72, 55, 26, 10, "ISTD test (540, 46 scenes),\nSBU test (638 images):\nread only at final evaluation", dashed=True)
    ev = sbox(ax, 72, 41, 26, 10, "BER (Eq 3.13), IoU, threshold\nsweep, boundary bands;\npaired bootstrap")
    arrow(ax, R(sel), L(ev)); arrow(ax, B(test), T(ev), dashed=True)
    # inference lane
    _panel(ax, 0.5, 1, 99, 33)
    ax.text(98, 30.3, "Shadow signature at inference (Secs 3.3.8, 3.4)", fontsize=6.6, fontweight="bold", ha="right",
            va="bottom")
    ax.add_patch(Rectangle((2, 6), 15, 21, facecolor="white", edgecolor=INK2, linewidth=0.8))
    ax.add_patch(Rectangle((7, 15), 4, 9, facecolor=FILL[POSE], edgecolor=POSE, linewidth=1.0))
    ax.add_patch(Rectangle((6.2, 10.5), 5.6, 4.5, fill=False, edgecolor=SHADOW, linewidth=1.0, linestyle=(0, (3, 2))))
    ax.text(9.5, 5.2, "detection box and\nROI below it", fontsize=5.0, ha="center", va="top", color=INK2)
    net = sbox(ax, 21, 11, 13, 11, "BiFPN-SIRAN\n(selected\nweights)", SHADOW)
    arrow(ax, (17, 16.5), L(net), col=SHADOW)
    arrow(ax, (57, 41), (27.5, 22), dashed=True, rad=0.2)
    # mask
    ax.add_patch(Rectangle((38, 6), 10, 21, facecolor="white", edgecolor=INK2, linewidth=0.8))
    t = np.linspace(0, 1, 40)
    left = 43 - 1.2 - 1.6 * np.sin(np.pi * t) ** 2
    right = 43 + 1.2 + 2.4 * np.sin(np.pi * t * 1.3) ** 2
    yy = 8 + 17 * t
    ax.add_patch(Polygon(np.c_[np.r_[left, right[::-1]], np.r_[yy, yy[::-1]]], closed=True, facecolor=INK,
                         edgecolor="none"))
    ax.text(43, 5.2, "shadow mask\n(≥ 100 px)", fontsize=5.0, ha="center", va="top", color=INK2)
    arrow(ax, R(net), (38, 16.5), col=SHADOW)
    # width profile
    ax.add_patch(Rectangle((52, 6), 10, 21, facecolor="white", edgecolor=INK2, linewidth=0.8))
    w = (right - left); w = (w - w.min()) / (w.max() - w.min())
    ax.plot(53 + 8 * w, yy, color=SHADOW, linewidth=1.2)
    ax.text(57, 5.2, "width profile\ns(r), Eq 3.7", fontsize=5.0, ha="center", va="top", color=INK2)
    arrow(ax, (48, 16.5), (52, 16.5), col=SHADOW)
    # stripe map
    r, c = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 64), indexing="ij")
    stripe = np.sin(np.pi * r) ** 2 * (0.6 + 0.4 * np.sin(2 * np.pi * (3 * c + r)))
    ax.imshow(stripe, extent=(66, 80, 6, 27), origin="lower", cmap="Greens", aspect="auto", zorder=2)
    ax.add_patch(Rectangle((66, 6), 14, 21, fill=False, edgecolor=INK2, linewidth=0.8, zorder=3))
    ax.text(73, 5.2, "stripe map\n128 × 64 frames", fontsize=5.0, ha="center", va="top", color=INK2)
    arrow(ax, (62, 16.5), (66, 16.5), col=SHADOW)
    enc = sbox(ax, 84, 9, 14.5, 15, "Stripe encoder\n3 conv blocks\n→ 256-D\n(pre-trained\non CASIA-B)", SHADOW)
    arrow(ax, (80, 16.5), L(enc), col=SHADOW)
    ax.set_xlim(0, 100); ax.set_ylim(0, 72)
    save(fig, "fig3_obj2_shadow")


FIGS = {f.__name__: f for f in (fig3_2_siran, fig3_obj2_shadow)}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
