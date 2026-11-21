"""Schematic of the three checkpoint-selection rules (Figure 1).

One training run produces a checkpoint per epoch; each rule keeps one of them, and all are then scored on
the test partition. The epochs marked are those chosen for FDRNet seed 0, read from the latest
results/fdrnet_analysis_*.json, so the figure contains no typed result. Partition sizes are read from the
same file's run configuration where available.

Writes paper/figures/fig_rules.pdf (and .png).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#222222", "#666666", "#bbbbbb"
C_FIXED, C_VAL, C_TEST = "#0072B2", "#009E73", "#D55E00"   # Okabe-Ito, as in the FDRNet figure
TRAIN_FILL, VAL_FILL, TEST_FILL = "#e8eef5", "#dff1ea", "#fbe6dc"


def box(ax, x, y, w, h, text, fc, ec=GRID, fs=7.5, color=INK, weight="normal", lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            weight=weight, linespacing=1.25)


def main():
    path = sorted((ROOT / "results").glob("fdrnet_analysis_*.json"))[-1]
    a = json.loads(path.read_text())
    seed0 = next(r for r in a["per_seed"] if r["seed"] == 0)["rules"]
    n_ep = len(a["curves"]["test_ber"]["mean"])
    chosen = {"last": seed0["last"]["epoch"], "val": seed0["val"]["epoch"], "test": seed0["test"]["epoch"]}
    split = a["config"]["run_config"].get("split", {})
    n_val_scenes = len(split.get("val_scenes", [])) or None

    fig, ax = plt.subplots(figsize=(7.0, 2.25))
    ax.set_xlim(0, 15.2); ax.set_ylim(0, 4.6); ax.axis("off")
    x0, w, gap = 3.55, 0.62, 0.12
    xs = {e: x0 + (e - 1) * (w + gap) for e in range(1, n_ep + 1)}
    xr = xs[n_ep] + w + 0.35                                   # start of the "data used" column

    ax.text(0.05, 4.25, "Rule", fontsize=7.8, weight="bold", color=INK, va="center")
    ax.text(x0, 4.25, "One training run, one checkpoint per epoch", fontsize=7.8, weight="bold",
            color=INK, va="center")
    ax.text(xr, 4.25, "Data used to choose", fontsize=7.8, weight="bold", color=INK, va="center")
    for e, x in xs.items():
        ax.text(x + w / 2, 3.78, f"{e}", ha="center", va="center", fontsize=7, color=MUTED)
    ax.text(x0 - 0.12, 3.78, "epoch", ha="right", va="center", fontsize=7, color=MUTED)

    vdesc = (f"validation: {n_val_scenes} scenes held\nout of the training partition" if n_val_scenes
             else "validation: scenes held\nout of the training partition")
    rules = [("last", C_FIXED, "Fixed schedule", "keep the last epoch", "none", TRAIN_FILL, 2.95),
             ("val", C_VAL, "Held-out selection", "lowest validation BER", vdesc, VAL_FILL, 1.95),
             ("test", C_TEST, "Test selection", "lowest test BER", "the test partition itself", TEST_FILL, 0.95)]
    h = 0.62
    for key, col, name, how, data, fill, y in rules:
        ax.text(0.05, y + 0.14, name, fontsize=7.8, weight="bold", color=col, va="center")
        ax.text(0.05, y - 0.2, how, fontsize=7, color=INK, va="center")
        for e, x in xs.items():
            if e == chosen[key]:
                box(ax, x, y - h / 2, w, h, "kept", col, ec=col, fs=6.5, color="white", weight="bold")
            else:
                box(ax, x, y - h / 2, w, h, "", "white", ec=GRID)
        box(ax, xr, y - h / 2 - 0.05, 15.15 - xr, h + 0.1, data, fill, fs=6.6)

    ax.text(0.05, 0.18, "Each kept checkpoint is then scored on the test partition. Only test selection has "
            "seen that partition before it is scored.", fontsize=6.8, color=MUTED, va="center", style="italic")

    out = ROOT / "paper" / "figures" / "fig_rules"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    print(out, "source", path.name, "epochs marked (FDRNet seed 0):", chosen)


if __name__ == "__main__":
    main()
