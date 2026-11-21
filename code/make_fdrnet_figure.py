"""Per-epoch FDRNet BER for the five seeds: ISTD test (left) and held-out validation (right).

Reads the latest results/fdrnet_analysis_*.json; writes paper/figures/fig_fdrnet_epochs.pdf (and .png).
Markers show the epoch each rule selects: square = last epoch, triangle = lowest validation BER,
star = lowest test BER. Colours: Okabe-Ito (colour-blind safe); seeds are also identified by marker
legend so colour is not the only cue.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]


def main():
    path = sorted((ROOT / "results").glob("fdrnet_analysis_*.json"))[-1]
    a = json.loads(path.read_text())
    test = np.array(a["curves"]["test_ber"]["per_seed"])
    val = np.array(a["curves"]["val_ber"]["per_seed"])
    ep = np.arange(1, test.shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)
    for ax, data, title in ((axes[0], test, "ISTD test (540 images)"),
                            (axes[1], val, "Held-out validation (120 images)")):
        for i, row in enumerate(a["per_seed"]):
            ax.plot(ep, data[i], color=COLORS[i], ls=LINESTYLES[i], lw=1.4, label=f"seed {row['seed']}")
            r = row["rules"]
            ax.plot(r["val"]["epoch"], data[i, r["val"]["epoch"] - 1], "^", color=COLORS[i], ms=7,
                    mec="black", mew=0.6, zorder=4)
            ax.plot(r["test"]["epoch"], data[i, r["test"]["epoch"] - 1], "*", color=COLORS[i], ms=10,
                    mec="black", mew=0.6, zorder=5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("epoch", fontsize=9)
        ax.set_xticks(ep)
        ax.set_ylim(1.0, 6.0)
        ax.grid(axis="y", lw=0.4, color="#dddddd")
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("BER (%)", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], ls="", marker="^", color="grey", mec="black", mew=0.6, ms=7),
          plt.Line2D([], [], ls="", marker="*", color="grey", mec="black", mew=0.6, ms=10)]
    l += ["chosen on validation", "chosen on test"]
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = ROOT / "paper" / "figures" / "fig_fdrnet_epochs"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200)
    clipped = int((test > 6).sum() + (val > 6).sum())
    print(out, "source", path.name, "points above y-limit (epochs 1-3):", clipped)


if __name__ == "__main__":
    main()
