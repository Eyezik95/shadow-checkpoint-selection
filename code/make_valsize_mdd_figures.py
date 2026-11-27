"""Figures for the validation-size and minimum-detectable-difference analyses.

fig_valsize.pdf: FDRNet, selection on k ISTD test scenes, scored on the other 46-k; mean over 5 seeds and 2,000
splits, band = 5th-95th percentile over splits. Diamond: the real 120-image validation set (9 training scenes).
fig_mdd.pdf: smallest BER gap two methods must show, against seeds per method, ISTD and SBU.
Okabe-Ito colours plus distinct line styles and direct labels, so colour is never the only cue.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "grey": "#555555"}


def style(ax):
    ax.grid(axis="y", lw=0.4, color="#dddddd"); ax.tick_params(labelsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def valsize():
    p = sorted((ROOT / "results").glob("val_size_analysis_*.json"))[-1]; a = json.loads(p.read_text())
    ks = [int(k) for k in a["results"]]; r = a["results"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhline(0, color="black", lw=0.8)
    for key, sgn, col, ls, lab in (("cost_vs_last", 1, C["blue"], "-", "chosen on k scenes $-$ last epoch (new scenes)"),
                                   ("oracle_gain", 1, C["orange"], "--", "chosen on test $-$ last epoch"),
                                   ("optimism", -1, C["green"], "-.", "flattery: chosen epoch looks this much better on its own k scenes")):
        m = sgn * np.array([r[str(k)][key]["mean"] for k in ks])
        lo, hi = sorted((sgn * np.array([r[str(k)][key]["p05"] for k in ks]), sgn * np.array([r[str(k)][key]["p95"] for k in ks])),
                        key=lambda v: v.mean())
        ax.fill_between(ks, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(ks, m, color=col, ls=ls, lw=1.6, marker="o", ms=3, label=lab)
    o = a["observed_training_scene_validation"]
    ax.plot([9], [o["val_minus_last_9_training_scenes"]], "D", color=C["red"], mec="black", mew=0.6, ms=7,
            label="real validation set (9 training scenes)")
    ax.set_xlabel("selection scenes k (of 46 ISTD test scenes)", fontsize=9)
    ax.set_ylabel("test BER difference (points)", fontsize=9)
    ax.set_xticks(ks); style(ax)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    fig.tight_layout()
    out = ROOT / "paper" / "figures" / "fig_valsize"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200); print(out, p.name)


def mdd():
    p = sorted((ROOT / "results").glob("mdd_analysis_*.json"))[-1]; a = json.loads(p.read_text())["results"]
    panels = (("istd_native_with_fdrnet_sddnet", "ISTD (46 test scenes)"), ("sbu_original", "SBU, original labels (638 images)"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        t = a[key]["mdd"]; ns = [int(n) for n in t]
        for f, col, ls, lab in (("mdd_pow80", C["red"], "-", "80% power"),
                                ("mdd_sig", C["blue"], "--", "interval excludes zero"),
                                ("mdd_seed_only", C["green"], "-.", "seed noise only (fixed test set)")):
            y = [t[str(n)][f] for n in ns]
            ax.plot(ns, y, color=col, ls=ls, lw=1.6, marker="o", ms=3, label=lab)
            ax.annotate(f"{y[-1]:.2f}", (ns[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, va="center", color="#333333")
        ax.set_title(title, fontsize=9); ax.set_xlabel("seeds per method", fontsize=9)
        ax.set_xticks(ns); ax.set_xlim(0.5, ns[-1] + 1.5); ax.set_ylim(0, None); style(ax)
    axes[0].set_ylabel("smallest detectable BER gap (points)", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = ROOT / "paper" / "figures" / "fig_mdd"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200); print(out, p.name)



def leaderboard():
    """fig_leaderboard.pdf: claimed gain over the best method in each paper's own table, in publication order,
    against the single-run thresholds from the MDD analysis (seed noise only; interval over test sampling + seeds)."""
    p = sorted((ROOT / "results").glob("leaderboard_analysis_*.json"))[-1]; a = json.loads(p.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
    for ax, d in zip(axes, ("ISTD", "SBU")):
        rs = [r for r in a["rows"] if r["dataset"] == d]
        t = a["summary"][d]["thresholds"]
        x = np.arange(len(rs))
        ax.axhspan(0, t["seed_only"], color=C["green"], alpha=0.12, lw=0)
        ax.axhline(t["seed_only"], color=C["green"], ls="-.", lw=1.2)
        ax.axhline(t["interval"], color=C["blue"], ls="--", lw=1.2)
        ax.text(len(rs) - 0.5, t["seed_only"], f"seed noise, 1 run ({t['seed_only']:.2f})", fontsize=6.5, ha="right", va="bottom", color="#333333")
        ax.text(len(rs) - 0.5, t["interval"], f"test sampling + seeds ({t['interval']:.2f})", fontsize=6.5, ha="right", va="bottom", color="#333333")
        g = [r["claimed_gain"] for r in rs]
        ax.vlines(x, 0, g, color="#888888", lw=1.0)
        ax.plot(x, g, "o", color="#222222", ms=4)
        ax.set_xticks(x); ax.set_xticklabels([f"{r['paper']} '{str(r['year'])[2:]}" for r in rs], rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{d}: {a['summary'][d]['n_below_seed_only']} of {len(rs)} gains below one-run seed noise", fontsize=8.5)
        ax.set_ylim(0, 2.1); style(ax)
    axes[0].set_ylabel("claimed BER gain over best\nmethod in own table (points)", fontsize=8.5)
    fig.tight_layout()
    out = ROOT / "paper" / "figures" / "fig_leaderboard"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200); print(out, p.name)
if __name__ == "__main__":
    valsize(); mdd(); leaderboard()
