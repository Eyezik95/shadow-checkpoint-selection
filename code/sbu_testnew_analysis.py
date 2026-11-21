"""SBU analyses repeated on the hand-relabelled SBU test set (SBU-TestNew; Yang et al., ICCV 2023).

Same three comparisons, protocol (thesis256, tau 0.50) and image bootstrap (10,000 draws, rng seed 0) as
bifpn-siran-shadow/scripts/obj2_sbu_analysis.py, computed from the per-image counts written by
`scripts/obj2_eval.py --dataset sbu --sbu <SBUTestNew>` (logs/sbu_testnew_eval.log). Adds a side-by-side
with the original-label analysis (results/obj2/obj2_sbu_analysis_20261111_115652.json in the PhD project,
read only) so the paper can say whether the conclusions survive cleaner labels.

The relabelled release renames the 638 images (1.jpg ... 638.jpg) and states no licence; it is used for
evaluation only and is not redistributed. Paired per-image comparisons with the original labels are
therefore not possible; set-level BERs and model differences are compared instead.
"""
from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1] / "dev_models"
sys.path.insert(0, str(REPO))
from src.data.hashing import file_sha256  # noqa: E402
from src.eval.shadow_metrics import THRESHOLDS, ber_from_counts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ORIG_ANALYSIS = REPO / "results" / "obj2" / "obj2_sbu_analysis_20261111_115652.json"
ARCHS = ("unetpp", "ran", "siran", "rarm")
KEY, DRAWS, RNG_SEED = "thesis256_counts", 10_000, 0
I50 = THRESHOLDS.index(0.5)


def find(name):
    hits = sorted(RES.glob(f"sbu_obj2_counts_{name}_*.npz"))
    if len(hits) != 1:
        raise AssertionError(f"{name}: expected one counts file, found {len(hits)}")
    return hits[0]


def ber(c):
    v = float(ber_from_counts(c.sum(0))["ber"][I50])
    if not np.isfinite(v):
        raise FloatingPointError("BER is not finite")
    return v


def main():
    PHASE1 = {a: find(f"{a}@sbu_phase1") for a in ARCHS}
    HELDOUT = {a: [find(f"{a}@heldout_seed{s}") for s in range(5)] for a in ARCHS}
    ORIGINAL = {"unetpp": find("unetpp@original"), "ran": find("ran@iter2"), "siran": find("siran@original"),
                "rarm": find("rarm@original")}
    files = [*PHASE1.values(), *itertools.chain(*HELDOUT.values()), *ORIGINAL.values()]
    data = {p: np.load(p) for p in files}
    ref = data[files[0]]
    for p, d in data.items():
        if not np.array_equal(d["images"], ref["images"]):
            raise AssertionError(f"{p}: image order differs")
    n = len(ref["images"])
    if n != 638:
        raise AssertionError(f"SBU-TestNew should have 638 images, found {n}")
    draws = np.random.default_rng(RNG_SEED).integers(0, n, size=(DRAWS, n))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    boot = lambda c: np.array([ber(c[dr]) for dr in draws])  # noqa: E731

    p1 = {a: ber(data[PHASE1[a]][KEY]) for a in ARCHS}
    fair, bmean = {}, {}
    for a in ARCHS:
        per = [ber(data[p][KEY]) for p in HELDOUT[a]]
        fair[a] = {"per_seed_ber": per, "mean": float(np.mean(per)), "sd": float(np.std(per, ddof=1))}
        bmean[a] = np.mean([boot(data[p][KEY]) for p in HELDOUT[a]], axis=0)
    fair_pairs = {f"{a}-{b}": {"ber_diff": fair[a]["mean"] - fair[b]["mean"], "ci95": ci(bmean[a] - bmean[b])}
                  for a, b in itertools.combinations(ARCHS, 2)}
    sel = {}
    for a in ARCHS:
        o = data[ORIGINAL[a]][KEY]
        sel[a] = {"original_ber": ber(o), "heldout_mean_ber": fair[a]["mean"],
                  "heldout_minus_original": fair[a]["mean"] - ber(o), "ci95": ci(bmean[a] - boot(o)),
                  "seeds_worse_than_original": int(sum(x > ber(o) for x in fair[a]["per_seed_ber"]))}

    orig = json.loads(ORIG_ANALYSIS.read_text())["results"]
    side = {a: {"phase1_ber": {"original_labels": orig["phase1_in_domain"][a]["ber"], "relabelled": p1[a]},
                "fair_mean_ber": {"original_labels": orig["fair_phase2_on_sbu"][a]["mean"], "relabelled": fair[a]["mean"]},
                "selection_effect": {"original_labels": orig["selection_effect_on_sbu"][a]["heldout_minus_original"],
                                     "relabelled": sel[a]["heldout_minus_original"]}} for a in ARCHS}
    rank = lambda d: sorted(ARCHS, key=lambda a: d[a])  # noqa: E731
    ranking = {"phase1": {"original_labels": rank({a: side[a]["phase1_ber"]["original_labels"] for a in ARCHS}),
                          "relabelled": rank(p1)},
               "fair": {"original_labels": rank({a: side[a]["fair_mean_ber"]["original_labels"] for a in ARCHS}),
                        "relabelled": rank({a: fair[a]["mean"] for a in ARCHS})}}

    result = {"experiment": "sbu_testnew_reanalysis", "created_utc": datetime.now(timezone.utc).isoformat(),
              "config": {"counts_files": {str(p.name): file_sha256(p) for p in files},
                         "original_label_analysis": str(ORIG_ANALYSIS), "protocol": "thesis256", "tau": 0.5,
                         "bootstrap": {"unit": "image", "images": n, "draws": DRAWS, "rng_seed": RNG_SEED},
                         "dataset": "SBU-TestNew (SILT release refined_sbu/relabeled_sbu_test.zip, sha256 "
                                    "64e9b1287413501cad8cf22085a7ca702cc19d9e5737498b871a28c6e50e05d3)"},
              "results": {"phase1_in_domain": p1, "fair_phase2": fair, "fair_pairs": fair_pairs,
                          "selection_effect": sel, "side_by_side": side, "ranking_best_first": ranking}}
    dest = RES / f"sbu_testnew_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    dest.write_text(json.dumps(result, indent=2))
    print(f"{'model':7s} {'phase1 orig/new':>18s} {'fair orig/new':>18s} {'sel.effect orig/new':>22s}")
    for a in ARCHS:
        s = side[a]
        print(f"{a:7s} {s['phase1_ber']['original_labels']:8.2f} / {s['phase1_ber']['relabelled']:6.2f}"
              f"  {s['fair_mean_ber']['original_labels']:8.2f} / {s['fair_mean_ber']['relabelled']:6.2f}"
              f"  {s['selection_effect']['original_labels']:+9.2f} / {s['selection_effect']['relabelled']:+6.2f}")
    print("fair pairs (relabelled):")
    for k, v in fair_pairs.items():
        print(f"  {k:14s} {v['ber_diff']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
    print("selection effect (relabelled):")
    for a in ARCHS:
        v = sel[a]
        print(f"  {a:7s} {v['heldout_minus_original']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]  worse {v['seeds_worse_than_original']}/5")
    print("ranking:", json.dumps(ranking))
    print("written:", dest)


if __name__ == "__main__":
    main()
