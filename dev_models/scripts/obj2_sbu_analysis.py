"""Shadow detectors on the SBU test set (638 images): second benchmark for the paper (Objective 2; Eq 3.13).

Three comparisons, from the per-image counts saved by `scripts/obj2_eval.py --dataset sbu` (no re-inference):

  1. phase-1 checkpoints (trained on SBU): in-domain BER of the four models after phase 1 alone;
  2. fair-protocol phase-2 checkpoints (five seeds each, 0-4, selected on held-out ISTD training scenes):
     seed-mean BER on SBU, a test set none of the phase-2 choices could see, with pairwise differences;
  3. selection effect on SBU: held-out seed-mean BER minus the original ISTD-test-selected checkpoint.

SBU has no scene labels, so the bootstrap resamples images (10,000 draws, percentile 95% intervals);
images of one SBU source photograph collection are not independent, so these intervals are, if
anything, too narrow. Protocol thesis256, tau = 0.50.

Usage:
    python scripts/obj2_sbu_analysis.py
"""

from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.coco import file_sha256  # noqa: E402
from src.eval.shadow_metrics import THRESHOLDS, ber_from_counts  # noqa: E402

STAMP = "20261111_010526"
F = lambda name: f"results/obj2/sbu_obj2_counts_{name}_{STAMP}.npz"  # noqa: E731
ARCHS = ("unetpp", "ran", "siran", "rarm")
PHASE1 = {a: F(f"{a}@sbu_phase1") for a in ARCHS}
HELDOUT = {a: [F(f"{a}@heldout_seed{s}") for s in range(3)]
           + [f"results/obj2/sbu_obj2_counts_{a}@heldout_seed{s}_20261111_115247.npz" for s in (3, 4)]  # experiment B
           for a in ARCHS}
ORIGINAL = {"unetpp": F("unetpp@original"), "ran": F("ran@iter2"), "siran": F("siran@original"), "rarm": F("rarm@original")}
KEY, DRAWS, RNG_SEED = "thesis256_counts", 10_000, 0
I50 = THRESHOLDS.index(0.5)


def ber(c):
    v = float(ber_from_counts(c.sum(0))["ber"][I50])
    if not np.isfinite(v):
        raise FloatingPointError("BER is not finite")
    return v


def iou(c):
    return float(ber_from_counts(c.sum(0))["iou_pooled"][I50])


def main():
    load = lambda p: np.load(ROOT / p)  # noqa: E731
    files = [*PHASE1.values(), *itertools.chain(*HELDOUT.values()), *ORIGINAL.values()]
    data = {p: load(p) for p in files}
    ref = data[files[0]]
    for p, d in data.items():
        if not np.array_equal(d["images"], ref["images"]):
            raise AssertionError(f"{p}: image order differs")
    n = len(ref["images"])
    if n != 638:
        raise AssertionError(f"SBU test should have 638 images, found {n}")
    draws = np.random.default_rng(RNG_SEED).integers(0, n, size=(DRAWS, n))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    boot = lambda c: np.array([ber(c[dr]) for dr in draws])  # noqa: E731

    p1 = {a: {"ber": ber(data[PHASE1[a]][KEY]), "iou": iou(data[PHASE1[a]][KEY])} for a in ARCHS}
    b1 = {a: boot(data[PHASE1[a]][KEY]) for a in ARCHS}
    p1_pairs = {f"{a}-{b}": {"ber_diff": p1[a]["ber"] - p1[b]["ber"], "ci95": ci(b1[a] - b1[b])}
                for a, b in itertools.combinations(ARCHS, 2)}

    fair, bmean = {}, {}
    for a in ARCHS:
        cs = [data[p][KEY] for p in HELDOUT[a]]
        per = [ber(c) for c in cs]
        fair[a] = {"per_seed_ber": per, "mean": float(np.mean(per)), "sd": float(np.std(per, ddof=1)),
                   "iou_mean": float(np.mean([iou(c) for c in cs]))}
        bmean[a] = np.mean([boot(c) for c in cs], axis=0)
    fair_pairs = {f"{a}-{b}": {"ber_diff": fair[a]["mean"] - fair[b]["mean"], "ci95": ci(bmean[a] - bmean[b])}
                  for a, b in itertools.combinations(ARCHS, 2)}

    sel = {}
    for a in ARCHS:
        o = data[ORIGINAL[a]][KEY]
        sel[a] = {"original_ber": ber(o), "heldout_mean_ber": fair[a]["mean"],
                  "heldout_minus_original": fair[a]["mean"] - ber(o), "ci95": ci(bmean[a] - boot(o)),
                  "seeds_worse_than_original": int(sum(x > ber(o) for x in fair[a]["per_seed_ber"]))}

    result = {
        "experiment": "obj2_sbu_second_benchmark",
        "thesis_refs": ["Section 4.4", "Eq 3.13"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"phase1": PHASE1, "heldout": HELDOUT, "original": ORIGINAL,
                   "sha256": {p: file_sha256(ROOT / p) for p in files}, "protocol": "thesis256", "tau": 0.5,
                   "bootstrap": {"unit": "image (SBU has no scene labels)", "images": n, "draws": DRAWS,
                                 "rng_seed": RNG_SEED, "ci": "percentile 2.5/97.5"}},
        "results": {"phase1_in_domain": p1, "phase1_pairs": p1_pairs, "fair_phase2_on_sbu": fair,
                    "fair_pairs": fair_pairs, "selection_effect_on_sbu": sel},
    }
    dest = ROOT / f"results/obj2/obj2_sbu_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    dest.write_text(json.dumps(result, indent=2))
    print("== phase 1, SBU test (in domain)")
    for a in ARCHS:
        print(f"  {a:7s} BER {p1[a]['ber']:.2f}  IoU {p1[a]['iou']:.4f}")
    for k, v in p1_pairs.items():
        print(f"    {k:14s} {v['ber_diff']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
    print("== fair phase 2 (5 seeds), SBU test (not seen by any phase-2 choice)")
    for a in ARCHS:
        print(f"  {a:7s} BER {fair[a]['mean']:.2f} ± {fair[a]['sd']:.2f}  IoU {fair[a]['iou_mean']:.4f}")
    for k, v in fair_pairs.items():
        print(f"    {k:14s} {v['ber_diff']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
    print("== selection effect on SBU (held-out mean - original)")
    for a in ARCHS:
        v = sel[a]
        print(f"  {a:7s} original {v['original_ber']:.2f}  held-out {v['heldout_mean_ber']:.2f}  "
              f"diff {v['heldout_minus_original']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]  worse {v['seeds_worse_than_original']}/{len(fair[a]['per_seed_ber'])}")
    print("written:", dest)


if __name__ == "__main__":
    main()
