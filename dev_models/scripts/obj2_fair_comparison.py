"""Fair comparison of the four shadow detectors of Table 4.2 (Objective 2; Section 4.4, Eq 3.13).

Every model here was trained with the same two-phase recipe and had its phase-2
checkpoint chosen on held-out ISTD *training* scenes (scripts/obj2_retrain_siran.py),
five seeds each (0-4): BiFPN-SIRAN, BiFPN-RAN (iteration 2), BiFPN-RARM, UNet++ (EfficientNet-B5).
Reads the per-image counts saved by scripts/obj2_eval.py (no re-inference).

Reported per architecture (both protocols): BER at tau = 0.50, cross-fitted BER and pooled
IoU at tau = 0.50, per seed and as mean / SD over seeds; the original test-selected
checkpoint for reference, with held-out mean - original.

Pairwise differences of seed-mean BER (tau 0.50): paired scene bootstrap, the 46 test
scenes resampled with replacement (10,000 draws); in each draw every seed's BER is
computed on the resampled scenes and averaged over seeds, so the interval covers scene
sampling only; seed spread is the SD.

Usage:
    python scripts/obj2_fair_comparison.py
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

from src.data.hashing import file_sha256  # noqa: E402
from src.eval.shadow_metrics import THRESHOLDS, ber_from_counts, cross_fit  # noqa: E402

R = "results/obj2/"
HELDOUT = {
    "siran": [R + "obj2_counts_siran@heldout_20261102_212753.npz",
              R + "obj2_counts_siran@heldout_seed1_20261103_024305.npz",
              R + "obj2_counts_siran@heldout_seed2_20261103_024305.npz"],
    "ran": [R + f"obj2_counts_ran@heldout_seed{s}_20261103_190019.npz" for s in range(3)],
    "rarm": [R + "obj2_counts_rarm@heldout_seed0_20261103_190019.npz",
             R + "obj2_counts_rarm@heldout_seed1_20261103_190019.npz",
             R + "obj2_counts_rarm@heldout_seed2_20261103_202904.npz"],     # rerun; see results/obj2/failed/README.txt
    "unetpp": [R + f"obj2_counts_unetpp@heldout_seed{s}_20261103_230632.npz" for s in range(3)],
}
# seeds 3 and 4 (paper experiment B; scripts/run_obj2_more_seeds.sh), evaluated together on 2026-11-11
for _a in HELDOUT:
    HELDOUT[_a] += [R + f"obj2_counts_{_a}@heldout_seed{s}_20261111_114930.npz" for s in (3, 4)]
SEEDS = [0, 1, 2, 3, 4]
ORIGINAL = {"siran": R + "obj2_counts_siran@original_20261102_212753.npz",
            "ran": R + "obj2_counts_ran@iter2_20261102_194216.npz",
            "rarm": R + "obj2_counts_rarm_20261102_194216.npz",
            "unetpp": R + "obj2_counts_unetpp_20261103_202935.npz"}
PROTOCOLS = ("thesis256", "native")
DRAWS, RNG_SEED = 10_000, 0
I50 = THRESHOLDS.index(0.5)


def at50(counts):
    m = ber_from_counts(counts.sum(0))
    v = {"ber": float(m["ber"][I50]), "iou": float(m["iou_pooled"][I50])}
    if not all(np.isfinite(x) for x in v.values()):
        raise FloatingPointError("BER or IoU is not finite")
    return v


def main():
    ref = np.load(ROOT / ORIGINAL["siran"])
    scenes = ref["scenes"]
    load = lambda p: np.load(ROOT / p)  # noqa: E731
    runs = {a: [load(p) for p in ps] for a, ps in HELDOUT.items()}
    orig = {a: load(p) for a, p in ORIGINAL.items()}
    for d in [*itertools.chain(*runs.values()), *orig.values()]:
        if not (np.array_equal(d["images"], ref["images"]) and np.array_equal(d["scenes"], scenes)):
            raise AssertionError("image order differs between evaluations")
    uniq = np.unique(scenes)
    idx = [np.flatnonzero(scenes == s) for s in uniq]
    draws = np.random.default_rng(RNG_SEED).integers(0, len(uniq), size=(DRAWS, len(uniq)))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731

    out = {}
    for prot in PROTOCOLS:
        key = f"{prot}_counts"
        arch = {}
        boot = {}
        for a, ds in runs.items():
            per = [{**at50(d[key]), "cross_fitted_ber": float(cross_fit(d[key], d["halves"])["ber"])} for d in ds]
            o = {**at50(orig[a][key]), "cross_fitted_ber": float(cross_fit(orig[a][key], orig[a]["halves"])["ber"])}
            stat = {k: {"mean": float(np.mean([p[k] for p in per])), "sd": float(np.std([p[k] for p in per], ddof=1))}
                    for k in ("ber", "cross_fitted_ber", "iou")}
            arch[a] = {"per_seed": per, "seed_mean": stat, "original_test_selected": o,
                       "heldout_minus_original_ber": stat["ber"]["mean"] - o["ber"]}
            sc = [np.stack([d[key][i].sum(0) for i in idx]) for d in ds]
            boot[a] = np.array([np.mean([float(ber_from_counts(s[dr].sum(0))["ber"][I50]) for s in sc])
                                for dr in draws])
            if not np.isfinite(boot[a]).all():
                raise FloatingPointError(f"{a}: bootstrap BER is not finite")
        pairs = {}
        for a, b in itertools.combinations(HELDOUT, 2):
            diff = boot[a] - boot[b]
            pairs[f"{a}-{b}"] = {"ber_diff": arch[a]["seed_mean"]["ber"]["mean"] - arch[b]["seed_mean"]["ber"]["mean"],
                                 "bootstrap_scenes_ci95": ci(diff), "share_of_draws_below_zero": float((diff < 0).mean())}
        ranking = sorted(HELDOUT, key=lambda a: arch[a]["seed_mean"]["ber"]["mean"])
        out[prot] = {"architectures": arch, "pairwise_seed_mean_ber": pairs, "ranking_by_ber": ranking}

    result = {
        "experiment": "obj2_fair_comparison_heldout_selection",
        "thesis_refs": ["Section 4.4", "Table 4.2", "Eq 3.13", "Section 3.3.2.1"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"heldout_counts": HELDOUT, "original_counts": ORIGINAL,
                   "sha256": {p: file_sha256(ROOT / p) for p in [*itertools.chain(*HELDOUT.values()), *ORIGINAL.values()]},
                   "tau": 0.5, "seeds": SEEDS,
                   "bootstrap": {"unit": "test scene", "scenes": int(len(uniq)), "draws": DRAWS, "rng_seed": RNG_SEED,
                                 "ci": "percentile 2.5/97.5", "statistic": "difference of seed-mean BER"},
                   "sd": "sample SD over seeds (ddof 1)"},
        "results": out,
    }
    dest = ROOT / f"results/obj2/obj2_fair_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    dest.write_text(json.dumps(result, indent=2))
    for prot in PROTOCOLS:
        print(f"== {prot}")
        for a, v in out[prot]["architectures"].items():
            s = v["seed_mean"]
            print(f"{a:7s} BER {s['ber']['mean']:.2f} ± {s['ber']['sd']:.2f}  cross-fitted {s['cross_fitted_ber']['mean']:.2f} "
                  f"± {s['cross_fitted_ber']['sd']:.2f}  IoU {s['iou']['mean']:.4f} ± {s['iou']['sd']:.4f}  "
                  f"| original {v['original_test_selected']['ber']:.2f}")
        for k, v in out[prot]["pairwise_seed_mean_ber"].items():
            print(f"  {k:14s} {v['ber_diff']:+.2f}  [{v['bootstrap_scenes_ci95'][0]:+.2f}, {v['bootstrap_scenes_ci95'][1]:+.2f}]")
    print("written:", dest)


if __name__ == "__main__":
    main()
