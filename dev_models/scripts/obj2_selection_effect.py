"""Effect of test-set checkpoint selection for every shadow detector (Objective 2; Section 4.4, Eq 3.13).

For each of BiFPN-SIRAN, BiFPN-RAN (iteration 2), BiFPN-RARM and UNet++: seed-mean BER (tau = 0.50)
of the five held-out-validation retrains (seeds 0-4) minus the BER of the original checkpoint, which was
selected on the ISTD test partition. Positive = the original looked better than fair selection gives.

Paired scene bootstrap as in scripts/obj2_seed_bias.py and scripts/obj2_fair_comparison.py: the 46
test scenes resampled with replacement (10,000 draws); in each draw every seed's BER and the
original's BER are computed on the same resampled scenes. The interval covers scene sampling only;
seed spread is the SD. Reads the saved per-image counts (no re-inference); file lists are imported
from scripts/obj2_fair_comparison.py so both analyses use the same evaluations.

Usage:
    python scripts/obj2_selection_effect.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from obj2_fair_comparison import HELDOUT, ORIGINAL, SEEDS  # noqa: E402
from src.data.coco import file_sha256  # noqa: E402
from src.eval.shadow_metrics import THRESHOLDS, ber_from_counts  # noqa: E402

PROTOCOLS = ("thesis256", "native")
DRAWS, RNG_SEED = 10_000, 0
I50 = THRESHOLDS.index(0.5)


def ber50(counts):
    v = float(ber_from_counts(counts.sum(0))["ber"][I50])
    if not np.isfinite(v):
        raise FloatingPointError("BER is not finite")
    return v


def main():
    load = lambda p: np.load(ROOT / p)  # noqa: E731
    ref = load(ORIGINAL["siran"])
    scenes = ref["scenes"]
    uniq = np.unique(scenes)
    idx = [np.flatnonzero(scenes == s) for s in uniq]
    draws = np.random.default_rng(RNG_SEED).integers(0, len(uniq), size=(DRAWS, len(uniq)))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731

    out = {}
    for prot in PROTOCOLS:
        key = f"{prot}_counts"
        res = {}
        for a in HELDOUT:
            runs, orig = [load(p) for p in HELDOUT[a]], load(ORIGINAL[a])
            for d in [*runs, orig]:
                if not (np.array_equal(d["images"], ref["images"]) and np.array_equal(d["scenes"], scenes)):
                    raise AssertionError(f"{a}: image order differs between evaluations")
            seed_ber = np.array([ber50(d[key]) for d in runs])
            o = ber50(orig[key])
            sc_runs = [np.stack([d[key][i].sum(0) for i in idx]) for d in runs]
            sc_orig = np.stack([orig[key][i].sum(0) for i in idx])
            boot = np.array([np.mean([ber50(s[dr]) for s in sc_runs]) - ber50(sc_orig[dr]) for dr in draws])
            res[a] = {"original_test_selected_ber": o,
                      "heldout_seed_ber": seed_ber.tolist(),
                      "heldout_mean_ber": float(seed_ber.mean()), "heldout_sd_ber": float(seed_ber.std(ddof=1)),
                      "heldout_minus_original": float(seed_ber.mean() - o),
                      "bootstrap_scenes_ci95": ci(boot),
                      "seeds_worse_than_original": int((seed_ber > o).sum())}
        out[prot] = res

    files = [p for ps in HELDOUT.values() for p in ps] + list(ORIGINAL.values())
    result = {
        "experiment": "obj2_selection_effect_all_models",
        "thesis_refs": ["Section 4.4.3", "Section 4.4.4", "Table 4.4", "Eq 3.13"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"heldout_counts": HELDOUT, "original_counts": ORIGINAL,
                   "sha256": {p: file_sha256(ROOT / p) for p in files}, "tau": 0.5, "seeds": SEEDS,
                   "bootstrap": {"unit": "test scene", "scenes": int(len(uniq)), "draws": DRAWS, "rng_seed": RNG_SEED,
                                 "ci": "percentile 2.5/97.5", "statistic": "held-out seed-mean BER - original BER"},
                   "sd": "sample SD over seeds (ddof 1)"},
        "results": out,
    }
    dest = ROOT / f"results/obj2/obj2_selection_effect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    dest.write_text(json.dumps(result, indent=2))
    for prot in PROTOCOLS:
        print(f"== {prot}")
        for a, v in out[prot].items():
            print(f"{a:7s} original {v['original_test_selected_ber']:.2f}  held-out {v['heldout_mean_ber']:.2f} ± "
                  f"{v['heldout_sd_ber']:.2f}  diff {v['heldout_minus_original']:+.2f} "
                  f"[{v['bootstrap_scenes_ci95'][0]:+.2f}, {v['bootstrap_scenes_ci95'][1]:+.2f}]  "
                  f"seeds worse {v['seeds_worse_than_original']}/{len(v['heldout_seed_ber'])}")
    print("written:", dest)


if __name__ == "__main__":
    main()
