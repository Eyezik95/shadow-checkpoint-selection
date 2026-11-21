"""Checkpoint-selection bias of BiFPN-SIRAN across held-out-validation seeds (Objective 2; Section 4.4, Eq 3.13).

Reads the per-image counts saved by scripts/obj2_eval.py (no re-inference) for the
original, test-selected SIRAN checkpoint and for every phase-2 redo with held-out
validation (seeds 0-4). Reports per seed and for the seed mean:

  * BER at tau = 0.50 and the cross-fitted BER, both protocols;
  * mean and SD over seeds;
  * difference held-out - original, with a paired scene bootstrap (the 46 test
    scenes resampled with replacement, 10,000 draws). For the seed mean the BER of
    each seed is computed on the resampled scenes and averaged, so the interval
    covers scene sampling only; seed spread is reported separately as the SD.

Usage:
    python scripts/obj2_seed_bias.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hashing import file_sha256  # noqa: E402
from src.eval.shadow_metrics import THRESHOLDS, ber_from_counts, cross_fit  # noqa: E402

ORIGINAL = "results/obj2/obj2_counts_siran@original_20261102_212753.npz"
SEEDS = {0: "results/obj2/obj2_counts_siran@heldout_20261102_212753.npz",
         1: "results/obj2/obj2_counts_siran@heldout_seed1_20261103_024305.npz",
         2: "results/obj2/obj2_counts_siran@heldout_seed2_20261103_024305.npz",
         3: "results/obj2/obj2_counts_siran@heldout_seed3_20261111_114930.npz",   # experiment B
         4: "results/obj2/obj2_counts_siran@heldout_seed4_20261111_114930.npz"}
PROTOCOLS = ("thesis256", "native")
DRAWS, RNG_SEED = 10_000, 0
I50 = THRESHOLDS.index(0.5)


def ber50(counts):
    v = float(ber_from_counts(counts.sum(0))["ber"][I50])
    if not np.isfinite(v):
        raise FloatingPointError("BER is not finite")
    return v


def main():
    ref = np.load(ROOT / ORIGINAL)
    runs = {s: np.load(ROOT / p) for s, p in SEEDS.items()}
    for s, d in runs.items():
        if not (np.array_equal(d["images"], ref["images"]) and np.array_equal(d["scenes"], ref["scenes"])):
            raise AssertionError(f"seed {s}: image order differs from the original evaluation")
    scenes = ref["scenes"]
    uniq = np.unique(scenes)
    idx = [np.flatnonzero(scenes == s) for s in uniq]
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, len(uniq), size=(DRAWS, len(uniq)))

    out = {}
    for prot in PROTOCOLS:
        key = f"{prot}_counts"
        c_ref = ref[key]
        per_seed = {s: {"ber_tau_0.50": ber50(d[key]), "cross_fitted_ber": float(cross_fit(d[key], d["halves"])["ber"])}
                    for s, d in runs.items()}
        b = np.array([v["ber_tau_0.50"] for v in per_seed.values()])
        cf = np.array([v["cross_fitted_ber"] for v in per_seed.values()])
        orig = {"ber_tau_0.50": ber50(c_ref), "cross_fitted_ber": float(cross_fit(c_ref, ref["halves"])["ber"])}
        # scene-level sums so each bootstrap draw is a sum over chosen scenes
        sc_ref = np.stack([c_ref[i].sum(0) for i in idx])
        sc_seed = {s: np.stack([d[key][i].sum(0) for i in idx]) for s, d in runs.items()}
        boot = {s: np.empty(DRAWS) for s in runs}
        boot_mean = np.empty(DRAWS)
        for j, dr in enumerate(draws):
            r = ber50(sc_ref[dr])
            vals = []
            for s in runs:
                v = ber50(sc_seed[s][dr])
                boot[s][j] = v - r
                vals.append(v)
            boot_mean[j] = np.mean(vals) - r
        ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
        out[prot] = {
            "original": orig,
            "heldout_per_seed": {str(s): {**v, "diff_vs_original_tau_0.50": v["ber_tau_0.50"] - orig["ber_tau_0.50"],
                                          "bootstrap_scenes_ci95": ci(boot[s])} for s, v in per_seed.items()},
            "heldout_seed_mean": {
                "ber_tau_0.50": {"mean": float(b.mean()), "sd": float(b.std(ddof=1))},
                "cross_fitted_ber": {"mean": float(cf.mean()), "sd": float(cf.std(ddof=1))},
                "diff_vs_original_tau_0.50": float(b.mean() - orig["ber_tau_0.50"]),
                "bootstrap_scenes_ci95": ci(boot_mean),
                "seeds_worse_than_original": int((b > orig["ber_tau_0.50"]).sum()),
            },
        }
    result = {
        "experiment": "obj2_siran_selection_bias_seeds",
        "thesis_refs": ["Section 4.4", "Table 4.2", "Eq 3.13"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"original_counts": ORIGINAL, "heldout_counts": {str(s): p for s, p in SEEDS.items()},
                   "sha256": {p: file_sha256(ROOT / p) for p in [ORIGINAL, *SEEDS.values()]},
                   "tau": 0.5, "bootstrap": {"unit": "test scene", "scenes": int(len(uniq)), "draws": DRAWS,
                                             "rng_seed": RNG_SEED, "ci": "percentile 2.5/97.5"},
                   "sd": "sample SD over seeds (ddof 1)"},
        "results": out,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ROOT / f"results/obj2/obj2_seed_bias_{stamp}.json"
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps(out, indent=1))
    print("written:", dest)


if __name__ == "__main__":
    main()
