"""Shadow-class / non-shadow-class error and boundary-band error, SIRAN vs RAN iteration 2
(Objective 2; Section 4.4.6, 4.4.8, Eq 3.13). Answers the draft's request to substantiate "SIRAN reduces
shadow-class error specifically" with figures.

From the per-image counts saved by scripts/obj2_eval.py (protocol thesis256, tau = 0.50), for
the original (test-selected) checkpoints and for the 5-seed held-out redos:

  * shadow error = FN / (TP + FN), non-shadow error = FP / (TN + FP), in % (pooled pixels);
  * boundary band k = 2 / 5 / 10 px: error rate inside the band (wrong pixels in band / band pixels, %)
    and the share of all misclassified pixels that lie inside the band (%);
  * SIRAN - RAN for each quantity; for the held-out redos the seed means are compared with a paired
    scene bootstrap (46 test scenes, 10,000 draws), as scripts/obj2_fair_comparison.py.

Usage:
    python scripts/obj2_error_components.py
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
from src.eval.shadow_metrics import BAND_WIDTHS, THRESHOLDS  # noqa: E402

R = "results/obj2/"
SETS = {
    "original": {"siran": [R + "obj2_counts_siran@original_20261102_212753.npz"],
                 "ran": [R + "obj2_counts_ran@iter2_20261102_194216.npz"]},
    "heldout": {"siran": [R + "obj2_counts_siran@heldout_20261102_212753.npz",
                          R + "obj2_counts_siran@heldout_seed1_20261103_024305.npz",
                          R + "obj2_counts_siran@heldout_seed2_20261103_024305.npz"],
                "ran": [R + f"obj2_counts_ran@heldout_seed{s}_20261103_190019.npz" for s in range(3)]},
}
for _a in ("siran", "ran"):  # seeds 3 and 4 (experiment B)
    SETS["heldout"][_a] += [R + f"obj2_counts_{_a}@heldout_seed{s}_20261111_114930.npz" for s in (3, 4)]
PROT = "thesis256"
DRAWS, RNG_SEED = 10_000, 0
I50 = THRESHOLDS.index(0.5)


def measures(c, be, bp):
    """c: (n, 4) TP FN TN FP at tau 0.50; be: (n, bands) wrong pixels in band; bp: (n, bands) band pixels."""
    tp, fn, tn, fp = c.sum(0).astype(float)
    wrong = fn + fp
    m = {"shadow_err": 100 * fn / (tp + fn), "nonshadow_err": 100 * fp / (tn + fp)}
    for j, k in enumerate(BAND_WIDTHS):
        m[f"band{k}_err_rate"] = 100 * be[:, j].sum() / bp[:, j].sum()
        m[f"band{k}_share_of_errors"] = 100 * be[:, j].sum() / wrong
    for k, v in m.items():
        if not np.isfinite(v):
            raise FloatingPointError(f"{k} is not finite")
    return m


def main():
    data = {s: {a: [np.load(ROOT / p) for p in ps] for a, ps in d.items()} for s, d in SETS.items()}
    ref = data["original"]["siran"][0]
    for d in [x for s in data.values() for ds in s.values() for x in ds]:
        if not np.array_equal(d["images"], ref["images"]):
            raise AssertionError("image order differs between evaluations")
    scenes = ref["scenes"]
    idx = [np.flatnonzero(scenes == s) for s in np.unique(scenes)]
    draws = np.random.default_rng(RNG_SEED).integers(0, len(idx), size=(DRAWS, len(idx)))
    get = lambda d, rows=slice(None): (d[f"{PROT}_counts"][rows, I50], d[f"{PROT}_band_err"][rows, I50],  # noqa: E731
                                      d[f"{PROT}_band_px"][rows])

    out = {}
    for s, d in data.items():
        per = {a: [measures(*get(x)) for x in ds] for a, ds in d.items()}
        mean = {a: {k: float(np.mean([p[k] for p in v])) for k in v[0]} for a, v in per.items()}
        sd = {a: ({k: float(np.std([p[k] for p in v], ddof=1)) for k in v[0]} if len(v) > 1 else None)
              for a, v in per.items()}
        keys = list(mean["siran"])
        boot = {k: np.empty(DRAWS) for k in keys}
        for j, dr in enumerate(draws):
            rows = np.concatenate([idx[i] for i in dr])
            ms = {a: [measures(*get(x, rows)) for x in ds] for a, ds in d.items()}
            for k in keys:
                boot[k][j] = np.mean([m[k] for m in ms["siran"]]) - np.mean([m[k] for m in ms["ran"]])
        out[s] = {"seeds": len(d["siran"]), "mean": mean, "sd": sd,
                  "siran_minus_ran": {k: {"diff": mean["siran"][k] - mean["ran"][k],
                                          "bootstrap_scenes_ci95": [float(np.percentile(boot[k], 2.5)),
                                                                    float(np.percentile(boot[k], 97.5))]}
                                      for k in keys}}
    result = {
        "experiment": "obj2_error_components_siran_vs_ran",
        "thesis_refs": ["Section 4.4.6", "Section 4.4.8", "Eq 3.13"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"counts": SETS, "sha256": {p: file_sha256(ROOT / p) for d in SETS.values() for ps in d.values() for p in ps},
                   "protocol": PROT, "tau": 0.5, "band_widths_px": list(BAND_WIDTHS),
                   "bootstrap": {"unit": "test scene", "scenes": len(idx), "draws": DRAWS, "rng_seed": RNG_SEED}},
        "results": out,
    }
    dest = ROOT / f"results/obj2/obj2_error_components_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    dest.write_text(json.dumps(result, indent=2))
    for s, v in out.items():
        print(f"== {s} ({v['seeds']} seed(s))")
        for k, x in v["siran_minus_ran"].items():
            print(f"  {k:24s} SIRAN {v['mean']['siran'][k]:6.2f}  RAN {v['mean']['ran'][k]:6.2f}  diff {x['diff']:+.2f} "
                  f"[{x['bootstrap_scenes_ci95'][0]:+.2f}, {x['bootstrap_scenes_ci95'][1]:+.2f}]")
    print("written:", dest)


if __name__ == "__main__":
    main()
