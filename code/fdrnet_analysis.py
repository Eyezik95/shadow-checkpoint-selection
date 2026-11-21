"""Five-seed FDRNet analysis: three checkpoint-selection rules applied to identical runs.

Reads results/fdrnet/fdrnet_istd_seed*_2*.json and the matching *_counts.npz (per-image TP, TN, P, N
for every epoch on the 120 held-out validation images and the 540 ISTD test images).

Rules (per seed): last = epoch 10 (the authors' fixed schedule); val = lowest BER on the held-out
validation scenes; test = lowest BER on the ISTD test set.

Reports, on ISTD test BER (pooled, %):
  * per seed and per rule: epoch and BER (recomputed from the counts and checked against the JSON);
  * per rule: mean and sample SD over seeds;
  * differences test-last, val-last, test-val, per seed and averaged over seeds, with 95% percentile
    intervals from a paired bootstrap over the 46 test scenes (10,000 resamples, rng seed 0). The
    selected epochs are held fixed in the bootstrap: the interval reflects test-set sampling of the
    difference between those checkpoints, not the variability of the selection itself;
  * per-epoch test and validation BER over seeds (mean, SD) for the figure.
Fails on NaN, on a mismatch with the stored selection, and if validation and test names share a scene.
"""
import glob
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "fdrnet"
DRAWS, RNG_SEED = 10_000, 0
RULES = ("last", "val", "test")
DIFFS = (("test", "last"), ("val", "last"), ("test", "val"))


def ber(c):
    """c: (..., 4) summed TP, TN, P, N -> pooled BER in %."""
    tp, tn, p, n = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    return 100.0 * (1.0 - (tp / p + tn / n) / 2.0)


def scene(name):
    return str(name).split("-")[0]


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return [float(lo), float(hi)]


def check(x, what):
    if not np.all(np.isfinite(x)):
        raise ValueError(f"non-finite value in {what}")


def main():
    files = sorted(glob.glob(str(RES / "fdrnet_istd_seed*_2*.json")))
    runs = []
    for f in files:
        d = json.loads(Path(f).read_text())
        if "selection" not in d:
            continue
        z = np.load(f.replace(".json", "_counts.npz"))
        assert list(z["columns"]) == ["TP", "TN", "P", "N"]
        runs.append((f, d, z))
    seeds = [d["config"]["seed"] for _, d, _ in runs]
    assert sorted(seeds) == list(range(5)), seeds
    cfg0 = {k: v for k, v in runs[0][1]["config"].items() if k not in ("seed",)}
    for _, d, _ in runs:
        c = {k: v for k, v in d["config"].items() if k not in ("seed",)}
        assert c == cfg0, "configs differ between seeds beyond the seed"

    test_names = runs[0][2]["test_names"]
    val_names = runs[0][2]["val_names"]
    for _, _, z in runs:
        assert (z["test_names"] == test_names).all() and (z["val_names"] == val_names).all()
    tscenes = sorted({scene(n) for n in test_names}, key=int)
    vscenes = {scene(n) for n in val_names}
    assert len(tscenes) == 46 and len(test_names) == 540 and len(val_names) == 120
    assert vscenes == set(runs[0][1]["config"]["split"]["val_scenes"])
    # validation images come from the ISTD training partition; they must share no scene with the test set
    assert not (vscenes & set(tscenes)), "validation and test scenes overlap"
    assert not ({str(n) for n in val_names} & {str(n) for n in test_names}), "validation and test images overlap"

    # per-scene summed counts: (seed, epoch, scene, 4)
    sidx = np.array([tscenes.index(scene(n)) for n in test_names])
    per_scene = []
    for _, _, z in runs:
        t = z["test"].astype(np.float64)                       # (epoch, image, 4)
        s = np.zeros((t.shape[0], len(tscenes), 4))
        for k in range(len(tscenes)):
            s[:, k] = t[:, sidx == k].sum(axis=1)
        per_scene.append(s)
    per_scene = np.stack(per_scene)                            # (5, 10, 46, 4)
    check(per_scene, "per-scene counts")

    val_ber = np.stack([ber(z["val"].astype(np.float64).sum(axis=1)) for _, _, z in runs])   # (5, 10)
    test_ber = ber(per_scene.sum(axis=2))                                                    # (5, 10)
    check(val_ber, "validation BER"); check(test_ber, "test BER")

    n_ep = test_ber.shape[1]
    sel = {}
    for i, (f, d, _) in enumerate(runs):
        e = {"last": n_ep - 1, "val": int(np.argmin(val_ber[i])), "test": int(np.argmin(test_ber[i]))}
        for r in RULES:
            stored = d["selection"][r]
            assert stored["epoch"] == e[r] + 1, (f, r, stored["epoch"], e[r] + 1)
            assert math.isclose(stored["test_ber"], test_ber[i, e[r]], abs_tol=1e-9), (f, r)
        sel[i] = e

    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, len(tscenes), size=(DRAWS, len(tscenes)))
    # bootstrap BER for each seed and rule: (5, rule, DRAWS)
    boot = np.zeros((len(runs), len(RULES), DRAWS))
    for i in range(len(runs)):
        for j, r in enumerate(RULES):
            boot[i, j] = ber(per_scene[i, sel[i][r]][draws].sum(axis=1))
    check(boot, "bootstrap")

    out_seeds = []
    for i, (f, d, _) in enumerate(runs):
        row = {"seed": d["config"]["seed"], "file": Path(f).name,
               "rules": {r: {"epoch": sel[i][r] + 1, "test_ber": float(test_ber[i, sel[i][r]]),
                             "val_ber": float(val_ber[i, sel[i][r]])} for r in RULES},
               "diffs": {}}
        for a, b in DIFFS:
            ja, jb = RULES.index(a), RULES.index(b)
            row["diffs"][f"{a}_minus_{b}"] = {
                "ber_diff": float(test_ber[i, sel[i][a]] - test_ber[i, sel[i][b]]),
                "bootstrap_scenes_ci95": ci(boot[i, ja] - boot[i, jb])}
        out_seeds.append(row)

    summary = {}
    for j, r in enumerate(RULES):
        v = np.array([test_ber[i, sel[i][r]] for i in range(len(runs))])
        summary[r] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                      "min": float(v.min()), "max": float(v.max()), "range": float(v.max() - v.min()),
                      "epochs": [sel[i][r] + 1 for i in range(len(runs))]}
    mean_diffs = {}
    for a, b in DIFFS:
        ja, jb = RULES.index(a), RULES.index(b)
        per = np.array([test_ber[i, sel[i][a]] - test_ber[i, sel[i][b]] for i in range(len(runs))])
        mean_diffs[f"{a}_minus_{b}"] = {
            "mean": float(per.mean()), "sd": float(per.std(ddof=1)),
            "min": float(per.min()), "max": float(per.max()),
            "bootstrap_scenes_ci95": ci(boot[:, ja].mean(axis=0) - boot[:, jb].mean(axis=0))}

    curves = {"test_ber": {"mean": test_ber.mean(0).tolist(), "sd": test_ber.std(0, ddof=1).tolist(),
                           "per_seed": test_ber.tolist()},
              "val_ber": {"mean": val_ber.mean(0).tolist(), "sd": val_ber.std(0, ddof=1).tolist(),
                          "per_seed": val_ber.tolist()}}

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = {"experiment": "fdrnet_selection_rules_five_seeds", "created": ts,
           "config": {"inputs": [Path(f).name for f, _, _ in runs], "run_config": cfg0,
                      "rules": {"last": f"epoch {n_ep} (fixed schedule)",
                                "val": "argmin BER on 120 held-out validation images (9 training scenes)",
                                "test": "argmin BER on 540 ISTD test images"},
                      "bootstrap": {"unit": "test scene", "scenes": len(tscenes), "draws": DRAWS,
                                    "rng_seed": RNG_SEED, "selected_epochs": "held fixed"},
                      "ber": "pooled 100*(1-(TP/P+TN/N)/2), native mask resolution",
                      "sd": "sample SD over seeds (ddof=1)"},
           "per_seed": out_seeds, "summary": summary, "mean_diffs": mean_diffs, "curves": curves}
    path = ROOT / "results" / f"fdrnet_analysis_{ts}.json"
    path.write_text(json.dumps(out, indent=1))
    print(path)
    for r in RULES:
        s = summary[r]; print(f"{r:5s} {s['mean']:.3f} +- {s['sd']:.3f}  range {s['range']:.3f}  epochs {s['epochs']}")
    for k, v in mean_diffs.items():
        print(f"{k:15s} mean {v['mean']:+.3f} sd {v['sd']:.3f} [{v['min']:+.3f},{v['max']:+.3f}] "
              f"CI {v['bootstrap_scenes_ci95'][0]:+.3f} {v['bootstrap_scenes_ci95'][1]:+.3f}")
    for row in out_seeds:
        print(row["seed"], {k: (round(v["ber_diff"], 3), [round(x, 3) for x in v["bootstrap_scenes_ci95"]])
                            for k, v in row["diffs"].items()})


if __name__ == "__main__":
    main()
