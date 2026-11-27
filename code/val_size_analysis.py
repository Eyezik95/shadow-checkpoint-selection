"""How large must a validation set be before checkpoint selection on it beats a fixed schedule? (FDRNet, ISTD)

Uses the per-epoch, per-image ISTD test counts saved for the five FDRNet seeds (results/fdrnet/*_counts.npz).
The 46 test scenes are split at random, many times, into a selection part of k scenes and a scoring part of
the remaining 46-k scenes. On each split and seed:
  * val-k rule : the epoch with the lowest BER on the k selection scenes;
  * last rule  : epoch 10 (the authors' fixed schedule);
  * oracle     : the epoch with the lowest BER on the scoring scenes themselves (= selecting on the test set).
All three are scored on the scoring scenes. Reported per k, averaged over seeds and splits:
  * cost_vs_last  = BER(val-k) - BER(last)         (> 0: honest selection did worse than the fixed schedule)
  * gap_to_oracle = BER(val-k) - BER(oracle)       (how far selection falls short of the best epoch)
  * oracle_gain   = BER(oracle) - BER(last)        (what selecting on the test set appears to gain)
  * optimism      = [BER_sel(e*) - BER_sel(last)] - [BER_score(e*) - BER_score(last)], e* = val-k epoch:
                    how much better the chosen checkpoint looks on the scenes that chose it than on new scenes;
                    the in-sample flattery that test-set selection adds to a reported number;
  * p_last        = share of splits in which val-k picks epoch 10; p_worse = share with BER(val-k) > BER(last).
Selection scenes come from the test distribution, so this is the most favourable case for a validation set
drawn from the same source; scenes are the sampling unit because images within an ISTD scene share a background.

Usage: python code/val_size_analysis.py [fdrnet|sddnet]  (default fdrnet -> results/val_size_analysis_<ts>.json;
sddnet -> results/sddnet_val_size_analysis_<ts>.json, kept apart so the FDRNet glob is unchanged)
"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHOD = sys.argv[1] if len(sys.argv) > 1 else "fdrnet"
assert METHOD in ("fdrnet", "sddnet"), METHOD
RES = ROOT / "results" / METHOD
KS = (2, 3, 4, 6, 9, 12, 15, 18, 23)
SPLITS, RNG_SEED = 2000, 0


def ber(c):
    tp, tn, p, n = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    return 100.0 * (1.0 - (tp / p + tn / n) / 2.0)


def load_per_scene():
    files = sorted(f for f in glob.glob(str(RES / f"{METHOD}_istd_seed*_2*.json"))
                   if "selection" in json.loads(Path(f).read_text()))
    assert len(files) == 5, files
    out, names0 = [], None
    for f in files:
        z = np.load(f.replace(".json", "_counts.npz"))
        assert list(z["columns"]) == ["TP", "TN", "P", "N"]
        names = z["test_names"]
        if names0 is None:
            names0 = names
            scenes = sorted({str(n).split("-")[0] for n in names}, key=int)
            sidx = np.array([scenes.index(str(n).split("-")[0]) for n in names])
        assert (names == names0).all()
        t = z["test"].astype(np.float64)                      # (epoch, image, 4)
        s = np.stack([t[:, sidx == k].sum(axis=1) for k in range(len(scenes))], axis=1)
        out.append(s)
    per_scene = np.stack(out)                                 # (seed, epoch, scene, 4)
    assert per_scene.shape[2] == 46 and np.isfinite(per_scene).all()
    return per_scene, [Path(f).name for f in files]


def main():
    per_scene, files = load_per_scene()
    n_seed, n_ep, n_sc, _ = per_scene.shape
    last = n_ep - 1
    rng = np.random.default_rng(RNG_SEED)
    res = {}
    for k in KS:
        m = np.zeros((SPLITS, n_sc), bool)
        for i in range(SPLITS):
            m[i, rng.choice(n_sc, k, replace=False)] = True
        # (split, seed, epoch, 4)
        sel = np.einsum("ij,sejc->isec", m.astype(float), per_scene)
        sco = np.einsum("ij,sejc->isec", (~m).astype(float), per_scene)
        bs, bc = ber(sel), ber(sco)                           # (split, seed, epoch)
        e = bs.argmin(axis=2)                                 # val-k epoch
        take = lambda b, idx: np.take_along_axis(b, idx[..., None], 2)[..., 0]
        v, l, o = take(bc, e), bc[..., last], bc.min(axis=2)
        opt = (take(bs, e) - bs[..., last]) - (v - l)
        stat = lambda x: {"mean": float(x.mean()), "sd_over_splits": float(x.mean(1).std(ddof=1)),
                          "p05": float(np.percentile(x.mean(1), 5)), "p95": float(np.percentile(x.mean(1), 95))}
        res[k] = {"scenes_select": k, "scenes_score": n_sc - k,
                  "cost_vs_last": stat(v - l), "gap_to_oracle": stat(v - o), "oracle_gain": stat(o - l),
                  "optimism": stat(opt), "p_last": float((e == last).mean()), "p_worse": float((v > l + 1e-12).mean()),
                  "epoch_hist": np.bincount(e.ravel(), minlength=n_ep).tolist()}
    # where does the observed cost of the real 120-image validation set (9 ISTD *training* scenes) fall among
    # selection sets of the same number of scenes drawn from the test distribution?
    ana = json.loads(sorted((ROOT / "results").glob(f"{METHOD}_analysis_*.json"))[-1].read_text())
    obs = ana["mean_diffs"]["val_minus_last"]["mean"]
    rng2 = np.random.default_rng(RNG_SEED + 1)
    costs = []
    for _ in range(20_000):
        mk = np.zeros(n_sc, bool); mk[rng2.choice(n_sc, 9, replace=False)] = True
        bs, bc = ber(per_scene[:, :, mk].sum(2)), ber(per_scene[:, :, ~mk].sum(2))
        costs.append(np.mean(bc[np.arange(n_seed), bs.argmin(1)] - bc[:, last]))
    costs = np.array(costs)
    observed = {"val_minus_last_9_training_scenes": obs, "k": 9, "draws": len(costs),
                "share_test_scene_splits_with_cost_ge_observed": float((costs >= obs).mean()),
                "test_scene_split_cost_max": float(costs.max()), "test_scene_split_cost_p95": float(np.percentile(costs, 95))}
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = {"experiment": f"validation_size_vs_selection_{METHOD}", "created": ts,
           "config": {"inputs": files, "ks": KS, "splits": SPLITS, "rng_seed": RNG_SEED,
                      "unit": "ISTD test scene (46)", "epochs": n_ep, "fixed_schedule_epoch": n_ep,
                      "averaging": "mean over seeds and splits; sd/p05/p95 are over splits of the seed-averaged value"},
           "results": {str(k): v for k, v in res.items()}, "observed_training_scene_validation": observed}
    p = ROOT / "results" / (f"val_size_analysis_{ts}.json" if METHOD == "fdrnet" else f"{METHOD}_val_size_analysis_{ts}.json")
    p.write_text(json.dumps(out, indent=1)); print(p)
    print("observed:", observed)
    print(" k  cost_vs_last  gap_to_oracle  oracle_gain  optimism  p_last  p_worse")
    for k, r in res.items():
        print(f"{k:2d}  {r['cost_vs_last']['mean']:+.3f}        {r['gap_to_oracle']['mean']:+.3f}         "
              f"{r['oracle_gain']['mean']:+.3f}      {r['optimism']['mean']:+.3f}    {r['p_last']:.2f}    {r['p_worse']:.2f}")


if __name__ == "__main__":
    main()
