"""Minimum detectable BER difference on ISTD and SBU as a function of the number of seeds per method.

Two sources of variation enter a comparison of two methods A and B, each trained with n seeds:
  * seed variance sigma_s^2 (measured: sample variance of BER over our five seeds per model);
  * test-set sampling of the difference, var_t(n) (measured: bootstrap variance of the BER difference over test
    scenes for ISTD, images for SBU, 10,000 draws), computed for single runs (n=1) and for 5-seed means (n=5)
    and interpolated as var_t(n) = a + b/n through those two points.
SE(n) = sqrt(2 sigma_s^2 / n + var_t(n)). Reported:
  * mdd_sig(n)  = 1.96 SE: the smallest observed gap whose 95% interval excludes zero;
  * mdd_pow(n)  = 2.80 SE: the true gap detected with 80% power at alpha 0.05 (two-sided);
  * mdd_seed(n) = 1.96 sqrt(2 sigma_s^2 / n): seed noise only, i.e. "is A better than B on this fixed test set".
sigma_s is pooled (root mean of variances) over the models of a benchmark, and also given per model.
Caveats written to the output: the four development-sequence models share phase-1 weights across seeds (their
sigma_s understates fully independent training); SBU has no scene labels, so its image bootstrap is too narrow;
five seeds estimate sigma_s loosely (chi-square 95% interval reported).
"""
import glob
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "dev_models" / "results" / "obj2"
I50 = 9                     # THRESHOLDS = 0.05 ... 0.95; index 9 = 0.50
DRAWS, RNG_SEED = 10_000, 0
NS = (1, 2, 3, 5, 10)
MODELS = ("unetpp", "ran", "siran", "rarm")


def ber(c):
    tp, tn, p, n = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    return 100.0 * (1.0 - (tp / p + tn / n) / 2.0)


def dev(c):
    """Development-model counts are (TP, FN, TN, FP) (dev_models/src/eval/shadow_metrics.py); -> (TP, TN, P, N)."""
    c = c.astype(np.float64)
    tp, fn, tn, fp = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    return np.stack([tp, tn, tp + fn, tn + fp], axis=-1)


def istd_dev_files(m):
    if m == "siran":
        f = ["obj2_counts_siran@heldout_20261102_212753.npz", "obj2_counts_siran@heldout_seed1_20261103_024305.npz",
             "obj2_counts_siran@heldout_seed2_20261103_024305.npz"]
    elif m == "rarm":
        f = ["obj2_counts_rarm@heldout_seed0_20261103_190019.npz", "obj2_counts_rarm@heldout_seed1_20261103_190019.npz",
             "obj2_counts_rarm@heldout_seed2_20261103_202904.npz"]
    elif m == "ran":
        f = [f"obj2_counts_ran@heldout_seed{s}_20261103_190019.npz" for s in range(3)]
    else:
        f = [f"obj2_counts_unetpp@heldout_seed{s}_20261103_230632.npz" for s in range(3)]
    return [DEV / x for x in f + [f"obj2_counts_{m}@heldout_seed{s}_20261111_114930.npz" for s in (3, 4)]]


def unit_counts(path, key, units_key="scenes"):
    """Per-image counts at tau 0.50 summed into bootstrap units -> (units, 4), unit labels."""
    z = np.load(path)
    c = z[key][:, I50].astype(np.float64)
    u = z[units_key] if units_key else z["images"]
    labels = sorted(set(map(str, u)), key=lambda x: (len(x), x))
    idx = np.array([labels.index(str(x)) for x in u])
    return np.stack([c[idx == k].sum(0) for k in range(len(labels))]), labels, z["images"]


def fdrnet_last(protocol_images, method="fdrnet"):
    """Last-epoch test counts of the five runs of a published detector (fdrnet or sddnet), aligned to dev order."""
    files = sorted(f for f in glob.glob(str(ROOT / "results" / method / f"{method}_istd_seed*_2*.json"))
                   if "selection" in json.loads(Path(f).read_text()))
    assert len(files) == 5
    out = []
    for f in files:
        z = np.load(f.replace(".json", "_counts.npz"))
        names = [str(n) for n in z["test_names"]]
        order = [names.index(Path(str(n)).stem + ".png") if Path(str(n)).stem + ".png" in names else names.index(str(n))
                 for n in protocol_images]
        out.append(z["test"][-1].astype(np.float64)[order])          # last epoch, (image, 4) aligned to dev order
    return out, [Path(f).name for f in files]


def analyse(runs, units_idx, n_units, label_unit):
    """runs: {model: [ (image,4) per seed ]}; units_idx: image -> unit. Returns sigma_s, var_t, mdd table."""
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, n_units, size=(DRAWS, n_units))
    per_unit = {m: np.stack([np.stack([r[units_idx == k].sum(0) for k in range(n_units)]) for r in rs])
                for m, rs in runs.items()}                            # (seed, unit, 4)
    seed_ber = {m: ber(pu.sum(1)) for m, pu in per_unit.items()}      # (seed,)
    boot = {m: ber(pu[:, draws].sum(2)) for m, pu in per_unit.items()}  # (seed, DRAWS)
    var_s = {m: float(b.var(ddof=1)) for m, b in seed_ber.items()}
    k = len(next(iter(seed_ber.values())))
    chi = stats.chi2.ppf([0.975, 0.025], k - 1)
    pooled = float(np.sqrt(np.mean(list(var_s.values()))))
    dfp = (k - 1) * len(var_s)
    chip = stats.chi2.ppf([0.975, 0.025], dfp)
    vt1, vt5, pairs = [], [], {}
    for a, b in combinations(runs, 2):
        single = [float((boot[a][s] - boot[b][s]).var(ddof=1)) for s in range(k)]
        mean5 = float((boot[a].mean(0) - boot[b].mean(0)).var(ddof=1))
        vt1 += single; vt5.append(mean5)
        pairs[f"{a}-{b}"] = {"sd_t_single_run_median": float(np.sqrt(np.median(single))), "sd_t_5seed_mean": float(np.sqrt(mean5))}
    v1, v5 = float(np.median(vt1)), float(np.median(vt5))
    bcoef = (v1 - v5) / (1 - 1 / 5); acoef = v1 - bcoef
    table = {}
    for n in NS:
        vt = max(acoef + bcoef / n, 0.0)
        se = np.sqrt(2 * pooled ** 2 / n + vt)
        table[n] = {"se": float(se), "mdd_sig": float(1.96 * se), "mdd_pow80": float(2.80 * se),
                    "mdd_seed_only": float(1.96 * np.sqrt(2 * pooled ** 2 / n)), "sd_t": float(np.sqrt(vt))}
    return {"bootstrap_unit": label_unit, "units": n_units, "seeds": k,
            "seed_ber": {m: v.tolist() for m, v in seed_ber.items()},
            "sigma_s": {m: float(np.sqrt(v)) for m, v in var_s.items()},
            "sigma_s_ci95": {m: [float(np.sqrt((k - 1) * v / chi[0])), float(np.sqrt((k - 1) * v / chi[1]))] for m, v in var_s.items()},
            "sigma_s_pooled": pooled,
            "sigma_s_pooled_ci95": [float(np.sqrt(dfp * pooled ** 2 / chip[0])), float(np.sqrt(dfp * pooled ** 2 / chip[1]))],
            "var_t": {"single_run_median": v1, "five_seed_mean_median": v5, "model": "a + b/n", "a": acoef, "b": bcoef},
            "pairs": pairs, "mdd": {str(n): v for n, v in table.items()}}


def main():
    out = {}
    # ---- ISTD, development-sequence models, protocol thesis256 (as reported in the paper)
    runs, files = {}, []
    for m in MODELS:
        fs = istd_dev_files(m); files += [f.name for f in fs]
        runs[m] = [dev(np.load(f)["thesis256_counts"][:, I50]) for f in fs]
    z = np.load(istd_dev_files("ran")[0]); scenes = [str(s) for s in z["scenes"]]
    lab = sorted(set(scenes), key=int); uidx = np.array([lab.index(s) for s in scenes])
    assert len(lab) == 46
    out["istd_dev_thesis256"] = analyse(runs, uidx, 46, "ISTD test scene")
    # ---- ISTD, native resolution, development models + FDRNet (last epoch), so FDRNet's sigma_s enters the pool
    runs_n = {m: [dev(np.load(f)["native_counts"][:, I50]) for f in istd_dev_files(m)] for m in MODELS}
    fdr, ffiles = fdrnet_last([str(x) for x in z["images"]])
    runs_n["fdrnet"] = fdr; files += ffiles
    out["istd_native_with_fdrnet"] = analyse(runs_n, uidx, 46, "ISTD test scene")
    out["istd_fdrnet_only_sigma_s"] = out["istd_native_with_fdrnet"]["sigma_s"]["fdrnet"]
    # ---- same, with SDDNet (last epoch, native resolution) added to the pool when its five runs exist
    if len(glob.glob(str(ROOT / "results" / "sddnet" / "sddnet_istd_seed*_2*.json"))) == 5:
        sdd, sfiles = fdrnet_last([str(x) for x in z["images"]], "sddnet")
        runs_n2 = dict(runs_n); runs_n2["sddnet"] = sdd; files += sfiles
        out["istd_native_with_fdrnet_sddnet"] = analyse(runs_n2, uidx, 46, "ISTD test scene")
    # ---- SBU original labels and relabelled (SBU-TestNew): development models, image bootstrap
    for name, pat in (("sbu_original", str(DEV / "sbu_obj2_counts_{m}@heldout_seed{s}_2026111*.npz")),
                      ("sbu_relabelled", str(ROOT / "results" / "sbu_obj2_counts_{m}@heldout_seed{s}_20261120_225818.npz"))):
        runs_s = {}
        for m in MODELS:
            fs = [sorted(glob.glob(pat.format(m=m, s=s))) for s in range(5)]
            assert all(len(f) == 1 for f in fs), (name, m, fs)
            files += [Path(f[0]).name for f in fs]
            runs_s[m] = [dev(np.load(f[0])["thesis256_counts"][:, I50]) for f in fs]
        n_img = runs_s[MODELS[0]][0].shape[0]
        out[name] = analyse(runs_s, np.arange(n_img), n_img, "SBU test image (not independent; intervals too narrow)")
    ts = time.strftime("%Y%m%d_%H%M%S")
    res = {"experiment": "minimum_detectable_difference", "created": ts,
           "config": {"inputs": files, "threshold": 0.5, "draws": DRAWS, "rng_seed": RNG_SEED, "ns": NS,
                      "caveats": ["development-sequence models share phase-1 weights across seeds: sigma_s understates independent training",
                                  "SBU bootstrap resamples images, which are not independent: var_t too small",
                                  "sigma_s from 5 seeds per model; chi-square intervals given"]},
           "results": out}
    p = ROOT / "results" / f"mdd_analysis_{ts}.json"; p.write_text(json.dumps(res, indent=1)); print(p)
    for k, r in out.items():
        if not isinstance(r, dict): print(k, round(r, 3)); continue
        print(f"\n{k}: sigma_s pooled {r['sigma_s_pooled']:.3f} {[round(x, 3) for x in r['sigma_s_pooled_ci95']]}; "
              f"per model { {m: round(v, 3) for m, v in r['sigma_s'].items()} }; seed BER means "
              f"{ {m: round(float(np.mean(v)), 3) for m, v in r['seed_ber'].items()} }")
        print("  n   sd_t   mdd_sig  mdd_pow80  mdd_seed_only")
        for n, v in r["mdd"].items():
            print(f"  {n:>2}  {v['sd_t']:.3f}  {v['mdd_sig']:.3f}    {v['mdd_pow80']:.3f}      {v['mdd_seed_only']:.3f}")


if __name__ == "__main__":
    main()
