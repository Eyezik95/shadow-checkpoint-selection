"""Every result number in the TMLR paper comes from here; none is typed into the LaTeX.

Reads the result files (dev_models/results/obj2 for the four development-sequence models; results/) and writes
paper/numbers.tex, which defines \\nv{key}. Using an unknown key stops LaTeX with an error, so a
mistyped key cannot silently print a wrong value. paper/numbers.tsv lists key, value, source file
and JSON path for audit. check_numbers.py then verifies that the .tex contains no result-like
numbers outside \\nv{...}.

Formatting: BER and error rates to 2 decimals (percentage points), IoU to 3 or 4 as stated,
signed differences with an explicit sign, intervals as [lo, hi] with signs.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

PHD = Path(__file__).resolve().parents[1] / "dev_models" / "results" / "obj2"  # copy of the four-model results
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper"

ARCH = {"unetpp": "UNet++", "ran": "BiFPN-RAN", "siran": "BiFPN-SIRAN", "rarm": "BiFPN-RARM"}
rows: list[tuple[str, str, str, str]] = []


def load(path: Path):
    return json.loads(path.read_text())


def get(d, path):
    for p in path:
        d = d[p]
    return d


def fmt(v, kind):
    if not np.isfinite(v):
        raise FloatingPointError(f"non-finite value {v}")
    if kind == "p2":
        return f"{v:.2f}"
    if kind == "s2":
        return f"{v:+.2f}".replace("-", "\\textminus ")
    if kind == "s3":
        return f"{v:+.3f}".replace("-", "\\textminus ")
    if kind == "r2":                                   # unsigned style, but a proper minus for negatives
        return f"{v:.2f}".replace("-", "\\textminus ")
    if kind == "p3":
        return f"{v:.3f}"
    if kind == "p4":
        return f"{v:.4f}"
    if kind == "p1":
        return f"{v:.1f}"
    if kind == "intc":                                 # integer with thousands separator, e.g. 20,000
        return f"{int(round(v)):,}"
    if kind == "int":
        return f"{int(round(v))}"
    raise ValueError(kind)


def put(key, value, kind, src, path):
    if any(k == key for k, *_ in rows):
        raise KeyError(f"duplicate key {key}")
    rows.append((key, fmt(value, kind), str(src), "/".join(map(str, path))))


def put_ci(key, ci, src, path, kind="s2"):
    put(key + "-lo", ci[0], kind, src, path + ["0"]); put(key + "-hi", ci[1], kind, src, path + ["1"])


def original_checkpoints():
    src = {"siran": ("obj2_eval_20261102_212753.json", "siran@original"),
           "ran": ("obj2_eval_20261102_194216.json", "ran@iter2"),
           "rarm": ("obj2_eval_20261102_194216.json", "rarm"),
           "ranone": ("obj2_eval_20261103_060340.json", "ran_iter1"),
           "unetpp": ("obj2_eval_20261103_202935.json", "unetpp")}
    for a, (f, m) in src.items():
        d = load(PHD / f)["models"][m]["thesis256"]
        base = ["models", m, "thesis256"]
        put(f"orig-{a}-ber", d["native_tau_0.50"]["ber"], "p2", f, base + ["native_tau_0.50", "ber"])
        put(f"orig-{a}-ber-testsel", d["test_selected_BIASED"]["ber"], "p2", f, base + ["test_selected_BIASED", "ber"])
        put(f"orig-{a}-tau-testsel", d["test_selected_BIASED"]["tau"], "p2", f, base + ["test_selected_BIASED", "tau"])
        put(f"orig-{a}-ber-xfit", d["cross_fitted"]["ber"], "p2", f, base + ["cross_fitted", "ber"])
        put(f"orig-{a}-iou-meanimg", d["iou_tau_0.50"]["mean_per_image"], "p3", f, base + ["iou_tau_0.50", "mean_per_image"])
        put(f"orig-{a}-shadowerr", d["native_tau_0.50"]["shadow_err"], "p2", f, base + ["native_tau_0.50", "shadow_err"])
        put(f"orig-{a}-nonshadowerr", d["native_tau_0.50"]["nonshadow_err"], "p2", f, base + ["native_tau_0.50", "nonshadow_err"])
        if a == "siran":
            sw = d["sweep"]
            for i, t in enumerate(sw["tau"]):
                tag = f"{int(round(t * 100)):02d}"
                for q, kind in (("shadow_err", "p2"), ("nonshadow_err", "p2"), ("ber", "p2"), ("iou_pooled", "p3")):
                    put(f"sweep-siran-{tag}-{q.replace('_', '')}", sw[q][i], kind, f, base + ["sweep", q, str(i)])
        for w, v in d["boundary_band_error_pct"].items():  # keys 'k=2', 'k=5', 'k=10'
            put(f"orig-{a}-band{w.split('=')[1]}", v["tau_0.50"], "p2", f, base + ["boundary_band_error_pct", w, "tau_0.50"])


def seed_bias():
    f = "obj2_seed_bias_20261111_115734.json"; d = load(PHD / f)["results"]["thesis256"]
    p = ["results", "thesis256"]
    put("sb-orig-ber", d["original"]["ber_tau_0.50"], "p2", f, p + ["original", "ber_tau_0.50"])
    put("sb-orig-xfit", d["original"]["cross_fitted_ber"], "p2", f, p + ["original", "cross_fitted_ber"])
    for s, v in d["heldout_per_seed"].items():
        q = p + ["heldout_per_seed", s]
        put(f"sb-s{s}-ber", v["ber_tau_0.50"], "p2", f, q + ["ber_tau_0.50"])
        put(f"sb-s{s}-xfit", v["cross_fitted_ber"], "p2", f, q + ["cross_fitted_ber"])
        put(f"sb-s{s}-diff", v["diff_vs_original_tau_0.50"], "s2", f, q + ["diff_vs_original_tau_0.50"])
        put_ci(f"sb-s{s}-diff", v["bootstrap_scenes_ci95"], f, q + ["bootstrap_scenes_ci95"])
    m = d["heldout_seed_mean"]; q = p + ["heldout_seed_mean"]
    put("sb-mean-ber", m["ber_tau_0.50"]["mean"], "p2", f, q + ["ber_tau_0.50", "mean"])
    put("sb-sd-ber", m["ber_tau_0.50"]["sd"], "p2", f, q + ["ber_tau_0.50", "sd"])
    put("sb-mean-xfit", m["cross_fitted_ber"]["mean"], "p2", f, q + ["cross_fitted_ber", "mean"])
    put("sb-sd-xfit", m["cross_fitted_ber"]["sd"], "p2", f, q + ["cross_fitted_ber", "sd"])
    put("sb-mean-diff", m["diff_vs_original_tau_0.50"], "s2", f, q + ["diff_vs_original_tau_0.50"])
    put_ci("sb-mean-diff", m["bootstrap_scenes_ci95"], f, q + ["bootstrap_scenes_ci95"])
    bers = [v["ber_tau_0.50"] for v in d["heldout_per_seed"].values()]
    put("sb-seed-range", max(bers) - min(bers), "p2", f, q[:-1] + ["heldout_per_seed", "*", "ber_tau_0.50 (max-min)"])


def fair_and_selection():
    f = "obj2_fair_comparison_20261111_115646.json"; d = load(PHD / f)["results"]["thesis256"]
    p = ["results", "thesis256"]
    for a in ARCH:
        sm = d["architectures"][a]["seed_mean"]; q = p + ["architectures", a, "seed_mean"]
        for m, kind in (("ber", "p2"), ("cross_fitted_ber", "p2"), ("iou", "p4")):
            put(f"fair-{a}-{m.replace('_', '')}-mean", sm[m]["mean"], kind, f, q + [m, "mean"])
            put(f"fair-{a}-{m.replace('_', '')}-sd", sm[m]["sd"], kind if kind != "p2" else "p2", f, q + [m, "sd"])
    for k, v in d["pairwise_seed_mean_ber"].items():
        q = p + ["pairwise_seed_mean_ber", k]
        put(f"pair-{k}", v["ber_diff"], "s2", f, q + ["ber_diff"]); put_ci(f"pair-{k}", v["bootstrap_scenes_ci95"], f, q + ["bootstrap_scenes_ci95"])
    f = "obj2_selection_effect_20261111_115647.json"; d = load(PHD / f)["results"]["thesis256"]
    for a in ARCH:
        v = d[a]; q = ["results", "thesis256", a]
        put(f"sel-{a}-orig", v["original_test_selected_ber"], "p2", f, q + ["original_test_selected_ber"])
        put(f"sel-{a}-diff", v["heldout_minus_original"], "s2", f, q + ["heldout_minus_original"])
        put(f"sel-{a}-absdiff", abs(v["heldout_minus_original"]), "p2", f, q + ["heldout_minus_original (abs)"])
        put_ci(f"sel-{a}-diff", v["bootstrap_scenes_ci95"], f, q + ["bootstrap_scenes_ci95"])
        put(f"sel-{a}-worse", v["seeds_worse_than_original"], "int", f, q + ["seeds_worse_than_original"])


def error_components():
    f = "obj2_error_components_20261111_115829.json"; d = load(PHD / f)["results"]
    for grp in ("original", "heldout"):
        for a in ("siran", "ran"):
            for k, v in d[grp]["mean"][a].items():
                put(f"ec-{grp}-{a}-{k.replace('_', '')}", v, "p2", f, ["results", grp, "mean", a, k])
        for k, v in d[grp]["siran_minus_ran"].items():
            q = ["results", grp, "siran_minus_ran", k]
            put(f"ec-{grp}-diff-{k.replace('_', '')}", v["diff"], "s2", f, q + ["diff"])
            put_ci(f"ec-{grp}-diff-{k.replace('_', '')}", v["bootstrap_scenes_ci95"], f, q + ["bootstrap_scenes_ci95"])


def sbu(tag, path, keys):
    d = load(path)["results"]; f = path.name
    p1k, fairk, selk = keys
    for a in ARCH:
        v = d[p1k][a]
        put(f"{tag}-p1-{a}", v["ber"] if isinstance(v, dict) else v, "p2", f, ["results", p1k, a])
        put(f"{tag}-fair-{a}-mean", d[fairk][a]["mean"], "p2", f, ["results", fairk, a, "mean"])
        put(f"{tag}-fair-{a}-sd", d[fairk][a]["sd"], "p2", f, ["results", fairk, a, "sd"])
        per = d[fairk][a]["per_seed_ber"]                     # seeds 0-4, in order (analysis scripts)
        assert len(per) == 5, (tag, a, len(per))
        for i, v in enumerate(per):
            put(f"{tag}-fair-{a}-s{i}", v, "p2", f, ["results", fairk, a, "per_seed_ber", str(i)])
        s = d[selk][a]
        put(f"{tag}-orig-{a}", s["original_ber"], "p2", f, ["results", selk, a, "original_ber"])
        put(f"{tag}-sel-{a}-diff", s["heldout_minus_original"], "s2", f, ["results", selk, a, "heldout_minus_original"])
        put_ci(f"{tag}-sel-{a}-diff", s["ci95"], f, ["results", selk, a, "ci95"])
    for pk in [k for k in d if k.endswith("pairs")]:
        for k, v in d[pk].items():
            put(f"{tag}-{pk}-{k}", v["ber_diff"], "s2", f, ["results", pk, k, "ber_diff"])
            put_ci(f"{tag}-{pk}-{k}", v["ci95"], f, ["results", pk, k, "ci95"])


def fdrnet():
    runs = sorted(glob.glob(str(RES / "fdrnet" / "fdrnet_istd_seed*_2*.json")))
    runs = [r for r in runs if "selection" in load(Path(r))]
    if not runs:
        return
    for r in runs:
        d = load(Path(r)); s = d["config"]["seed"]; f = Path(r).name
        for rule in ("last", "val", "test"):
            put(f"fdr-s{s}-{rule}-ber", d["selection"][rule]["test_ber"], "p2", f, ["selection", rule, "test_ber"])
            put(f"fdr-s{s}-{rule}-epoch", d["selection"][rule]["epoch"], "int", f, ["selection", rule, "epoch"])
    ana = sorted(RES.glob("fdrnet_analysis_*.json"))
    if not ana:
        return
    a = load(ana[-1]); f = ana[-1].name
    for rule in ("last", "val", "test"):
        q = ["summary", rule]
        for k in ("mean", "sd", "min", "max", "range"):
            put(f"fdr-{rule}-{k}", a["summary"][rule][k], "p2", f, q + [k])
    for k, v in a["mean_diffs"].items():
        key = "fdr-d-" + k.replace("_minus_", "").replace("_", "")
        q = ["mean_diffs", k]
        put(key, v["mean"], "s3", f, q + ["mean"]); put(key + "-sd", v["sd"], "p3", f, q + ["sd"])
        put(key + "-min", v["min"], "s3", f, q + ["min"]); put(key + "-max", v["max"], "s3", f, q + ["max"])
        put_ci(key, v["bootstrap_scenes_ci95"], f, q + ["bootstrap_scenes_ci95"], "s3")
    for i, row in enumerate(a["per_seed"]):
        s = row["seed"]
        for k, v in row["diffs"].items():
            key = f"fdr-s{s}-d-" + k.replace("_minus_", "").replace("_", "")
            q = ["per_seed", str(i), "diffs", k]
            put(key, v["ber_diff"], "s3", f, q + ["ber_diff"]); put_ci(key, v["bootstrap_scenes_ci95"], f, q + ["bootstrap_scenes_ci95"], "s3")
        for rule in ("val",):
            put(f"fdr-s{s}-{rule}-valber", row["rules"][rule]["val_ber"], "p2", f, ["per_seed", str(i), "rules", rule, "val_ber"])
    readme = "code/upstream/FDRNet/README.md @ ea7e147 (Results table, L30-33)"
    put("fdr-pub-istd-nocrf", 1.53, "p2", readme, ["ISTD", "w/o CRF"])   # verified 21 Nov 2026
    put("fdr-pub-istd-crf", 1.55, "p2", readme, ["ISTD", "w/ CRF"])      # verified 21 Nov 2026
    c = a["curves"]["test_ber"]
    put("fdr-ep1-test-mean", c["mean"][0], "p2", f, ["curves", "test_ber", "mean", "0"])
    put("fdr-ep3-test-mean", c["mean"][2], "p2", f, ["curves", "test_ber", "mean", "2"])


def sddnet():
    """SDDNet, five seeds, 20 epochs (code/sddnet_dual_select.py; code/fdrnet_analysis.py sddnet;
    code/val_size_analysis.py sddnet; bercheck report on FDRNet + SDDNet). Keys prefixed sdd-, svs-, cmp-."""
    ana = sorted(RES.glob("sddnet_analysis_*.json"))
    if not ana:
        return
    a = load(ana[-1]); f = ana[-1].name
    for rule in ("last", "val", "test"):
        for k in ("mean", "sd", "min", "max", "range"):
            put(f"sdd-{rule}-{k}", a["summary"][rule][k], "p2", f, ["summary", rule, k])
        ep = a["summary"][rule]["epochs"]
        put(f"sdd-{rule}-epmin", min(ep), "int", f, ["summary", rule, "epochs", "min"])
        put(f"sdd-{rule}-epmax", max(ep), "int", f, ["summary", rule, "epochs", "max"])
    put("sdd-epochs", a["config"]["run_config"]["epochs"], "int", f, ["config", "run_config", "epochs"])
    for k, v in a["mean_diffs"].items():
        key = "sdd-d-" + k.replace("_minus_", "").replace("_", "")
        q = ["mean_diffs", k]
        put(key, v["mean"], "s3", f, q + ["mean"]); put(key + "-sd", v["sd"], "p3", f, q + ["sd"])
        put(key + "-min", v["min"], "s3", f, q + ["min"]); put(key + "-max", v["max"], "s3", f, q + ["max"])
        put_ci(key, v["bootstrap_scenes_ci95"], f, q + ["bootstrap_scenes_ci95"], "s3")
    for i, row in enumerate(a["per_seed"]):
        s = row["seed"]
        for rule in ("last", "val", "test"):
            put(f"sdd-s{s}-{rule}-ber", row["rules"][rule]["test_ber"], "p2", f, ["per_seed", str(i), "rules", rule, "test_ber"])
            put(f"sdd-s{s}-{rule}-epoch", row["rules"][rule]["epoch"], "int", f, ["per_seed", str(i), "rules", rule, "epoch"])
        for k, v in row["diffs"].items():
            key = f"sdd-s{s}-d-" + k.replace("_minus_", "").replace("_", "")
            q = ["per_seed", str(i), "diffs", k]
            put(key, v["ber_diff"], "s3", f, q + ["ber_diff"]); put_ci(key, v["bootstrap_scenes_ci95"], f, q + ["bootstrap_scenes_ci95"], "s3")
    c = a["curves"]["test_ber"]["mean"]
    put("sdd-ep3-test-mean", c[2], "p2", f, ["curves", "test_ber", "mean", "2"])
    put("sdd-late-test-min", min(c[9:]), "p2", f, ["curves", "test_ber", "mean", "epochs 10-20 min"])
    put("sdd-late-test-max", max(c[9:]), "p2", f, ["curves", "test_ber", "mean", "epochs 10-20 max"])
    v = a["curves"]["val_ber"]["per_seed"]
    put("sdd-s0-val-min", min(v[0]), "p1", f, ["curves", "val_ber", "per_seed", "0", "min"])
    put("sdd-others-val-max-late", max(max(r[9:]) for r in v[1:]), "p1", f, ["curves", "val_ber", "per_seed", "1-4", "epochs 10-20 max"])
    # validation-size repeat
    vp = sorted(RES.glob("sddnet_val_size_analysis_*.json"))
    if vp:
        b = load(vp[-1]); r = b["results"]; g = vp[-1].name
        for k in ("2", "6", "9", "23"):
            put(f"svs-k{k}-cost", r[k]["cost_vs_last"]["mean"], "s3", g, ["results", k, "cost_vs_last", "mean"])
            put(f"svs-k{k}-pworse", 100 * r[k]["p_worse"], "int", g, ["results", k, "p_worse x100"])
        put("svs-oracle", np.mean([r[k]["oracle_gain"]["mean"] for k in r]), "s3", g, ["results", "*", "oracle_gain", "mean (avg over k)"])
        o = b["observed_training_scene_validation"]
        put("svs-obs-cost", o["val_minus_last_9_training_scenes"], "s3", g, ["observed_training_scene_validation", "val_minus_last_9_training_scenes"])
        put("svs-obs-share", 100 * o["share_test_scene_splits_with_cost_ge_observed"], "int", g,
            ["observed_training_scene_validation", "share_test_scene_splits_with_cost_ge_observed x100"])
        put("svs-obs-p95", o["test_scene_split_cost_p95"], "s3", g, ["observed_training_scene_validation", "test_scene_split_cost_p95"])
    # two-method report (bercheck, rule = last epoch) and validation-test curve agreement
    bp = RES / "bercheck" / "fdrnet_sddnet_last.json"
    if bp.exists():
        b = load(bp); g = "bercheck/" + bp.name
        pair = next(iter(b["comparison"]["pairs"])); v = b["comparison"]["pairs"][pair]
        assert pair == "FDRNet - SDDNet", pair
        put("cmp-diff", v["diff"], "s3", g, ["comparison", "pairs", pair, "diff"])
        put_ci("cmp-diff", v["ci95"], g, ["comparison", "pairs", pair, "ci95"], "s2")
        put("cmp-psingle", 100 * v["p_single_run_order_holds"], "int", g, ["comparison", "pairs", pair, "p_single_run_order_holds x100"])
        for m, tag in (("FDRNet", "fdr"), ("SDDNet", "sdd")):
            ca = b["curve_agreement"][m]
            put(f"{tag}-rho-mean", ca["mean"], "p2", g, ["curve_agreement", m, "mean"])
            for i, x in enumerate(ca["spearman_per_run"]):
                put(f"{tag}-s{i}-rho", x, "r2", g, ["curve_agreement", m, "spearman_per_run", i])
    lp = sorted(RES.glob("leaderboard_analysis_*.json"))[-1]
    row = [r for r in load(lp)["rows"] if r["paper"] == "SDDNet" and r["dataset"] == "ISTD"]
    assert len(row) == 1 and row[0]["checked"] == "yes" and row[0]["note"] == "CRF", row
    put("sdd-pub-istd", float(row[0]["own_ber"]), "p2", lp.name, ["rows", "SDDNet/ISTD", "own_ber (with CRF, arXiv 2308.08935 Table 1)"])
    up = RES / "bercheck" / "sddnet_up512.json"
    if up.exists():
        u = load(up); g = "bercheck/" + up.name; r = u["rules"]["SDDNet-up512"]
        put("sdd-up-last-mean", r["summary"]["last"]["mean"], "p2", g, ["rules", "SDDNet-up512", "summary", "last", "mean"])
        put("sdd-up-d-testlast", r["differences"]["test-last"]["mean"], "s3", g, ["rules", "SDDNet-up512", "differences", "test-last", "mean"])
        put("sdd-up-d-vallast", r["differences"]["val-last"]["mean"], "s3", g, ["rules", "SDDNet-up512", "differences", "val-last", "mean"])
    # pool with SDDNet added (mdd_analysis.py, istd_native_with_fdrnet_sddnet)
    mp = sorted(RES.glob("mdd_analysis_*.json"))[-1]; m = load(mp)["results"]
    if "istd_native_with_fdrnet_sddnet" in m:
        x = m["istd_native_with_fdrnet_sddnet"]; key = "istd_native_with_fdrnet_sddnet"
        put("mdd-sdd-sigma", x["sigma_s"]["sddnet"], "p3", mp.name, ["results", key, "sigma_s", "sddnet"])
        put("mdd-sdd-sigma-lo", x["sigma_s_ci95"]["sddnet"][0], "p3", mp.name, ["results", key, "sigma_s_ci95", "sddnet", 0])
        put("mdd-sdd-sigma-hi", x["sigma_s_ci95"]["sddnet"][1], "p3", mp.name, ["results", key, "sigma_s_ci95", "sddnet", 1])


def histories():
    h = ROOT / "dev_models" / "data" / "raw" / "obj2_histories"
    p1 = load(h / "phase1_history_bifpn_rarm.json"); p2 = load(h / "phase2_history_bifpn_rarm.json")
    up = load(h / "phase2_history_unetplusplus.json")
    put("hist-rarm-p1-iou-first", p1["train_iou"][0], "p3", "phase1_history_bifpn_rarm.json", ["train_iou", "0"])
    put("hist-rarm-p1-iou-last", p1["train_iou"][-1], "p3", "phase1_history_bifpn_rarm.json", ["train_iou", "-1"])
    put("hist-rarm-p2-valiou-first", p2["val_iou"][0], "p3", "phase2_history_bifpn_rarm.json", ["val_iou", "0"])
    put("hist-rarm-p2-valiou-last", p2["val_iou"][-1], "p3", "phase2_history_bifpn_rarm.json", ["val_iou", "-1"])
    put("hist-unetpp-p2-valiou-first", up["val_iou"][0], "p3", "phase2_history_unetplusplus.json", ["val_iou", "0"])
    put("hist-unetpp-p2-valiou-max", max(up["val_iou"]), "p3", "phase2_history_unetplusplus.json", ["val_iou", "max"])


def derived():
    """Quantities computed from keys already put (recomputed from the stored strings is not allowed:
    they are recomputed here from the source files)."""
    best = None
    for a, (f, m) in {"siran": ("obj2_eval_20261102_212753.json", "siran@original"),
                      "ran": ("obj2_eval_20261102_194216.json", "ran@iter2"),
                      "rarm": ("obj2_eval_20261102_194216.json", "rarm"),
                      "ranone": ("obj2_eval_20261103_060340.json", "ran_iter1"),
                      "unetpp": ("obj2_eval_20261103_202935.json", "unetpp")}.items():
        d = load(PHD / f)["models"][m]["thesis256"]
        g = d["native_tau_0.50"]["ber"] - d["test_selected_BIASED"]["ber"]
        if best is None or g > best[0]:
            best = (g, a, f, m)
    put("orig-max-threshold-gain", best[0], "p2", best[2], ["models", best[3], "thesis256", "native_tau_0.50/ber - test_selected_BIASED/ber (max over models)"])


def selection_evidence():
    """Stored validation BER of each test-selected phase-2 checkpoint vs its ISTD test BER (thesis256)."""
    worst = (0.0, None)
    for f, m in (("obj2_eval_20261102_212753.json", "siran@original"), ("obj2_eval_20261102_194216.json", "ran@iter2"),
                 ("obj2_eval_20261102_194216.json", "rarm"), ("obj2_eval_20261103_202935.json", "unetpp")):
        v = load(PHD / f)["models"][m]
        diff = abs(v["meta"]["val_ber"] - v["thesis256"]["native_tau_0.50"]["ber"])
        if diff >= worst[0]:
            worst = (diff, f, m)
    put("orig-valtest-maxdiff", np.ceil(worst[0] * 1000) / 1000, "p3",  # an upper bound, so round up
        worst[1], ["models", worst[2], "|meta/val_ber - thesis256/native_tau_0.50/ber| (max of 4)"])


def sbu_label_rise():
    o = load(PHD / "obj2_sbu_analysis_20261111_115652.json")["results"]
    newp = sorted(RES.glob("sbu_testnew_analysis_*.json"))[-1]; n = load(newp)["results"]
    rises = []
    for a in ARCH:
        rises.append(n["phase1_in_domain"][a] - o["phase1_in_domain"][a]["ber"])
        rises.append(n["fair_phase2"][a]["mean"] - o["fair_phase2_on_sbu"][a]["mean"])
        rises.append(n["selection_effect"][a]["original_ber"] - o["selection_effect_on_sbu"][a]["original_ber"])
    put("sbunew-rise-min", min(rises), "p1", newp.name, ["relabelled minus original, phase1/fair-mean/original, min of 12"])
    put("sbunew-rise-max", max(rises), "p1", newp.name, ["relabelled minus original, phase1/fair-mean/original, max of 12"])


def valsize_numbers():
    """Validation-size simulation on FDRNet (code/val_size_analysis.py)."""
    p = sorted(RES.glob("val_size_analysis_*.json"))[-1]; a = load(p); r = a["results"]
    for k in ("2", "4", "6", "9", "15", "23"):
        put(f"vs-k{k}-cost", r[k]["cost_vs_last"]["mean"], "s3", p.name, ["results", k, "cost_vs_last", "mean"])
        put(f"vs-k{k}-flatter", -r[k]["optimism"]["mean"], "p2", p.name, ["results", k, "-optimism", "mean"])
        put(f"vs-k{k}-pworse", 100 * r[k]["p_worse"], "int", p.name, ["results", k, "p_worse x100"])
    put("vs-oracle", np.mean([r[k]["oracle_gain"]["mean"] for k in r]), "s3", p.name, ["results", "*", "oracle_gain", "mean (avg over k)"])
    put("vs-splits", a["config"]["splits"], "intc", p.name, ["config", "splits"])
    o = a["observed_training_scene_validation"]
    put("vs-obs-cost", o["val_minus_last_9_training_scenes"], "s3", p.name, ["observed_training_scene_validation", "val_minus_last_9_training_scenes"])
    put("vs-obs-draws", o["draws"], "intc", p.name, ["observed_training_scene_validation", "draws"])
    put("vs-obs-max", o["test_scene_split_cost_max"], "s3", p.name, ["observed_training_scene_validation", "test_scene_split_cost_max"])
    put("vs-obs-p95", o["test_scene_split_cost_p95"], "s3", p.name, ["observed_training_scene_validation", "test_scene_split_cost_p95"])


def mdd_numbers():
    """Minimum detectable difference (code/mdd_analysis.py)."""
    p = sorted(RES.glob("mdd_analysis_*.json"))[-1]; m = load(p)["results"]
    for tag, key in (("istd", "istd_native_with_fdrnet_sddnet"), ("sbu", "sbu_original"), ("sbunew", "sbu_relabelled")):
        x = m[key]
        put(f"mdd-{tag}-sigma", x["sigma_s_pooled"], "p3", p.name, ["results", key, "sigma_s_pooled"])
        put(f"mdd-{tag}-sigma-lo", x["sigma_s_pooled_ci95"][0], "p3", p.name, ["results", key, "sigma_s_pooled_ci95", 0])
        put(f"mdd-{tag}-sigma-hi", x["sigma_s_pooled_ci95"][1], "p3", p.name, ["results", key, "sigma_s_pooled_ci95", 1])
        put(f"mdd-{tag}-sdt", x["mdd"]["1"]["sd_t"], "p2", p.name, ["results", key, "mdd", "1", "sd_t"])
        for n in ("1", "3", "5", "10"):
            for f, short in (("mdd_sig", "sig"), ("mdd_pow80", "pow"), ("mdd_seed_only", "seed")):
                put(f"mdd-{tag}-{short}-n{n}", x["mdd"][n][f], "p2", p.name, ["results", key, "mdd", n, f])
    put("mdd-fdr-sigma", m["istd_native_with_fdrnet_sddnet"]["sigma_s"]["fdrnet"], "p3", p.name,
        ["results", "istd_native_with_fdrnet_sddnet", "sigma_s", "fdrnet"])
    # earlier pool without SDDNet (four development models + FDRNet), kept for the limitations paragraph
    x = m["istd_native_with_fdrnet"]; key = "istd_native_with_fdrnet"
    put("mdd-istd5-sigma", x["sigma_s_pooled"], "p3", p.name, ["results", key, "sigma_s_pooled"])
    for n in ("1", "5"):
        put(f"mdd-istd5-seed-n{n}", x["mdd"][n]["mdd_seed_only"], "p2", p.name, ["results", key, "mdd", n, "mdd_seed_only"])
        put(f"mdd-istd5-sig-n{n}", x["mdd"][n]["mdd_sig"], "p2", p.name, ["results", key, "mdd", n, "mdd_sig"])


def leaderboard_numbers():
    """Published claimed gains against the thresholds (code/leaderboard_analysis.py)."""
    p = sorted(RES.glob("leaderboard_analysis_*.json"))[-1]; a = load(p)
    rows_ = a["rows"]
    put("lb-papers", len({r["paper"] for r in rows_}), "int", p.name, ["rows", "distinct paper"])
    put("lb-rows", len(rows_), "int", p.name, ["rows", "count"])
    for d, tag in (("ISTD", "istd"), ("SBU", "sbu")):
        s = a["summary"][d]
        put(f"lb-{tag}-n", s["papers"], "int", p.name, ["summary", d, "papers"])
        put(f"lb-{tag}-below1", s["n_below_seed_only"], "int", p.name, ["summary", d, "n_below_seed_only"])
        put(f"lb-{tag}-belowint", s["n_below_interval"], "int", p.name, ["summary", d, "n_below_interval"])
        put(f"lb-{tag}-below5", s["n_below_seed_only_n5"], "int", p.name, ["summary", d, "n_below_seed_only_n5"])
        rec = [r for r in rows_ if r["dataset"] == d and int(r["year"]) >= 2019]
        g = [r["claimed_gain"] for r in rec]
        put(f"lb-{tag}-recent-n", len(rec), "int", p.name, ["rows", d, "year>=2019", "count"])
        put(f"lb-{tag}-recent-min", min(g), "p2", p.name, ["rows", d, "year>=2019", "min claimed_gain"])
        put(f"lb-{tag}-recent-max", max(g), "p2", p.name, ["rows", d, "year>=2019", "max claimed_gain"])
        put(f"lb-{tag}-recent-below1", sum(r["below_seed_only"] for r in rec), "int", p.name, ["rows", d, "year>=2019", "below_seed_only"])
        put(f"lb-{tag}-recent-below5", sum(r["below_seed_only_n5"] for r in rec), "int", p.name, ["rows", d, "year>=2019", "below_seed_only_n5"])
        old = [r["claimed_gain"] for r in rows_ if r["dataset"] == d and int(r["year"]) < 2019]
        put(f"lb-{tag}-early-min", min(old), "p2", p.name, ["rows", d, "year<2019", "min claimed_gain"])
        put(f"lb-{tag}-early-max", max(old), "p2", p.name, ["rows", d, "year<2019", "max claimed_gain"])


def main():
    original_checkpoints(); seed_bias(); fair_and_selection(); error_components()
    valsize_numbers(); mdd_numbers(); leaderboard_numbers()
    sbu("sbu", PHD / "obj2_sbu_analysis_20261111_115652.json",
        ("phase1_in_domain", "fair_phase2_on_sbu", "selection_effect_on_sbu"))
    new = sorted(RES.glob("sbu_testnew_analysis_*.json"))[-1]
    sbu("sbunew", new, ("phase1_in_domain", "fair_phase2", "selection_effect"))
    histories(); derived(); selection_evidence(); sbu_label_rise(); fdrnet(); sddnet()
    tex = ["% generated by code/make_numbers.py -- do not edit",
           "\\makeatletter",
           "\\newcommand{\\nv}[1]{\\ifcsname nv@#1\\endcsname\\csname nv@#1\\endcsname\\else"
           "\\errmessage{numbers.tex: undefined number key #1}\\fi}"]
    tex += [f"\\expandafter\\def\\csname nv@{k}\\endcsname{{{v}}}" for k, v, *_ in rows]
    tex.append("\\makeatother")
    (OUT / "numbers.tex").write_text("\n".join(tex) + "\n")
    (OUT / "numbers.tsv").write_text("key\tvalue\tsource\tpath\n" + "\n".join("\t".join(r) for r in rows) + "\n")
    print(f"{len(rows)} numbers written")


if __name__ == "__main__":
    main()
