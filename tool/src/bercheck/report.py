"""Full analysis of one or more methods and a Markdown report with a filled-in reporting checklist."""
from __future__ import annotations

import numpy as np
from scipy import stats

from .core import Method, Units, align, compare, mdd, rule_analysis


def analyse(methods, unit_fn=None, draws=10_000, seed=0, ns=(1, 2, 3, 5, 10)) -> dict:
    """methods: list of Method. All runs are aligned to the test images of the first run."""
    ref = methods[0].runs[0].names
    for m in methods:
        m.runs = [align(r, ref) for r in m.runs]
    kw = {} if unit_fn is None else {"unit_fn": unit_fn}
    units = Units(ref, draws=draws, seed=seed, **kw)
    res = {"config": {"draws": draws, "rng_seed": seed, "bootstrap_units": units.n, "test_images": len(ref),
                      "methods": {m.name: {"runs": [r.label for r in m.runs], "epochs": m.runs[0].epochs,
                                           "rule": m.rule, "has_val": all(r.val is not None for r in m.runs)}
                                  for m in methods}},
           "rules": {m.name: rule_analysis(m, units) for m in methods if m.runs[0].epochs > 1},
           "curve_agreement": {m.name: curve_agreement(m) for m in methods
                               if m.runs[0].epochs > 2 and all(r.val is not None for r in m.runs)},
           "comparison": compare(methods, units)}
    if all(len(m.runs) >= 2 for m in methods):
        res["mdd"] = mdd(methods, units, ns=ns)
        res["claims"] = size_claims(res)
    return res


def curve_agreement(m: Method) -> dict:
    """Spearman correlation between the per-epoch validation and test BER curves of each run."""
    rho = [float(stats.spearmanr(r.val_ber(), r.test_ber())[0]) for r in m.runs]
    return {"spearman_per_run": rho, "mean": float(np.mean(rho))}


def size_claims(res: dict) -> dict:
    """Classify each pairwise difference against the thresholds at the number of seeds used."""
    k = res["mdd"]["seeds_per_method"]
    sig = res["mdd"]["sigma_s_pooled"]
    seed_thr = 1.96 * np.sqrt(2 * sig ** 2 / k)
    out = {}
    for pair, v in res["comparison"]["pairs"].items():
        gap = abs(v["diff"])
        if gap < seed_thr:
            verdict = "within seed noise: not shown even on this test set"
        elif not v["excludes_zero"]:
            verdict = "beyond seed noise on this test set; not established beyond it (interval includes zero)"
        else:
            verdict = "established: paired interval over test units excludes zero"
        out[pair] = {"gap": gap, "seed_threshold": float(seed_thr), "ci95": v["ci95"], "verdict": verdict}
    return out


def f(x, d=3, sign=False):
    return "–" if x is None else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def markdown(res: dict) -> str:
    c = res["config"]
    L = ["# BER reporting check", "",
         f"Bootstrap: {c['draws']:,} paired resamples of {c['bootstrap_units']} test units "
         f"({c['test_images']} images), rng seed {c['rng_seed']}. Selected epochs are held fixed in the bootstrap.", ""]
    L += ["## Inputs", "", "| Method | Runs | Epochs per run | Validation counts | Rule used for comparison |", "|---|---|---|---|---|"]
    for n, m in c["methods"].items():
        L.append(f"| {n} | {len(m['runs'])} | {m['epochs']} | {'yes' if m['has_val'] else 'no'} | {m['rule']} |")
    if res["rules"]:
        L += ["", "## Checkpoint-selection rules (same runs, test BER %)", ""]
        for n, r in res["rules"].items():
            L += [f"### {n}", "", "| Rule | Mean | SD | Epochs chosen |", "|---|---|---|---|"]
            for rule, s in r["summary"].items():
                L.append(f"| {rule} | {f(s['mean'])} | {f(s['sd'])} | {', '.join(map(str, s['epochs']))} |")
            L += ["", "| Difference | Mean over seeds | 95% interval | Per seed |", "|---|---|---|---|"]
            for k, d in r["differences"].items():
                L.append(f"| {k} | {f(d['mean'], sign=True)} | [{f(d['ci95'][0], sign=True)}, {f(d['ci95'][1], sign=True)}] | "
                         f"{', '.join(f(x, sign=True) for x in d['per_seed'])} |")
            if n in res["curve_agreement"]:
                ca = res["curve_agreement"][n]
                L += ["", f"Validation–test curve agreement (Spearman over epochs, per run): "
                          f"{', '.join(f(x, 2) for x in ca['spearman_per_run'])}; mean {f(ca['mean'], 2)}."]
            L.append("")
    cmp_ = res["comparison"]
    L += ["## Methods", "", "| Method | Seeds | Mean BER | SD | SD 95% interval | Range |", "|---|---|---|---|---|---|"]
    for n in cmp_["order_by_mean"]:
        s = cmp_["methods"][n]
        ci = "–" if s["sd_ci95"] is None else f"{f(s['sd_ci95'][0])}–{f(s['sd_ci95'][1])}"
        L.append(f"| {n} | {s['seeds']} | {f(s['mean'])} | {f(s['sd'])} | {ci} | {f(s.get('range'))} |")
    L.append("")
    if cmp_["pairs"]:
        L += ["", "| Pair | Difference of means | 95% interval | Excludes 0 | P(single runs keep this order) |", "|---|---|---|---|---|"]
        for p, v in cmp_["pairs"].items():
            L.append(f"| {p} | {f(v['diff'], sign=True)} | [{f(v['ci95'][0], sign=True)}, {f(v['ci95'][1], sign=True)}] | "
                     f"{'yes' if v['excludes_zero'] else 'no'} | {f(v['p_single_run_order_holds'], 2)} |")
        rs = cmp_["rank_stability"]
        L += ["", f"Rank stability: drawing one run per method ({rs['single_run_draws']:,} combinations), the order of the "
                  f"means is reproduced in {100 * rs['p_same_order_as_means']:.0f}% of draws."]
    if "mdd" in res:
        m = res["mdd"]
        L += ["", "## Minimum detectable difference (BER points)", "",
              f"Seed SD pooled over methods: {f(m['sigma_s_pooled'])} (95% interval {f(m['sigma_s_pooled_ci95'][0])}–"
              f"{f(m['sigma_s_pooled_ci95'][1])}). {m.get('note', '')}", ""]
        cols = [k for k in ("seed_only", "interval", "power80") if k in next(iter(m["mdd"].values()))]
        L += ["| Seeds per method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for n, v in m["mdd"].items():
            L.append(f"| {n} | " + " | ".join(f(v[c], 2) for c in cols) + " |")
        L += ["", "seed_only: gap needed to exceed seed noise on this fixed test set. interval: observed gap whose 95% "
                  "interval over seeds and test units excludes zero. power80: true gap detected with 80% power.", ""]
    L += checklist(res)
    return "\n".join(L) + "\n"


def checklist(res: dict) -> list:
    c, cmp_ = res["config"], res["comparison"]
    L = ["## Reporting checklist", "", "| # | Item | Status from these files |", "|---|---|---|"]
    rules = ", ".join(f"{n}: {m['rule']}" for n, m in c["methods"].items())
    L.append(f"| 1 | Checkpoint rule stated | {rules}. State it in the paper; 'test' means selection on the test set. |")
    if res["curve_agreement"]:
        s = "; ".join(f"{n}: Spearman mean {f(v['mean'], 2)}, lowest run {f(min(v['spearman_per_run']), 2)}"
                      for n, v in res["curve_agreement"].items())
        L.append(f"| 2 | Validation set checked against test conditions | {s}. Low or negative agreement: prefer a fixed schedule. |")
    else:
        L.append("| 2 | Validation set checked against test conditions | no per-epoch validation counts supplied |")
    seeds = ", ".join(f"{n}: {s['seeds']} (SD {f(s['sd'])})" for n, s in cmp_["methods"].items())
    L.append(f"| 3 | Several seeds, mean and SD reported | {seeds} |")
    if "claims" in res and res["claims"]:
        s = "; ".join(f"{p}: {v['verdict']}" for p, v in res["claims"].items())
        L.append(f"| 4 | Claims sized against thresholds | {s} |")
    else:
        L.append("| 4 | Claims sized against thresholds | needs two or more methods with at least two runs each; "
                 "the seed-only threshold above applies to any comparison on this test set |")
    L.append("| 5 | Both label sets reported where relabelled data exist | not checkable from counts; state it |")
    L.append("| 6 | Per-image, per-epoch counts released | these files; release them with the paper |")
    return L
