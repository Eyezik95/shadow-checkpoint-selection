"""Published state-of-the-art steps on ISTD and SBU against the minimum detectable difference.

Input: paper/leaderboard.tsv, one row per (paper, dataset), values copied from each paper's OWN main comparison
table and checked against the source (column `checked` must be 'yes' for a row to be used):
  paper, venue, year, dataset, own_ber, best_other_ber, best_other_method, source, table, note, checked
claimed_gain = best_other_ber - own_ber (positive = the paper reports an improvement over the best method in its table).
Thresholds come from the latest results/mdd_analysis_*.json at n = 1 (single run per method, as published):
  seed-only   : gap needed to beat seed noise on the fixed test set (1.96 sqrt(2) sigma_s);
  interval    : gap needed for a 95% interval over test scenes/images and seeds to exclude zero.
Caveat written to the output: thresholds were measured at our models' BER level (ISTD about 1.7-3.2%,
SBU about 5.6-7.3%); published BERs are lower, and absolute noise may shrink with BER.
"""
import csv
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = {"ISTD": "istd_native_with_fdrnet_sddnet", "SBU": "sbu_original"}


def main():
    mdd_p = sorted((ROOT / "results").glob("mdd_analysis_*.json"))[-1]
    mdd = json.loads(mdd_p.read_text())["results"]
    thr = {d: {"seed_only": mdd[k]["mdd"]["1"]["mdd_seed_only"], "interval": mdd[k]["mdd"]["1"]["mdd_sig"],
               "seed_only_n5": mdd[k]["mdd"]["5"]["mdd_seed_only"]} for d, k in KEYS.items()}
    rows = [r for r in csv.DictReader(open(ROOT / "paper" / "leaderboard.tsv"), delimiter="\t")]
    used = [r for r in rows if r["checked"].strip().lower() == "yes" and r["own_ber"] and r["best_other_ber"]]
    out_rows, summ = [], {}
    for r in used:
        d = r["dataset"]
        g = float(r["best_other_ber"]) - float(r["own_ber"])
        t = thr[d]
        out_rows.append({**r, "claimed_gain": round(g, 3), "below_seed_only": g < t["seed_only"],
                         "below_interval": g < t["interval"], "below_seed_only_n5": g < t["seed_only_n5"]})
    for d in KEYS:
        rs = [x for x in out_rows if x["dataset"] == d]
        summ[d] = {"papers": len(rs), "thresholds": thr[d],
                   "n_below_seed_only": sum(x["below_seed_only"] for x in rs),
                   "n_below_interval": sum(x["below_interval"] for x in rs),
                   "n_below_seed_only_n5": sum(x["below_seed_only_n5"] for x in rs),
                   "gains": sorted(x["claimed_gain"] for x in rs)}
    ts = time.strftime("%Y%m%d_%H%M%S")
    res = {"experiment": "leaderboard_vs_mdd", "created": ts,
           "config": {"input": "paper/leaderboard.tsv", "rows_total": len(rows), "rows_used": len(used),
                      "mdd_source": mdd_p.name,
                      "caveat": "thresholds measured at our models' BER level; published BERs are lower"},
           "summary": summ, "rows": out_rows}
    p = ROOT / "results" / f"leaderboard_analysis_{ts}.json"; p.write_text(json.dumps(res, indent=1)); print(p)
    for d, s in summ.items():
        print(d, {k: v for k, v in s.items() if k != "gains"}, "gains", s["gains"])
    for x in out_rows:
        print(f"{x['dataset']:4s} {x['paper']:18s} {x['year']} own {x['own_ber']:>6} best-other {x['best_other_ber']:>6} "
              f"({x['best_other_method']}) gain {x['claimed_gain']:+.2f}")


if __name__ == "__main__":
    main()
