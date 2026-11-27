"""bercheck command line.

  bercheck report --method FDRNet runs/fdrnet/*_counts.npz --method SDDNet runs/sddnet/*_counts.npz \
                  --rule last --out report.md --json report.json

Each --method takes a name followed by one .npz file per seed (see bercheck.io for the format).
"""
from __future__ import annotations

import argparse
import json
import sys

from .core import Method
from .io import load_run
from .report import analyse, markdown


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bercheck", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="analyse runs and write a Markdown report with a filled-in checklist")
    r.add_argument("--method", nargs="+", action="append", required=True, metavar=("NAME", "NPZ"),
                   help="method name followed by one counts file per seed")
    r.add_argument("--rule", default="last", help="rule used to compare methods: last, val, test or epochN (default last)")
    r.add_argument("--units", choices=("scene", "image"), default="scene",
                   help="bootstrap unit: scene = file-name stem before the first separator (ISTD), image = each image")
    r.add_argument("--sep", default="-", help="scene separator in file names (default '-')")
    r.add_argument("--test-key", default="test"); r.add_argument("--val-key", default="val")
    r.add_argument("--names-key", default="test_names")
    r.add_argument("--draws", type=int, default=10_000); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", help="Markdown report path (default: stdout)")
    r.add_argument("--json", help="also write the full results as JSON")
    a = ap.parse_args(argv)

    methods = []
    for spec in a.method:
        if len(spec) < 2:
            ap.error("--method needs a name and at least one file")
        runs = [load_run(p, test_key=a.test_key, val_key=a.val_key, names_key=a.names_key) for p in spec[1:]]
        methods.append(Method(spec[0], runs, rule=a.rule))
    unit_fn = None if a.units == "scene" else (lambda n: n)
    if a.units == "scene" and a.sep != "-":
        unit_fn = lambda n: str(n).rsplit("/", 1)[-1].split(a.sep)[0]  # noqa: E731
    res = analyse(methods, unit_fn=unit_fn, draws=a.draws, seed=a.seed)
    md = markdown(res)
    if a.out:
        open(a.out, "w").write(md)
    else:
        sys.stdout.write(md)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
