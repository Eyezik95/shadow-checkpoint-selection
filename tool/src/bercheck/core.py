"""Core computations: BER, selection rules, paired unit bootstrap, seed variance, rank stability, MDD.

Conventions
-----------
* Counts are arrays whose last axis is (TP, TN, P, N): true positives, true negatives, positives (shadow pixels
  in the reference) and negatives. BER = 100 * (1 - (TP/P + TN/N) / 2), pooled over every pixel of every image
  it is computed on.
* A run is one training run (one seed): test counts per epoch and image, shape (E, I, 4), and optionally
  validation counts (E, J, 4). A single evaluated checkpoint is E = 1.
* Bootstrap units are the independent units of the test set (ISTD: scenes; without scene labels: images).
  Image counts are summed into units; resampling draws units with replacement. The same draws are shared by
  every run, rule and method, so all differences are paired. Selected epochs are held fixed in the bootstrap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats

DRAWS = 10_000
RNG_SEED = 0
Z95, Z80 = 1.96, 0.84          # as in the paper: 1.96 (two-sided 95%), 1.96 + 0.84 = 2.80 (80% power)


def ber(c: np.ndarray) -> np.ndarray:
    """(..., 4) counts (TP, TN, P, N) -> pooled BER in percent."""
    c = np.asarray(c, dtype=np.float64)
    tp, tn, p, n = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    return 100.0 * (1.0 - (tp / p + tn / n) / 2.0)


def to_tp_tn_p_n(c: np.ndarray, columns: Sequence[str]) -> np.ndarray:
    """Convert counts in any of the usual column orders to (TP, TN, P, N)."""
    cols = [str(x).upper() for x in columns]
    c = np.asarray(c, dtype=np.float64)
    g = {k: c[..., cols.index(k)] for k in cols}
    if set(cols) == {"TP", "TN", "P", "N"}:
        return np.stack([g["TP"], g["TN"], g["P"], g["N"]], axis=-1)
    if set(cols) == {"TP", "FN", "TN", "FP"}:
        return np.stack([g["TP"], g["TN"], g["TP"] + g["FN"], g["TN"] + g["FP"]], axis=-1)
    raise ValueError(f"unsupported columns {columns}; use (TP, TN, P, N) or (TP, FN, TN, FP)")


def default_unit(name: str) -> str:
    """ISTD convention: the scene is the part of the file stem before the first '-'."""
    return str(name).rsplit("/", 1)[-1].split("-")[0]


def _unit_key(u: str):
    return (len(u), u)          # numeric strings sort numerically; others lexicographically within a length


@dataclass
class Run:
    test: np.ndarray                       # (E, I, 4) TP, TN, P, N
    names: Sequence[str]                   # I test image names
    val: Optional[np.ndarray] = None       # (E, J, 4)
    label: str = ""

    def __post_init__(self):
        self.test = np.asarray(self.test, dtype=np.float64)
        if self.test.ndim == 2:
            self.test = self.test[None]
        if self.val is not None:
            self.val = np.asarray(self.val, dtype=np.float64)
            if self.val.ndim == 2:
                self.val = self.val[None]
            if self.val.shape[0] != self.test.shape[0]:
                raise ValueError("val and test must have the same number of epochs")
        if self.test.shape[1] != len(self.names):
            raise ValueError("number of test names differs from the counts")
        if not np.all(np.isfinite(self.test)):
            raise ValueError(f"non-finite counts in {self.label}")

    @property
    def epochs(self) -> int:
        return self.test.shape[0]

    def test_ber(self) -> np.ndarray:
        return ber(self.test.sum(axis=1))

    def val_ber(self) -> Optional[np.ndarray]:
        return None if self.val is None else ber(self.val.sum(axis=1))


RULES = {
    "last": "final epoch (fixed schedule)",
    "val": "lowest validation BER",
    "test": "lowest test BER (selection on the test set)",
}


def select(run: Run, rule: str) -> int:
    """0-based epoch index chosen by a rule."""
    if rule == "last":
        return run.epochs - 1
    if rule == "test":
        return int(np.argmin(run.test_ber()))
    if rule == "val":
        if run.val is None:
            raise ValueError("rule 'val' needs validation counts")
        return int(np.argmin(run.val_ber()))
    if rule.startswith("epoch"):
        return int(rule[5:]) - 1
    raise ValueError(rule)


class Units:
    """Maps the test images of a set of runs to bootstrap units and holds the shared resamples."""

    def __init__(self, names: Sequence[str], unit_fn: Optional[Callable[[str], str]] = default_unit,
                 draws: int = DRAWS, seed: int = RNG_SEED):
        self.names = [str(n) for n in names]
        lab = [unit_fn(n) for n in self.names] if unit_fn else self.names
        self.labels = sorted(set(lab), key=_unit_key)
        pos = {u: k for k, u in enumerate(self.labels)}
        self.index = np.array([pos[u] for u in lab])
        self.n = len(self.labels)
        rng = np.random.default_rng(seed)
        self.draws = rng.integers(0, self.n, size=(draws, self.n))

    def sum(self, counts: np.ndarray) -> np.ndarray:
        """(..., I, 4) -> (..., U, 4)."""
        out = np.zeros(counts.shape[:-2] + (self.n, 4))
        for k in range(self.n):
            out[..., k, :] = counts[..., self.index == k, :].sum(axis=-2)
        return out

    def boot(self, unit_counts: np.ndarray) -> np.ndarray:
        """(U, 4) -> bootstrap BER (DRAWS,)."""
        return ber(unit_counts[self.draws].sum(axis=1))


def align(run: Run, names: Sequence[str]) -> Run:
    """Reorder a run's test images to `names` (matching on file stem)."""
    stem = lambda s: str(s).rsplit("/", 1)[-1].rsplit(".", 1)[0]  # noqa: E731
    have = {stem(n): i for i, n in enumerate(run.names)}
    try:
        order = [have[stem(n)] for n in names]
    except KeyError as e:
        raise ValueError(f"{run.label}: test image {e} missing") from None
    return Run(run.test[:, order], [run.names[i] for i in order], run.val, run.label)


def ci95(x: np.ndarray) -> list:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


@dataclass
class Method:
    name: str
    runs: list
    rule: str = "last"
    _u: dict = field(default_factory=dict, repr=False)


def rule_analysis(m: Method, units: Units, rules=("last", "val", "test")) -> dict:
    """Per seed and over seeds: BER under each rule and paired differences between rules."""
    rules = [r for r in rules if r != "val" or all(r_.val is not None for r_ in m.runs)]
    if m.runs[0].epochs == 1:
        rules = ["last"]
    per_unit = [units.sum(r.test) for r in m.runs]                     # (E, U, 4) per seed
    sel = [{r: select(run, r) for r in rules} for run in m.runs]
    boot = np.array([[units.boot(per_unit[i][sel[i][r]]) for r in rules] for i in range(len(m.runs))])
    tb = [run.test_ber() for run in m.runs]
    vb = [run.val_ber() for run in m.runs]
    out = {"rules": rules, "per_seed": [], "summary": {}, "differences": {}}
    for i, run in enumerate(m.runs):
        row = {"run": run.label, "rules": {}, "differences": {}}
        for r in rules:
            e = sel[i][r]
            row["rules"][r] = {"epoch": e + 1, "test_ber": float(tb[i][e]),
                               "val_ber": None if vb[i] is None else float(vb[i][e])}
        for a, b in [d for d in (("test", "last"), ("val", "last"), ("test", "val")) if set(d) <= set(rules)]:
            ja, jb = rules.index(a), rules.index(b)
            row["differences"][f"{a}-{b}"] = {"diff": float(tb[i][sel[i][a]] - tb[i][sel[i][b]]),
                                              "ci95": ci95(boot[i, ja] - boot[i, jb]),
                                              "same_checkpoint": sel[i][a] == sel[i][b]}
        out["per_seed"].append(row)
    for j, r in enumerate(rules):
        v = np.array([tb[i][sel[i][r]] for i in range(len(m.runs))])
        out["summary"][r] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                             "epochs": [sel[i][r] + 1 for i in range(len(m.runs))]}
    for key in out["per_seed"][0]["differences"]:
        a, b = key.split("-")
        ja, jb = rules.index(a), rules.index(b)
        per = np.array([row["differences"][key]["diff"] for row in out["per_seed"]])
        out["differences"][key] = {"mean": float(per.mean()), "ci95": ci95(boot[:, ja].mean(0) - boot[:, jb].mean(0)),
                                   "per_seed": per.tolist()}
    return out


def chosen(m: Method, units: Units) -> np.ndarray:
    """(seeds, U, 4) unit counts of the checkpoint each run keeps under the method's rule."""
    return np.stack([units.sum(run.test)[select(run, m.rule)] for run in m.runs])


def seed_stats(per_seed_ber: np.ndarray) -> dict:
    k = len(per_seed_ber)
    if k < 2:
        return {"seeds": k, "mean": float(per_seed_ber.mean()), "sd": None, "sd_ci95": None}
    v = per_seed_ber.var(ddof=1)
    chi = stats.chi2.ppf([0.975, 0.025], k - 1)
    return {"seeds": k, "mean": float(per_seed_ber.mean()), "sd": float(np.sqrt(v)),
            "sd_ci95": [float(np.sqrt((k - 1) * v / chi[0])), float(np.sqrt((k - 1) * v / chi[1]))],
            "range": float(per_seed_ber.max() - per_seed_ber.min())}


def compare(methods: Sequence[Method], units: Units, max_combos: int = 100_000, seed: int = RNG_SEED) -> dict:
    """Seed means, paired differences with unit-bootstrap intervals, and rank stability over single runs."""
    pu = {m.name: chosen(m, units) for m in methods}
    sb = {n: ber(c.sum(axis=1)) for n, c in pu.items()}                                  # (seeds,)
    boot = {n: np.stack([units.boot(c[s]) for s in range(len(c))]) for n, c in pu.items()}  # (seeds, DRAWS)
    means = {n: float(v.mean()) for n, v in sb.items()}
    order = sorted(means, key=means.get)
    pairs = {}
    for a, b in combinations(order, 2):
        d = boot[a].mean(0) - boot[b].mean(0)
        single = [float(np.mean(sb[a][i] < sb[b][j])) for i in range(len(sb[a])) for j in range(len(sb[b]))]
        pairs[f"{a} - {b}"] = {"diff": means[a] - means[b], "ci95": ci95(d),
                               "excludes_zero": bool(ci95(d)[0] > 0 or ci95(d)[1] < 0),
                               "p_single_run_order_holds": float(np.mean(single))}
    # rank stability: one seed per method, all combinations (or a random sample)
    sizes = [len(sb[n]) for n in order]
    total = int(np.prod(sizes))
    if total <= max_combos:
        combos = np.array(list(product(*[range(s) for s in sizes])))
    else:
        rng = np.random.default_rng(seed)
        combos = np.stack([rng.integers(0, s, max_combos) for s in sizes], axis=1)
    vals = np.stack([sb[n][combos[:, j]] for j, n in enumerate(order)], axis=1)            # (combos, methods)
    ranks = vals.argsort(axis=1).argsort(axis=1)
    rank_freq = {n: np.bincount(ranks[:, j], minlength=len(order)).astype(float) / len(combos)
                 for j, n in enumerate(order)}
    same_order = float(np.mean(np.all(ranks == np.arange(len(order)), axis=1)))
    return {"order_by_mean": order,
            "methods": {n: {**seed_stats(sb[n]), "rule": next(m.rule for m in methods if m.name == n),
                            "per_seed": sb[n].tolist()} for n in order},
            "pairs": pairs,
            "rank_stability": {"single_run_draws": int(len(combos)), "p_same_order_as_means": same_order,
                               "rank_probability": {n: f.tolist() for n, f in rank_freq.items()}}}


def mdd(methods: Sequence[Method], units: Units, ns=(1, 2, 3, 5, 10)) -> dict:
    """Minimum detectable difference between two methods with n seeds each.

    sigma_s^2 : seed variance, pooled over methods (mean of per-method sample variances).
    sigma_t^2(n): paired bootstrap variance of a between-method difference, median over pairs, for single runs
                  (n=1) and for k-seed means (k = seeds per method), interpolated as a + b/n.
    Thresholds: seed_only = 1.96 sqrt(2 s^2/n); interval = 1.96 SE; power80 = (1.96 + 0.84) SE.
    """
    pu = {m.name: chosen(m, units) for m in methods}
    sb = {n: ber(c.sum(axis=1)) for n, c in pu.items()}
    ks = {len(v) for v in sb.values()}
    if min(ks) < 2:
        raise ValueError("MDD needs at least two seeds per method")
    k = min(ks)
    var_s = {n: float(v.var(ddof=1)) for n, v in sb.items()}
    pooled = float(np.sqrt(np.mean(list(var_s.values()))))
    dfp = sum(len(v) - 1 for v in sb.values())
    chip = stats.chi2.ppf([0.975, 0.025], dfp)
    out = {"seeds_per_method": k, "units": units.n,
           "sigma_s": {n: float(np.sqrt(v)) for n, v in var_s.items()}, "sigma_s_pooled": pooled,
           "sigma_s_pooled_ci95": [float(np.sqrt(dfp * pooled ** 2 / chip[0])), float(np.sqrt(dfp * pooled ** 2 / chip[1]))]}
    if len(methods) < 2:
        out["note"] = "one method: test-sampling variance of a difference needs two methods; seed-only thresholds only"
        out["mdd"] = {str(n): {"seed_only": float(Z95 * np.sqrt(2 * pooled ** 2 / n))} for n in ns}
        return out
    boot = {n: np.stack([units.boot(c[s]) for s in range(len(c))]) for n, c in pu.items()}
    v1, vk = [], []
    for a, b in combinations(boot, 2):
        v1 += [float((boot[a][s] - boot[b][s]).var(ddof=1)) for s in range(k)]
        vk.append(float((boot[a][:k].mean(0) - boot[b][:k].mean(0)).var(ddof=1)))
    v1m, vkm = float(np.median(v1)), float(np.median(vk))
    bcoef = (v1m - vkm) / (1 - 1 / k); acoef = v1m - bcoef
    table = {}
    for n in ns:
        vt = max(acoef + bcoef / n, 0.0)
        se = float(np.sqrt(2 * pooled ** 2 / n + vt))
        table[str(n)] = {"seed_only": float(Z95 * np.sqrt(2 * pooled ** 2 / n)), "interval": Z95 * se,
                         "power80": (Z95 + Z80) * se, "sd_test_sampling": float(np.sqrt(vt)), "se": se}
    out.update({"var_t": {"single_run_median": v1m, f"{k}_seed_mean_median": vkm, "a": acoef, "b": bcoef},
                "mdd": table})
    return out
