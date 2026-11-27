"""Tests: synthetic checks, and regression against the paper's own analyses when the repository's result files
are present (they are skipped otherwise, e.g. when the package is installed on its own)."""
import glob
import json
from pathlib import Path

import numpy as np
import pytest

from bercheck import Method, Run, Units, analyse, ber, compare, mdd, rule_analysis, select, to_tp_tn_p_n
from bercheck.io import load_run

REPO = Path(__file__).resolve().parents[2]


def synth(e=6, scenes=12, per=5, bias=0.0, noise=0.002, seed=0, val=True):
    rng = np.random.default_rng(seed)
    names = [f"{s}-{i}.png" for s in range(1, scenes + 1) for i in range(1, per + 1)]
    p = rng.integers(2_000, 8_000, size=len(names)).astype(float)
    n = 60_000 - p
    def counts(err):
        tp = np.clip(p * (1 - err[..., None] - rng.normal(0, noise, (e, len(names)))), 0, p)
        tn = np.clip(n * (1 - err[..., None] - rng.normal(0, noise, (e, len(names)))), 0, n)
        return np.stack([tp, tn, np.broadcast_to(p, tp.shape), np.broadcast_to(n, tn.shape)], -1)
    err = np.linspace(0.06, 0.02, e) + bias
    return Run(counts(err), names, counts(err) if val else None, label=f"s{seed}")


def test_ber_and_columns():
    c = np.array([90, 900, 100, 1000])
    assert ber(c) == pytest.approx(100 * (1 - (0.9 + 0.9) / 2))
    alt = to_tp_tn_p_n(np.array([90, 10, 900, 100]), ["TP", "FN", "TN", "FP"])
    assert np.allclose(alt, c)
    with pytest.raises(ValueError):
        to_tp_tn_p_n(c, ["A", "B", "C", "D"])


def test_rules_select_expected_epochs():
    r = synth()
    assert select(r, "last") == r.epochs - 1
    assert select(r, "test") == int(np.argmin(r.test_ber()))
    assert select(r, "epoch2") == 1


def test_units_group_by_scene():
    r = synth(scenes=7, per=3)
    u = Units(r.names)
    assert u.n == 7 and u.labels[:3] == ["1", "2", "3"]
    assert np.allclose(u.sum(r.test).sum(-2), r.test.sum(-2))


def test_planted_difference_is_found_and_null_is_not():
    a = Method("A", [synth(seed=s) for s in range(5)])
    b = Method("B", [synth(seed=10 + s, bias=0.01) for s in range(5)])
    c = Method("C", [synth(seed=20 + s) for s in range(5)])
    u = Units(a.runs[0].names, draws=2000)
    res = compare([a, b, c], u)
    assert res["pairs"]["A - B"]["excludes_zero"] or res["pairs"]["C - B"]["excludes_zero"]
    assert res["order_by_mean"][-1] == "B"
    m = mdd([a, b, c], u)
    assert m["mdd"]["1"]["seed_only"] > m["mdd"]["5"]["seed_only"]
    assert m["mdd"]["5"]["interval"] >= m["mdd"]["5"]["seed_only"]


def test_report_runs_end_to_end():
    a = Method("A", [synth(seed=s) for s in range(3)])
    b = Method("B", [synth(seed=10 + s, bias=0.002) for s in range(3)])
    res = analyse([a, b], draws=500)
    from bercheck import markdown
    md = markdown(res)
    assert "Reporting checklist" in md and "test-last" in md and ("A - B" in md or "B - A" in md)


# ---------------------------------------------------------------- regression against the paper's result files
FDR = sorted(glob.glob(str(REPO / "results" / "fdrnet" / "fdrnet_istd_seed*_2*_counts.npz")))


@pytest.mark.skipif(len(FDR) != 5, reason="FDRNet result files not present")
def test_reproduces_fdrnet_analysis():
    ref = json.loads(sorted((REPO / "results").glob("fdrnet_analysis_*.json"))[-1].read_text())
    m = Method("FDRNet", [load_run(f) for f in FDR])
    u = Units(m.runs[0].names)
    r = rule_analysis(m, u)
    for key, rk in (("test-last", "test_minus_last"), ("val-last", "val_minus_last"), ("test-val", "test_minus_val")):
        assert r["differences"][key]["mean"] == pytest.approx(ref["mean_diffs"][rk]["mean"], abs=1e-12)
        assert r["differences"][key]["ci95"] == pytest.approx(ref["mean_diffs"][rk]["bootstrap_scenes_ci95"], abs=1e-12)
        for row, rrow in zip(r["per_seed"], ref["per_seed"]):
            if not row["differences"][key]["same_checkpoint"]:
                assert row["differences"][key]["ci95"] == pytest.approx(rrow["diffs"][rk]["bootstrap_scenes_ci95"], abs=1e-12)
    for rule in ("last", "val", "test"):
        assert r["summary"][rule]["mean"] == pytest.approx(ref["summary"][rule]["mean"], abs=1e-12)
        assert r["summary"][rule]["epochs"] == ref["summary"][rule]["epochs"]


DEV = REPO / "dev_models" / "results" / "obj2"


SDD = sorted(glob.glob(str(REPO / "results" / "sddnet" / "sddnet_istd_seed*_2*_counts.npz")))


@pytest.mark.skipif(not DEV.exists() or len(FDR) != 5, reason="development-model result files not present")
@pytest.mark.parametrize("with_sddnet", [False, True])
def test_reproduces_mdd_analysis_istd(with_sddnet):
    if with_sddnet and len(SDD) != 5:
        pytest.skip("SDDNet result files not present")
    import importlib.util
    spec = importlib.util.spec_from_file_location("mdd_analysis", REPO / "code" / "mdd_analysis.py")
    ma = importlib.util.module_from_spec(spec); spec.loader.exec_module(ma)
    ref = json.loads(sorted((REPO / "results").glob("mdd_analysis_*.json"))[-1].read_text())["results"][
        "istd_native_with_fdrnet_sddnet" if with_sddnet else "istd_native_with_fdrnet"]
    z = np.load(ma.istd_dev_files("ran")[0]); names = [str(x) for x in z["images"]]
    methods = []
    for m in ma.MODELS:
        runs = [Run(ma.dev(np.load(f)["native_counts"][:, ma.I50]), names, label=f.name) for f in ma.istd_dev_files(m)]
        methods.append(Method(m, runs))
    fdr = [load_run(f) for f in FDR]
    methods.append(Method("fdrnet", [Run(r.test[-1], r.names, label=r.label) for r in fdr]))
    from bercheck.core import align
    methods[-1].runs = [align(r, names) for r in methods[-1].runs]
    if with_sddnet:
        sdd = [load_run(f) for f in SDD]
        methods.append(Method("sddnet", [align(Run(r.test[-1], r.names, label=r.label), names) for r in sdd]))
    out = mdd(methods, Units(names))
    assert out["sigma_s_pooled"] == pytest.approx(ref["sigma_s_pooled"], abs=1e-12)
    for n in ("1", "3", "5", "10"):
        assert out["mdd"][n]["seed_only"] == pytest.approx(ref["mdd"][n]["mdd_seed_only"], abs=1e-9)
        assert out["mdd"][n]["interval"] == pytest.approx(ref["mdd"][n]["mdd_sig"], abs=1e-9)
        assert out["mdd"][n]["power80"] == pytest.approx(ref["mdd"][n]["mdd_pow80"], abs=1e-9)
