"""Objective 2 metrics: BER from pooled counts (Eq 3.13), boundary band, and
cross-fitted thresholds that never evaluate an image at a threshold chosen on it."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.shadow_metrics import THRESHOLDS, ber_from_counts, cross_fit, image_counts  # noqa: E402
from src.models.losses import balanced_error_rate  # noqa: E402


def test_ber_matches_reference_implementation():
    rng = np.random.default_rng(0)
    prob = rng.random((40, 50)); target = rng.random((40, 50)) > 0.7
    c = image_counts(prob, target)
    for i, th in enumerate(THRESHOLDS):
        ref = balanced_error_rate(torch.tensor(prob), torch.tensor(target, dtype=torch.float32), th)
        assert ber_from_counts(c["counts"][i])["ber"] == pytest.approx(ref["ber"])


def test_boundary_band_counts_only_errors_near_the_edge():
    target = np.zeros((60, 60), bool); target[20:40, 20:40] = True
    prob = target.astype(float).copy()
    prob[0, 0] = 1.0                     # error far from any boundary
    prob[20, 25] = 0.0                   # error on the boundary
    c = image_counts(prob, target, band_widths=(2,))
    i = THRESHOLDS.index(0.5)
    assert c["band_err"][i, 0] == 1 and c["counts"][i, 3] == 1 and c["counts"][i, 1] == 1


def test_cross_fit_uses_the_other_groups_threshold():
    # group 0 is best at a low threshold, group 1 at a high one
    counts = np.zeros((2, len(THRESHOLDS), 4), np.int64)
    for i, th in enumerate(THRESHOLDS):
        counts[0, i] = (100, int(th * 100), 100, 0)          # FN grows with threshold
        counts[1, i] = (100, 0, 100, int((1 - th) * 100))     # FP shrinks with threshold
    r = cross_fit(counts, np.array([0, 1]))
    assert r["thresholds_chosen"] == {"on_group_0_applied_to_1": 0.05, "on_group_1_applied_to_0": 0.95}
    expected = counts[1, 0] + counts[0, -1]
    assert np.array_equal(r["pooled_counts"], expected)


def test_size_mismatch_and_nan_fail_loudly():
    with pytest.raises(ValueError):
        image_counts(np.zeros((4, 4)), np.zeros((5, 4), bool))
    with pytest.raises(FloatingPointError):
        image_counts(np.full((4, 4), np.nan), np.zeros((4, 4), bool))
