"""Shadow-detection metrics (Objective 2; Equation 3.13).

    BER = 0.5 * ( FN / (TP + FN) + FP / (TN + FP) ) * 100                   (3.13)

computed from pixel counts pooled over the whole evaluation set (as the
training notebook's ConfusionAccumulator does), not averaged per image.

`image_counts` returns, for one image, the confusion counts at every threshold
of the sweep plus the boundary-band counts: the boundary is the set of pixels
whose 3x3 neighbourhood in the reference mask is not uniform, and the band of
width k is that boundary dilated by a (2k+1)x(2k+1) square. Boundary-band error
= misclassified pixels inside the band / pixels in the band. Storing counts per
image lets thresholds be chosen on one subset of images and applied to another.

`cross_fit` implements the threshold selection used here: images are split
into two groups by scene, the BER-minimising threshold of each group is applied
to the other group, and the resulting counts are pooled. No image is ever
evaluated at a threshold chosen on itself.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["THRESHOLDS", "BAND_WIDTHS", "image_counts", "ber_from_counts", "cross_fit"]

THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))      # 0.05 ... 0.95, as Section 4.4.1
BAND_WIDTHS = (2, 5, 10)


def _band(mask: np.ndarray, k: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(m, kernel3) != cv2.erode(m, kernel3)
    return cv2.dilate(edge.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool)


def image_counts(prob: np.ndarray, target: np.ndarray, thresholds=THRESHOLDS, band_widths=BAND_WIDTHS) -> dict:
    """prob: HxW shadow probability; target: HxW binary reference mask (same size)."""
    if prob.shape != target.shape:
        raise ValueError(f"prediction {prob.shape} and target {target.shape} differ in size")
    if not np.isfinite(prob).all():
        raise FloatingPointError("probability map is not finite")
    t = target.astype(bool)
    pos, neg = t.sum(), (~t).sum()
    counts = np.zeros((len(thresholds), 4), dtype=np.int64)              # TP, FN, TN, FP
    bands = {k: _band(t, k) for k in band_widths}
    band_err = np.zeros((len(thresholds), len(band_widths)), dtype=np.int64)
    for i, th in enumerate(thresholds):
        p = prob > th
        tp = int((p & t).sum()); fp = int((p & ~t).sum())
        counts[i] = (tp, int(pos) - tp, int(neg) - fp, fp)
        wrong = p != t
        for j, k in enumerate(band_widths):
            band_err[i, j] = int((wrong & bands[k]).sum())
    return {"counts": counts, "band_err": band_err,
            "band_px": np.array([int(bands[k].sum()) for k in band_widths], dtype=np.int64)}


def ber_from_counts(c: np.ndarray) -> dict:
    """c: (..., 4) TP, FN, TN, FP summed over images -> BER and its components (%)."""
    tp, fn, tn, fp = (c[..., i].astype(np.float64) for i in range(4))
    shadow_err = fn / np.maximum(tp + fn, 1) * 100.0
    nonshadow_err = fp / np.maximum(tn + fp, 1) * 100.0
    iou = tp / np.maximum(tp + fn + fp, 1)
    return {"ber": 0.5 * (shadow_err + nonshadow_err), "shadow_err": shadow_err,
            "nonshadow_err": nonshadow_err, "iou_pooled": iou}


def cross_fit(counts: np.ndarray, groups: np.ndarray, thresholds=THRESHOLDS) -> dict:
    """counts: (n_images, n_thresholds, 4); groups: (n_images,) in {0, 1}.
    Threshold chosen on each group by minimum pooled BER, applied to the other."""
    groups = np.asarray(groups)
    if set(np.unique(groups)) != {0, 1}:
        raise ValueError("groups must contain exactly the labels 0 and 1")
    chosen, pooled = {}, np.zeros(4, dtype=np.int64)
    for g in (0, 1):
        ber_g = ber_from_counts(counts[groups == g].sum(0))["ber"]
        i = int(np.argmin(ber_g))
        chosen[g] = thresholds[i]
        pooled += counts[groups == 1 - g, i].sum(0)                  # applied to the other group
    return {"thresholds_chosen": {"on_group_0_applied_to_1": chosen[0], "on_group_1_applied_to_0": chosen[1]},
            "pooled_counts": pooled, **{k: float(v) for k, v in ber_from_counts(pooled).items()}}
