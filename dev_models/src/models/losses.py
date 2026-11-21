"""Loss functions for the multi-cue framework.

The triplet loss here replaces the original implementation, which produced NaN.
Two faults were present in that version and both are corrected below:

  1. The self-distance on the diagonal was treated as a valid positive pair.
     `dist * mask` leaves the diagonal at 0 and `mask` is True there, so when an
     identity appeared exactly once in a batch the "hardest positive" was the
     anchor itself at distance 0. The loss was then well defined but meaningless,
     and no attracting force acted on the embedding.

  2. `dist.clamp(min=1e-12).sqrt()` is finite in the forward pass but its
     gradient, d/dx sqrt(x) = 1/(2*sqrt(x)), is ~5e5 at x = 1e-12. Backward
     through the zero-valued diagonal produced gradients large enough to
     destabilise training and, combined with (1), yielded NaN.

The corrected version masks the diagonal explicitly, adds an epsilon before the
square root, and skips anchors that have no valid positive or no valid negative
in the batch rather than silently producing a degenerate value.

Note that fix (1) only matters if batches actually contain repeated identities.
Use `PKSampler` from `src.data.sampler` to guarantee that; without it, a batch
drawn uniformly from a corpus with one clip per identity contains no positive
pairs at all and metric learning cannot work regardless of the loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BatchHardTripletLoss", "BCEDiceLoss", "DeepSupervisionLoss", "balanced_error_rate"]


def _pairwise_euclidean(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Numerically stable pairwise Euclidean distance matrix.

    Uses the squared-norm expansion, clamps negatives introduced by floating
    point error, adds `eps` *before* the square root so the gradient stays
    bounded, and zeroes the diagonal afterwards.
    """
    sq = x.pow(2).sum(dim=1, keepdim=True)              # (n, 1)
    dist_sq = sq + sq.t() - 2.0 * (x @ x.t())           # (n, n)
    dist_sq = dist_sq.clamp(min=0.0)
    dist = torch.sqrt(dist_sq + eps)
    # the diagonal is exactly zero by definition; remove the eps artefact and
    # detach it from the graph so it contributes no gradient
    n = x.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=x.device)
    dist = dist.masked_fill(eye, 0.0)
    return dist


class BatchHardTripletLoss(nn.Module):
    """Batch-hard triplet loss (Hermans, Beyer & Leibe, 2017).

    For every anchor the most distant positive and the closest negative within
    the mini-batch are selected, and the margin is enforced between them.

    Anchors with no valid positive (identity appears once) or no valid negative
    (batch is a single identity) are excluded from the mean. If no anchor in the
    batch is valid the loss is zero, which keeps the step a no-op instead of
    poisoning the parameters with NaN.
    """

    def __init__(self, margin: float = 0.3, eps: float = 1e-8):
        super().__init__()
        self.margin = float(margin)
        self.eps = float(eps)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        n = embeddings.size(0)
        dist = _pairwise_euclidean(embeddings, self.eps)

        same = labels.view(n, 1).eq(labels.view(1, n))
        eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
        pos_mask = same & ~eye        # same identity, not the anchor itself
        neg_mask = ~same

        has_pos = pos_mask.any(dim=1)
        has_neg = neg_mask.any(dim=1)
        valid = has_pos & has_neg

        stats = {
            "valid_anchors": int(valid.sum().item()),
            "batch_size": n,
            "mean_pos_per_anchor": float(pos_mask.sum(dim=1).float().mean().item()),
        }

        if not bool(valid.any()):
            # No usable triplet in this batch. Return a real zero that is still
            # connected to the graph so the optimiser step is well defined.
            return embeddings.sum() * 0.0, stats

        # hardest positive: maximum distance among same-identity pairs
        d_pos = dist.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
        # hardest negative: minimum distance among different-identity pairs
        d_neg = dist.masked_fill(~neg_mask, float("inf")).min(dim=1).values

        d_pos = d_pos[valid]
        d_neg = d_neg[valid]

        loss = F.relu(d_pos - d_neg + self.margin).mean()

        stats["mean_d_pos"] = float(d_pos.mean().item())
        stats["mean_d_neg"] = float(d_neg.mean().item())
        stats["active_fraction"] = float((d_pos - d_neg + self.margin > 0).float().mean().item())
        return loss, stats


class BCEDiceLoss(nn.Module):
    """Equally weighted binary cross-entropy and Dice loss (Equation 3.5).

    BCE supplies a dense per-pixel signal; the Dice term optimises overlap
    directly and is insensitive to the dominance of the background class, which
    matters because shadow pixels are a small minority of image area.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        num = 2.0 * (prob * target).sum(dim=(1, 2, 3)) + self.smooth
        den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.smooth
        dice = 1.0 - (num / den).mean()
        return self.bce_weight * bce + self.dice_weight * dice


class DeepSupervisionLoss(nn.Module):
    """Applies BCEDiceLoss at every cascade stage (Equation 3.6).

    Each stage prediction is resampled to the reference resolution before the
    loss is taken. Deep supervision matters for this architecture specifically
    because the reverse-attention gate of each stage is driven by the output of
    the stage above it: an unsupervised coarse prediction would propagate a
    meaningless gate through the rest of the cascade.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 stage_weights: tuple | None = None):
        super().__init__()
        self.base = BCEDiceLoss(bce_weight, dice_weight)
        self.stage_weights = stage_weights

    def forward(self, predictions, target: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(predictions):
            predictions = [predictions]
        weights = self.stage_weights or [1.0] * len(predictions)
        total = 0.0
        for w, pred in zip(weights, predictions):
            if pred.shape[-2:] != target.shape[-2:]:
                pred = F.interpolate(pred, size=target.shape[-2:],
                                     mode="bilinear", align_corners=False)
            total = total + w * self.base(pred, target)
        return total


@torch.no_grad()
def balanced_error_rate(prob: torch.Tensor, target: torch.Tensor, threshold: float):
    """Balanced Error Rate (Equation 3.13), returned with its two components.

    BER = 0.5 * (FN / (TP + FN) + FP / (TN + FP)), expressed as a percentage.
    Reported alongside the shadow-class and non-shadow-class error rates so the
    operating trade-off is visible rather than hidden inside the aggregate.
    """
    pred = (prob > threshold)
    tgt = target.bool()

    tp = (pred & tgt).sum().double()
    fn = (~pred & tgt).sum().double()
    tn = (~pred & ~tgt).sum().double()
    fp = (pred & ~tgt).sum().double()

    shadow_err = (fn / (tp + fn).clamp(min=1)).item() * 100.0
    nonshadow_err = (fp / (tn + fp).clamp(min=1)).item() * 100.0
    return {
        "threshold": threshold,
        "ber": 0.5 * (shadow_err + nonshadow_err),
        "shadow_error_fn": shadow_err,
        "nonshadow_error_fp": nonshadow_err,
        "tp": int(tp.item()), "fn": int(fn.item()),
        "tn": int(tn.item()), "fp": int(fp.item()),
    }
