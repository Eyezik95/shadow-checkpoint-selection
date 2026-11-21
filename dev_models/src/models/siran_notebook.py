"""Verbatim copy of reference/notebooks/obj2_shadow_detector.ipynb cell 0 (notebook sha256 5f16e738d422e492...): the training-capable
BiFPN-SIRAN class (deep-supervision outputs in train mode, sigmoid in eval mode)
and the phase-2 helpers (freeze_for_finetuning, get_discriminative_param_groups).
Used only to redo phase 2 faithfully; inference uses src/models/shadow.py.
"""

"""
BiFPN-SIRAN: Skip-Injected Reverse Attention Network for Shadow Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None


# ==============================================================================
# 1. Shared building blocks
# ==============================================================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                    stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class BiFPNLayer(nn.Module):
    """One BiFPN fusion round (top-down pass with learnable weighted fusion)."""
    def __init__(self, channels, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.w_td = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.w_out = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.conv_td = DepthwiseSeparableConv(channels, channels)
        self.conv_out = DepthwiseSeparableConv(channels, channels)

    def forward(self, p_in, p_skip=None):
        w_td = F.relu(self.w_td)
        w_td = w_td / (torch.sum(w_td) + self.epsilon)

        if p_skip is not None:
            p_td = self.conv_td(w_td[0] * p_in + w_td[1] * p_skip)
        else:
            p_td = self.conv_td(p_in)

        w_out = F.relu(self.w_out)
        w_out = w_out / (torch.sum(w_out) + self.epsilon)
        return self.conv_out(w_out[0] * p_in + w_out[1] * p_td)


# ==============================================================================
# 2. Option A: distraction-aware auxiliary branch (optional, off by default)
# ==============================================================================

class DistractionAwareHead(nn.Module):
    """Predicts a per-pixel 'distraction' score (shadow-like-but-not-shadow
    regions, e.g. painted lines / dark surfaces) used to suppress false
    positives. Optional -- your EfficientNet-b5 run already has low false-
    positive error (0.74%), so this is unlikely to be your bottleneck right
    now, but it's wired in for completeness / future use."""
    def __init__(self, channels, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1)
        )

    def forward(self, features):
        return self.net(features)


class DistractionAwareLoss(nn.Module):
    """Deep-supervision BCE+Dice plus a distraction term whose pseudo-label
    is mined on the fly from the model's own current predictions. Only
    used if the model was built with use_distraction_head=True."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5,
                 distraction_weight=0.3, hard_negative_thresh=0.3):
        super().__init__()
        self.deep_supervision = DeepSupervisionLoss(bce_weight, dice_weight)
        self.distraction_bce = nn.BCEWithLogitsLoss()
        self.distraction_weight = distraction_weight
        self.hard_negative_thresh = hard_negative_thresh

    def forward(self, outputs, targets):
        *main_preds, distraction_logits = outputs
        main_loss = self.deep_supervision(main_preds, targets)

        pred1 = main_preds[0]  # finest, full-resolution prediction
        with torch.no_grad():
            pred1_resized = pred1
            if pred1.shape[2:] != targets.shape[2:]:
                pred1_resized = F.interpolate(pred1, size=targets.shape[2:],
                                               mode='bilinear', align_corners=False)
            pred1_prob = torch.sigmoid(pred1_resized)
            pseudo_label = ((targets < 0.5) & (pred1_prob > self.hard_negative_thresh)).float()

        distraction_logits_resized = distraction_logits
        if distraction_logits.shape[2:] != targets.shape[2:]:
            distraction_logits_resized = F.interpolate(
                distraction_logits, size=targets.shape[2:], mode='bilinear', align_corners=False
            )
        distraction_loss = self.distraction_bce(distraction_logits_resized, pseudo_label)

        return main_loss + self.distraction_weight * distraction_loss


# ==============================================================================
# 3. Losses
# ==============================================================================

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, logits=True):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.logits = logits
        self.bce = nn.BCEWithLogitsLoss() if logits else nn.BCELoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        inputs_prob = torch.sigmoid(inputs) if self.logits else inputs
        smooth = 1e-5
        inputs_flat = inputs_prob.reshape(-1)
        targets_flat = targets.reshape(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1.0 - ((2.0 * intersection + smooth) /
                            (inputs_flat.sum() + targets_flat.sum() + smooth))
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)


class DeepSupervisionLoss(nn.Module):
    """Applies BCEDiceLoss to every intermediate prediction returned in train mode."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.base_loss = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight, logits=True)

    def forward(self, predictions, targets):
        target_size = targets.shape[2:]
        total_loss = 0.0
        for pred in predictions:
            if pred.shape[2:] != target_size:
                pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
            total_loss = total_loss + self.base_loss(pred, targets)
        return total_loss


class AsymmetricBCEDiceLoss(nn.Module):
    """
    BCE (with pos_weight penalizing missed-shadow pixels more than false
    positives) + Tversky loss (alpha weights false positives, beta weights
    false negatives). Use this if shadow_err >> non_shadow_err, as in your
    EfficientNet-b5 run (5.29% vs 0.74%) -- it's a much cheaper first test
    than an architecture change, since it's a pure loss-function swap on
    your EXISTING model. tversky_beta > tversky_alpha biases toward
    recall; start at alpha=0.3, beta=0.7 and adjust based on the resulting
    split (if non_shadow_err rises above shadow_err, you've overshot).
    """
    def __init__(self, bce_weight=0.5, tversky_weight=0.5, pos_weight=1.5,
                 tversky_alpha=0.3, tversky_beta=0.7, logits=True):
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.logits = logits
        self.register_buffer('pos_weight', torch.tensor(pos_weight))

    def forward(self, inputs, targets):
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight)
            prob = torch.sigmoid(inputs)
        else:
            eps = 1e-7
            prob = inputs.clamp(eps, 1 - eps)
            bce = -(self.pos_weight * targets * torch.log(prob) +
                    (1 - targets) * torch.log(1 - prob)).mean()

        smooth = 1e-5
        p_flat = prob.reshape(-1)
        t_flat = targets.reshape(-1)
        tp = (p_flat * t_flat).sum()
        fp = (p_flat * (1 - t_flat)).sum()
        fn = ((1 - p_flat) * t_flat).sum()
        tversky = 1.0 - (tp + smooth) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + smooth)

        return self.bce_weight * bce + self.tversky_weight * tversky


class DeepSupervisionAsymmetricLoss(nn.Module):
    """AsymmetricBCEDiceLoss applied to every intermediate prediction."""
    def __init__(self, bce_weight=0.5, tversky_weight=0.5, pos_weight=1.5,
                 tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        self.base_loss = AsymmetricBCEDiceLoss(bce_weight, tversky_weight, pos_weight,
                                                tversky_alpha, tversky_beta, logits=True)

    def forward(self, predictions, targets):
        target_size = targets.shape[2:]
        total_loss = 0.0
        for pred in predictions:
            if pred.shape[2:] != target_size:
                pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
            total_loss = total_loss + self.base_loss(pred, targets)
        return total_loss


def sweep_threshold_for_ber(model, dataloader, device, thresholds=None):
    """
    No-retraining diagnostic: one forward pass over dataloader, then BER
    evaluated at every candidate threshold against the same collected
    predictions. Returns (threshold, ber, shadow_err, non_shadow_err)
    tuples sorted best-BER-first. Run this on an existing checkpoint
    before deciding an architecture change is necessary.
    """
    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(1, 19)]  # 0.05..0.90

    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            probs = model(images)  # eval mode: already sigmoid
            all_probs.append(probs.cpu())
            all_targets.append(masks.cpu())
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    results = []
    for t in thresholds:
        pred = (all_probs > t).float()
        target = (all_targets > 0.5).float()
        tp = (pred * target).sum().item()
        fp = (pred * (1 - target)).sum().item()
        tn = ((1 - pred) * (1 - target)).sum().item()
        fn = ((1 - pred) * target).sum().item()
        eps = 1e-6
        tpr = tp / (tp + fn + eps)
        tnr = tn / (tn + fp + eps)
        shadow_err = 100.0 * (1.0 - tpr)
        non_shadow_err = 100.0 * (1.0 - tnr)
        ber = 100.0 * (1.0 - 0.5 * (tpr + tnr))
        results.append((t, ber, shadow_err, non_shadow_err))

    return sorted(results, key=lambda r: r[1])


# ==============================================================================
# 4. SI-RAN Block
# ==============================================================================

def _safe_group_count(channels, requested_groups=32):
    """GroupNorm requires num_channels % num_groups == 0. Falls back to
    the largest divisor of `channels` <= requested_groups."""
    for g in range(min(requested_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class SIRANBlock(nn.Module):
    """
    Skip-Injected Reverse Attention block. Same underlying mechanism as
    plain reverse attention (erase what a coarser prediction was
    confident about, refine what's left), but fuses in the RAW backbone
    feature at this pyramid level BEFORE the (1-P) multiplication.

    Uses GroupNorm rather than BatchNorm2d: this is a heavier block on
    top of a pipeline that likely runs at small batch size with a large
    pretrained backbone (e.g. EfficientNet-b5 at batch~4), and BatchNorm's
    running statistics get less reliable as batch size drops. GroupNorm
    sidesteps that dependency entirely.
    """
    def __init__(self, bifpn_channels, skip_channels, out_channels, dropout_p=0.2, norm_groups=32):
        super().__init__()
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, bifpn_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_safe_group_count(bifpn_channels, norm_groups), bifpn_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(bifpn_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_safe_group_count(out_channels, norm_groups), out_channels),
            nn.ReLU(inplace=True)
        )
        self.refine_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_safe_group_count(out_channels, norm_groups), out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.mask_predictor = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)

    def forward(self, bifpn_feat, skip_feat, prev_mask):
        """
        bifpn_feat: BiFPN-fused context feature at this scale, (B, bifpn_channels, H, W)
        skip_feat: RAW backbone feature at this scale (pre-BiFPN)
        prev_mask: prediction logits from the deeper (coarser) cascade stage
        """
        if prev_mask.shape[2:] != bifpn_feat.shape[2:]:
            prev_mask = F.interpolate(prev_mask, size=bifpn_feat.shape[2:],
                                       mode='bilinear', align_corners=False)

        skip_proj = self.skip_proj(skip_feat)
        # Defensive resize: FPN levels and raw backbone levels at "the same"
        # stride can differ by a pixel from rounding on non-power-of-2 input
        # sizes -- concatenation requires exact spatial match.
        if skip_proj.shape[2:] != bifpn_feat.shape[2:]:
            skip_proj = F.interpolate(skip_proj, size=bifpn_feat.shape[2:],
                                       mode='bilinear', align_corners=False)

        concat_feat = torch.cat([bifpn_feat, skip_proj], dim=1)
        fused_feat = self.fusion_conv(concat_feat)

        rev_attn_weight = 1.0 - torch.sigmoid(prev_mask)
        gated_feat = fused_feat * rev_attn_weight

        refined_feat = self.refine_conv(gated_feat)
        refined_feat = self.dropout(refined_feat)
        current_mask_logits = self.mask_predictor(refined_feat)

        return refined_feat, current_mask_logits


# ==============================================================================
# 5. BiFPN-SIRAN model
# ==============================================================================

class BiFPN_SIRAN_ShadowDetector(nn.Module):
    """
    Encoder (custom from-scratch, or any 5-stage timm backbone) -> BiFPN
    neck -> cascaded Skip-Injected Reverse Attention decoder.

    backbone: 'custom' (from-scratch stem + ResidualBlock stages, random
    init) or any timm model name producing exactly 5 feature levels via
    features_only=True, out_indices=(0,1,2,3,4) -- verified for
    'resnet50' ([64,256,512,1024,2048]) and 'efficientnet_b5'
    ([24,40,64,176,512]). Other backbones aren't guaranteed to have 5
    usable stages; check model.timm_backbone.feature_info.channels() if
    unsure.
    """
    def __init__(self, in_channels=3, out_channels=1,
                 features=[64, 128, 256, 512, 1024],
                 siran_channels=64, dropout_p=0.2,
                 use_distraction_head=False,
                 backbone='custom', pretrained_backbone=True):
        super().__init__()
        self.use_distraction_head = use_distraction_head
        self.backbone_name = backbone
        bifpn_channels = features[3]

        # --- Encoder ---
        if backbone == 'custom':
            self.timm_backbone = None
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, features[0], 7, 2, 3, bias=False),
                nn.BatchNorm2d(features[0]), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1)
            )
            self.encoder1 = self._make_encoder_block(features[0], features[0], 2)
            self.encoder2 = self._make_encoder_block(features[0], features[1], 2, stride=2)
            self.encoder3 = self._make_encoder_block(features[1], features[2], 2, stride=2)
            self.encoder4 = self._make_encoder_block(features[2], features[3], 2, stride=2)
            self.encoder5 = self._make_encoder_block(features[3], features[4], 2, stride=2)
            skip_channels = list(features)
            self.lateral5 = nn.Conv2d(features[4], bifpn_channels, 1)
            self.lateral4 = nn.Conv2d(features[3], bifpn_channels, 1)
            self.lateral3 = nn.Conv2d(features[2], bifpn_channels, 1)
            self.lateral2 = nn.Conv2d(features[1], bifpn_channels, 1)
            self.lateral1 = nn.Conv2d(features[0], bifpn_channels, 1)
        else:
            if timm is None:
                raise ImportError("timm is required for pretrained backbones: pip install timm")
            self.timm_backbone = timm.create_model(
                backbone, pretrained=pretrained_backbone,
                features_only=True, out_indices=(0, 1, 2, 3, 4), in_chans=in_channels,
            )
            skip_channels = self.timm_backbone.feature_info.channels()
            if len(skip_channels) != 5:
                raise ValueError(
                    f"Backbone '{backbone}' produced {len(skip_channels)} feature levels "
                    f"(channels={skip_channels}), expected exactly 5 to match this "
                    f"architecture's 5-level BiFPN neck."
                )
            c1, c2, c3, c4, c5 = skip_channels
            self.lateral1 = nn.Conv2d(c1, bifpn_channels, 1)
            self.lateral2 = nn.Conv2d(c2, bifpn_channels, 1)
            self.lateral3 = nn.Conv2d(c3, bifpn_channels, 1)
            self.lateral4 = nn.Conv2d(c4, bifpn_channels, 1)
            self.lateral5 = nn.Conv2d(c5, bifpn_channels, 1)

        # --- BiFPN neck ---
        self.bifpn_channels = bifpn_channels
        self.bifpn1 = BiFPNLayer(bifpn_channels)
        self.bifpn2 = BiFPNLayer(bifpn_channels)
        self.bifpn3 = BiFPNLayer(bifpn_channels)

        # --- SI-RAN cascade ---
        # NOT weight-shared across levels (unlike RARM/RA in other
        # architectures): each level's raw skip has a different channel
        # count, so skip_proj can't be a shared module across stages.
        c1_skip, c2_skip, c3_skip, c4_skip, c5_skip = skip_channels
        self.global_map_head = nn.Conv2d(bifpn_channels, 1, kernel_size=3, padding=1)
        self.siran4 = SIRANBlock(bifpn_channels, c4_skip, siran_channels, dropout_p=dropout_p)
        self.siran3 = SIRANBlock(bifpn_channels, c3_skip, siran_channels, dropout_p=dropout_p)
        self.siran2 = SIRANBlock(bifpn_channels, c2_skip, siran_channels, dropout_p=dropout_p)
        self.siran1 = SIRANBlock(bifpn_channels, c1_skip, siran_channels, dropout_p=dropout_p)

        self.final = nn.Conv2d(siran_channels, out_channels, 1)

        if use_distraction_head:
            self.distraction_head = DistractionAwareHead(siran_channels)
            self.distraction_suppression_scale = nn.Parameter(torch.tensor(0.5))

    def _make_encoder_block(self, in_channels, out_channels, num_blocks, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _encode(self, x):
        if self.timm_backbone is not None:
            return self.timm_backbone(x)
        x = self.stem(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        return e1, e2, e3, e4, e5

    def _fuse(self, e1, e2, e3, e4, e5):
        p5 = self.lateral5(e5)
        p4 = self.lateral4(e4)
        p3 = self.lateral3(e3)
        p2 = self.lateral2(e2)
        p1 = self.lateral1(e1)

        p5_up = F.interpolate(p5, size=p4.shape[2:], mode='bilinear', align_corners=False)
        p4_td = self.bifpn1(p4, p5_up)
        p4_up = F.interpolate(p4_td, size=p3.shape[2:], mode='bilinear', align_corners=False)
        p3_td = self.bifpn2(p3, p4_up)
        p3_up = F.interpolate(p3_td, size=p2.shape[2:], mode='bilinear', align_corners=False)
        p2_td = self.bifpn3(p2, p3_up)
        p2_up = F.interpolate(p2_td, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p1_td = self.bifpn1(p1, p2_up)

        return p5, p4_td, p3_td, p2_td, p1_td

    def forward(self, x):
        orig_shape = x.shape[2:]

        e1, e2, e3, e4, e5 = self._encode(x)
        p5, p4_td, p3_td, p2_td, p1_td = self._fuse(e1, e2, e3, e4, e5)

        global_pred = self.global_map_head(p5)
        # Each stage gets: this level's BiFPN context, this level's RAW
        # backbone feature (the novel part), and the previous (coarser)
        # stage's mask.
        feat4, pred4 = self.siran4(p4_td, e4, global_pred)
        feat3, pred3 = self.siran3(p3_td, e3, pred4)
        feat2, pred2 = self.siran2(p2_td, e2, pred3)
        feat1, pred1 = self.siran1(p1_td, e1, pred2)

        out = self.final(feat1)

        distraction_logits = None
        if self.use_distraction_head:
            distraction_logits = self.distraction_head(feat1)
            distraction_up = distraction_logits
            if distraction_logits.shape[2:] != out.shape[2:]:
                distraction_up = F.interpolate(distraction_logits, size=out.shape[2:],
                                                mode='bilinear', align_corners=False)
            out = out - self.distraction_suppression_scale * torch.sigmoid(distraction_up)

        if out.shape[2:] != orig_shape:
            out = F.interpolate(out, size=orig_shape, mode='bilinear', align_corners=False)
            if distraction_logits is not None and distraction_logits.shape[2:] != orig_shape:
                distraction_logits = F.interpolate(distraction_logits, size=orig_shape,
                                                    mode='bilinear', align_corners=False)

        if self.training:
            if self.use_distraction_head:
                return out, global_pred, pred4, pred3, pred2, pred1, distraction_logits
            return out, global_pred, pred4, pred3, pred2, pred1
        return torch.sigmoid(out)


# ==============================================================================
# 6. Phase-2 transfer-learning helpers
# ==============================================================================

def freeze_for_finetuning(model, frozen_modules=('stem',)):
    """Freezes the given submodules and pins them to eval() (so frozen
    BatchNorm/GroupNorm-adjacent stats don't drift -- GroupNorm doesn't
    use running stats so this mainly matters if backbone='custom', but
    kept for a uniform freeze API)."""
    frozen = []
    for name in frozen_modules:
        module = getattr(model, name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = False
        module.eval()
        frozen.append(name)
        print(f"  Frozen: {name} ({sum(p.numel() for p in module.parameters()):,} params)")
    return frozen


def reassert_frozen_eval(model, frozen_module_names):
    """Call after model.train() each epoch -- train() recursively puts
    every submodule back in train mode, including frozen ones."""
    for name in frozen_module_names:
        module = getattr(model, name, None)
        if module is not None:
            module.eval()


def get_discriminative_param_groups(model, base_lr, frozen_module_names=()):
    """
    Discriminative LR groups: pretrained/custom encoder at a low tier,
    BiFPN neck at a mid tier, SI-RAN decoder + distraction head at full
    base_lr. Raises if any trainable parameter isn't captured by any
    group (would otherwise silently never be optimized).
    """
    groups = []
    seen_ids = set()

    def add_group(names_or_modules, lr_mult):
        params = []
        for item in names_or_modules:
            module = getattr(model, item, None) if isinstance(item, str) else item
            if module is None:
                continue
            for p in module.parameters():
                if p.requires_grad and id(p) not in seen_ids:
                    params.append(p)
                    seen_ids.add(id(p))
        if params:
            groups.append({'params': params, 'lr': base_lr * lr_mult})

    if getattr(model, 'timm_backbone', None) is not None:
        add_group([n for n in ['timm_backbone'] if n not in frozen_module_names], 0.1)
    else:
        add_group([n for n in ['stem'] if n not in frozen_module_names], 0.02)
        add_group([n for n in ['encoder1', 'encoder2'] if n not in frozen_module_names], 0.05)
        add_group([n for n in ['encoder3', 'encoder4', 'encoder5'] if n not in frozen_module_names], 0.1)

    add_group(['lateral5', 'lateral4', 'lateral3', 'lateral2', 'lateral1',
               'bifpn1', 'bifpn2', 'bifpn3'], 0.5)

    decoder_items = ['global_map_head', 'siran4', 'siran3', 'siran2', 'siran1', 'final']
    if getattr(model, 'use_distraction_head', False):
        decoder_items.append('distraction_head')
    add_group(decoder_items, 1.0)

    suppression_param = getattr(model, 'distraction_suppression_scale', None)
    if suppression_param is not None and suppression_param.requires_grad \
            and id(suppression_param) not in seen_ids:
        groups[-1]['params'].append(suppression_param)
        seen_ids.add(id(suppression_param))

    grouped_ids = seen_ids
    all_trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    missing = all_trainable_ids - grouped_ids
    if missing:
        missing_names = [name for name, p in model.named_parameters()
                          if p.requires_grad and id(p) in missing]
        raise RuntimeError(
            f"{len(missing)} trainable parameter(s) not assigned to any optimizer "
            f"param group: {missing_names[:10]}{'...' if len(missing_names) > 10 else ''}."
        )

    return groups


# if __name__ == "__main__":
#     dummy = torch.randn(2, 3, 128, 128)
#     dummy_mask = torch.randint(0, 2, (2, 1, 128, 128)).float()

#     for backbone, pretrained in [('custom', True), ('efficientnet_b5', False)]:
#         print(f"--- backbone={backbone} ---")
#         model = BiFPN_SIRAN_ShadowDetector(backbone=backbone, pretrained_backbone=pretrained)

#         model.train()
#         outs = model(dummy)
#         assert len(outs) == 6 and outs[0].shape == (2, 1, 128, 128)
#         loss = DeepSupervisionLoss()(outs, dummy_mask)
#         assert torch.isfinite(loss)
#         loss.backward()
#         n_missing = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
#         assert n_missing == 0, f"{n_missing} params never received gradients"
#         print(f"  train OK: loss={loss.item():.4f}, params_missing_grad={n_missing}")

#         model.eval()
#         with torch.no_grad():
#             out = model(dummy)
#         assert out.shape == (2, 1, 128, 128)
#         print(f"  eval OK: shape={tuple(out.shape)}, range=[{out.min():.3f}, {out.max():.3f}]")

#         total = sum(p.numel() for p in model.parameters())
#         print(f"  total params: {total:,}")

#         frozen = freeze_for_finetuning(model, frozen_modules=('stem',) if backbone == 'custom' else ())
#         groups = get_discriminative_param_groups(model, base_lr=1e-5, frozen_module_names=frozen)
#         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         grouped = sum(p.numel() for g in groups for p in g['params'])
#         assert trainable == grouped
#         print(f"  param-group check OK: trainable={trainable:,}, grouped={grouped:,}")

#     print("\nTesting Option A (distraction head) + AsymmetricBCEDiceLoss...")
#     model = BiFPN_SIRAN_ShadowDetector(backbone='custom', use_distraction_head=True)
#     model.train()
#     outs = model(dummy)
#     assert len(outs) == 7
#     loss = DistractionAwareLoss()(outs, dummy_mask)
#     assert torch.isfinite(loss)
#     print(f"  Option A loss OK: {loss.item():.4f}")

#     model2 = BiFPN_SIRAN_ShadowDetector(backbone='custom')
#     model2.train()
#     outs2 = model2(dummy)
#     asym_loss = DeepSupervisionAsymmetricLoss(pos_weight=1.5, tversky_alpha=0.3, tversky_beta=0.7)(outs2, dummy_mask)
#     assert torch.isfinite(asym_loss)
#     print(f"  AsymmetricBCEDiceLoss OK: {asym_loss.item():.4f}")

#     print("\nAll checks passed.")