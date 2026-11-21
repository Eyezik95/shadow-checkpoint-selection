"""Shadow-detector variants from the Objective 2 notebook (Section 3.3.2, Table 3.2).

Verbatim copy of reference/notebooks/obj2_shadow_detector.ipynb cell 3 (written there as bifpn_rarm_ran_model.py;
notebook sha256 5f16e738d422e492...): BiFPN-RAN, BiFPN-RARM and
BiFPN-RARM-RAN with their building blocks. Kept unedited so the stored
checkpoints load with strict=True; BiFPN-SIRAN lives in src/models/shadow.py.
In eval mode these models return sigmoid probabilities.
"""

"""
BiFPN-RARM-RAN: Combined Shadow Detection Architecture
========================================================
Merges two complementary refinement mechanisms on top of the same
BiFPN neck used in your dissertation's BiFPN-RARM model:

  1. RARM (Recurrent Attention Residual Module) — from your documented
     model. Applied independently at each of the 5 pyramid levels.
     2 recurrent iterations of channel-attention -> spatial-attention
     -> residual add. This sharpens WHAT each level's features encode
     (discriminative channels, roughly-localised shadow regions) but
     does not use information from other levels while doing so.

  2. Cascaded Reverse Attention (from BiFPN-RAN). Takes a coarse
     prediction from a deeper level, inverts it, and uses it to mask
     out the already-confident regions in the next, higher-resolution
     level's features -- forcing that level to focus specifically on
     what the coarser prediction got wrong or missed: typically
     boundaries, thin/weak shadow regions, and merged/overlapping
     shadow edges.

  These two mechanisms operate on different axes (per-level feature
  quality vs. cross-level error correction), so they compose without
  redundancy: RARM refines each level's features first, then the
  reverse-attention cascade uses those already-refined features to do
  boundary-focused inter-level correction, replacing the plain
  transposed-conv decoder from the original BiFPN-RARM.

  IMPORTANT CAVEATS (read before swapping this in for your documented
  model):
  - This roughly doubles decoder-side parameters/compute vs. either
    component alone.
  - Your ISTD fine-tuning set is only 1,330 images. A larger, more
    expressive model raises overfitting risk during Phase 2 -- watch
    train/val IoU divergence closely, and consider stronger
    augmentation or a lower Phase 2 LR if you see it.
  - There is no guarantee this beats plain BiFPN-RARM (0.8442 IoU) --
    it targets a specific documented failure mode (weak/fragmented
    boundaries at shadow onset, merged overlapping shadows) rather
    than a general accuracy problem. Validate with an ablation:
    BiFPN-RARM alone vs. BiFPN-RAN alone vs. this combined model,
    same data/schedule, before adopting it as your final model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Shared building blocks
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
        p_out = self.conv_out(w_out[0] * p_in + w_out[1] * p_td)

        return p_out


# ==============================================================================
# RARM: Recurrent Attention Residual Module (as documented for BiFPN-RARM)
# ==============================================================================

class ChannelAttention(nn.Module):
    """Squeeze-and-excite style channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = torch.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention(nn.Module):
    """Localises shadow boundary regions via max/avg-pooled spatial map."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class RARMBlock(nn.Module):
    """
    Recurrent Attention Residual Module: `num_iterations` rounds of
    channel attention -> spatial attention -> residual add, applied
    independently to a single feature-pyramid level.

    dropout_p: Dropout2d probability applied to the refined feature map
    before the residual add. Drops whole channels (not individual
    pixels), which is the effective form of dropout for conv features
    since adjacent pixels are highly correlated.
    """
    def __init__(self, channels, num_iterations=2, reduction=16, dropout_p=0.2):
        super().__init__()
        self.num_iterations = num_iterations
        self.channel_attn = ChannelAttention(channels, reduction=reduction)
        self.spatial_attn = SpatialAttention()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        out = x
        for _ in range(self.num_iterations):
            attended = self.channel_attn(out)
            attended = self.spatial_attn(attended)
            attended = self.refine(attended)
            attended = self.dropout(attended)
            out = out + attended  # residual connection
        return out


# ==============================================================================
# Reverse Attention: cascaded boundary-focused decoder (from BiFPN-RAN)
# ==============================================================================

class ReverseAttentionBlock(nn.Module):
    """Erases confident foreground from a coarser prediction to mine
    the regions that prediction got wrong -- typically boundaries."""
    def __init__(self, in_channels, out_channels=64, dropout_p=0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.pred_head = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)

    def forward(self, features, prior_prediction):
        if prior_prediction.shape[2:] != features.shape[2:]:
            prior_prediction = F.interpolate(
                prior_prediction, size=features.shape[2:], mode='bilinear', align_corners=False
            )

        prob_map = torch.sigmoid(prior_prediction)
        reverse_map = 1.0 - prob_map
        erased_features = features * reverse_map

        refined = self.conv1(erased_features)
        refined = self.conv2(refined)
        refined = self.dropout(refined)
        new_prediction = self.pred_head(refined)

        return new_prediction, refined


# ==============================================================================
# Loss functions (shared by both plain and distraction-aware training)
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


# ==============================================================================
# Option A: Distraction-aware auxiliary branch (DSDNet-inspired)
# ==============================================================================

class DistractionAwareHead(nn.Module):
    """
    Auxiliary branch that predicts a per-pixel "distraction" score: how
    likely this pixel is a shadow-like-but-not-shadow region (dark
    surfaces, painted lines, low-albedo materials) the main branch is at
    risk of false-positiving on. Reads the finest-resolution,
    fully-refined decoder feature (the output right before the final 1x1
    conv), since false positives on fine detail show up most clearly at
    that resolution.

    ADAPTATION NOTE: the original DSDNet (Zheng et al., CVPR 2019) trains
    its distraction module using extra supervision from other models /
    datasets. We don't have that external supervision here, so this is a
    self-supervised, online variant: DistractionAwareLoss below generates
    its own pseudo-labels each batch from the model's current
    predictions vs. ground truth (see that class's docstring). Same
    underlying idea -- explicitly model and suppress shadow-like
    distractor regions -- implemented without needing an external
    annotated distraction dataset.
    """
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
        return self.net(features)  # logits, same spatial size as input


class DistractionAwareLoss(nn.Module):
    """
    Wraps DeepSupervisionLoss with an additional distraction term.

    The distraction pseudo-label is generated ON THE FLY each batch: any
    pixel the network CURRENTLY, confidently predicts as shadow
    (probability > hard_negative_thresh) while ground truth says it is
    NOT shadow is treated as a distraction example for that batch. No
    gradient flows through pseudo-label construction. This directly
    targets your documented false-positive failure mode (e.g. painted
    lines) without needing a separately annotated distraction dataset --
    the model effectively mines its own hard negatives as training
    progresses.

    Expects `outputs` to be the model's train-mode tuple WITH the
    distraction logits appended as the last element:
        (out, global_pred, pred4, pred3, pred2, pred1, distraction_logits)
    i.e. the model must be constructed with use_distraction_head=True.
    """
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

        # main_preds[0] is always "out" -- the finest, full-resolution
        # prediction -- across all three architectures, unlike main_preds[-1]
        # which is the RA cascade's finest stage in RAN-based models but only
        # a coarse aux head in the transposed-conv BiFPN-RARM decoder.
        pred1 = main_preds[0]
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
# Shared encoder (used by all three architectures below)
# ==============================================================================

try:
    import timm
except ImportError:
    timm = None


class _SharedEncoderMixin:
    """Common stem/encoder/lateral construction, reused identically by all
    three architectures so the encoder is never accidentally different
    between ablation runs.

    backbone='custom' (default) builds the original from-scratch stem +
    5 ResidualBlock stages (random init, no ImageNet pretraining).

    backbone=<any timm model name, e.g. 'resnet50', 'efficientnet_b4'>
    uses timm.create_model(..., features_only=True, out_indices=(0,1,2,3,4))
    to extract 5 pyramid levels directly from an ImageNet-pretrained
    backbone. This is the single highest-leverage change available: every
    competitive published shadow detector (BDRAR, DSDNet, MTMT, FDRNet,
    SwinShadow) uses an ImageNet-pretrained encoder, and the from-scratch
    'custom' encoder here never had that starting point. The lateral convs
    (self.lateral1-5) already project an arbitrary channel count to
    bifpn_channels, so swapping the channel source (timm backbone vs.
    scratch ResidualBlocks) requires no other code changes -- BiFPN neck,
    RARM, RAN, and Option A are all identical either way.
    """
    def _build_encoder(self, in_channels, features, backbone='custom', pretrained_backbone=True):
        self.backbone_name = backbone
        bifpn_channels = features[3]

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

            self.lateral5 = nn.Conv2d(features[4], bifpn_channels, 1)
            self.lateral4 = nn.Conv2d(features[3], bifpn_channels, 1)
            self.lateral3 = nn.Conv2d(features[2], bifpn_channels, 1)
            self.lateral2 = nn.Conv2d(features[1], bifpn_channels, 1)
            self.lateral1 = nn.Conv2d(features[0], bifpn_channels, 1)
        else:
            if timm is None:
                raise ImportError(
                    "timm is required for pretrained backbones: pip install timm"
                )
            self.timm_backbone = timm.create_model(
                backbone, pretrained=pretrained_backbone,
                features_only=True, out_indices=(0, 1, 2, 3, 4),
                in_chans=in_channels,
            )
            channels = self.timm_backbone.feature_info.channels()
            if len(channels) != 5:
                raise ValueError(
                    f"Backbone '{backbone}' produced {len(channels)} feature levels "
                    f"(channels={channels}), expected exactly 5 to match this "
                    f"architecture's 5-level BiFPN neck. Check out_indices or pick "
                    f"a different backbone -- not every timm model has 5 usable stages."
                )
            c1, c2, c3, c4, c5 = channels
            self.lateral1 = nn.Conv2d(c1, bifpn_channels, 1)
            self.lateral2 = nn.Conv2d(c2, bifpn_channels, 1)
            self.lateral3 = nn.Conv2d(c3, bifpn_channels, 1)
            self.lateral4 = nn.Conv2d(c4, bifpn_channels, 1)
            self.lateral5 = nn.Conv2d(c5, bifpn_channels, 1)

    def _make_encoder_block(self, in_channels, out_channels, num_blocks, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _encode_and_fuse(self, x):
        """Runs stem->encoder->lateral->BiFPN neck. Returns the 5
        BiFPN-fused pyramid levels (p5..p1) shared by all three archs."""
        if self.timm_backbone is not None:
            e1, e2, e3, e4, e5 = self.timm_backbone(x)
        else:
            x = self.stem(x)
            e1 = self.encoder1(x)
            e2 = self.encoder2(e1)
            e3 = self.encoder3(e2)
            e4 = self.encoder4(e3)
            e5 = self.encoder5(e4)

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


# ==============================================================================
# Architecture 1: BiFPN-RAN (no RARM -- BiFPN neck + cascaded reverse
# attention decoder only). Ablation baseline: isolates what RAN alone
# contributes without RARM's per-level refinement.
# ==============================================================================

class BiFPN_RAN_ShadowDetector(nn.Module, _SharedEncoderMixin):
    def __init__(self, in_channels=3, out_channels=1,
                 features=[64, 128, 256, 512, 1024],
                 ra_channels=64, dropout_p=0.2, share_weights=True,
                 use_distraction_head=False,
                 backbone='custom', pretrained_backbone=True):
        super().__init__()
        self.use_distraction_head = use_distraction_head
        self._build_encoder(in_channels, features, backbone=backbone,
                             pretrained_backbone=pretrained_backbone)

        self.bifpn_channels = features[3]
        self.bifpn1 = BiFPNLayer(self.bifpn_channels)
        self.bifpn2 = BiFPNLayer(self.bifpn_channels)
        self.bifpn3 = BiFPNLayer(self.bifpn_channels)

        self.global_map_head = nn.Conv2d(self.bifpn_channels, 1, kernel_size=3, padding=1)
        if share_weights:
            shared_ra = ReverseAttentionBlock(self.bifpn_channels, ra_channels, dropout_p=dropout_p)
            self.ra4 = self.ra3 = self.ra2 = self.ra1 = shared_ra
        else:
            self.ra4 = ReverseAttentionBlock(self.bifpn_channels, ra_channels, dropout_p=dropout_p)
            self.ra3 = ReverseAttentionBlock(self.bifpn_channels, ra_channels, dropout_p=dropout_p)
            self.ra2 = ReverseAttentionBlock(self.bifpn_channels, ra_channels, dropout_p=dropout_p)
            self.ra1 = ReverseAttentionBlock(self.bifpn_channels, ra_channels, dropout_p=dropout_p)

        self.final = nn.Conv2d(ra_channels, out_channels, 1)

        if use_distraction_head:
            self.distraction_head = DistractionAwareHead(ra_channels)
            self.distraction_suppression_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        orig_shape = x.shape[2:]
        p5, p4_td, p3_td, p2_td, p1_td = self._encode_and_fuse(x)

        # No RARM here -- RA cascade operates directly on BiFPN-fused features
        global_pred = self.global_map_head(p5)
        pred4, _ = self.ra4(p4_td, global_pred)
        pred3, _ = self.ra3(p3_td, pred4)
        pred2, _ = self.ra2(p2_td, pred3)
        pred1, refined1 = self.ra1(p1_td, pred2)

        out = self.final(refined1)

        distraction_logits = None
        if self.use_distraction_head:
            distraction_logits = self.distraction_head(refined1)
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
# Architecture 2: BiFPN-RARM (no RAN -- BiFPN neck + RARM refinement +
# transposed-conv decoder with skip concatenation). This is a
# REIMPLEMENTATION from your methodology doc's description of the
# documented 0.8442-IoU model, not the literal original file (we never
# had that code in this conversation) -- built from the same shared
# blocks so it's a fair ablation partner to the other two, with deep
# supervision added at each decoder stage for consistency (the original
# may not have had this; flagging so you don't mistake this for a byte-
# identical reproduction).
# ==============================================================================

class BiFPN_RARM_ShadowDetector(nn.Module, _SharedEncoderMixin):
    def __init__(self, in_channels=3, out_channels=1,
                 features=[64, 128, 256, 512, 1024],
                 rarm_iterations=2, dropout_p=0.2, share_weights=True,
                 use_distraction_head=False,
                 backbone='custom', pretrained_backbone=True):
        super().__init__()
        self.use_distraction_head = use_distraction_head
        self._build_encoder(in_channels, features, backbone=backbone,
                             pretrained_backbone=pretrained_backbone)

        self.bifpn_channels = bc = features[3]
        self.bifpn1 = BiFPNLayer(bc)
        self.bifpn2 = BiFPNLayer(bc)
        self.bifpn3 = BiFPNLayer(bc)

        if share_weights:
            shared_rarm = RARMBlock(bc, num_iterations=rarm_iterations, dropout_p=dropout_p)
            self.rarm5 = self.rarm4 = self.rarm3 = self.rarm2 = self.rarm1 = shared_rarm
        else:
            self.rarm5 = RARMBlock(bc, num_iterations=rarm_iterations, dropout_p=dropout_p)
            self.rarm4 = RARMBlock(bc, num_iterations=rarm_iterations, dropout_p=dropout_p)
            self.rarm3 = RARMBlock(bc, num_iterations=rarm_iterations, dropout_p=dropout_p)
            self.rarm2 = RARMBlock(bc, num_iterations=rarm_iterations, dropout_p=dropout_p)
            self.rarm1 = RARMBlock(bc, num_iterations=rarm_iterations, dropout_p=dropout_p)

        # Transposed-conv decoder with skip concatenation (deep -> shallow)
        def up_block():
            return nn.ConvTranspose2d(bc, bc, kernel_size=2, stride=2)

        def fuse_block():
            return nn.Sequential(
                nn.Conv2d(bc * 2, bc, 3, padding=1, bias=False),
                nn.BatchNorm2d(bc), nn.ReLU(inplace=True), nn.Dropout2d(p=dropout_p)
            )

        self.up4, self.fuse4 = up_block(), fuse_block()
        self.up3, self.fuse3 = up_block(), fuse_block()
        self.up2, self.fuse2 = up_block(), fuse_block()
        self.up1, self.fuse1 = up_block(), fuse_block()

        # Deep-supervision aux heads at each decoder stage
        self.aux4 = nn.Conv2d(bc, 1, kernel_size=3, padding=1)
        self.aux3 = nn.Conv2d(bc, 1, kernel_size=3, padding=1)
        self.aux2 = nn.Conv2d(bc, 1, kernel_size=3, padding=1)

        self.final = nn.Conv2d(bc, out_channels, 1)

        if use_distraction_head:
            self.distraction_head = DistractionAwareHead(bc)
            self.distraction_suppression_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        orig_shape = x.shape[2:]
        p5, p4_td, p3_td, p2_td, p1_td = self._encode_and_fuse(x)

        # RARM refinement at every level (no RA cascade here)
        p5_r = self.rarm5(p5)
        p4_r = self.rarm4(p4_td)
        p3_r = self.rarm3(p3_td)
        p2_r = self.rarm2(p2_td)
        p1_r = self.rarm1(p1_td)

        # Transposed-conv decoder, skip concatenation, deep -> shallow
        d4 = self.up4(p5_r)
        d4 = self.fuse4(torch.cat([d4, p4_r], dim=1))
        aux4 = self.aux4(d4)

        d3 = self.up3(d4)
        d3 = self.fuse3(torch.cat([d3, p3_r], dim=1))
        aux3 = self.aux3(d3)

        d2 = self.up2(d3)
        d2 = self.fuse2(torch.cat([d2, p2_r], dim=1))
        aux2 = self.aux2(d2)

        d1 = self.up1(d2)
        d1 = self.fuse1(torch.cat([d1, p1_r], dim=1))

        out = self.final(d1)

        distraction_logits = None
        if self.use_distraction_head:
            distraction_logits = self.distraction_head(d1)
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
                return out, aux4, aux3, aux2, distraction_logits
            return out, aux4, aux3, aux2
        return torch.sigmoid(out)


# ==============================================================================
# Combined model
# ==============================================================================

class BiFPN_RARM_RAN_ShadowDetector(nn.Module, _SharedEncoderMixin):
    """
    Encoder -> BiFPN neck (3 stacked layers, as in BiFPN-RARM) ->
    per-level RARM refinement (as in BiFPN-RARM) -> cascaded reverse
    attention decoder (as in BiFPN-RAN), with deep supervision at
    every cascade stage.
    """
    def __init__(self, in_channels=3, out_channels=1,
                 features=[64, 128, 256, 512, 1024],
                 rarm_iterations=2, ra_channels=64,
                 dropout_p=0.2, share_weights=True, use_distraction_head=False,
                 backbone='custom', pretrained_backbone=True):
        """
        share_weights: if True, ties one RARMBlock across all 5 pyramid
        levels and one ReverseAttentionBlock across all 4 cascade stages,
        instead of instantiating 5 + 4 separate sets of weights. Cuts
        decoder-side parameters roughly 5x / 4x respectively. Shadow
        boundary refinement doesn't inherently need level-specific
        parameters, and with only 1,330 ISTD fine-tuning images this is
        the single highest-leverage change for reducing overfitting risk
        -- prefer this over adding more dropout/weight-decay on top of an
        over-parameterized model. Set False only if an ablation shows
        per-level weights are worth the added capacity.

        use_distraction_head: if True (Option A), adds the DSDNet-inspired
        auxiliary branch (see DistractionAwareHead docstring) that learns
        to suppress shadow-like-but-not-shadow false positives. Must be
        paired with DistractionAwareLoss during training -- plain
        DeepSupervisionLoss won't know what to do with the extra 7th
        output element. Default False keeps this class's forward()
        signature identical to before, so existing training code that
        doesn't opt in is unaffected.

        backbone: 'custom' (default, from-scratch encoder) or any timm
        model name (e.g. 'resnet50') for an ImageNet-pretrained encoder.
        See _SharedEncoderMixin docstring.
        """
        super().__init__()
        self.features = features
        self.share_weights = share_weights
        self.use_distraction_head = use_distraction_head

        self._build_encoder(in_channels, features, backbone=backbone,
                             pretrained_backbone=pretrained_backbone)

        self.bifpn_channels = features[3]
        self.bifpn1 = BiFPNLayer(self.bifpn_channels)
        self.bifpn2 = BiFPNLayer(self.bifpn_channels)
        self.bifpn3 = BiFPNLayer(self.bifpn_channels)

        # RARM: shared weights across levels by default (see docstring above)
        if share_weights:
            shared_rarm = RARMBlock(self.bifpn_channels, num_iterations=rarm_iterations,
                                     dropout_p=dropout_p)
            self.rarm5 = self.rarm4 = self.rarm3 = self.rarm2 = self.rarm1 = shared_rarm
        else:
            self.rarm5 = RARMBlock(self.bifpn_channels, num_iterations=rarm_iterations, dropout_p=dropout_p)
            self.rarm4 = RARMBlock(self.bifpn_channels, num_iterations=rarm_iterations, dropout_p=dropout_p)
            self.rarm3 = RARMBlock(self.bifpn_channels, num_iterations=rarm_iterations, dropout_p=dropout_p)
            self.rarm2 = RARMBlock(self.bifpn_channels, num_iterations=rarm_iterations, dropout_p=dropout_p)
            self.rarm1 = RARMBlock(self.bifpn_channels, num_iterations=rarm_iterations, dropout_p=dropout_p)

        # Cascaded reverse-attention decoder, operating on RARM-refined features.
        # Shared weights across cascade stages by default (see docstring above).
        self.global_map_head = nn.Conv2d(self.bifpn_channels, 1, kernel_size=3, padding=1)
        if share_weights:
            shared_ra = ReverseAttentionBlock(self.bifpn_channels, ra_channels, dropout_p=dropout_p)
            self.ra4 = self.ra3 = self.ra2 = self.ra1 = shared_ra
        else:
            self.ra4 = ReverseAttentionBlock(self.bifpn_channels, ra_channels, dropout_p=dropout_p)
            self.ra3 = ReverseAttentionBlock(self.bifpn_channels, ra_channels, dropout_p=dropout_p)
            self.ra2 = ReverseAttentionBlock(self.bifpn_channels, ra_channels, dropout_p=dropout_p)
            self.ra1 = ReverseAttentionBlock(self.bifpn_channels, ra_channels, dropout_p=dropout_p)

        self.final = nn.Conv2d(ra_channels, out_channels, 1)

        # Option A: distraction-aware auxiliary branch (opt-in)
        if use_distraction_head:
            self.distraction_head = DistractionAwareHead(ra_channels)
            # Learnable rather than a fixed magic constant, so the model
            # decides how much to trust the distraction signal.
            self.distraction_suppression_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        orig_shape = x.shape[2:]

        # --- Encoder -> BiFPN neck (shared with the other two architectures) ---
        p5, p4_td, p3_td, p2_td, p1_td = self._encode_and_fuse(x)

        # --- RARM: per-level feature refinement (channel + spatial attn, recurrent) ---
        p5_r = self.rarm5(p5)
        p4_r = self.rarm4(p4_td)
        p3_r = self.rarm3(p3_td)
        p2_r = self.rarm2(p2_td)
        p1_r = self.rarm1(p1_td)

        # --- Cascaded reverse attention: deep -> shallow boundary refinement ---
        global_pred = self.global_map_head(p5_r)
        pred4, _ = self.ra4(p4_r, global_pred)
        pred3, _ = self.ra3(p3_r, pred4)
        pred2, _ = self.ra2(p2_r, pred3)
        pred1, refined1 = self.ra1(p1_r, pred2)

        out = self.final(refined1)

        distraction_logits = None
        if self.use_distraction_head:
            distraction_logits = self.distraction_head(refined1)
            distraction_up = distraction_logits
            if distraction_logits.shape[2:] != out.shape[2:]:
                distraction_up = F.interpolate(distraction_logits, size=out.shape[2:],
                                                mode='bilinear', align_corners=False)
            # Logit-domain suppression: reduce shadow confidence in
            # proportion to how "distraction-like" the head thinks this
            # pixel is.
            out = out - self.distraction_suppression_scale * torch.sigmoid(distraction_up)

        if out.shape[2:] != orig_shape:
            out = F.interpolate(out, size=orig_shape, mode='bilinear', align_corners=False)
            if distraction_logits is not None and distraction_logits.shape[2:] != orig_shape:
                distraction_logits = F.interpolate(distraction_logits, size=orig_shape,
                                                    mode='bilinear', align_corners=False)

        if self.training:
            # Deep supervision over every cascade stage
            if self.use_distraction_head:
                return out, global_pred, pred4, pred3, pred2, pred1, distraction_logits
            return out, global_pred, pred4, pred3, pred2, pred1
        else:
            return torch.sigmoid(out)


# ==============================================================================
# Phase-2 transfer-learning helpers
# ==============================================================================

def freeze_for_finetuning(model, frozen_modules=('stem', 'encoder1', 'encoder2')):
    """
    Freezes the given submodules for Phase 2 fine-tuning.

    Critically, freezing requires_grad alone does NOT stop BatchNorm's
    running_mean/running_var from drifting -- those update from forward-
    pass batch statistics regardless of requires_grad. Left unaddressed,
    a frozen-but-train()-mode encoder will silently re-normalise to
    ISTD's much smaller batch statistics, undermining the point of
    freezing. This also permanently sets those submodules to eval mode
    (dropout off, BN using frozen running stats) and returns the list of
    frozen submodule names so the training loop can re-assert .eval() on
    them every epoch after the top-level model.train() call flips
    everything back to train mode.
    """
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
    """Call after model.train() in the training loop -- train() recursively
    puts every submodule back in train mode, including frozen ones, so
    frozen BatchNorm layers must be pinned back to eval() each epoch."""
    for name in frozen_module_names:
        module = getattr(model, name, None)
        if module is not None:
            module.eval()


def get_discriminative_param_groups(model, base_lr, frozen_module_names=()):
    """
    Builds optimizer param groups with per-region learning-rate
    multipliers instead of a hard frozen/full-LR cliff:
      - frozen_module_names: excluded entirely (requires_grad already False)
      - encoder1-2 (if not frozen): 0.05x base_lr -- closest to the frozen
        stem, most generic low-level features; gentlest LR of any
        unfrozen region if these ever get unfrozen (e.g. freeze_modules
        narrowed to just ['stem'])
      - encoder3-5 (if not frozen): 0.1x base_lr -- still adapting low/mid
        level features, but gently
      - BiFPN neck: 0.5x base_lr -- these fusion weights are exactly what
        your discussion section credits with adapting to shadow scale, so
        they should keep learning, just not as fast as the decoder
      - RARM / reverse-attention decoder + final head: 1.0x base_lr -- the
        newest, least-trained-on-SBU parts of the model

    Any module NOT listed above (encoder1/2/3/4/5, BiFPN, decoder) that
    still has requires_grad=True params would silently never be optimized
    if omitted from every group -- an easy way to accidentally "freeze"
    something without meaning to when freeze_modules changes. The
    assertion at the end catches that class of bug: it fails loudly
    instead of training with part of the model silently un-updated.

    Pretrained-backbone models (backbone != 'custom' at construction,
    exposed as model.timm_backbone) get ONE group instead of the
    stem/encoder1-2/encoder3-5 tiers above, at 0.1x base_lr -- gentle
    enough not to destroy ImageNet-pretrained features on a 1,330-image
    fine-tuning set, but still adapting rather than fully frozen, matching
    what published shadow detectors (BDRAR, DSDNet, FDRNet) do with their
    pretrained backbones.
    """
    stem_names = ['stem']
    low_level_encoder_names = ['encoder1', 'encoder2']
    mid_encoder_names = ['encoder3', 'encoder4', 'encoder5']
    bifpn_names = ['lateral5', 'lateral4', 'lateral3', 'lateral2', 'lateral1',
                    'bifpn1', 'bifpn2', 'bifpn3']

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
        add_group([n for n in stem_names if n not in frozen_module_names], 0.02)
        add_group([n for n in low_level_encoder_names if n not in frozen_module_names], 0.05)
        add_group([n for n in mid_encoder_names if n not in frozen_module_names], 0.1)
    add_group(bifpn_names, 0.5)
    # Superset of decoder module names across all three architectures --
    # getattr(model, name, None) inside add_group() skips whichever names
    # a given architecture doesn't have, so this one function works for
    # BiFPN-RAN, BiFPN-RARM, and BiFPN-RARM-RAN without special-casing.
    decoder_items = [
        'global_map_head', 'rarm5', 'rarm4', 'rarm3', 'rarm2', 'rarm1',   # RARM (if present)
        'ra4', 'ra3', 'ra2', 'ra1',                                        # RAN cascade (if present)
        'up4', 'fuse4', 'aux4', 'up3', 'fuse3', 'aux3',                    # transposed-conv decoder (if present)
        'up2', 'fuse2', 'aux2', 'up1', 'fuse1',
        'final',
    ]
    if getattr(model, 'use_distraction_head', False):
        decoder_items.append('distraction_head')
    add_group(decoder_items, 1.0)

    # distraction_suppression_scale is a raw nn.Parameter (not inside a
    # module), so add_group's getattr-based module walk can't reach it --
    # add it to the decoder group directly.
    suppression_param = getattr(model, 'distraction_suppression_scale', None)
    if suppression_param is not None and suppression_param.requires_grad \
            and id(suppression_param) not in seen_ids:
        groups[-1]['params'].append(suppression_param)
        seen_ids.add(id(suppression_param))

    # Safety net: every trainable param in the model must end up in
    # exactly one group above, or it would silently never receive an
    # optimizer update despite requires_grad=True.
    grouped_ids = seen_ids
    all_trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    missing = all_trainable_ids - grouped_ids
    if missing:
        missing_names = [name for name, p in model.named_parameters()
                          if p.requires_grad and id(p) in missing]
        raise RuntimeError(
            f"{len(missing)} trainable parameter(s) not assigned to any "
            f"optimizer param group (would never be updated): "
            f"{missing_names[:10]}{'...' if len(missing_names) > 10 else ''}. "
            f"Add their parent module to get_discriminative_param_groups."
        )

    return groups


if __name__ == "__main__":
    dummy = torch.randn(2, 3, 256, 256)
    dummy_mask = torch.randint(0, 2, (2, 1, 256, 256)).float()

    architectures = {
        'BiFPN-RAN': BiFPN_RAN_ShadowDetector,
        'BiFPN-RARM': BiFPN_RARM_ShadowDetector,
        'BiFPN-RARM-RAN': BiFPN_RARM_RAN_ShadowDetector,
    }

    for name, cls in architectures.items():
        for use_distraction in (False, True):
            model = cls(share_weights=True, use_distraction_head=use_distraction)

            model.train()
            outs = model(dummy)
            assert outs[0].shape == (2, 1, 256, 256)
            if use_distraction:
                assert outs[-1].shape[1] == 1  # distraction logits, single channel

            model.eval()
            with torch.no_grad():
                out = model(dummy)
            assert out.shape == (2, 1, 256, 256)

            total_params = sum(p.numel() for p in model.parameters())
            tag = f"{name} (distraction_head={use_distraction})"
            print(f"{tag}: {total_params:,} params, train_outputs={len(outs)}, "
                  f"eval_shape={tuple(out.shape)}")

            # Freeze + discriminative LR groups must cover every trainable
            # param for every architecture x Option A combination
            frozen = freeze_for_finetuning(model, frozen_modules=('stem',))
            groups = get_discriminative_param_groups(model, base_lr=1e-5, frozen_module_names=frozen)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            grouped = sum(p.numel() for g in groups for p in g['params'])
            assert trainable == grouped, f"{tag}: grouped params don't match trainable params"

            # Loss forward pass
            if use_distraction:
                model.train()
                outs = model(dummy)  # re-run in train mode (freeze doesn't affect forward shapes)
                loss = DistractionAwareLoss()(outs, dummy_mask)
            else:
                model.train()
                outs = model(dummy)
                loss = DeepSupervisionLoss()(outs, dummy_mask)
            assert torch.isfinite(loss)
            print(f"  Loss forward OK: {loss.item():.4f}, trainable={trainable:,}, grouped={grouped:,}")

    print("\nAll architecture x Option A combinations passed (custom encoder).")

    # Pretrained-backbone path: use pretrained=False here to avoid a
    # network call in this test (Colab should use pretrained=True, the
    # default). This still exercises the exact same timm feature-
    # extraction, lateral-conv sizing, freeze, and param-group code paths.
    if timm is not None:
        print("\nTesting pretrained-backbone path (timm resnet50, pretrained=False)...")
        for name, cls in architectures.items():
            model = cls(share_weights=True, backbone='resnet50', pretrained_backbone=False)
            assert model.timm_backbone is not None

            model.train()
            outs = model(dummy)
            assert outs[0].shape == (2, 1, 256, 256)

            model.eval()
            with torch.no_grad():
                out = model(dummy)
            assert out.shape == (2, 1, 256, 256)

            total_params = sum(p.numel() for p in model.parameters())
            print(f"  {name} + resnet50 backbone: {total_params:,} params, "
                  f"eval_shape={tuple(out.shape)}")

            frozen = freeze_for_finetuning(model, frozen_modules=('timm_backbone',))
            groups = get_discriminative_param_groups(model, base_lr=1e-5, frozen_module_names=frozen)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            grouped = sum(p.numel() for g in groups for p in g['params'])
            assert trainable == grouped, f"{name}+resnet50: grouped params don't match trainable"
            print(f"    Freeze/param-group check OK: trainable={trainable:,}, grouped={grouped:,}")

        print("\nAll architecture x pretrained-backbone combinations passed.")
    else:
        print("\ntimm not installed -- skipping pretrained-backbone test.")