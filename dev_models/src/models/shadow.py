"""BiFPN-SIRAN shadow detector (Section 3.3, Figure 3.2), inference only.

Ported from the training notebook (obj2). The BiFPN width is `features[3]`
(512), not the backbone's 4th-level width (176) that the obj4 inference copy
used; with 176 the selected checkpoint cannot be loaded at all. Loading here is
strict: every parameter must be present and correctly shaped, so no layer can
silently stay at its random initialisation.

Preprocessing matches the deployed ShadowSilhouetteGenerator: RGB, resize to
256x256, ImageNet mean/std normalisation.
"""

from __future__ import annotations

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BiFPN_SIRAN_ShadowDetector", "load_siran", "ShadowMasker"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.depthwise = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin, bias=False)
        self.pointwise = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class BiFPNLayer(nn.Module):
    def __init__(self, channels, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.w_td = nn.Parameter(torch.ones(3))
        self.w_out = nn.Parameter(torch.ones(2))
        self.conv_td = DepthwiseSeparableConv(channels, channels)
        self.conv_out = DepthwiseSeparableConv(channels, channels)

    def forward(self, p_in, p_skip=None):
        w_td = F.relu(self.w_td); w_td = w_td / (w_td.sum() + self.epsilon)
        p_td = self.conv_td(w_td[0] * p_in + w_td[1] * p_skip) if p_skip is not None \
            else self.conv_td(p_in)
        w_out = F.relu(self.w_out); w_out = w_out / (w_out.sum() + self.epsilon)
        return self.conv_out(w_out[0] * p_in + w_out[1] * p_td)


def _groups(c, g=32):
    for k in range(min(g, c), 0, -1):
        if c % k == 0:
            return k
    return 1


class SIRANBlock(nn.Module):
    def __init__(self, bifpn_c, skip_c, out_c, dropout_p=0.2):
        super().__init__()
        self.skip_proj = nn.Sequential(nn.Conv2d(skip_c, bifpn_c, 1, bias=False),
                                       nn.GroupNorm(_groups(bifpn_c), bifpn_c), nn.ReLU(inplace=True))
        self.fusion_conv = nn.Sequential(nn.Conv2d(bifpn_c * 2, out_c, 3, padding=1, bias=False),
                                         nn.GroupNorm(_groups(out_c), out_c), nn.ReLU(inplace=True))
        self.refine_conv = nn.Sequential(nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                                         nn.GroupNorm(_groups(out_c), out_c), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.mask_predictor = nn.Conv2d(out_c, 1, 3, padding=1)

    def forward(self, bifpn_feat, skip_feat, prev_mask):
        size = bifpn_feat.shape[2:]
        if prev_mask.shape[2:] != size:
            prev_mask = F.interpolate(prev_mask, size=size, mode="bilinear", align_corners=False)
        s = self.skip_proj(skip_feat)
        if s.shape[2:] != size:
            s = F.interpolate(s, size=size, mode="bilinear", align_corners=False)
        f = self.fusion_conv(torch.cat([bifpn_feat, s], 1)) * (1.0 - torch.sigmoid(prev_mask))
        f = self.dropout(self.refine_conv(f))
        return f, self.mask_predictor(f)


class BiFPN_SIRAN_ShadowDetector(nn.Module):
    def __init__(self, backbone="efficientnet_b5", bifpn_channels=512, siran_channels=64):
        super().__init__()
        self.timm_backbone = timm.create_model(backbone, pretrained=False, features_only=True,
                                               out_indices=(0, 1, 2, 3, 4))
        c = self.timm_backbone.feature_info.channels()
        for i in range(5):
            setattr(self, f"lateral{i + 1}", nn.Conv2d(c[i], bifpn_channels, 1))
        self.bifpn1, self.bifpn2, self.bifpn3 = (BiFPNLayer(bifpn_channels) for _ in range(3))
        self.global_map_head = nn.Conv2d(bifpn_channels, 1, 3, padding=1)
        self.siran4 = SIRANBlock(bifpn_channels, c[3], siran_channels)
        self.siran3 = SIRANBlock(bifpn_channels, c[2], siran_channels)
        self.siran2 = SIRANBlock(bifpn_channels, c[1], siran_channels)
        self.siran1 = SIRANBlock(bifpn_channels, c[0], siran_channels)
        self.final = nn.Conv2d(siran_channels, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        e1, e2, e3, e4, e5 = self.timm_backbone(x)
        p1, p2, p3, p4, p5 = (getattr(self, f"lateral{i}")(e) for i, e in
                              enumerate((e1, e2, e3, e4, e5), start=1))
        up = lambda a, b: F.interpolate(a, size=b.shape[2:], mode="bilinear", align_corners=False)
        p4_td = self.bifpn1(p4, up(p5, p4))
        p3_td = self.bifpn2(p3, up(p4_td, p3))
        p2_td = self.bifpn3(p2, up(p3_td, p2))
        p1_td = self.bifpn1(p1, up(p2_td, p1))          # bifpn1 reused, as in training
        g = self.global_map_head(p5)
        f4, m4 = self.siran4(p4_td, e4, g)
        f3, m3 = self.siran3(p3_td, e3, m4)
        f2, m2 = self.siran2(p2_td, e2, m3)
        f1, _ = self.siran1(p1_td, e1, m2)
        out = self.final(f1)
        if out.shape[2:] != size:
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)
        return out                                       # logits


def load_siran(path, map_location="cpu") -> tuple[BiFPN_SIRAN_ShadowDetector, dict]:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    sd = ck.get("model_state_dict", ck)
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    width = sd["lateral1.weight"].shape[0]
    model = BiFPN_SIRAN_ShadowDetector(bifpn_channels=width)
    model.load_state_dict(sd, strict=True)
    meta = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in ck.items()
            if k in ("epoch", "val_iou", "val_loss", "val_ber")}
    meta["bifpn_channels"] = int(width)
    return model.eval(), meta


class ShadowMasker:
    """Crop -> binary shadow mask at crop resolution (Section 3.3.8 / 3.5.3)."""

    def __init__(self, model: nn.Module, threshold: float = 0.5, size: int = 256, device="cpu",
                 max_batch: int = 4):
        # max_batch bounds peak memory: at full resolution each crop costs hundreds
        # of MB of activations, and on Apple silicon that comes out of system RAM.
        self.model, self.threshold, self.size, self.device = model.to(device), threshold, size, device
        self.max_batch = max_batch

    def _prep(self, crop_bgr):
        # Crops are slices of the frame; on sequence 15 cv2.resize segfaulted (frame 621)
        # on such views and ran cleanly on contiguous copies. Copying changes no pixel.
        crop_bgr = np.ascontiguousarray(crop_bgr)
        rgb = cv2.cvtColor(cv2.resize(crop_bgr, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        return ((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)

    @torch.no_grad()
    def batch(self, crops_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """Same as calling the masker on each crop; one forward pass (eval mode, so
        BatchNorm uses running statistics and the batch does not mix samples)."""
        out = [np.zeros(c.shape[:2], np.uint8) for c in crops_bgr]
        idx = [i for i, c in enumerate(crops_bgr) if c.shape[0] > 0 and c.shape[1] > 0]
        if not idx:
            return out
        prob = []
        for j in range(0, len(idx), self.max_batch):
            x = torch.from_numpy(np.stack([self._prep(crops_bgr[i]) for i in idx[j:j + self.max_batch]]))
            prob.append(torch.sigmoid(self.model(x.to(self.device)))[:, 0].cpu().numpy())
        prob = np.concatenate(prob)
        if not np.isfinite(prob).all():
            raise FloatingPointError("shadow probability map is not finite")
        for j, i in enumerate(idx):
            h, w = crops_bgr[i].shape[:2]
            out[i] = cv2.resize((prob[j] > self.threshold).astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST)
        return out

    def __call__(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.batch([crop_bgr])[0]
