"""BiFPN-RAN, iteration 1 (custom backbone), as it was trained (Objective 2; Table 3.2).

Verbatim copy of reference/notebooks/bifpn_ran_model_training.ipynb cell 5, lines 15-24 and
40-232 (imports and model classes; notebook sha256 53d32ecdb1fd32a7...). That cell's printed
output ends "Best validation IoU: 0.8523", the val_iou stored in
data/raw/checkpoints/best_model_bifpn_ran.pth, so this is the code that produced the
checkpoint. The later BiFPN_RAN_ShadowDetector in src.models.obj2_variants loads the
same weights with strict key matching but computes a different forward pass.
Kept unedited; in eval mode the model returns sigmoid probabilities if the notebook does.
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. MODEL ARCHITECTURE: BiFPN-RAN (Reverse Attention Network)
# ==============================================================================

class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise separable convolution"""
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


class BiFPNLayer(nn.Module):
    """Single BiFPN Layer with weighted feature fusion"""
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


class ReverseAttentionBlock(nn.Module):
    """Mines boundary features by erasing confident predictions"""
    def __init__(self, in_channels, out_channels=64):
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
        new_prediction = self.pred_head(refined)

        return new_prediction, refined


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


class BiFPN_RAN_ShadowDetector(nn.Module):
    """BiFPN neck + cascaded Reverse Attention Network decoder"""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.features = features
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(features[0]), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1)
        )

        self.encoder1 = self._make_encoder_block(features[0], features[0], 2)
        self.encoder2 = self._make_encoder_block(features[0], features[1], 2, stride=2)
        self.encoder3 = self._make_encoder_block(features[1], features[2], 2, stride=2)
        self.encoder4 = self._make_encoder_block(features[2], features[3], 2, stride=2)
        self.encoder5 = self._make_encoder_block(features[3], features[4], 2, stride=2)

        self.lateral5 = nn.Conv2d(features[4], features[3], 1)
        self.lateral4 = nn.Conv2d(features[3], features[3], 1)
        self.lateral3 = nn.Conv2d(features[2], features[3], 1)
        self.lateral2 = nn.Conv2d(features[1], features[3], 1)
        self.lateral1 = nn.Conv2d(features[0], features[3], 1)

        self.bifpn_channels = features[3]
        self.bifpn1 = BiFPNLayer(self.bifpn_channels)
        self.bifpn2 = BiFPNLayer(self.bifpn_channels)
        self.bifpn3 = BiFPNLayer(self.bifpn_channels)

        self.global_map_head = nn.Conv2d(self.bifpn_channels, 1, kernel_size=3, padding=1)

        self.ra4 = ReverseAttentionBlock(in_channels=self.bifpn_channels, out_channels=64)
        self.ra3 = ReverseAttentionBlock(in_channels=self.bifpn_channels, out_channels=64)
        self.ra2 = ReverseAttentionBlock(in_channels=self.bifpn_channels, out_channels=64)
        self.ra1 = ReverseAttentionBlock(in_channels=self.bifpn_channels, out_channels=64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def _make_encoder_block(self, in_channels, out_channels, num_blocks, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        orig_shape = x.shape[2:]

        # Encoder
        x = self.stem(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # BiFPN input projection
        p5 = self.lateral5(e5)
        p4 = self.lateral4(e4)
        p3 = self.lateral3(e3)
        p2 = self.lateral2(e2)
        p1 = self.lateral1(e1)

        # BiFPN neck fusion
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode='bilinear', align_corners=False)
        p4_td = self.bifpn1(p4, p5_up)
        p4_up = F.interpolate(p4_td, size=p3.shape[2:], mode='bilinear', align_corners=False)
        p3_td = self.bifpn2(p3, p4_up)
        p3_up = F.interpolate(p3_td, size=p2.shape[2:], mode='bilinear', align_corners=False)
        p2_td = self.bifpn3(p2, p3_up)
        p2_up = F.interpolate(p2_td, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p1_td = self.bifpn1(p1, p2_up)

        # Cascaded reverse attention (deep -> shallow)
        global_pred = self.global_map_head(p5)
        pred4, _ = self.ra4(p4_td, global_pred)
        pred3, _ = self.ra3(p3_td, pred4)
        pred2, _ = self.ra2(p2_td, pred3)
        pred1, refined1 = self.ra1(p1_td, pred2)

        out = self.final(refined1)
        if out.shape[2:] != orig_shape:
            out = F.interpolate(out, size=orig_shape, mode='bilinear', align_corners=False)

        if self.training:
            # Deep supervision: return all intermediate predictions
            return out, global_pred, pred4, pred3, pred2, pred1
        else:
            return torch.sigmoid(out)


