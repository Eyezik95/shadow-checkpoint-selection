"""UNet++ (EfficientNet-B5) internal baseline of Table 4.2 (Objective 2; Section 3.3.2.1).

Copy of reference/notebooks/Untitled16_unetplusplus.ipynb cell 1, lines 56-79
(notebook sha256 0a26b5f2ae8db7b3...), the class that trained best_model_unetplusplus.pth.
One change: encoder_weights is a parameter (notebook: fixed "imagenet"), so that a model
about to receive a checkpoint does not download ImageNet weights first. With the default
the class is the notebook's.

segmentation_models_pytorch 0.6.0 (the Colab version is not recorded); checkpoints load
with strict=True, which checks that the parameter names and shapes agree.
"""

import torch
import torch.nn as nn

import segmentation_models_pytorch as smp


class UNetPlusPlus_EfficientNetB5(nn.Module):
    """
    Standard UNet++ with EfficientNet-b5 backbone control model.
    Uses pre-trained ImageNet weights for the encoder to ensure a fair,
    highly-optimized baseline comparison against your custom architectures.
    """
    def __init__(self, in_channels=3, out_channels=1, encoder_weights="imagenet", **kwargs):
        super().__init__()
        # SMP does not have a distraction head
        self.use_distraction_head = False

        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b5",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
        )

    def forward(self, x):
        logits = self.model(x)
        if self.training:
            # Wrap in a tuple because DeepSupervisionLoss expects an iterable of intermediate predictions
            return (logits,)
        return torch.sigmoid(logits)
