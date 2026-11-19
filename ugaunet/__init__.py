"""UGA-UNet: Uncertainty-Guided Boundary Anchoring for lightweight segmentation.

This package is a cleaned, runnable implementation of the method described in

    UGA-UNet: Lightweight Segmentation with Uncertainty-Guided Boundary Anchoring
    ARRL-Health'26, in conjunction with IEEE BIBM 2026

The core contributions are:

* :class:`ugaunet.models.UGAUNet` -- a 3.9M-parameter U-Net-style encoder/decoder
  with a Local-Global Bottleneck (LGB).
* :func:`ugaunet.losses.anchor_loss_from_maps` -- the Uncertainty-Guided Anchor
  Loss operating on decoder features.
"""

from .models import UGAUNet, build_model
from .losses import LossCEDice, DiceLoss, anchor_loss_from_maps

__all__ = [
    "UGAUNet",
    "build_model",
    "LossCEDice",
    "DiceLoss",
    "anchor_loss_from_maps",
]

__version__ = "0.1.0"
