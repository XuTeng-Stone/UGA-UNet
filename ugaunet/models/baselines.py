"""Baseline segmentation models used in the paper.

The model definitions are the same source files as the original experiments
(see ``ugaunet/models/third_party``).  This module only provides a uniform
factory and the feature adapter used when the anchor loss is applied to a
baseline.

Available names
---------------
``unet``, ``unetplusplus``, ``deeplabv3``, ``anato``, ``medt``,
``transunet``, ``cmunext``, ``ukan``, ``ugaunet``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_baseline(name: str):
    """Import a baseline lazily so optional dependencies stay optional."""
    if name == "unet":
        from .third_party.unet import U_Net

        return U_Net
    if name == "unetplusplus":
        from .third_party.unetplus import ResNet34UnetPlus

        return ResNet34UnetPlus
    if name == "anato":
        from .third_party.AnatoNet import UResNetWithAttention

        return UResNetWithAttention
    if name == "medt":
        from .third_party.axialnet import MedT

        return MedT
    if name == "cmunext":
        from .third_party.CMUNeXt import cmunext_l

        return cmunext_l
    if name == "ukan":
        from .third_party.ukan import UKAN

        return UKAN
    if name == "transunet":
        from .third_party.transunet.vit_seg_modeling import (
            CONFIGS as CONFIGS_ViT_seg,
            VisionTransformer as ViT_seg,
        )

        return ViT_seg, CONFIGS_ViT_seg
    raise ValueError(f"Unknown baseline model: {name!r}")


def build_model(
    name: str,
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained: bool = True,
    image_size: int = 256,
    **kwargs: Any,
) -> nn.Module:
    """Build a baseline or the proposed UGA-UNet.

    Parameters
    ----------
    name:
        Model name (see module docstring).
    num_classes:
        Number of output classes.  The paper uses two (background + foreground).
    in_channels:
        Number of input image channels.
    pretrained:
        Load ImageNet/ResNet initialization where the baseline supports it.
        This only affects the initial weights, not the architecture.
    image_size:
        Spatial input size; TransUNet and MedT need it at construction time.
    **kwargs:
        Forwarded to :class:`~ugaunet.models.ugaunet.UGAUNet` when
        ``name == "ugaunet"``.
    """
    name = name.lower()

    if name == "ugaunet":
        from .ugaunet import UGAUNet

        return UGAUNet(in_channels=in_channels, n_classes=num_classes, **kwargs)

    if name == "deeplabv3":
        import torchvision.models as tv_models

        weights = "DEFAULT" if pretrained else None
        weights_backbone = "DEFAULT" if pretrained else None
        model = tv_models.segmentation.deeplabv3_resnet50(
            weights=weights, weights_backbone=weights_backbone
        )
        model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        model.aux_classifier = None
        return model

    if name == "transunet":
        import copy

        ViT_seg, configs = _import_baseline(name)
        config = copy.deepcopy(configs["R50-ViT-B_16"])
        config.n_classes = num_classes
        config.n_skip = 3
        if image_size % 16 == 0:
            config.patches.grid = (image_size // 16, image_size // 16)
        return ViT_seg(config, img_size=image_size, num_classes=num_classes)

    builder = _import_baseline(name)

    if name == "unet":
        return builder(img_ch=in_channels, output_ch=num_classes)
    if name == "unetplusplus":
        return builder(num_channels=in_channels, num_class=num_classes)
    if name == "anato":
        return builder(n_classes=num_classes, beta=3.5, gamma=1.0, pretrained=pretrained)
    if name == "medt":
        return builder(img_size=image_size, imgchan=in_channels, num_classes=num_classes)
    if name == "cmunext":
        return builder(n_class=num_classes)
    if name == "ukan":
        return builder(in_channels=in_channels, num_classes=num_classes)

    raise ValueError(f"Unknown baseline model: {name!r}")


class AnchorFeatureWrapper(nn.Module):
    """Expose a full-resolution decoder feature map from a baseline model.

    The original experiments registered a forward pre-hook on the last
    ``nn.Conv2d`` of the baseline and treated that convolution's input as the
    feature map.  This wrapper keeps exactly that behaviour while making the
    output contract explicit:

        wrapper(x) -> (logits, features)

    For single-channel baselines (e.g. some CMUNeXt configurations), the logits
    are converted to two channels as ``[-logits, logits]``.
    """

    def __init__(self, base_model: nn.Module, num_classes: int = 2) -> None:
        super().__init__()
        self.model = base_model
        self.num_classes = num_classes
        self.extracted_features: torch.Tensor | None = None

        target_conv: nn.Conv2d | None = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                target_conv = module
        if target_conv is not None:
            target_conv.register_forward_pre_hook(self._capture_input)

    def _capture_input(self, _module, inputs) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            self.extracted_features = inputs[0]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.extracted_features = None
        logits = self.model(x)

        if isinstance(logits, dict):
            logits = logits.get("out", logits.get("logits"))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Unexpected model output type: {type(logits)!r}")

        features = self.extracted_features
        if features is None or features.shape[1] < 2:
            features = logits

        if logits.shape[1] == 1 and self.num_classes == 2:
            logits = torch.cat([-logits, logits], dim=1)

        if features.shape[-2:] != logits.shape[-2:]:
            features = F.interpolate(
                features, size=logits.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits, features
