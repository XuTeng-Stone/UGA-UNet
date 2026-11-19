"""Model definitions.

``UGAUNet`` is the proposed architecture.  ``build_model`` provides a single
factory for UGA-UNet and the eight baselines used in the paper.
"""

from .baselines import AnchorFeatureWrapper, build_model
from .ugaunet import UGAUNet, MobileUVitUNet_Ablation

__all__ = ["UGAUNet", "MobileUVitUNet_Ablation", "build_model", "AnchorFeatureWrapper"]
