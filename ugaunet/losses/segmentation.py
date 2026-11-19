"""Standard logit-level segmentation losses (paper Eq. 8)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss on the foreground class.

    Parameters
    ----------
    smooth:
        Small constant for numerical stability.
    """

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits: (B, 2, H, W)
        target: (B, H, W), values in {0, 1}
        """
        probs = torch.softmax(logits, dim=1)[:, 1]
        pred_flat = probs.reshape(-1)
        target_flat = target.float().reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)


class LossCEDice(nn.Module):
    """Weighted cross-entropy + Dice, ``w_ce * CE + w_dice * Dice``."""

    def __init__(self, w_ce: float = 1.0, w_dice: float = 1.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.w_ce = w_ce
        self.w_dice = w_dice

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.w_ce * self.ce(logits, target) + self.w_dice * self.dice(logits, target)
