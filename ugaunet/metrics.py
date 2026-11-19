"""Evaluation metrics and model-complexity helpers.

Dice/F1 and IoU are computed from binarized predictions.  ASSD and HD95 follow
the surface-distance implementation used in the original experiments.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, generate_binary_structure
from scipy.spatial import cKDTree

try:  # optional dependency
    from thop import profile
except ImportError:  # pragma: no cover
    profile = None


def cal_params_flops(
    model: torch.nn.Module,
    input_size: tuple[int, int, int, int] = (1, 3, 256, 256),
    device: str | torch.device = "cuda",
) -> tuple[float, float]:
    """Return ``(params_millions, gflops)`` using thop on a temporary copy."""
    if profile is None:
        return 0.0, 0.0
    try:
        model_copy = copy.deepcopy(model).to(device).eval()
        input_tensor = torch.randn(input_size, device=device)
        flops, params = profile(model_copy, inputs=(input_tensor,), verbose=False)
        del model_copy, input_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return params / 1e6, flops / 1e9
    except Exception as exc:  # pragma: no cover - thop is fragile
        print(f"[warning] FLOPs calculation failed: {exc}")
        return 0.0, 0.0


def get_surface_distance(mask_gt: np.ndarray, mask_pred: np.ndarray) -> tuple[float, float]:
    """Return ``(ASSD, HD95)`` for two binary masks."""
    if mask_gt.sum() == 0 or mask_pred.sum() == 0:
        return np.nan, np.nan

    def border_pixels(mask: np.ndarray) -> np.ndarray:
        struct = generate_binary_structure(2, 1)
        eroded = distance_transform_edt(mask) > 1
        border = np.logical_xor(mask, eroded)
        return np.argwhere(border)

    gt_border = border_pixels(mask_gt)
    pred_border = border_pixels(mask_pred)
    if len(gt_border) == 0 or len(pred_border) == 0:
        return np.nan, np.nan

    tree_gt = cKDTree(gt_border)
    tree_pred = cKDTree(pred_border)
    d_pred_to_gt, _ = tree_gt.query(pred_border)
    d_gt_to_pred, _ = tree_pred.query(gt_border)

    assd = (d_pred_to_gt.sum() + d_gt_to_pred.sum()) / (
        len(d_pred_to_gt) + len(d_gt_to_pred)
    )
    hd95 = np.percentile(np.hstack([d_pred_to_gt, d_gt_to_pred]), 95)
    return float(assd), float(hd95)


def calculate_advanced_metrics(
    logits: torch.Tensor, target: torch.Tensor
) -> tuple[float, float]:
    """Batch-level ASSD and HD95, matching the original training scripts."""
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    targets = target.cpu().numpy()

    assd_list: list[float] = []
    hd95_list: list[float] = []
    for pred, gt in zip(preds, targets):
        if gt.sum() > 0 and pred.sum() > 0:
            assd, hd95 = get_surface_distance(gt, pred)
            if not np.isnan(assd):
                assd_list.append(assd)
                hd95_list.append(hd95)

    return (
        float(np.mean(assd_list)) if assd_list else 0.0,
        float(np.mean(hd95_list)) if hd95_list else 0.0,
    )


def binary_dice_iou(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Dice and IoU for flattened binary arrays."""
    pred = pred.astype(bool).reshape(-1)
    target = target.astype(bool).reshape(-1)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    denom = pred.sum() + target.sum()
    dice = (2.0 * intersection + 1e-6) / (denom + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return float(dice), float(iou)


def count_parameters(model: torch.nn.Module) -> int:
    """Direct parameter count (no thop hooks)."""
    return sum(p.numel() for p in model.parameters())
