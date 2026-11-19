"""Uncertainty-Guided Anchor Loss (paper Sec. 3.3, Eq. 3-7).

Given a dense decoder feature map ``F``, a ground-truth mask ``Y``, and an
offline uncertainty map ``U``, the loss

1. builds a local background ring around the foreground,
2. splits each class region into stable and ambiguous pixels using
   region-specific uncertainty quantiles,
3. forms class prototypes from the L2-normalized stable-pixel features
   (with stop-gradient), and
4. applies a margin-based cosine loss to ambiguous pixels.

The implementation mirrors the original experiment code, including its
per-image loop and OpenCV morphology for the background ring.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def l2_normalize_features(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize along the channel dimension."""
    return x / (torch.norm(x, dim=1, keepdim=True) + eps)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine distance between ``a`` (N, C) and ``b`` (C,)."""
    if b.ndim == 1:
        b = b.unsqueeze(0)
    return 1.0 - (a * b).sum(dim=1)


def make_outer_ring_from_gt(
    gt_01: torch.Tensor,
    rin: int = 3,
    rout: int = 8,
) -> torch.Tensor:
    """Return the local background ring ``dilate(Y, rout) \\ dilate(Y, rin)``.

    Parameters
    ----------
    gt_01:
        ``(H, W)`` binary ground-truth mask on any device.
    rin, rout:
        Inner and outer dilation radii in pixels (paper: 3 and 8).
    """
    gt = gt_01.detach().cpu().numpy().astype(np.uint8)
    kernel_out = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * rout + 1, 2 * rout + 1)
    )
    kernel_in = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * rin + 1, 2 * rin + 1)
    )
    dil_out = cv2.dilate(gt, kernel_out)
    dil_in = cv2.dilate(gt, kernel_in)
    ring = (dil_out > 0).astype(np.uint8) - (dil_in > 0).astype(np.uint8)
    ring = (ring > 0).astype(np.uint8)
    return torch.from_numpy(ring).to(gt_01.device).float()


def anchor_loss_from_maps(
    feat: torch.Tensor,
    gt: torch.Tensor,
    u_fg: torch.Tensor,
    w_fg: torch.Tensor | None = None,
    rin: int = 3,
    rout: int = 8,
    q_low: float = 0.2,
    q_high: float = 0.8,
    margin: float = 0.3,
    lam_bg: float = 0.25,
    min_pts: int = 10,
) -> torch.Tensor:
    """Compute the Uncertainty-Guided Anchor Loss for one batch.

    Parameters
    ----------
    feat:
        ``(B, C, H, W)`` decoder feature map (full resolution in the paper).
    gt:
        ``(B, H, W)`` binary ground-truth masks.
    u_fg:
        ``(B, H, W)`` normalized uncertainty maps in ``[0, 1]``.  Higher means
        more ambiguous.
    w_fg:
        Optional confidence maps.  When omitted, confidence is ``1 - u_fg``.
    rin, rout:
        Background-ring inner/outer dilation radii.
    q_low, q_high:
        Region-wise uncertainty quantiles for stable / ambiguous pixels.
    margin:
        Margin ``m`` in the softplus anchoring term.
    lam_bg:
        Weight of the background-ring term (paper: 0.25).
    min_pts:
        Minimum number of pixels required to form a prototype or to apply a
        loss term.  The original training scripts use 10.

    Returns
    -------
    Scalar loss averaged over the images that contain usable prototypes.
    """
    if feat.ndim != 4:
        raise ValueError("feat must have shape (B, C, H, W)")
    if gt.ndim != 3:
        raise ValueError("gt must have shape (B, H, W)")
    if u_fg.ndim != 3:
        raise ValueError("u_fg must have shape (B, H, W)")

    b, c, fh, fw = feat.shape
    h, w = gt.shape[-2:]
    if (fh != h) or (fw != w):
        feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)

    feat = l2_normalize_features(feat)
    total: torch.Tensor | float = 0.0
    used = 0

    for i in range(b):
        gt_i = gt[i].float()
        u_i = u_fg[i].float()
        # The original experiments pass uncertainty maps and use 1 - U as the
        # confidence weight for stable foreground selection.
        w_i = (1.0 - u_i) if w_fg is None else w_fg[i].float()

        fg = gt_i > 0.5
        if fg.sum() < min_pts:
            continue

        ring = make_outer_ring_from_gt(gt_i, rin=rin, rout=rout) > 0.5
        if ring.sum() < min_pts:
            continue

        w_fg_vals = w_i[fg]
        u_fg_vals = u_i[fg]
        ring_u_vals = u_i[ring]
        if w_fg_vals.numel() < min_pts:
            continue

        # Stable foreground: high confidence (equivalently low uncertainty).
        t_w_high = torch.quantile(w_fg_vals, q_high)
        fg_st = fg & (w_i >= t_w_high)

        # Ambiguous foreground: high uncertainty.
        t_u_high = torch.quantile(u_fg_vals, q_high)
        fg_am = fg & (u_i >= t_u_high)

        # Stable background ring: low uncertainty.
        t_ring_low = torch.quantile(ring_u_vals, q_low)
        bg_st = ring & (u_i <= t_ring_low)

        # Ambiguous background ring: high uncertainty.
        t_ring_high = torch.quantile(ring_u_vals, q_high)
        bg_am = ring & (u_i >= t_ring_high)

        f_i = feat[i].permute(1, 2, 0).reshape(-1, c)
        idx_fg_st = torch.where(fg_st.reshape(-1))[0]
        idx_bg_st = torch.where(bg_st.reshape(-1))[0]
        if idx_fg_st.numel() < min_pts or idx_bg_st.numel() < min_pts:
            continue

        # Class prototypes from stable pixels, with stop-gradient.
        p_fg = f_i[idx_fg_st].mean(dim=0).detach()
        p_bg = f_i[idx_bg_st].mean(dim=0).detach()
        p_fg = l2_normalize_features(p_fg.unsqueeze(0)).squeeze(0)
        p_bg = l2_normalize_features(p_bg.unsqueeze(0)).squeeze(0)

        loss_i = feat.new_tensor(0.0)

        idx_fg_am = torch.where(fg_am.reshape(-1))[0]
        if idx_fg_am.numel() > 0:
            fa = f_i[idx_fg_am]
            d_pos = cosine_distance(fa, p_fg)
            d_neg = cosine_distance(fa, p_bg)
            loss_i = loss_i + F.softplus(d_pos - d_neg + margin).mean()

        idx_bg_am = torch.where(bg_am.reshape(-1))[0]
        if idx_bg_am.numel() > 0 and lam_bg > 0:
            fb = f_i[idx_bg_am]
            d_pos = cosine_distance(fb, p_bg)
            d_neg = cosine_distance(fb, p_fg)
            loss_i = loss_i + lam_bg * F.softplus(d_pos - d_neg + margin).mean()

        total = total + loss_i
        used += 1

    if used == 0:
        # Keep a gradient path so that ``backward`` on the total objective is
        # always well-defined, even when a batch has no usable masks.
        return feat.sum() * 0.0
    return total / used
