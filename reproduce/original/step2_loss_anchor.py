import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

# =========================================================
# 1. Standard Segmentation Losses
# =========================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: (B, C, H, W)
        target: (B, H, W) 0/1
        """
        probs = torch.softmax(logits, dim=1)
        pred_flat = probs[:, 1, ...].contiguous().view(-1)
        target_flat = target.float().contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        return loss

class LossCEDice(nn.Module):
    def __init__(self, w_ce=0.5, w_dice=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.w_ce = w_ce
        self.w_dice = w_dice

    def forward(self, logits, target):
        l_ce = self.ce(logits, target)
        l_dice = self.dice(logits, target)
        return self.w_ce * l_ce + self.w_dice * l_dice

# =========================================================
# 2. Anchor Loss Helpers
# =========================================================
def l2_normalize_feat(x, eps=1e-6):
    return x / (torch.norm(x, dim=1, keepdim=True) + eps)

def cosine_distance(a, b):
    # a: (N,C), b: (C,) or (1,C)
    if b.ndim == 1: b = b.unsqueeze(0)
    sim = (a * b).sum(dim=1)
    return 1.0 - sim

def make_outer_ring_from_gt(gt_01: torch.Tensor, rin=3, rout=8):
    """
    生成肿瘤周围的环形区域
    gt_01: (H,W) float 0/1
    """
    gt = gt_01.detach().cpu().numpy().astype(np.uint8)
    kernel_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rout+1, 2*rout+1))
    kernel_in  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rin+1, 2*rin+1))
    dil_out = cv2.dilate(gt, kernel_out)
    dil_in  = cv2.dilate(gt, kernel_in)
    ring = (dil_out > 0).astype(np.uint8) - (dil_in > 0).astype(np.uint8)
    ring = (ring > 0).astype(np.uint8)
    return torch.from_numpy(ring).to(gt_01.device).float()

# =========================================================
# 3. Anchor Loss Function
# =========================================================
def anchor_loss_from_maps(
    feat, gt, u_fg, w_fg,
    rin=3, rout=8,
    q_low=0.2, q_high=0.8,
    margin=0.2,
    lam_bg=0.5,
    min_pts=20
):
    """
    feat: (B,C,h,w) 特征图
    gt:   (B,H,W) 标签
    u_fg: (B,H,W) 不确定性图 (Uncertainty)
    w_fg: (B,H,W) 置信度图 (Confidence/Weight)
    """
    B, C, fh, fw = feat.shape
    H, W = gt.shape[-2], gt.shape[-1]

    # 对齐尺寸
    if (fh != H) or (fw != W):
        feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

    feat = l2_normalize_feat(feat)
    total = 0.0
    used = 0

    for i in range(B):
        gt_i = gt[i].float()
        u_i = u_fg[i]
        
        # Dataset 只返回 u_fg (uncertainty)，用 1-u 近似 w (confidence)
        if w_fg is None:
            w_i = 1.0 - u_i
        else:
            w_i = w_fg[i]

        # 1. 基础筛选
        fg = (gt_i > 0.5)
        if fg.sum() < min_pts: continue

        ring = (make_outer_ring_from_gt(gt_i, rin=rin, rout=rout) > 0.5)
        if ring.sum() < min_pts: continue

        w_fg_vals = w_i[fg]
        u_fg_vals = u_i[fg]
        ring_u_vals = u_i[ring]
        
        if w_fg_vals.numel() < min_pts: continue

        # 2. 锚点与样本分类
        # Stable FG (Anchor): High Confidence
        t_w_high = torch.quantile(w_fg_vals, q_high)
        fg_st = fg & (w_i >= t_w_high)

        # Ambiguous FG (Target): High Uncertainty
        t_u_high = torch.quantile(u_fg_vals, q_high)
        fg_am = fg & (u_i >= t_u_high)

        # Stable BG (Anchor): Low Uncertainty in Ring
        t_ring_low = torch.quantile(ring_u_vals, q_low)
        bg_st = ring & (u_i <= t_ring_low)

        # Ambiguous BG (Target): High Uncertainty in Ring
        t_ring_high = torch.quantile(ring_u_vals, q_high)
        bg_am = ring & (u_i >= t_ring_high)

        # 3. 提取特征
        f_i = feat[i].permute(1, 2, 0).reshape(-1, C)
        
        idx_fg_st = torch.where(fg_st.reshape(-1))[0]
        idx_bg_st = torch.where(bg_st.reshape(-1))[0]
        
        if idx_fg_st.numel() < min_pts or idx_bg_st.numel() < min_pts:
            continue

        # 计算 Prototypes (Anchors)
        p_fg = f_i[idx_fg_st].mean(dim=0).detach() # Detach! 不优化 Anchor 本身
        p_bg = f_i[idx_bg_st].mean(dim=0).detach()
        p_fg = l2_normalize_feat(p_fg.unsqueeze(0)).squeeze(0)
        p_bg = l2_normalize_feat(p_bg.unsqueeze(0)).squeeze(0)

        loss_i = 0.0
        
        # 4. 计算 Ambiguous FG Loss
        idx_fg_am = torch.where(fg_am.reshape(-1))[0]
        if idx_fg_am.numel() > 0:
            fa = f_i[idx_fg_am]
            d_pos = cosine_distance(fa, p_fg) # 应靠近 p_fg
            d_neg = cosine_distance(fa, p_bg) # 应远离 p_bg
            loss_fg = F.softplus(d_pos - d_neg + margin).mean()
            loss_i += loss_fg

        # 5. 计算 Ambiguous BG Loss
        idx_bg_am = torch.where(bg_am.reshape(-1))[0]
        if idx_bg_am.numel() > 0 and lam_bg > 0:
            fb = f_i[idx_bg_am]
            d_pos = cosine_distance(fb, p_bg) # 应靠近 p_bg
            d_neg = cosine_distance(fb, p_fg) # 应远离 p_fg
            loss_bg = F.softplus(d_pos - d_neg + margin).mean()
            loss_i += lam_bg * loss_bg

        total += loss_i
        used += 1

    if used == 0:
        return torch.tensor(0.0, device=feat.device)
    
    return total / used