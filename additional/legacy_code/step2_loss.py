import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: (B, C, H, W) - Unnormalized logits
        target: (B, H, W) - Ground truth mask (0 or 1)
        """
        # 1. Softmax 归一化
        probs = torch.softmax(logits, dim=1)
        
        # 2. 取前景类概率
        pred_flat = probs[:, 1, ...].contiguous().view(-1)
        
        # 3. 强制转为 Float
        target_flat = target.float().contiguous().view(-1)
        
        # 4. 计算 Dice
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        return loss

class AnchorLoss(nn.Module):
    def __init__(self, lambda_bg=0.5, margin=0.2):
        super(AnchorLoss, self).__init__()
        self.lambda_bg = lambda_bg
        self.margin = margin
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, features, mask, uncertainty_map):
        """
        features: (B, C, H, W)
        mask: (B, H, W)
        uncertainty_map: (B, H, W)
        """
        mask = mask.float()
        uncertainty_map = uncertainty_map.float()
        
        # 对齐尺寸
        if mask.shape[-2:] != features.shape[-2:]:
            mask = F.interpolate(mask.unsqueeze(1), size=features.shape[-2:], mode='nearest').squeeze(1)
            uncertainty_map = F.interpolate(uncertainty_map.unsqueeze(1), size=features.shape[-2:], mode='nearest').squeeze(1)

        b, c, h, w = features.shape
        feats_flat = features.view(b, c, -1)     # (B, C, N)
        mask_flat = mask.view(b, 1, -1)          # (B, 1, N)
        
        # 计算 Anchor
        fg_mask = (mask_flat > 0.5).float()
        bg_mask = (mask_flat <= 0.5).float()

        fg_sum = (feats_flat * fg_mask).sum(dim=2)
        bg_sum = (feats_flat * bg_mask).sum(dim=2)
        
        fg_count = fg_mask.sum(dim=2) + 1e-6
        bg_count = bg_mask.sum(dim=2) + 1e-6

        fg_anchor = fg_sum / fg_count
        bg_anchor = bg_sum / bg_count

        # 计算相似度
        feats_norm = F.normalize(feats_flat, p=2, dim=1)
        fg_anchor_norm = F.normalize(fg_anchor.unsqueeze(2), p=2, dim=1)
        bg_anchor_norm = F.normalize(bg_anchor.unsqueeze(2), p=2, dim=1)
        
        sim_fg = torch.bmm(feats_norm.transpose(1, 2), fg_anchor_norm).squeeze(2)
        
        # 将相似度映射到概率 (0~1)
        prob_fg = (sim_fg + 1) / 2.0
        
        with torch.cuda.amp.autocast(enabled=False):
            loss_map = self.bce(prob_fg.float(), mask_flat.squeeze(1).float())
        # ==========================================
        
        # 引入不确定性加权
        uncertainty_flat = uncertainty_map.view(b, -1)
        pixel_weights = 1.0 - uncertainty_flat
        
        weighted_loss = (loss_map * pixel_weights).mean()
        
        return weighted_loss