import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score

# 导入 Dataset 和 Loss
from step2_dataset_busi import BUSIDataset
from step2_loss_anchor import LossCEDice, anchor_loss_from_maps

try:
    from small_unet_lcf import MobileUVitUNet
except ImportError:
    print("[Error] 找不到 small_unet_lcf.py")
    exit()

# ================= 配置 =================
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/BUSI"
TRAIN_CSV = os.path.join(DATA_ROOT, "train_split.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test_split.csv")
WEIGHT_DIR = os.path.join(DATA_ROOT, "uncertainty_weights")

# Loss Weights
W_DICE = 1.0
W_CE   = 1.0
ALPHA_MAX = 0.1  # Anchor Loss 最大权重
MARGIN = 0.3     # Anchor Loss Margin
LAM_BG = 0.1     # 背景 Anchor 权重

DIR_NAME = f"results_busi_anchor_W{W_DICE}-{W_CE}_Alpha{ALPHA_MAX}"
RESULTS_DIR = os.path.join("/data/xuten/Code", DIR_NAME)
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

DEVICE = "cuda:1"
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4

# ================= 辅助函数 =================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_alpha(epoch, num_epochs, alpha_max=0.3, warmup_ratio=0.1):
    warm = int(num_epochs * warmup_ratio)
    if epoch < warm: return 0.0
    progress = (epoch - warm) / (num_epochs - warm)
    return alpha_max * min(1.0, progress)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ious, f1s = [], []
    for imgs, msks, _ in loader:
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
        
        # [修改] 模型现在返回 (logits, feat)，验证时只需要 logits
        logits, _ = model(imgs) 
        
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        target = msks.cpu().numpy()
        
        for i in range(pred.shape[0]):
            p, t = pred[i].flatten(), target[i].flatten()
            if np.sum(t) > 0:
                inter = np.logical_and(t, p).sum()
                union = np.logical_or(t, p).sum()
                ious.append(inter / (union + 1e-6))
                f1s.append(f1_score(t, p, average='binary', zero_division=1))
    
    return np.mean(ious) if ious else 0, np.mean(f1s) if f1s else 0

def generate_report(path, iou, f1, best_alpha):
    with open(path, "w") as f:
        f.write(f"=== BUSI Anchor Loss Report ===\n")
        f.write(f"Best IoU: {iou:.4f}\nBest F1: {f1:.4f}\nBest Alpha: {best_alpha:.4f}\n")

# ================= 主流程 =================
def main():
    seed_everything(42)
    print(f"[Info] Anchor Loss Training | Dir: {RESULTS_DIR}")

    # Transforms
    train_tf = A.Compose([A.Resize(256, 256), A.HorizontalFlip(p=0.5), A.Rotate(limit=30), A.Normalize(), ToTensorV2()])
    val_tf = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])

    # Dataset
    train_ds = BUSIDataset(TRAIN_CSV, DATA_ROOT, WEIGHT_DIR, transform=train_tf, only_tumor=True)
    val_ds = BUSIDataset(TEST_CSV, DATA_ROOT, WEIGHT_DIR, transform=val_tf, only_tumor=True)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = MobileUVitUNet(in_channels=3, n_classes=2).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    
    seg_criterion = LossCEDice(w_ce=W_CE, w_dice=W_DICE).to(DEVICE)

    best_iou = 0.0
    best_alpha = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        alpha = get_alpha(epoch, NUM_EPOCHS, ALPHA_MAX)
        
        epoch_seg = 0
        epoch_anc = 0
        
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        
        # Train Loop
        for imgs, msks, u_fg in loop:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            u_fg = u_fg.to(DEVICE)
            
            optimizer.zero_grad()
            with autocast():
                # [关键修改] 直接获取特征 feat
                logits, feat = model(imgs) 
                
                # 1. 基础分割 Loss
                l_seg = seg_criterion(logits, msks)
                
                # 2. Anchor Loss (直接使用 feat)
                if (feat is not None) and (alpha > 0):
                    l_anc = anchor_loss_from_maps(
                        feat=feat, gt=msks, u_fg=u_fg, w_fg=None,
                        margin=MARGIN, lam_bg=LAM_BG
                    )
                else:
                    l_anc = torch.tensor(0.0, device=DEVICE)
                
                loss = l_seg + alpha * l_anc

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_seg += l_seg.item()
            epoch_anc += l_anc.item()
            
            loop.set_postfix(seg=l_seg.item(), anc=l_anc.item(), alpha=alpha)

        # Validation
        val_iou, val_f1 = evaluate(model, val_loader)
        print(f"Ep {epoch+1} | Seg: {epoch_seg/len(train_loader):.4f} | Anc: {epoch_anc/len(train_loader):.4f} | Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            best_alpha = alpha
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pth"))

    # Report
    print(f"Training Finished. Best IoU: {best_iou:.4f}")
    generate_report(os.path.join(RESULTS_DIR, "final_report.txt"), best_iou, 0, best_alpha)

if __name__ == "__main__":
    main()