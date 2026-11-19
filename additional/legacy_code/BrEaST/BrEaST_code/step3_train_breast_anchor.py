import os
import random
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

# [关键修改] 导入 BrEaST Dataset
from step2_dataset_breast import BrEaSTDataset
from step2_loss_anchor import LossCEDice, anchor_loss_from_maps

try:
    from small_unet_lcf import MobileUVitUNet
except ImportError:
    print("[Error] 找不到 small_unet_lcf.py")
    exit()

# ================= 1. 核心配置 =================
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/BrEaST"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
WEIGHT_DIR = os.path.join(DATA_ROOT, "uncertainty_weights")

# [配置保持与你提供的代码一致]
W_DICE = 1.0     
W_CE   = 1.0     
ALPHA_MAX = 0.1  
MARGIN = 0.3
LAM_BG = 0.1     

DIR_NAME = f"results_breast_anchor_FIXED_D{W_DICE}_C{W_CE}_A{ALPHA_MAX}"
RESULTS_DIR = os.path.join("/data/xuten/Code", DIR_NAME)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

DEVICE = "cuda:0"
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4 

# ================= 2. 辅助函数 =================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_alpha(epoch, num_epochs, alpha_max=0.3, warmup_ratio=0.2):
    warm = int(num_epochs * warmup_ratio)
    if epoch < warm: return 0.0
    progress = (epoch - warm) / (num_epochs - warm)
    return alpha_max * min(1.0, progress)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ious, f1s = [], []
    
    total_pred_pixels = 0
    
    for imgs, msks, _ in loader:
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
        
        logits, _ = model(imgs) 
        
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        target = msks.cpu().numpy()
        
        total_pred_pixels += np.sum(pred == 1)
        
        for i in range(pred.shape[0]):
            p, t = pred[i].flatten(), target[i].flatten()
            if np.sum(t) > 0:
                inter = np.logical_and(t, p).sum()
                union = np.logical_or(t, p).sum()
                ious.append(inter / (union + 1e-6))
                f1s.append(f1_score(t, p, average='binary', zero_division=1))
    
    if total_pred_pixels == 0:
        print("  [Warning] Val Prediction is ALL BLACK (No Tumor Detected)!")

    return np.mean(ious) if ious else 0, np.mean(f1s) if f1s else 0

def generate_report(path, iou, f1, best_alpha):
    with open(path, "w") as f:
        f.write(f"=== BrEaST Fixed Training Report ===\n")
        f.write(f"Weights: Dice={W_DICE}, CE={W_CE}, Alpha={ALPHA_MAX}\n")
        f.write(f"Best IoU: {iou:.4f}\nBest F1: {f1:.4f}\nBest Alpha: {best_alpha:.4f}\n")

# ================= 3. 主训练流程 =================
def main():
    seed_everything(42)
    print(f"[Info] Starting BrEaST Fixed Training | Output: {RESULTS_DIR}")

    train_tf = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5), 
        A.Normalize(),
        ToTensorV2()
    ])
    val_tf = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])

    # [关键修改] 使用 BrEaSTDataset，且不传 only_tumor 参数 (BrEaSTDataset 没这个参数)
    train_ds = BrEaSTDataset(TRAIN_CSV, DATA_ROOT, WEIGHT_DIR, transform=train_tf)
    val_ds = BrEaSTDataset(TEST_CSV, DATA_ROOT, WEIGHT_DIR, transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    model = MobileUVitUNet(in_channels=3, n_classes=2).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    
    seg_criterion = LossCEDice(w_ce=W_CE, w_dice=W_DICE).to(DEVICE)

    best_iou = 0.0
    best_f1 = 0.0 
    best_alpha = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        alpha = get_alpha(epoch, NUM_EPOCHS, ALPHA_MAX)
        
        epoch_seg = 0
        epoch_anc = 0
        
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        
        for imgs, msks, u_fg in loop:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            u_fg = u_fg.to(DEVICE)
            
            optimizer.zero_grad()
            with autocast():
                logits, feat = model(imgs) 
                
                l_seg = seg_criterion(logits, msks)
                
                if (alpha > 1e-5):
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
        
        print(f"Ep {epoch+1} | SegLoss: {epoch_seg/len(train_loader):.4f} | AncLoss: {epoch_anc/len(train_loader):.4f} | Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f}")

        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            best_f1 = val_f1
            best_alpha = alpha
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pth"))
            print(f"  >>> Best Model Saved! IoU: {best_iou:.4f}, F1: {best_f1:.4f}")

    print(f"\nTraining Finished. Best IoU: {best_iou:.4f}, F1: {best_f1:.4f}")
    generate_report(os.path.join(RESULTS_DIR, "final_report.txt"), best_iou, best_f1, best_alpha)

if __name__ == "__main__":
    main()