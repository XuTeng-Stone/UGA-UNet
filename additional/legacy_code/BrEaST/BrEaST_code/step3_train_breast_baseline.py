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

# 导入 Dataset 和 Loss
from step2_dataset_breast import BrEaSTDataset
from step2_loss_anchor import LossCEDice # 复用这个简单的 Loss

try:
    from small_unet_lcf import MobileUVitUNet
except ImportError:
    print("[Error] 找不到 small_unet_lcf.py")
    exit()

# ================= 配置 =================
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/BrEaST"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
WEIGHT_DIR = None # Baseline 不需要权重图

W_DICE = 1.0     
W_CE   = 1.0     

DIR_NAME = f"results_breast_baseline_D{W_DICE}_C{W_CE}"
RESULTS_DIR = os.path.join("/data/xuten/Code", DIR_NAME)
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

DEVICE = "cuda:3"
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

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ious, f1s = [], []
    for imgs, msks, _ in loader:
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
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

def generate_report(path, iou, f1):
    with open(path, "w") as f:
        f.write(f"=== BrEaST Baseline Report ===\n")
        f.write(f"Weights: Dice={W_DICE}, CE={W_CE}\n")
        f.write(f"Best IoU: {iou:.4f}\nBest F1: {f1:.4f}\n")

# ================= 主流程 =================
def main():
    seed_everything(42)
    print(f"[Info] Starting BrEaST Baseline Training | Output: {RESULTS_DIR}")

    train_tf = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])
    val_tf = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])

    train_ds = BrEaSTDataset(TRAIN_CSV, DATA_ROOT, WEIGHT_DIR, transform=train_tf)
    val_ds = BrEaSTDataset(TEST_CSV, DATA_ROOT, WEIGHT_DIR, transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = MobileUVitUNet(in_channels=3, n_classes=2).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    
    criterion = LossCEDice(w_ce=W_CE, w_dice=W_DICE).to(DEVICE)

    best_iou = 0.0
    best_f1 = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        epoch_loss = 0
        
        for imgs, msks, _ in loop:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                logits, _ = model(imgs) 
                loss = criterion(logits, msks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation
        val_iou, val_f1 = evaluate(model, val_loader)
        print(f"Ep {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pth"))
            print(f"  >>> Best Model Saved! IoU: {best_iou:.4f}, F1: {best_f1:.4f}")

    print(f"\nTraining Finished. Best IoU: {best_iou:.4f}, F1: {best_f1:.4f}")
    generate_report(os.path.join(RESULTS_DIR, "final_report.txt"), best_iou, best_f1)

if __name__ == "__main__":
    main()