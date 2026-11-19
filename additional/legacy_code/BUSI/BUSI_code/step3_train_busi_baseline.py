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

from step2_dataset_busi import BUSIDataset
from step2_loss import DiceLoss

try:
    from small_unet_lcf import MobileUVitUNet
except ImportError:
    print("[Error] 找不到 small_unet_lcf.py")
    exit()

# ================= 配置 =================
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/BUSI"
TRAIN_CSV = os.path.join(DATA_ROOT, "train_split.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test_split.csv")
WEIGHT_DIR = None # Baseline

WEIGHT_DICE = 2.0   
WEIGHT_CE   = 1.0   

DIR_NAME = f"results_busi_baseline_Dice{WEIGHT_DICE}_CE{WEIGHT_CE}"
RESULTS_DIR = os.path.join("/data/xuten/Code", DIR_NAME)

if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

DEVICE = "cuda:5"
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def calculate_metrics_tumor_only(logits, target):
    pred = torch.argmax(logits, dim=1).cpu().numpy()
    target = target.cpu().numpy()
    ious, f1s = [], []
    for i in range(pred.shape[0]):
        p, t = pred[i].flatten(), target[i].flatten()
        if np.sum(t) > 0: 
            inter = np.logical_and(t, p).sum()
            union = np.logical_or(t, p).sum()
            ious.append(inter / (union + 1e-6))
            f1s.append(f1_score(t, p, average='binary', zero_division=1))
    return ious, f1s

def evaluate_dataset(model, loader, desc="Eval"):
    model.eval()
    all_ious = []
    with torch.no_grad():
        for imgs, msks, _ in tqdm(loader, desc=desc, leave=False):
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            outs, _ = model(imgs) 
            ious, _ = calculate_metrics_tumor_only(outs, msks)
            all_ious.extend(ious)
    return np.mean(all_ious) if all_ious else 0.0

def check_overfitting(model, train_loader, val_loader, log_path):
    print("\n[Analysis] Checking Overfitting...")
    train_iou = evaluate_dataset(model, train_loader, desc="Train Set")
    val_iou = evaluate_dataset(model, val_loader, desc="Val Set")
    gap = train_iou - val_iou
    msg = f"\n=== Overfitting Analysis ===\nTrain IoU: {train_iou:.4f}\nVal IoU: {val_iou:.4f}\nGap: {gap:.4f}\n"
    print(msg)
    with open(log_path, "a") as f: f.write(msg)

def generate_report(model, loader, path):
    model.eval()
    all_ious, all_f1s = [], []
    with torch.no_grad():
        for imgs, msks, _ in tqdm(loader, desc="Report"):
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            outs, _ = model(imgs) 
            ious, f1s = calculate_metrics_tumor_only(outs, msks)
            all_ious.extend(ious)
            all_f1s.extend(f1s)
    
    mean_iou = np.mean(all_ious) if all_ious else 0
    mean_f1 = np.mean(all_f1s) if all_f1s else 0
    content = f"=== BUSI Baseline ===\nIoU: {mean_iou:.4f}\nF1: {mean_f1:.4f}\n"
    with open(path, "w") as f: f.write(content)

def main():
    seed_everything(42)
    print(f"[Info] Baseline Training | Dir: {RESULTS_DIR}")

    train_tf = A.Compose([A.Resize(256, 256), A.HorizontalFlip(p=0.5), A.Rotate(limit=30, p=0.5), A.Normalize(), ToTensorV2()])
    val_tf = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])

    train_ds = BUSIDataset(TRAIN_CSV, DATA_ROOT, weight_dir=None, transform=train_tf, only_tumor=True)
    val_ds = BUSIDataset(TEST_CSV, DATA_ROOT, weight_dir=None, transform=val_tf, only_tumor=True)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = MobileUVitUNet(in_channels=3, n_classes=2).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    
    criterion_dice = DiceLoss().to(DEVICE)
    criterion_bce = nn.CrossEntropyLoss().to(DEVICE)

    best_iou = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        for imgs, msks, _ in loop:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                logits, _ = model(imgs) 
                l_seg = WEIGHT_DICE * criterion_dice(logits, msks) + WEIGHT_CE * criterion_bce(logits, msks)
                loss = l_seg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_iou = evaluate_dataset(model, val_loader)
        print(f"Ep {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pth"))

    print(f"Training Finished. Best IoU: {best_iou:.4f}")
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_model.pth")))
    generate_report(model, val_loader, os.path.join(RESULTS_DIR, "final_report.txt"))
    check_overfitting(model, train_loader, val_loader, os.path.join(RESULTS_DIR, "final_report.txt"))

if __name__ == "__main__":
    main()