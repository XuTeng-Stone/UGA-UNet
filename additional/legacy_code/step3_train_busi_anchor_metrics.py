import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
from scipy.ndimage import distance_transform_edt, binary_erosion

from step2_dataset_busi import BUSIDataset
from step2_loss_anchor import LossCEDice, anchor_loss_from_maps

try:
    from small_unet_lcf import MobileUVitUNet
except ImportError:
    print("[Error] 找不到 small_unet_lcf.py")
    exit()

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# ================= 1. 解析命令行参数 =================
parser = argparse.ArgumentParser(description='BUSI Anchor Loss Tuning')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--lam_bg', type=float, default=0.25)
parser.add_argument('--w_dice', type=float, default=1.0)
parser.add_argument('--w_ce', type=float, default=1.0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4)

args = parser.parse_args()

# ================= 2. 核心配置 =================
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/BUSI"
TRAIN_CSV = os.path.join(DATA_ROOT, "train_split.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test_split.csv")
WEIGHT_DIR = os.path.join(DATA_ROOT, "uncertainty_weights")

W_DICE = args.w_dice
W_CE   = args.w_ce
ALPHA_MAX = args.alpha
MARGIN = args.margin
LAM_BG = args.lam_bg

EXP_ID = f"A{ALPHA_MAX}_M{MARGIN}_Lbg{LAM_BG}_lr{args.lr}"
RESULTS_DIR = os.path.join("/data/xuten/Code/results_busi_metrics", EXP_ID)
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

DEVICE = "cuda:4"

# ================= 3. 指标计算 =================
def compute_hd95(pred, target):
    if np.sum(pred) == 0 or np.sum(target) == 0: return np.nan
    dt_target = distance_transform_edt(1 - target)
    dt_pred = distance_transform_edt(1 - pred)
    dist_pred_to_target = dt_target[pred > 0]
    dist_target_to_pred = dt_pred[target > 0]
    all_dists = np.concatenate([dist_pred_to_target, dist_target_to_pred])
    return np.percentile(all_dists, 95)

def compute_assd(pred, target):
    if np.sum(pred) == 0 or np.sum(target) == 0: return np.nan
    pred_border = pred ^ binary_erosion(pred)
    target_border = target ^ binary_erosion(target)
    if np.sum(pred_border) == 0 or np.sum(target_border) == 0: return np.nan
    dt_target = distance_transform_edt(1 - target_border)
    dt_pred = distance_transform_edt(1 - pred_border)
    dists_pred_to_target = dt_target[pred_border > 0]
    dists_target_to_pred = dt_pred[target_border > 0]
    assd = (np.mean(dists_pred_to_target) + np.mean(dists_target_to_pred)) / 2.0
    return assd

# ================= 4. 辅助函数 =================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ious, f1s, hd95s, assds = [], [], [], []
    
    for imgs, msks, _ in loader: 
        imgs = imgs.to(DEVICE)
        msks = torch.clamp(msks, 0, 1).long().to(DEVICE)
        if msks.dim() == 4 and msks.shape[1] == 1: msks = msks.squeeze(1)
            
        logits, _ = model(imgs) 
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        target = msks.cpu().numpy()
        
        for i in range(pred.shape[0]):
            p, t = pred[i], target[i]
            if np.sum(t) > 0:
                inter = np.logical_and(t, p).sum()
                union = np.logical_or(t, p).sum()
                ious.append(inter / (union + 1e-6))
                f1s.append(f1_score(t.flatten(), p.flatten(), average='binary', zero_division=1))
                if np.sum(p) > 0:
                    hd95 = compute_hd95(p, t)
                    assd = compute_assd(p, t)
                    if not np.isnan(hd95): hd95s.append(hd95)
                    if not np.isnan(assd): assds.append(assd)
                    
    return (np.mean(ious) if ious else 0.0, 
            np.mean(f1s) if f1s else 0.0, 
            np.mean(hd95s) if hd95s else 0.0, 
            np.mean(assds) if assds else 0.0)

def append_summary_report(iou, f1, hd95, assd):
    summary_path = "/data/xuten/Code/results_busi_metrics/all_experiments_summary.csv"
    write_header = not os.path.exists(summary_path)
    data = {
        "Exp_ID": [EXP_ID],
        "Alpha": [ALPHA_MAX],
        "Margin": [MARGIN],
        "Lam_BG": [LAM_BG],
        "mIoU": [iou],
        "mF1": [f1],
        "mHD95": [hd95],
        "mASSD": [assd]
    }
    pd.DataFrame(data).to_csv(summary_path, mode='a', header=write_header, index=False)

# ================= 5. 主流程 =================
def main():
    seed_everything(42)
    print(f"[Start BUSI] Exp: {EXP_ID}")

    train_tf = A.Compose([A.Resize(256, 256), A.HorizontalFlip(p=0.5), A.Rotate(limit=30, p=0.5), 
                          A.Normalize(), ToTensorV2()])
    val_tf = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])

    train_ds = BUSIDataset(TRAIN_CSV, DATA_ROOT, WEIGHT_DIR, transform=train_tf, only_tumor=True)
    val_ds = BUSIDataset(TEST_CSV, DATA_ROOT, WEIGHT_DIR, transform=val_tf, only_tumor=True)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = MobileUVitUNet(in_channels=3, n_classes=2).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler('cuda')
    
    # [关键对接] 采用您的 LossCEDice
    criterion_seg = LossCEDice(w_ce=W_CE, w_dice=W_DICE).to(DEVICE)

    best_iou, best_f1, best_hd95, best_assd = 0.0, 0.0, 0.0, 0.0
    
    # [关键设定] Pipeline的预热轮数为 30 (即前 30% epochs)
    warmup_epochs = int(0.3 * args.epochs)
    
    for epoch in range(args.epochs):
        model.train()
        
        if epoch < warmup_epochs: alpha = 0.0
        else: alpha = ALPHA_MAX * ((epoch - warmup_epochs) / (args.epochs - warmup_epochs))
            
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        for imgs, msks, wgts in loop:
            imgs = imgs.to(DEVICE); msks = torch.clamp(msks, 0, 1).long().to(DEVICE)
            if msks.dim() == 4 and msks.shape[1] == 1: msks = msks.squeeze(1)
            wgts = wgts.float().squeeze(1).to(DEVICE) if wgts.dim() == 4 else wgts.float().to(DEVICE)
            
            optimizer.zero_grad()
            with autocast('cuda'):
                logits, feats = model(imgs) 
                l_seg = criterion_seg(logits, msks)
                
                # [关键对接] 采用您的 anchor_loss_from_maps
                if alpha > 1e-6:
                    l_anchor = anchor_loss_from_maps(feat=feats.float(), gt=msks, u_fg=wgts, w_fg=None, margin=MARGIN, lam_bg=LAM_BG, min_pts=10)
                else:
                    l_anchor = torch.tensor(0.0, device=DEVICE)
                
                loss = l_seg + alpha * l_anchor

            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loop.set_postfix(seg=l_seg.item(), anc=l_anchor.item(), a=alpha)

        # Validation
        if (epoch + 1) % 5 == 0 or epoch > args.epochs - 10:
            val_iou, val_f1, val_hd95, val_assd = evaluate(model, val_loader)
            if val_iou > best_iou:
                best_iou, best_f1, best_hd95, best_assd = val_iou, val_f1, val_hd95, val_assd
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pth"))
                print(f"Ep {epoch+1}: New Best mIoU: {best_iou:.4f} | mF1: {best_f1:.4f}")

    print(f"=== Finished: {EXP_ID} === Best mIoU: {best_iou:.4f}")
    with open(os.path.join(RESULTS_DIR, "report.txt"), "w") as f:
        f.write(f"Alpha: {ALPHA_MAX}\nMargin: {MARGIN}\nLamBG: {LAM_BG}\nmIoU: {best_iou:.4f}\nmF1: {best_f1:.4f}\nmHD95: {best_hd95:.4f}\nmASSD: {best_assd:.4f}\n")
    append_summary_report(best_iou, best_f1, best_hd95, best_assd)

if __name__ == "__main__":
    main()