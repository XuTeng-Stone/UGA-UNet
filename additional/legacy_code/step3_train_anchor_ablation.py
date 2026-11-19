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
import pandas as pd

# =========================================================
# Imports
# =========================================================
from step2_dataset_busi import BUSIDataset
from small_unet_ablation import MobileUVitUNet_Ablation 
from metrics_utils import cal_params_flops, calculate_advanced_metrics

# 引入您提供的 Anchor Loss 模块
try:
    from step2_loss_anchor import LossCEDice, anchor_loss_from_maps
except ImportError:
    print("[Error] 找不到 step2_loss_anchor.py，请确保文件在当前目录")
    exit()

# =========================================================
# Configuration
# =========================================================
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/BUSI"
TRAIN_CSV = os.path.join(DATA_ROOT, "train_split.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test_split.csv")

# [关键修改] 指定外部权重文件夹
WEIGHT_DIR = os.path.join(DATA_ROOT, "uncertainty_weights")

SAVE_DIR_ROOT = os.path.join("/data/xuten/Code", "results_anchor_ablation")
if not os.path.exists(SAVE_DIR_ROOT): os.makedirs(SAVE_DIR_ROOT)

DEVICE = "cuda:5"
BATCH_SIZE = 8
NUM_EPOCHS = 100 
LEARNING_RATE = 1e-4

# Anchor Loss Hyperparams
ALPHA_MAX = 0.1   
WARMUP_RATIO = 0.1
MARGIN = 0.3      
LAM_BG = 0.1      

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_alpha(epoch, num_epochs, alpha_max, warmup_ratio):
    warm = int(num_epochs * warmup_ratio)
    if epoch < warm: return 0.0
    progress = (epoch - warm) / (num_epochs - warm)
    return alpha_max * min(1.0, progress)

# =========================================================
# Evaluation Function
# =========================================================
@torch.no_grad()
def evaluate_metrics(model, loader):
    model.eval()
    all_ious, all_f1s = [], []
    all_assd, all_hd95 = [], []
    
    # 验证集可能没有 weight，这里只取前两个
    for batch in tqdm(loader, desc="Testing", leave=False):
        imgs = batch[0].to(DEVICE)
        msks = batch[1].to(DEVICE)
        
        outs, _ = model(imgs) 
        
        pred = torch.argmax(outs, dim=1).cpu().numpy()
        target = msks.cpu().numpy()
        
        for i in range(pred.shape[0]):
            p, t = pred[i].flatten(), target[i].flatten()
            if np.sum(t) > 0: 
                inter = np.logical_and(t, p).sum()
                union = np.logical_or(t, p).sum()
                all_ious.append(inter / (union + 1e-6))
                all_f1s.append(f1_score(t, p, average='binary', zero_division=1))
        
        assd, hd95 = calculate_advanced_metrics(outs, msks)
        if assd > 0: all_assd.append(assd)
        if hd95 > 0: all_hd95.append(hd95)

    return {
        "mIoU": np.mean(all_ious) if all_ious else 0,
        "mF1": np.mean(all_f1s) if all_f1s else 0,
        "ASSD": np.mean(all_assd) if all_assd else 0,
        "HD95": np.mean(all_hd95) if all_hd95 else 0
    }

# =========================================================
# Main Training Function
# =========================================================
def train_anchor_ablation():
    run_name = "fusion_k2_anchor_external"
    print(f"\n{'='*20} Start Training: {run_name} {'='*20}")
    
    save_dir = os.path.join(SAVE_DIR_ROOT, run_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # 1. Dataset
    # 必须传入 weight_dir 才能读取外部权重
    train_tf = A.Compose([A.Resize(256, 256), A.HorizontalFlip(p=0.5), A.Rotate(limit=30, p=0.5), A.Normalize(), ToTensorV2()])
    val_tf = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
    
    print(f"[Info] Loading weights from: {WEIGHT_DIR}")
    train_ds = BUSIDataset(TRAIN_CSV, DATA_ROOT, weight_dir=WEIGHT_DIR, transform=train_tf, only_tumor=True)
    val_ds = BUSIDataset(TEST_CSV, DATA_ROOT, weight_dir=WEIGHT_DIR, transform=val_tf, only_tumor=True)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4) 

    # 2. Model (Fusion, K=2)
    model = MobileUVitUNet_Ablation(
        mode='fusion', 
        in_channels=3, 
        n_classes=2, 
        num_bottleneck_blocks=2 
    ).to(DEVICE)
    
    params, flops = cal_params_flops(model, input_size=(1, 3, 256, 256), device=DEVICE)
    print(f"[{run_name}] Params: {params:.2f}M | FLOPs: {flops:.2f}G")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    criterion_seg = LossCEDice(w_ce=1.0, w_dice=1.0).to(DEVICE)

    best_iou = 0.0

    # 3. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        alpha = get_alpha(epoch, NUM_EPOCHS, ALPHA_MAX, WARMUP_RATIO)
        
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        epoch_loss_seg = 0
        epoch_loss_anc = 0
        
        # [关键修改] 获取外部权重 u_map
        for imgs, msks, u_map in loop:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            u_map = u_map.to(DEVICE) # External Uncertainty Map
            
            optimizer.zero_grad()
            with autocast():
                logits, feats = model(imgs) 
                
                l_seg = criterion_seg(logits, msks)
                
                if alpha > 0:
                    # 使用外部加载的 u_map
                    l_anc = anchor_loss_from_maps(
                        feat=feats, 
                        gt=msks, 
                        u_fg=u_map,   # 传入外部 Map
                        w_fg=None,    # 内部自动处理 (1-u)
                        margin=MARGIN,
                        lam_bg=LAM_BG,
                        min_pts=10
                    )
                else:
                    l_anc = torch.tensor(0.0, device=DEVICE)
                
                loss = l_seg + alpha * l_anc

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss_seg += l_seg.item()
            epoch_loss_anc += l_anc.item()
            loop.set_postfix(seg=l_seg.item(), anc=l_anc.item(), alpha=alpha)

        # Validation
        if (epoch + 1) % 5 == 0 or epoch > NUM_EPOCHS - 10:
            metrics = evaluate_metrics(model, val_loader)
            current_iou = metrics['mIoU']
            print(f"Ep {epoch+1} | Seg: {epoch_loss_seg/len(train_loader):.3f} | Anc: {epoch_loss_anc/len(train_loader):.3f} | Val IoU: {current_iou:.4f}")
            
            if current_iou > best_iou:
                best_iou = current_iou
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    # 4. Final Report
    print(f"Training finished. Best IoU: {best_iou:.4f}. Generating Final Report...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    
    final_metrics = evaluate_metrics(model, val_loader)
    final_metrics['Mode'] = 'Fusion_K2_Anchor_External'
    final_metrics['Alpha'] = ALPHA_MAX
    final_metrics['Params(M)'] = params
    final_metrics['FLOPs(G)'] = flops
    
    df = pd.DataFrame([final_metrics])
    df.to_csv(os.path.join(SAVE_DIR_ROOT, "anchor_ablation_results.csv"), index=False)
    print("\n=== Final Anchor Ablation Report ===")
    print(df)

if __name__ == "__main__":
    seed_everything(42)
    train_anchor_ablation()