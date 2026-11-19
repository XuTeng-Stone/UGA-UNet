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
# 1. 导入公共模块
# =========================================================
# [关键修改] 导入 UCLM 数据集
from step2_dataset_bus_uclm import BUSUCLMDataset
from step2_loss import DiceLoss 
from metrics_utils import cal_params_flops, calculate_advanced_metrics

# 导入 AnatoNet 模型
try:
    from models.AnatoNet import UResNetWithAttention
except ImportError:
    print("[Error] 找不到 models.AnatoNet。请确保 AnatoNet 代码在当前目录的 models 文件夹中。")
    exit()

# =========================================================
# 2. 配置
# =========================================================
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/BUS-UCLM"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")

SAVE_DIR_ROOT = os.path.join("/data/xuten/Code", "results_anato_uclm")
if not os.path.exists(SAVE_DIR_ROOT): os.makedirs(SAVE_DIR_ROOT)

DEVICE = "cuda:1" 
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

# =========================================================
# 3. 评估函数
# =========================================================
@torch.no_grad()
def evaluate_metrics(model, loader):
    model.eval()
    all_ious, all_f1s = [], []
    all_assd, all_hd95 = [], []
    
    for imgs, msks, _ in tqdm(loader, desc="Testing", leave=False):
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
        
        # AnatoNet Forward
        outs = model(imgs) 
        
        pred = torch.argmax(outs, dim=1).cpu().numpy()
        target = msks.cpu().numpy()
        
        # 1. IoU & F1
        for i in range(pred.shape[0]):
            p, t = pred[i].flatten(), target[i].flatten()
            if np.sum(t) > 0: 
                inter = np.logical_and(t, p).sum()
                union = np.logical_or(t, p).sum()
                all_ious.append(inter / (union + 1e-6))
                all_f1s.append(f1_score(t, p, average='binary', zero_division=1))
        
        # 2. ASSD & HD95
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
# 4. 主训练流程
# =========================================================
def main():
    seed_everything(42)
    run_name = "AnatoNet_UCLM"
    print(f"\n{'='*20} Start Training: {run_name} {'='*20}")
    
    # --- Transforms ---
    train_tf = A.Compose([A.Resize(256, 256), A.HorizontalFlip(p=0.5), A.Rotate(limit=30, p=0.5), A.Normalize(), ToTensorV2()])
    val_tf = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
    
    # --- Dataset ---
    print(f"[Info] Loading UCLM Dataset from: {DATA_ROOT}")
    try:
        train_ds = BUSUCLMDataset(TRAIN_CSV, DATA_ROOT, weight_dir=None, transform=train_tf, only_tumor=True)
        val_ds = BUSUCLMDataset(TEST_CSV, DATA_ROOT, weight_dir=None, transform=val_tf, only_tumor=True)
    except Exception as e:
        print(f"[Error] 数据集加载失败: {e}")
        return
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4) 

    # --- Model Init ---
    # AnatoNet (保留原始参数设置)
    model = UResNetWithAttention(n_classes=2, beta=3.5, gamma=1.0).to(DEVICE)
    
    # Metrics
    try:
        params, flops = cal_params_flops(model, input_size=(1, 3, 256, 256), device=DEVICE)
        print(f"[{run_name}] Params: {params:.2f}M | FLOPs: {flops:.2f}G")
    except Exception as e:
        print(f"Metrics calc failed: {e}")
        params, flops = 0, 0

    # --- Optimizer & Loss ---
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    
    criterion_dice = DiceLoss().to(DEVICE)
    criterion_ce = nn.CrossEntropyLoss().to(DEVICE)

    best_iou = 0.0

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        epoch_loss = 0
        
        for imgs, msks, _ in loop:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            
            optimizer.zero_grad()
            with autocast():
                # Forward
                outs = model(imgs)
                
                # Loss Calculation
                l_dice = criterion_dice(outs, msks)
                l_ce = criterion_ce(outs, msks)
                loss = l_dice + l_ce

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation
        if (epoch + 1) % 5 == 0 or epoch > NUM_EPOCHS - 10:
            metrics = evaluate_metrics(model, val_loader)
            current_iou = metrics['mIoU']
            print(f"Ep {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | Val IoU: {current_iou:.4f}")
            
            if current_iou > best_iou:
                best_iou = current_iou
                torch.save(model.state_dict(), os.path.join(SAVE_DIR_ROOT, "best_model.pth"))

    # --- Final Test ---
    print(f"Training finished. Best IoU: {best_iou:.4f}. Generating Final Report...")
    if os.path.exists(os.path.join(SAVE_DIR_ROOT, "best_model.pth")):
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR_ROOT, "best_model.pth")))
    
    final_metrics = evaluate_metrics(model, val_loader)
    final_metrics['Model'] = 'AnatoNet_UCLM'
    final_metrics['Params(M)'] = params
    final_metrics['FLOPs(G)'] = flops
    
    df = pd.DataFrame([final_metrics])
    df.to_csv(os.path.join(SAVE_DIR_ROOT, "anato_uclm_results.csv"), index=False)
    print("\n=== Final AnatoNet (UCLM) Report ===")
    print(df)

if __name__ == "__main__":
    main()