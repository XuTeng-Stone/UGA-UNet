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
# 1. 导入公共模块 (保证环境一致)
# =========================================================
from step2_dataset_busi import BUSIDataset
from step2_loss import DiceLoss 
from metrics_utils import cal_params_flops, calculate_advanced_metrics

# 导入 CMUNeXt 模型 (假设 models 文件夹在当前目录下)
try:
    from models.CMUNeXt import cmunext_l
except ImportError:
    print("[Error] 找不到 models.CMUNeXt。请确保 CMUNeXt 代码在当前目录的 models 文件夹中。")
    exit()

# =========================================================
# 2. 配置 (与 LocalGlobal/TransUNet 保持完全一致)
# =========================================================
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/BUSI"
TRAIN_CSV = os.path.join(DATA_ROOT, "train_split.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test_split.csv")

SAVE_DIR_ROOT = os.path.join("/data/xuten/Code", "results_cmunext_baseline")
if not os.path.exists(SAVE_DIR_ROOT): os.makedirs(SAVE_DIR_ROOT)

DEVICE = "cuda:0" 
BATCH_SIZE = 8
NUM_EPOCHS = 100 
LEARNING_RATE = 1e-4 # 对齐 LR

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# =========================================================
# 3. 评估函数 (包含 ASSD/HD95)
# =========================================================
@torch.no_grad()
def evaluate_metrics(model, loader):
    model.eval()
    all_ious, all_f1s = [], []
    all_assd, all_hd95 = [], []
    
    for imgs, msks, _ in tqdm(loader, desc="Testing", leave=False):
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
        
        # CMUNeXt Forward
        outs = model(imgs) # [B, 1, H, W]
        
        # [关键修复] 转为双通道，适配后面的 argmax 和 metrics 计算
        outs = torch.cat([-outs, outs], dim=1) # [B, 2, H, W]
        
        pred = torch.argmax(outs, dim=1).cpu().numpy()
        # ... (后续代码不变)
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
    run_name = "CMUNeXt_L"
    print(f"\n{'='*20} Start Training: {run_name} {'='*20}")
    
    # --- Transforms (完全一致) ---
    train_tf = A.Compose([A.Resize(256, 256), A.HorizontalFlip(p=0.5), A.Rotate(limit=30, p=0.5), A.Normalize(), ToTensorV2()])
    val_tf = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
    
    # --- Dataset (使用 BUSIDataset) ---
    train_ds = BUSIDataset(TRAIN_CSV, DATA_ROOT, transform=train_tf, only_tumor=True)
    val_ds = BUSIDataset(TEST_CSV, DATA_ROOT, transform=val_tf, only_tumor=True)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4) 

    # --- Model Init ---
    # 尝试传入 num_classes=2 以适配 CrossEntropyLoss
    # 如果您的 CMUNeXt 代码不支持参数，请修改源码或告诉我
    try:
        model = cmunext_l(num_classes=2).to(DEVICE)
    except TypeError:
        print("[Warning] cmunext_l 不接受 num_classes 参数，尝试默认初始化...")
        model = cmunext_l().to(DEVICE)
        # 注意：如果默认输出是单通道，下面的 CrossEntropy 会报错，需要改为 BCE
    
    # Metrics (计算 Params 和 FLOPs)
    # 使用 metrics_utils 中的修复版函数，避免 thop 报错
    try:
        params, flops = cal_params_flops(model, input_size=(1, 3, 256, 256), device=DEVICE)
        print(f"[{run_name}] Params: {params:.2f}M | FLOPs: {flops:.2f}G")
    except Exception as e:
        print(f"Metrics calc failed: {e}")
        params, flops = 0, 0

    # --- Optimizer & Loss (完全对齐) ---
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE) # 改为 AdamW
    scaler = GradScaler()
    
    # Loss: CE + Dice (权重 1:1)
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
                # 1. Forward (模型输出单通道)
                outs = model(imgs) # Shape: [B, 1, H, W]
                
                # 2. [关键修复] 将单通道转换为双通道 Logits
                # 假设 outs 是 logit(p)，那么 -outs 就是 logit(1-p)
                # 拼接得到 [B, 2, H, W]，分别代表 [背景 logits, 前景 logits]
                outs_2ch = torch.cat([-outs, outs], dim=1) 
                
                # 3. Loss Calculation (现在输入是双通道了)
                l_dice = criterion_dice(outs_2ch, msks)
                l_ce = criterion_ce(outs_2ch, msks)
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
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR_ROOT, "best_model.pth")))
    
    final_metrics = evaluate_metrics(model, val_loader)
    final_metrics['Model'] = 'CMUNeXt_L'
    final_metrics['Params(M)'] = params
    final_metrics['FLOPs(G)'] = flops
    
    df = pd.DataFrame([final_metrics])
    df.to_csv(os.path.join(SAVE_DIR_ROOT, "cmunext_results.csv"), index=False)
    print("\n=== Final CMUNeXt Baseline Report ===")
    print(df)

if __name__ == "__main__":
    main()