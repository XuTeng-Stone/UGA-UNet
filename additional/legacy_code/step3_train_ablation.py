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

# 导入你的模块
from step2_dataset_busi import BUSIDataset
from step2_loss import DiceLoss
from small_unet_ablation import MobileUVitUNet_Ablation # 引用修改后的模型
from metrics_utils import cal_params_flops, calculate_advanced_metrics # 引用指标工具

# ================= 配置 =================
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/BUSI"
TRAIN_CSV = os.path.join(DATA_ROOT, "train_split.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test_split.csv")
SAVE_DIR_ROOT = os.path.join("/data/xuten/Code", "results_ablation_study")

if not os.path.exists(SAVE_DIR_ROOT): os.makedirs(SAVE_DIR_ROOT)

DEVICE = "cuda:5"
BATCH_SIZE = 8
NUM_EPOCHS = 100 # 可以根据消融实验需求适当减少，比如 50-80
LEARNING_RATE = 1e-4

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def evaluate_metrics(model, loader):
    model.eval()
    all_ious, all_f1s = [], []
    all_assd, all_hd95 = [], []
    
    for imgs, msks, _ in tqdm(loader, desc="Testing", leave=False):
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
        outs, _ = model(imgs) 
        
        # 1. IoU & F1 (Tumor Only)
        pred = torch.argmax(outs, dim=1).cpu().numpy()
        target = msks.cpu().numpy()
        
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

def train_one_mode(mode_name):
    print(f"\n{'='*20} Start Training Mode: {mode_name} {'='*20}")
    save_dir = os.path.join(SAVE_DIR_ROOT, mode_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # 1. 初始化数据
    train_tf = A.Compose([A.Resize(256, 256), A.HorizontalFlip(p=0.5), A.Rotate(limit=30, p=0.5), A.Normalize(), ToTensorV2()])
    val_tf = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
    
    train_ds = BUSIDataset(TRAIN_CSV, DATA_ROOT, transform=train_tf, only_tumor=True)
    val_ds = BUSIDataset(TEST_CSV, DATA_ROOT, transform=val_tf, only_tumor=True)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4) # BS=1 for metrics

    # 2. 初始化模型 (Num_bottleneck=1)
    model = MobileUVitUNet_Ablation(
        mode=mode_name, 
        in_channels=3, 
        n_classes=2, 
        num_bottleneck_blocks=1 # 按照你的要求设置为 1
    ).to(DEVICE)
    
    # 计算参数量和FLOPs
    params, flops = cal_params_flops(model, input_size=(1, 3, 256, 256), device=DEVICE)
    print(f"[{mode_name}] Params: {params:.2f}M | FLOPs: {flops:.2f}G")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion_dice = DiceLoss().to(DEVICE)
    criterion_bce = nn.CrossEntropyLoss().to(DEVICE)
    scaler = GradScaler()

    best_iou = 0.0

    # 3. 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        for imgs, msks, _ in loop:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                logits, _ = model(imgs) 
                loss = criterion_dice(logits, msks) + criterion_bce(logits, msks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=loss.item())

        # 简单的验证以保存模型
        if (epoch + 1) % 5 == 0 or epoch > NUM_EPOCHS - 10:
            metrics = evaluate_metrics(model, val_loader)
            current_iou = metrics['mIoU']
            print(f"Ep {epoch+1} | IoU: {current_iou:.4f}")
            
            if current_iou > best_iou:
                best_iou = current_iou
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    # 4. 最终测试 (加载最佳模型)
    print(f"Training {mode_name} finished. Best IoU: {best_iou:.4f}. Running Final Test...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    
    final_metrics = evaluate_metrics(model, val_loader)
    final_metrics['Mode'] = mode_name
    final_metrics['Params(M)'] = params
    final_metrics['FLOPs(G)'] = flops
    
    return final_metrics

def main():
    seed_everything(42)
    
    # [关键修改] 加入 'conv' 进行对比
    # conv: 纯 3x3 卷积
    # local: 只有局部卷积分支
    # global: 只有 Transformer 分支
    # fusion: Local + Global
    modes_to_test = ['conv', 'local', 'global', 'fusion'] 
    
    results = []
    
    for mode in modes_to_test:
        res = train_one_mode(mode)
        results.append(res)
        
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(SAVE_DIR_ROOT, "ablation_results.csv"), index=False)
        print(f"Results for {mode} saved.")

    print("\n=== Final Ablation Report ===")
    print(df)

if __name__ == "__main__":
    main()