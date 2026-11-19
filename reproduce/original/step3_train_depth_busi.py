import os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.amp import autocast, GradScaler; from torch.utils.data import DataLoader; from tqdm import tqdm
import albumentations as A; from albumentations.pytorch import ToTensorV2; from sklearn.metrics import f1_score; import pandas as pd

from step2_dataset_busi import BUSIDataset
from step2_loss_anchor import LossCEDice  # 仅需基础分割 Loss
from small_unet_ablation import MobileUVitUNet_Ablation; from metrics_utils import cal_params_flops, calculate_advanced_metrics

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/BUSI"
TRAIN_CSV = os.path.join(DATA_ROOT, "train_split.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test_split.csv")
# 基线模型不需要 weight_dir，可以传 None
WEIGHT_DIR = None

# 保存路径区分开，这是纯基线目录
SAVE_DIR_ROOT = "results_depth_ablation_busi"
if not os.path.exists(SAVE_DIR_ROOT): os.makedirs(SAVE_DIR_ROOT)

DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE = "cuda:0", 8, 100, 1e-4

def seed_everything(seed=42): random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.backends.cudnn.deterministic = True

@torch.no_grad()
def evaluate_dataset(model, loader, desc="Eval"):
    model.eval(); ious, f1s, assds, hd95s = [], [], [], []
    for imgs, msks, _ in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(DEVICE); msks = torch.clamp(msks, 0, 1).long().to(DEVICE)
        if msks.dim() == 4 and msks.shape[1] == 1: msks = msks.squeeze(1)
        outs, _ = model(imgs); pred, target = torch.argmax(outs, dim=1).cpu().numpy(), msks.cpu().numpy()
        for i in range(pred.shape[0]):
            p, t = pred[i].flatten(), target[i].flatten()
            if np.sum(t) > 0: 
                inter, union = np.logical_and(t, p).sum(), np.logical_or(t, p).sum()
                ious.append(inter / (union + 1e-6)); f1s.append(f1_score(t, p, average='binary', zero_division=1))
        assd, hd95 = calculate_advanced_metrics(outs, msks)
        if assd > 0: assds.append(assd)
        if hd95 > 0: hd95s.append(hd95)
    return (np.mean(ious) if ious else 0.0, np.mean(f1s) if f1s else 0.0, np.mean(assds) if assds else 0.0, np.mean(hd95s) if hd95s else 0.0)

def train_one_depth(n_blocks):
    mode_name = 'fusion'; run_name = f"{mode_name}_{n_blocks}blocks_BUSI_Baseline"
    print(f"\n{'='*20} Start Training: {run_name} {'='*20}")
    save_dir = os.path.join(SAVE_DIR_ROOT, run_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    train_tf = A.Compose([A.Resize(256, 256), A.HorizontalFlip(p=0.5), A.Rotate(limit=30, p=0.5), A.Normalize(), ToTensorV2()])
    val_tf = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
    train_ds = BUSIDataset(TRAIN_CSV, DATA_ROOT, WEIGHT_DIR, transform=train_tf)
    val_ds = BUSIDataset(TEST_CSV, DATA_ROOT, WEIGHT_DIR, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True) 

    model = MobileUVitUNet_Ablation(mode=mode_name, in_channels=3, n_classes=2, num_bottleneck_blocks=n_blocks).to(DEVICE)
    try: params, flops = cal_params_flops(model, input_size=(1, 3, 256, 256), device=DEVICE)
    except: params, flops = 0, 0

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE); scaler = GradScaler('cuda')
    criterion_seg = LossCEDice(w_ce=1.0, w_dice=1.0).to(DEVICE)
    best_iou = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train(); epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1} [Baseline]", leave=False)
        for imgs, msks, _ in loop:  # 忽略 weight map
            imgs = imgs.to(DEVICE); msks = torch.clamp(msks, 0, 1).long().to(DEVICE)
            if msks.dim() == 4 and msks.shape[1] == 1: msks = msks.squeeze(1)
            optimizer.zero_grad()
            with autocast('cuda'):
                logits, _ = model(imgs)
                loss = criterion_seg(logits, msks)  # 只有纯基础 Loss
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            epoch_loss += loss.item(); loop.set_postfix(loss=loss.item())

        if (epoch + 1) % 5 == 0 or epoch > NUM_EPOCHS - 10:
            val_iou, _, _, _ = evaluate_dataset(model, val_loader)
            print(f"Ep {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | Val mIoU: {val_iou:.4f}")
            if val_iou > best_iou: best_iou = val_iou; torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    if os.path.exists(os.path.join(save_dir, "best_model.pth")): model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth"), weights_only=True))
    iou, f1, assd, hd95 = evaluate_dataset(model, val_loader, desc="Final Test")
    return {'Mode': mode_name, 'Blocks': n_blocks, 'Params(M)': params, 'FLOPs(G)': flops, 'mIoU': iou, 'mF1': f1, 'ASSD': assd, 'HD95': hd95, 'Best_Alpha': 0.0}

def main():
    seed_everything(42); blocks_to_test = [2]; results = []
    for n in blocks_to_test:
        res = train_one_depth(n)
        if res is not None:
            results.append(res); pd.DataFrame(results).to_csv(os.path.join(SAVE_DIR_ROOT, "depth_ablation_busi_results.csv"), index=False)
    if results: print(pd.DataFrame(results))

if __name__ == "__main__": main()