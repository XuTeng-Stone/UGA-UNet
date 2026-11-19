import os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
import pandas as pd

from step2_dataset_busi import BUSIDataset
from step2_loss_anchor import LossCEDice, anchor_loss_from_maps
from small_unet_ablation import MobileUVitUNet_Ablation
from metrics_utils import cal_params_flops, calculate_advanced_metrics

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

DATA_ROOT  = "/data/xuten/Breast Ultrasound Datasets/BUSI"
TRAIN_CSV  = os.path.join(DATA_ROOT, "train_split.csv")
TEST_CSV   = os.path.join(DATA_ROOT, "test_split.csv")
WEIGHT_DIR = os.path.join(DATA_ROOT, "uncertainty_weights")

# 结果保存到与其他模型相同的父目录下
SAVE_DIR_ROOT = "BUSI/BUSI_Anchor_result/results_depth_ablation_busi_proposed"
if not os.path.exists(SAVE_DIR_ROOT):
    os.makedirs(SAVE_DIR_ROOT)

DEVICE        = "cuda:0"
BATCH_SIZE    = 8
NUM_EPOCHS    = 100
LEARNING_RATE = 1e-4

MAX_ALPHA    = 0.1
ANCHOR_MARGIN = 0.3
LAMBDA_BG    = 0.25
WARMUP_RATIO = 0.1


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate_dataset(model, loader, desc="Eval"):
    model.eval()
    ious, f1s, assds, hd95s = [], [], [], []
    for imgs, msks, _ in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(DEVICE)
        msks = torch.clamp(msks, 0, 1).long().to(DEVICE)
        if msks.dim() == 4 and msks.shape[1] == 1:
            msks = msks.squeeze(1)
        outs, _ = model(imgs)
        pred, target = torch.argmax(outs, dim=1).cpu().numpy(), msks.cpu().numpy()
        for i in range(pred.shape[0]):
            p, t = pred[i].flatten(), target[i].flatten()
            if np.sum(t) > 0:
                inter = np.logical_and(t, p).sum()
                union = np.logical_or(t, p).sum()
                ious.append(inter / (union + 1e-6))
                f1s.append(f1_score(t, p, average='binary', zero_division=1))
        assd, hd95 = calculate_advanced_metrics(outs, msks)
        if assd > 0: assds.append(assd)
        if hd95 > 0: hd95s.append(hd95)
    return (
        np.mean(ious)  if ious  else 0.0,
        np.mean(f1s)   if f1s   else 0.0,
        np.mean(assds) if assds else 0.0,
        np.mean(hd95s) if hd95s else 0.0,
    )


def train_one_depth(n_blocks):
    mode_name = 'fusion'
    run_name  = f"{mode_name}_{n_blocks}blocks_BUSI_Proposed"
    print(f"\n{'='*20} Start Training: {run_name} {'='*20}")

    save_dir = os.path.join(SAVE_DIR_ROOT, run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_tf = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2(),
    ])

    # only_tumor=True 与 DeepLabV3 脚本对齐
    train_ds = BUSIDataset(TRAIN_CSV, DATA_ROOT, WEIGHT_DIR,
                           transform=train_tf, only_tumor=True)
    val_ds   = BUSIDataset(TEST_CSV,  DATA_ROOT, WEIGHT_DIR,
                           transform=val_tf,   only_tumor=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = MobileUVitUNet_Ablation(
        mode=mode_name, in_channels=3,
        n_classes=2, num_bottleneck_blocks=n_blocks,
    ).to(DEVICE)

    try:
        params, flops = cal_params_flops(model, input_size=(1, 3, 256, 256), device=DEVICE)
    except:
        params, flops = 0, 0

    optimizer     = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler        = GradScaler('cuda')
    criterion_seg = LossCEDice(w_ce=1.0, w_dice=1.0).to(DEVICE)

    best_iou, best_alpha = 0.0, 0.0
    warmup_epochs = int(WARMUP_RATIO * NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        if epoch < warmup_epochs:
            alpha, stage = 0.0, "Warmup"
        else:
            alpha = MAX_ALPHA * ((epoch - warmup_epochs) / (NUM_EPOCHS - warmup_epochs))
            stage = "Ramp"

        loop = tqdm(train_loader, desc=f"Ep {epoch+1} [{stage}]", leave=False)
        for imgs, msks, wgts in loop:
            imgs = imgs.to(DEVICE)
            msks = torch.clamp(msks, 0, 1).long().to(DEVICE)
            if msks.dim() == 4 and msks.shape[1] == 1:
                msks = msks.squeeze(1)
            wgts = (wgts.float().squeeze(1).to(DEVICE)
                    if wgts.dim() == 4 else wgts.float().to(DEVICE))

            optimizer.zero_grad()
            with autocast('cuda'):
                logits, feats = model(imgs)
                l_seg = criterion_seg(logits, msks)
                if alpha > 1e-6:
                    l_anchor = anchor_loss_from_maps(
                        feat=feats.float(), gt=msks, u_fg=wgts, w_fg=None,
                        margin=ANCHOR_MARGIN, lam_bg=LAMBDA_BG, min_pts=10,
                    )
                else:
                    l_anchor = torch.tensor(0.0, device=DEVICE)
                loss = l_seg + alpha * l_anchor

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 每个 epoch 都评估，与 DeepLabV3 脚本对齐
        val_iou, _, _, _ = evaluate_dataset(model, val_loader)
        print(f"Ep {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} "
              f"| Val mIoU: {val_iou:.4f} | Alpha: {alpha:.4f}")
        if val_iou > best_iou:
            best_iou, best_alpha = val_iou, alpha
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "best_model.pth"))

    # 最终评估
    ckpt_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    iou, f1, assd, hd95 = evaluate_dataset(model, val_loader, desc="Final Test")

    # 保存报告（格式与 DeepLabV3 脚本一致）
    open(os.path.join(save_dir, "final_report.txt"), "w").write(
        f"mIoU: {iou:.4f}\nmF1: {f1:.4f}\nASSD: {assd:.4f}\n"
        f"HD95: {hd95:.4f}\nBest_Alpha: {best_alpha:.4f}"
    )
    return {
        'Mode': mode_name, 'Blocks': n_blocks,
        'Params(M)': params, 'FLOPs(G)': flops,
        'mIoU': iou, 'mF1': f1, 'ASSD': assd, 'HD95': hd95,
        'Best_Alpha': best_alpha,
    }


def main():
    seed_everything(42)
    blocks_to_test = [2]
    results = []
    for n in blocks_to_test:
        res = train_one_depth(n)
        if res is not None:
            results.append(res)
            pd.DataFrame(results).to_csv(
                os.path.join(SAVE_DIR_ROOT, "depth_ablation_busi_proposed_results.csv"),
                index=False,
            )
    if results:
        print(pd.DataFrame(results))


if __name__ == "__main__":
    main()
