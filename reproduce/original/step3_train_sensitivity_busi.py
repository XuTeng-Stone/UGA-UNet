"""
Sensitivity ablation on BUSI for two Anchor-Loss design choices that were
NOT covered by the existing P1/P2/P3 sensitivity table:

    P4: ring radius (rin / rout)
    P5: stable / ambiguous quantile thresholds (q_low / q_high)

Each config below trains a fresh `fusion_2blocks` model from scratch.
Everything else (alpha=0.1, margin=0.3, lam_bg=0.25, warmup=10%, 100 ep,
seed=42, AdamW 1e-4, batch 8) is held fixed.

Usage
-----
Run all configs sequentially:
    python step3_train_sensitivity_busi.py

Run only one config (useful when distributing across GPUs):
    python step3_train_sensitivity_busi.py --config ring_r10
    python step3_train_sensitivity_busi.py --config q_25_75 --device cuda:1

Outputs
-------
    BUSI/BUSI_Sensitivity/
        fusion_2blocks_BUSI_<name>/
            best_model.pth
        sensitivity_summary.csv     # mIoU / mF1 / ASSD / HD95 per config
"""
import os, random, argparse
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import f1_score

from step2_dataset_busi import BUSIDataset
from step2_loss_anchor  import LossCEDice, anchor_loss_from_maps
from small_unet_ablation import MobileUVitUNet_Ablation
from metrics_utils      import calculate_advanced_metrics

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# ===================== Paths (unchanged from main train script) =====================
DATA_ROOT  = "/data/xuten/Breast Ultrasound Datasets/BUSI"
TRAIN_CSV  = os.path.join(DATA_ROOT, "train_split.csv")
TEST_CSV   = os.path.join(DATA_ROOT, "test_split.csv")
WEIGHT_DIR = os.path.join(DATA_ROOT, "uncertainty_weights")

SAVE_DIR_ROOT = "BUSI/BUSI_Sensitivity"
os.makedirs(SAVE_DIR_ROOT, exist_ok=True)

# ===================== Fixed hyperparameters (matches main paper config) ============
DEVICE_DEFAULT = "cuda:0"
BATCH_SIZE     = 8
NUM_EPOCHS     = 100
LEARNING_RATE  = 1e-4

MAX_ALPHA      = 0.1
ANCHOR_MARGIN  = 0.3
LAMBDA_BG      = 0.25
WARMUP_RATIO   = 0.1

N_BLOCKS       = 2          # fusion_2blocks is your reported best
SEED           = 42

# ===================== Sensitivity configurations =====================
# Four cases from the supervisor's spec:
#   P4 (ring radius):       narrow ring (2,6)      vs  wider ring (4,10)
#   P5 (quantile threshold): less selective (0.3/0.7) vs more selective (0.1/0.9)
# Default (rin=3, rout=8, q=0.2/0.8) comes from the main BUSI experiment and
# is NOT re-trained here -- reuse the number from your main results table.
CONFIGS = [
    # ---- P4: ring radius sensitivity ----
    {"name": "narrow_ring",  "rin": 2, "rout": 6,  "q_low": 0.20, "q_high": 0.80},
    {"name": "wider_ring",   "rin": 4, "rout": 10, "q_low": 0.20, "q_high": 0.80},

    # ---- P5: quantile sensitivity ----
    {"name": "less_selective", "rin": 3, "rout": 8, "q_low": 0.30, "q_high": 0.70},
    {"name": "more_selective", "rin": 3, "rout": 8, "q_low": 0.10, "q_high": 0.90},
]

# ====================================================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate_dataset(model, loader, device, desc="Eval"):
    model.eval()
    ious, f1s, assds, hd95s = [], [], [], []
    for imgs, msks, _ in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(device)
        msks = torch.clamp(msks, 0, 1).long().to(device)
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
        if assd > 0:  assds.append(assd)
        if hd95 > 0:  hd95s.append(hd95)
    return (
        float(np.mean(ious))  if ious  else 0.0,
        float(np.mean(f1s))   if f1s   else 0.0,
        float(np.mean(assds)) if assds else 0.0,
        float(np.mean(hd95s)) if hd95s else 0.0,
    )


def train_one_config(cfg, device):
    """Train one sensitivity configuration end-to-end and return its final metrics."""
    run_name = f"fusion_{N_BLOCKS}blocks_BUSI_{cfg['name']}"
    print(f"\n{'='*22} Start: {run_name} {'='*22}")
    print(f"  rin={cfg['rin']}  rout={cfg['rout']}  "
          f"q_low={cfg['q_low']}  q_high={cfg['q_high']}")
    print(f"  (fixed: alpha_max={MAX_ALPHA}  m={ANCHOR_MARGIN}  "
          f"lam_bg={LAMBDA_BG}  warmup={WARMUP_RATIO})")

    save_dir = os.path.join(SAVE_DIR_ROOT, run_name)
    os.makedirs(save_dir, exist_ok=True)

    seed_everything(SEED)

    train_tf = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_ds = BUSIDataset(
        csv_file  = TRAIN_CSV,
        root_dir  = DATA_ROOT,
        weight_dir= WEIGHT_DIR,
        transform = train_tf,
        only_tumor= True,
    )
    test_ds = BUSIDataset(
        csv_file  = TEST_CSV,
        root_dir  = DATA_ROOT,
        weight_dir= WEIGHT_DIR,
        transform = val_tf,
        only_tumor= True,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4)

    model = MobileUVitUNet_Ablation(
        mode='fusion',
        in_channels=3,
        n_classes=2,
        num_bottleneck_blocks=N_BLOCKS,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = LossCEDice(w_ce=1.0, w_dice=1.0)
    scaler    = GradScaler()

    warmup_epochs = int(NUM_EPOCHS * WARMUP_RATIO)
    best_iou   = 0.0
    best_ckpt  = os.path.join(save_dir, "best_model.pth")

    for epoch in range(NUM_EPOCHS):
        model.train()
        if epoch < warmup_epochs:
            alpha = 0.0
        else:
            alpha = MAX_ALPHA * (epoch - warmup_epochs) / max(1, NUM_EPOCHS - warmup_epochs)

        for imgs, msks, wgts in train_loader:
            imgs = imgs.to(device)
            msks = torch.clamp(msks, 0, 1).long().to(device)
            if msks.dim() == 4 and msks.shape[1] == 1:
                msks = msks.squeeze(1)
            wgts = wgts.float().squeeze(1).to(device) if wgts.dim() == 4 else wgts.float().to(device)

            optimizer.zero_grad()
            with autocast("cuda"):
                logits, feats = model(imgs)
                l_seg = criterion(logits, msks)
                if alpha > 1e-6:
                    l_anc = anchor_loss_from_maps(
                        feat=feats.float(), gt=msks, u_fg=wgts, w_fg=None,
                        rin=cfg['rin'],   rout=cfg['rout'],          # <-- swept
                        q_low=cfg['q_low'], q_high=cfg['q_high'],   # <-- swept
                        margin=ANCHOR_MARGIN, lam_bg=LAMBDA_BG, min_pts=10,
                    )
                else:
                    l_anc = torch.tensor(0.0, device=device)
                loss = l_seg + alpha * l_anc

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        iou, f1, assd, hd95 = evaluate_dataset(model, test_loader, device,
                                               desc=f"E{epoch+1}/{NUM_EPOCHS}")
        print(f"  [{cfg['name']}] Ep {epoch+1:3d}/{NUM_EPOCHS} | "
              f"iou={iou:.4f}  f1={f1:.4f}  assd={assd:.2f}  hd95={hd95:.2f}  "
              f"alpha={alpha:.4f}", flush=True)

        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), best_ckpt)

    # Final eval with best ckpt
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    iou, f1, assd, hd95 = evaluate_dataset(model, test_loader, device, desc="Final")
    print(f"  [{cfg['name']}] FINAL: iou={iou:.4f}  f1={f1:.4f}  "
          f"assd={assd:.2f}  hd95={hd95:.2f}  (best_ckpt={best_ckpt})")

    return {
        "name":   cfg['name'],
        "rin":    cfg['rin'],
        "rout":   cfg['rout'],
        "q_low":  cfg['q_low'],
        "q_high": cfg['q_high'],
        "mIoU":   round(iou,  4),
        "mF1":    round(f1,   4),
        "ASSD":   round(assd, 4),
        "HD95":   round(hd95, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        help="Run only one named config (e.g. ring_r10). "
                             "If omitted, runs all configs sequentially.")
    parser.add_argument("--device", default=DEVICE_DEFAULT,
                        help="CUDA device (default: cuda:0)")
    args = parser.parse_args()

    if args.config is None:
        configs_to_run = CONFIGS
    else:
        configs_to_run = [c for c in CONFIGS if c['name'] == args.config]
        if not configs_to_run:
            raise ValueError(f"Unknown config '{args.config}'. "
                             f"Choices: {[c['name'] for c in CONFIGS]}")

    all_results = []
    for cfg in configs_to_run:
        try:
            res = train_one_config(cfg, args.device)
            all_results.append(res)
        except Exception as e:
            print(f"!!! FAILED: {cfg['name']}: {e}")
            import traceback; traceback.print_exc()

    if all_results:
        df = pd.DataFrame(all_results)
        # Append-mode: don't overwrite if you ran configs in separate invocations
        out_csv = os.path.join(SAVE_DIR_ROOT, "sensitivity_summary.csv")
        if os.path.exists(out_csv):
            old = pd.read_csv(out_csv)
            df  = pd.concat([old, df], ignore_index=True).drop_duplicates(
                subset=["name"], keep="last")
        df.to_csv(out_csv, index=False)
        print(f"\n{'='*60}\nSUMMARY  ->  {out_csv}\n")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()