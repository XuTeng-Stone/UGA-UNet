import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import f1_score
from dataset_new import BreastCancerSegmentation
from models.dual_swin_unet_dualblock import DualSwinUNet

# =========================================================
# Utils & losses
# =========================================================
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def calculate_f1_score(logits, target):
    pred = torch.argmax(logits, dim=1).cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    return f1_score(target, pred, average="binary")

@torch.no_grad()
def calculate_iou(logits, target):
    pred = torch.argmax(logits, dim=1)
    pred_tumor = (pred == 1).float()
    target_tumor = (target == 1).float()
    intersection = (pred_tumor * target_tumor).sum(dim=(1, 2))
    union = pred_tumor.sum(dim=(1, 2)) + target_tumor.sum(dim=(1, 2)) - intersection
    iou = intersection / (union + 1e-6)
    iou[union == 0] = 1
    return iou.mean()

def soft_dice_loss_from_logits(logits_2ch, target_01, eps=1e-6):
    # logits_2ch: (B,2,H,W), target_01: (B,H,W) in {0,1}
    probs1 = F.softmax(logits_2ch, dim=1)[:, 1]        # (B,H,W)
    target = target_01.float()
    inter = (probs1 * target).sum(dim=(1, 2))
    denom = probs1.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + eps
    dice = (2 * inter + eps) / denom
    return 1 - dice.mean()

class LossCEDice(nn.Module):
    def __init__(self, w_ce=0.5, w_dice=0.5, ignore_index=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100 if ignore_index is None else ignore_index)
        self.w_ce = w_ce
        self.w_dice = w_dice
    def forward(self, logits, target_01):
        ce = self.ce(logits, target_01)               # (B,2,H,W) vs (B,H,W)
        dice = soft_dice_loss_from_logits(logits, target_01)
        return self.w_ce * ce + self.w_dice * dice

def freeze_batchnorm(module, freeze=True):
    # put BN layers to eval and freeze params (encoder stabilization)
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if freeze:
                for p in m.parameters():
                    p.requires_grad = False

# =========================================================
# Data helpers
# =========================================================
def load_pairs_from_csv(csv_file, root):
    df = pd.read_csv(csv_file)
    pairs = []
    for _, row in df.iterrows():
        img_path = os.path.join(root, "Images", row["Image"])
        mask_path = os.path.join(root, "Labels", row["Mask"])
        pairs.append((img_path, mask_path))
    return pairs

# =========================================================
# Train / Validate
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, max_grad_norm):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0
    for inputs, masks in tqdm(loader, desc="Train", leave=False):
        inputs, masks = inputs.to(device), masks.to(device)  # masks: (B,H,W) long {0,1}

        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.bfloat16 if torch.cuda.is_available() else None):
            outputs = model(inputs)                          # (B,2,H,W)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        # gradient clipping (stabilizes transformers/decoders)
        scaler.unscale_(optimizer)
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_iou = calculate_iou(outputs, masks).item()
        batch_f1 = calculate_f1_score(outputs, masks)

        bs = inputs.size(0)
        running_loss += loss.item() * bs
        running_iou += batch_iou * bs
        running_f1 += batch_f1 * bs

    n = len(loader.dataset)
    return running_loss / n, running_iou / n, running_f1 / n

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0
    for inputs, masks in tqdm(loader, desc="Val", leave=False):
        inputs, masks = inputs.to(device), masks.to(device)
        with autocast(dtype=torch.bfloat16 if torch.cuda.is_available() else None):
            outputs = model(inputs)
            loss = criterion(outputs, masks)
        batch_iou = calculate_iou(outputs, masks).item()
        batch_f1 = calculate_f1_score(outputs, masks)

        bs = inputs.size(0)
        running_loss += loss.item() * bs
        running_iou += batch_iou * bs
        running_f1 += batch_f1 * bs

    n = len(loader.dataset)
    return running_loss / n, running_iou / n, running_f1 / n

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    seed_everything(42)

    dataset_root = "../BUS-BRA/"
    train_csv = os.path.join(dataset_root, "train.csv")
    test_csv  = os.path.join(dataset_root, "test.csv")

    train_pairs = load_pairs_from_csv(train_csv, dataset_root)
    test_pairs  = load_pairs_from_csv(test_csv, dataset_root)

    H = W = 256
    train_dataset = BreastCancerSegmentation(train_pairs, H, W, is_augment=True)
    test_dataset  = BreastCancerSegmentation(test_pairs,  H, W, is_augment=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    print("Training samples #: ", len(train_dataset))
    print("Test samples #: ", len(test_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------- Model ----------------
    base_model = DualSwinUNet(
        in_channels=3,
        n_classes=2,
        base_ch=32,               # try 24, 32, 48
        bottleneck_window_size=4, # with 256x256 input -> bottleneck 32x32, ws=4 divides 32
        bottleneck_heads=4,
    ).to(device)
    base_model = base_model.to(memory_format=torch.channels_last)

    # Optional: wrap DP if multiple GPUs
    model = base_model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(base_model, device_ids=list(range(torch.cuda.device_count())), output_device=0)

    # ---------------- Loss / Optim / Sched ----------------
    # Stronger objective for IoU/F1
    criterion = LossCEDice(w_ce=0.5, w_dice=0.5)

    # AdamW + lighter weight decay than before
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-5)

    # Cosine schedule with warmup
    num_epochs = 100
    steps_per_epoch = math.ceil(len(train_loader))
    warmup_steps = min(1000, steps_per_epoch * 3)   # ~3 epochs or 1000 iters
    total_steps = steps_per_epoch * num_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine decay to 10% of base LR
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # AMP scaler
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Freeze BatchNorm in encoder for first few epochs for stability
    # (applies to BN layers living inside the model)
    freeze_bn_epochs = 5

    # ---------------- Train Loop ----------------
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    global_step = 0
    best_val_iou = -1.0

    for epoch in range(num_epochs):
        # Freeze BN for first N epochs, then unfreeze
        if epoch < freeze_bn_epochs:
            freeze_batchnorm(model, freeze=True)
        else:
            # unfreeze BN params after warm-up phase
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        p.requires_grad = True

        train_loss, train_iou, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, max_grad_norm=1.0
        )

        val_loss, val_iou, val_f1 = validate_one_epoch(
            model, test_loader, criterion, device
        )

        # step the scheduler by steps-per-epoch
        for _ in range(steps_per_epoch):
            scheduler.step()
            global_step += 1

        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train: loss {train_loss:.4f} IoU {train_iou:.4f} F1 {train_f1:.4f} | "
              f"Val:   loss {val_loss:.4f}   IoU {val_iou:.4f}   F1 {val_f1:.4f} | "
              f"LR {optimizer.param_groups[0]['lr']:.2e}")

        # Save best by IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_path = os.path.join(checkpoint_dir, "best_mobileuvit.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ New best IoU: {best_val_iou:.4f} — saved to {best_path}")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            cp = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), cp)
            print(f"  💾 Checkpoint saved at {cp}")

    final_model_path = os.path.join(checkpoint_dir, "final_mobileuvit.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")
