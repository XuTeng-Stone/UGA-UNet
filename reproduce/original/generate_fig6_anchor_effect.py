"""
generate_fig6_anchor_effect.py
──────────────────────────────
1. 训练 Without Anchor 模型（纯 CE+Dice）
2. 加载 With Anchor 模型（已有权重）
3. 遍历 BUSI 测试集，选 Anchor Loss 改善最明显的 2-3 个案例
4. 生成 Fig 6：Image | SAM Uncertainty | Without Anchor | With Anchor | GT
"""

import os, random, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image as PILImage

from small_unet_ablation import MobileUVitUNet_Ablation
from step2_dataset_busi   import BUSIDataset
from step2_loss_anchor    import LossCEDice, anchor_loss_from_maps

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════
DEVICE     = "cuda:2"
SEED       = 42
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_BUSI_ROOT   = "/data/xuten/Breast Ultrasound Datasets/BUSI"
_TRAIN_CSV   = os.path.join(_BUSI_ROOT, "train_split.csv")
_TEST_CSV    = os.path.join(_BUSI_ROOT, "test_split.csv")
_WEIGHT_DIR  = os.path.join(_BUSI_ROOT, "uncertainty_weights")

CKPT_NO_ANCHOR   = "results_depth_ablation_busi/fusion_2blocks_BUSI_Baseline/best_model.pth"
CKPT_WITH_ANCHOR = "BUSI/BUSI_Anchor_results/results_depth_ablation_busi_proposed/fusion_2blocks_BUSI_Proposed/best_model.pth"

NUM_EPOCHS    = 100
BATCH_SIZE    = 8
LR            = 1e-4
MAX_ALPHA     = 0.1
ANCHOR_MARGIN = 0.3
LAMBDA_BG     = 0.25
WARMUP_RATIO  = 0.1
N_CASES       = 3

COLOR_GT       = (0.18, 0.80, 0.44)
COLOR_NO_ANC   = (0.20, 0.50, 0.85)
COLOR_WITH_ANC = (0.85, 0.15, 0.25)
COL_NAMES = ["Image", "GT", "Uncertainty Map", "Without Anchor", "With Anchor"]

# ══════════════════════════════════════════════════════════════════════════
#  Utils
# ══════════════════════════════════════════════════════════════════════════
def seed_everything(seed=42):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.backends.cudnn.deterministic = True


def make_loaders():
    train_tf = A.Compose([A.Resize(256,256), A.HorizontalFlip(p=0.5),
                          A.Rotate(limit=30, p=0.5), A.Normalize(), ToTensorV2()])
    val_tf   = A.Compose([A.Resize(256,256), A.Normalize(), ToTensorV2()])
    train_ds = BUSIDataset(_TRAIN_CSV, _BUSI_ROOT, _WEIGHT_DIR,
                           transform=train_tf, only_tumor=True)
    val_ds   = BUSIDataset(_TEST_CSV, _BUSI_ROOT, _WEIGHT_DIR,
                           transform=val_tf, only_tumor=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader


@torch.no_grad()
def eval_iou(model, loader):
    model.eval()
    ious = []
    for batch in loader:
        imgs, msks = batch[0].to(DEVICE), batch[1]
        msks = torch.clamp(msks, 0, 1).long().to(DEVICE)
        if msks.dim() == 4 and msks.shape[1] == 1: msks = msks.squeeze(1)
        out = model(imgs)
        preds = torch.argmax(out[0] if isinstance(out,(tuple,list)) else out, dim=1)
        for i in range(preds.shape[0]):
            p = preds[i].cpu().numpy().flatten()
            t = msks[i].cpu().numpy().flatten()
            if t.sum() > 0:
                ious.append(np.logical_and(t,p).sum() / (np.logical_or(t,p).sum()+1e-6))
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def predict(model, img_tensor):
    x      = img_tensor.unsqueeze(0).to(DEVICE)
    out    = model(x)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(bool)


# ══════════════════════════════════════════════════════════════════════════
#  Train Without Anchor
# ══════════════════════════════════════════════════════════════════════════
def train_without_anchor():
    print("\n" + "="*60)
    print("  Training Without Anchor (CE + Dice only) ...")
    print("="*60, flush=True)

    os.makedirs(os.path.dirname(CKPT_NO_ANCHOR), exist_ok=True)
    train_loader, val_loader = make_loaders()

    model     = MobileUVitUNet_Ablation(mode="fusion", in_channels=3,
                                        n_classes=2, num_bottleneck_blocks=2).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler    = GradScaler("cuda")
    criterion = LossCEDice(w_ce=1.0, w_dice=1.0).to(DEVICE)
    best_iou  = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            imgs, msks, _ = batch
            imgs = imgs.to(DEVICE)
            msks = torch.clamp(msks, 0, 1).long().to(DEVICE)
            if msks.dim() == 4 and msks.shape[1] == 1: msks = msks.squeeze(1)
            optimizer.zero_grad()
            with autocast("cuda"):
                logits, _ = model(imgs)
                loss = criterion(logits, msks)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            epoch_loss += loss.item()

        val_iou = eval_iou(model, val_loader)
        print(f"  [No-Anchor] Ep {epoch+1:3d}/{NUM_EPOCHS} | "
              f"loss={epoch_loss/len(train_loader):.4f} | iou={val_iou:.4f}", flush=True)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), CKPT_NO_ANCHOR)

    print(f"  [No-Anchor] Done. Best IoU={best_iou:.4f} → {CKPT_NO_ANCHOR}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
#  Uncertainty map loading
#  命名规则来自 step2_dataset_busi.py:
#    wgt_name = os.path.splitext(img_base)[0] + "_weight.png"
# ══════════════════════════════════════════════════════════════════════════
def load_uncertainty(real_img_path):
    if not real_img_path:
        return None
    img_base  = os.path.basename(real_img_path)
    wgt_name  = os.path.splitext(img_base)[0] + "_weight.png"
    umap_path = os.path.join(_WEIGHT_DIR, wgt_name)
    if not os.path.exists(umap_path):
        return None
    umap = np.array(PILImage.open(umap_path).convert("L"), dtype=np.float32)
    umap = (umap - umap.min()) / (umap.max() - umap.min() + 1e-6)
    return umap


# ══════════════════════════════════════════════════════════════════════════
#  Sample selection
# ══════════════════════════════════════════════════════════════════════════
def pick_anchor_effect_cases(model_no, model_with, loader, dataset, n=3):
    """
    遍历全部测试样本，选 gain = iou_with - iou_no 最大的 n 个案例。
    用 global_idx + dataset.df + dataset.resolve_path 取真实图像路径，
    再按 step2_dataset_busi.py 的命名规则加载 uncertainty map。
    """
    candidates = []
    global_idx = 0

    for batch in loader:
        imgs, msks = batch[0], batch[1]

        for i in range(imgs.shape[0]):
            img  = imgs[i]
            mask = msks[i].squeeze().numpy().astype(bool)

            # 取真实图像路径
            csv_img       = dataset.df.iloc[global_idx]["image"]
            real_img_path = dataset.resolve_path(csv_img, dataset.image_index)
            global_idx   += 1

            if mask.sum() < 100:
                continue

            pred_no   = predict(model_no,   img)
            pred_with = predict(model_with, img)

            def iou(p, t):
                return float(np.logical_and(p,t).sum() /
                             (np.logical_or(p,t).sum() + 1e-6))

            iou_no   = iou(pred_no,   mask)
            iou_with = iou(pred_with, mask)
            gain     = iou_with - iou_no

            candidates.append({
                "img":          img.cpu().numpy(),
                "mask":         mask,
                "pred_no":      pred_no,
                "pred_with":    pred_with,
                "iou_no":       iou_no,
                "iou_with":     iou_with,
                "gain":         gain,
                "real_img_path": real_img_path or "",
            })

    if not candidates:
        return []

    gains = [c["gain"] for c in candidates]
    print(f"  Gain stats: max={max(gains):.3f} mean={np.mean(gains):.3f} "
          f"min={min(gains):.3f}", flush=True)

    valid = [c for c in candidates if c["iou_with"] >= 0.3]
    if not valid:
        valid = candidates
    valid.sort(key=lambda x: x["gain"], reverse=True)
    return valid[:n]


# ══════════════════════════════════════════════════════════════════════════
#  Visualisation
# ══════════════════════════════════════════════════════════════════════════
_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])

def to_rgb(img_np):
    return np.clip(img_np.transpose(1,2,0) * _STD + _MEAN, 0, 1)

def show_image(ax, img_np):
    ax.imshow(to_rgb(img_np)); ax.axis("off")

def show_uncertainty(ax, img_np, umap):
    H, W = img_np.shape[1], img_np.shape[2]
    if umap is not None:
        umap_r = np.array(
            PILImage.fromarray((umap * 255).astype(np.uint8))
            .resize((W, H), PILImage.BILINEAR),
            dtype=np.float32
        ) / 255.0
        ax.imshow(umap_r, cmap="hot", vmin=0, vmax=1)
    else:
        ax.imshow(np.zeros((H, W)), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

def show_overlay(ax, img_np, mask, color, alpha=0.40):
    ax.imshow(to_rgb(img_np))
    if mask is not None and mask.sum() > 0:
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask] = [*color, alpha]
        ax.imshow(rgba)
    ax.axis("off")

def show_contour(ax, img_np, mask, color, lw=1.5):
    ax.imshow(to_rgb(img_np))
    if mask is not None and mask.sum() > 0:
        ax.contour(mask.astype(float), levels=[0.5],
                   colors=[color], linewidths=[lw])
    ax.axis("off")


def build_fig6(cases):
    n_rows = len(cases)
    n_cols = len(COL_NAMES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.4, n_rows * 2.4),
        gridspec_kw={"wspace": 0.06, "hspace": 0.06},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, c in enumerate(cases):
        img       = c["img"]
        mask      = c["mask"]
        pred_no   = c["pred_no"]
        pred_with = c["pred_with"]
        umap      = c.get("umap")

        show_image(axes[r, 0], img)
        show_overlay(axes[r, 1], img, mask,      COLOR_GT)
        show_uncertainty(axes[r, 2], img, umap)
        show_contour(axes[r, 3], img, pred_no,   COLOR_NO_ANC)
        show_contour(axes[r, 4], img, pred_with, COLOR_WITH_ANC)

        axes[r, 0].set_ylabel(
            f"IoU: {c['iou_no']:.2f}→{c['iou_with']:.2f}",
            fontsize=13, rotation=90, labelpad=10, va="center"
        )

    for col, cname in enumerate(COL_NAMES):
        axes[0, col].set_title(cname, fontsize=13, pad=10)

    legend_elems = [
        mpatches.Patch(facecolor=COLOR_GT,       label="GT"),
        mpatches.Patch(facecolor=COLOR_NO_ANC,   label="Without Anchor"),
        mpatches.Patch(facecolor=COLOR_WITH_ANC, label="With Anchor"),
    ]
    axes[n_rows - 1, 2].legend(
        handles=legend_elems, loc="upper center", ncol=3,
        fontsize=13, bbox_to_anchor=(0.5, -0.06),
        frameon=False, borderaxespad=0,
        handletextpad=0.4, columnspacing=1.2
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    seed_everything(SEED)

    # Step 1: train Without Anchor if needed
    if not os.path.exists(CKPT_NO_ANCHOR):
        train_without_anchor()
    else:
        print(f"[Skip] Without Anchor checkpoint found: {CKPT_NO_ANCHOR}")

    # Step 2: load both models
    print("\nLoading models ...", flush=True)
    model_no = MobileUVitUNet_Ablation(mode="fusion", in_channels=3,
                                       n_classes=2, num_bottleneck_blocks=2).to(DEVICE)
    model_no.load_state_dict(
        torch.load(CKPT_NO_ANCHOR, map_location=DEVICE, weights_only=True))
    model_no.eval()

    model_with = MobileUVitUNet_Ablation(mode="fusion", in_channels=3,
                                         n_classes=2, num_bottleneck_blocks=2).to(DEVICE)
    model_with.load_state_dict(
        torch.load(CKPT_WITH_ANCHOR, map_location=DEVICE, weights_only=True))
    model_with.eval()

    # Step 3: scan test set
    print("\nScanning test set ...", flush=True)
    val_tf = A.Compose([A.Resize(256,256), A.Normalize(), ToTensorV2()])
    val_ds = BUSIDataset(_TEST_CSV, _BUSI_ROOT, _WEIGHT_DIR,
                         transform=val_tf, only_tumor=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True)

    cases = pick_anchor_effect_cases(model_no, model_with, val_loader, val_ds, n=N_CASES)
    if not cases:
        print("No valid cases found. Exiting.")
        return

    # Step 4: load uncertainty maps
    for c in cases:
        c["umap"] = load_uncertainty(c["real_img_path"])
        status = "loaded" if c["umap"] is not None else "not found"
        print(f"  gain={c['gain']:+.3f} | iou {c['iou_no']:.2f}→{c['iou_with']:.2f} | "
              f"umap={status} | {os.path.basename(c['real_img_path'])}")

    # Step 5: render
    print(f"\nRendering Fig 6 ({len(cases)} rows) ...", flush=True)
    fig = build_fig6(cases)
    pdf = os.path.join(OUTPUT_DIR, "fig6_anchor_effect.pdf")
    png = os.path.join(OUTPUT_DIR, "fig6_anchor_effect.png")
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved:\n  {pdf}\n  {png}")


if __name__ == "__main__":
    main()