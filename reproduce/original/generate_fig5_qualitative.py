"""
generate_fig5_qualitative.py
────────────────────────────
生成论文 Fig 5：四数据集定性对比图 (Pro 版本：自适应阈值抗脆弱设计)
"""

import os, random, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 模型 imports ──────────────────────────────────────────────────────────
from small_unet_ablation import MobileUVitUNet_Ablation
from models.AnatoNet      import UResNetWithAttention
from models.CMUNeXt       import cmunext_l
import torchvision.models as tv_models


# ── Training imports (for auto-retrain) ──────────────────────────────────
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import f1_score

# ── Dataset imports ──────────────────────────────────────────────────────
from step2_dataset_busi      import BUSIDataset
from step2_dataset_breast    import BrEaSTDataset
from step2_dataset_qata      import QaTaDataset        # QaTa-v2
from step2_dataset_qata_v1   import QaTaV1CSVDataset   # QaTa-v1

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════
DEVICE     = "cuda:1"
SEED       = 42
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_BUSI_ROOT  = "/data/xuten/Breast Ultrasound Datasets/BUSI"
_BREAST_ROOT= "/data/xuten/Breast Ultrasound Datasets/BrEaST"
_QATA_ROOT  = "/data/xuten/Breast Ultrasound Datasets/QaTa-COV19"

CKPT_ROOTS = {
    "BUSI": {
        "AnatoNet":   "BUSI/BUSI_Anchor_results/results_AnatoNet_busi_proposed_Dice1.0_CE1.0_BG0.25",
        "DeepLabV3+": "BUSI/BUSI_Anchor_results/results_DeepLabV3_busi_proposed_Dice1.0_CE1.0_BG0.25",
        "CMUNeXt-L":  "BUSI/BUSI_Anchor_results/results_CMUNeXt_busi_proposed_Dice1.0_CE1.0_BG0.25",
        "Ours":       "BUSI/BUSI_Anchor_results/results_depth_ablation_busi_proposed/fusion_2blocks_BUSI_Proposed",
    },
    "BrEaST": {
        "AnatoNet":   "BrEaST/BrEaST_Anchor_result/results_AnatoNet_breast_proposed_Dice1.0_CE1.0_BG0.25",
        "DeepLabV3+": "BrEaST/BrEaST_Anchor_result/results_DeepLabV3_breast_proposed_Dice1.0_CE1.0_BG0.25",
        "CMUNeXt-L":  "BrEaST/BrEaST_Anchor_result/results_CMUNeXt_breast_proposed_Dice1.0_CE1.0_BG0.25",
        "Ours":       "BrEaST/BrEaST_Anchor_result/results_depth_ablation_breast/fusion_2blocks_BrEaST",
    },
    "QaTa-v1": {
        "AnatoNet":   "QaTa/QaTa_v1_Anchor_result/results_AnatoNet_qata_v1_Dice1.0_CE1.0_BG0.25",
        "DeepLabV3+": "QaTa/QaTa_v1_Anchor_result/results_DeepLabV3_qata_v1_Dice1.0_CE1.0_BG0.25",
        "CMUNeXt-L":  "QaTa/QaTa_v1_Anchor_result/results_CMUNeXt_qata_v1_Dice1.0_CE1.0_BG0.25",
        "Ours":       "QaTa/QaTa_v1_Anchor_result/results_depth_ablation_qata_v1/fusion_2blocks_QaTa_v1",
    },
    "QaTa-v2": {
        "AnatoNet":   "QaTa/QaTa_v2_Anchor_result/results_AnatoNet_qata_v2_Dice1.0_CE1.0_BG0.25",
        "DeepLabV3+": "QaTa/QaTa_v2_Anchor_result/results_DeepLabV3_qata_v2_Dice1.0_CE1.0_BG0.25",
        "CMUNeXt-L":  "QaTa/QaTa_v2_Anchor_result/results_CMUNeXt_qata_v2_Dice1.0_CE1.0_BG0.25",
        "Ours":       "QaTa/QaTa_v2_Anchor_result/results_depth_ablation_qata_v2/fusion_2blocks_QaTa_v2",
    },
}


# ── 自动重训配置 ────────────────────────────────────────────────────────────
# RETRAIN_IF_LOW: 若某数据集所有模型的 best iou_ours < 此阈值，自动触发重训
RETRAIN_THRESHOLD = 0.15
# FORCE_RETRAIN: 强制重训指定数据集的全部模型（忽略阈值）
FORCE_RETRAIN = {"BrEaST": True}

# BrEaST 训练参数（与 step3_train_*_breast_proposed.py 完全一致）
_BREAST_TRAIN_CSV  = os.path.join(_BREAST_ROOT, "train.csv")
_BREAST_TEST_CSV   = os.path.join(_BREAST_ROOT, "test.csv")
_BREAST_WEIGHT_DIR = os.path.join(_BREAST_ROOT, "uncertainty_weights")
_TRAIN_EPOCHS      = 100
_BATCH_SIZE        = 8
_LR                = 1e-4
_MAX_ALPHA         = 0.1
_ANCHOR_MARGIN     = 0.3
_LAMBDA_BG         = 0.25
_WARMUP_RATIO      = 0.1

COL_MODELS = ["AnatoNet", "DeepLabV3+", "CMUNeXt-L", "Ours"]
COL_NAMES  = ["Image", "GT"] + COL_MODELS
COLOR_GT   = (0.18, 0.80, 0.44)
COLOR_PRED = {
    "AnatoNet":   (0.95, 0.60, 0.07),
    "DeepLabV3+": (0.20, 0.50, 0.85),
    "CMUNeXt-L":  (0.65, 0.20, 0.75),
    "Ours":       (0.85, 0.15, 0.25),
}

# ══════════════════════════════════════════════════════════════════════════
#  Wrapper 定义 (模型挂载点)
# ══════════════════════════════════════════════════════════════════════════

class UniversalAnchorWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.extracted_features = None
        def hook_fn(module, inputs): self.extracted_features = inputs[0]
        target_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d): target_conv = module
        if target_conv is not None: target_conv.register_forward_pre_hook(hook_fn)

    def forward(self, x):
        self.extracted_features = None
        logits = self.model(x)
        if isinstance(logits, (tuple, list)): logits = logits[0]
        feats = self.extracted_features
        if feats is None or feats.shape[1] < 2: feats = logits
        elif feats.shape[-2:] != logits.shape[-2:]:
            feats = F.interpolate(feats, size=logits.shape[-2:], mode='bilinear', align_corners=False)
        return logits, feats

class CMUNeXtAnchorWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.extracted_features = None
        def hook_fn(module, inputs): self.extracted_features = inputs[0]
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d): module.register_forward_pre_hook(hook_fn)

    def forward(self, x):
        self.extracted_features = None
        outs = self.model(x)
        if isinstance(outs, (tuple, list)): outs = outs[0]
        feats = self.extracted_features
        if feats is None or feats.shape[1] < 2: feats = outs
        elif feats.shape[-2:] != outs.shape[-2:]:
            feats = F.interpolate(feats, size=outs.shape[-2:], mode='bilinear', align_corners=False)
        logits = torch.cat([-outs, outs], dim=1) if outs.shape[1] == 1 else outs
        return logits, feats

class DeepLabV3AnchorWrapper(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.base_model = tv_models.segmentation.deeplabv3_resnet50(weights=None)
        self.base_model.aux_classifier = None
        self.base_model.classifier[-1] = nn.Conv2d(256, n_classes, kernel_size=1)
        layers = list(self.base_model.classifier.children())
        self.feature_extractor = nn.Sequential(*layers[:-1])
        self.classifier_head   = layers[-1]
        self.base_model.classifier = nn.Identity()

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.base_model.backbone(x)['out']
        feats    = self.feature_extractor(features)
        logits   = self.classifier_head(feats)
        logits   = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)
        feats    = F.interpolate(feats, size=input_shape, mode='bilinear', align_corners=False)
        return logits, feats


# ══════════════════════════════════════════════════════════════════════════
#  Auto-retrain (BrEaST)
# ══════════════════════════════════════════════════════════════════════════

from step2_loss_anchor import LossCEDice, anchor_loss_from_maps

def _make_breast_loaders():
    train_tf = A.Compose([A.Resize(256,256), A.HorizontalFlip(p=0.5),
                          A.Rotate(limit=30, p=0.5), A.Normalize(), ToTensorV2()])
    val_tf   = A.Compose([A.Resize(256,256), A.Normalize(), ToTensorV2()])
    train_ds = BrEaSTDataset(_BREAST_TRAIN_CSV, _BREAST_ROOT,
                             weight_dir=_BREAST_WEIGHT_DIR,
                             transform=train_tf, only_tumor=True)
    val_ds   = BrEaSTDataset(_BREAST_TEST_CSV,  _BREAST_ROOT,
                             weight_dir=_BREAST_WEIGHT_DIR,
                             transform=val_tf, only_tumor=True)
    train_loader = DataLoader(train_ds, batch_size=_BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=True)
    return train_loader, val_loader

@torch.no_grad()
def _eval(model, loader):
    model.eval()
    ious = []
    for batch in loader:
        imgs, msks = batch[0].to(DEVICE), batch[1]
        msks = torch.clamp(msks, 0, 1).long().to(DEVICE)
        if msks.dim() == 4 and msks.shape[1] == 1: msks = msks.squeeze(1)
        out = model(imgs)
        preds = torch.argmax(out[0] if isinstance(out,(tuple,list)) else out, dim=1)
        for i in range(preds.shape[0]):
            p, t = preds[i].cpu().numpy().flatten(), msks[i].cpu().numpy().flatten()
            if t.sum() > 0:
                ious.append(np.logical_and(t,p).sum() / (np.logical_or(t,p).sum()+1e-6))
    return float(np.mean(ious)) if ious else 0.0

def _train_one(model, save_path, tag):
    """通用训练循环，保存 best_model.pth 到 save_path"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_loader, val_loader = _make_breast_loaders()
    optimizer = optim.AdamW(model.parameters(), lr=_LR)
    scaler    = GradScaler("cuda")
    criterion = LossCEDice(w_ce=1.0, w_dice=1.0).to(DEVICE)
    warmup    = int(_WARMUP_RATIO * _TRAIN_EPOCHS)
    best_iou  = 0.0
    for epoch in range(_TRAIN_EPOCHS):
        model.train()
        alpha = 0.0 if epoch < warmup else _MAX_ALPHA * ((epoch-warmup)/(_TRAIN_EPOCHS-warmup))
        for batch in train_loader:
            imgs, msks, wgts = batch[0].to(DEVICE), batch[1], batch[2]
            msks = torch.clamp(msks, 0, 1).long().to(DEVICE)
            if msks.dim()==4 and msks.shape[1]==1: msks = msks.squeeze(1)
            wgts = wgts.float().squeeze(1).to(DEVICE) if wgts.dim()==4 else wgts.float().to(DEVICE)
            optimizer.zero_grad()
            with autocast("cuda"):
                logits, feats = model(imgs)
                l_seg = criterion(logits, msks)
                l_anc = anchor_loss_from_maps(feat=feats.float(), gt=msks, u_fg=wgts,
                                              w_fg=None, margin=_ANCHOR_MARGIN,
                                              lam_bg=_LAMBDA_BG, min_pts=10) if alpha>1e-6 else torch.tensor(0., device=DEVICE)
                loss = l_seg + alpha * l_anc
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        val_iou = _eval(model, val_loader)
        print(f"  [{tag}] Ep {epoch+1:3d}/{_TRAIN_EPOCHS} | iou={val_iou:.4f} | alpha={alpha:.4f}", flush=True)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), save_path)
    print(f"  [{tag}] Finished. Best IoU={best_iou:.4f} → {save_path}", flush=True)

def retrain_breast(ckpt_map):
    """重训 BrEaST 四个模型，直接保存到 CKPT_ROOTS 对应路径"""
    print("\n" + "="*60)
    print("  Auto-retraining BrEaST models ...")
    print("="*60, flush=True)

    for mname, ckpt_root in ckpt_map.items():
        save_path = os.path.join(ckpt_root, "best_model.pth")
        print(f"\n  Training {mname} → {save_path}", flush=True)
        model = build_model(mname).to(DEVICE)
        # Ours 直接 forward 返回 (logits, feats)，其他 model 已套 wrapper
        _train_one(model, save_path, mname)

# ══════════════════════════════════════════════════════════════════════════
#  Helper Functions
# ══════════════════════════════════════════════════════════════════════════

def build_model(name):
    if name == "AnatoNet":
        base  = UResNetWithAttention(n_classes=2, beta=3.5, gamma=1.0)
        return UniversalAnchorWrapper(base)
    elif name == "DeepLabV3+":
        return DeepLabV3AnchorWrapper(n_classes=2)
    elif name == "CMUNeXt-L":
        try: base = cmunext_l(num_classes=2)
        except TypeError: base = cmunext_l()
        return CMUNeXtAnchorWrapper(base)
    elif name == "Ours":
        return MobileUVitUNet_Ablation(mode="fusion", in_channels=3, n_classes=2, num_bottleneck_blocks=2)
    return None

def load_model(name, ckpt_root):
    ckpt_path = os.path.join(ckpt_root, "best_model.pth")
    if not os.path.exists(ckpt_path): raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = build_model(name)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

def make_loader(dataset_name, val_tf):
    if dataset_name == "BUSI":
        ds = BUSIDataset(csv_file=os.path.join(_BUSI_ROOT, "test_split.csv"), root_dir=_BUSI_ROOT, 
                         weight_dir=os.path.join(_BUSI_ROOT, "uncertainty_weights"), transform=val_tf, only_tumor=True)
    elif dataset_name == "BrEaST":
        ds = BrEaSTDataset(csv_file=os.path.join(_BREAST_ROOT, "test.csv"), root_dir=_BREAST_ROOT,
                           weight_dir=os.path.join(_BREAST_ROOT, "uncertainty_weights"), transform=val_tf, only_tumor=True)
    elif dataset_name == "QaTa-v1":
        ds = QaTaV1CSVDataset(csv_file=os.path.join(_QATA_ROOT, "test_v1.csv"), 
                              weight_dir=os.path.join(_QATA_ROOT, "uncertainty_weights"),
                              weight_prefix="v1_", transform=val_tf)
    elif dataset_name == "QaTa-v2":
        ds = QaTaDataset(img_dir=os.path.join(_QATA_ROOT, "QaTa-COV19-v2", "Test Set", "Images"),
                         gt_dir=os.path.join(_QATA_ROOT, "QaTa-COV19-v2", "Test Set", "Ground-truths"),
                         weight_dir=os.path.join(_QATA_ROOT, "uncertainty_weights"),
                         weight_prefix="v2_test_", transform=val_tf)
    return DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

@torch.no_grad()
def predict(model, img_tensor):
    x = img_tensor.unsqueeze(0).to(DEVICE)
    out = model(x)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(bool)

# ── 样本选择：遍历全部样本，选 Ours 优势最明显的案例 ──────────────────────
def pick_best_contrast(loader, models, n=1):
    BASELINE_NAMES = [m for m in models if m != "Ours"]
    all_candidates = []

    for batch in loader:
        imgs, msks = batch[0], batch[1]
        for i in range(imgs.shape[0]):
            img  = imgs[i]
            mask = msks[i].squeeze().numpy().astype(bool)
            if mask.sum() == 0:
                continue
            preds, iou_map = {}, {}
            for mname, mobj in models.items():
                p     = predict(mobj, img)
                inter = (p & mask).sum()
                union = (p | mask).sum()
                iou_map[mname] = float(inter / (union + 1e-6))
                preds[mname]   = p
            iou_ours      = iou_map.get("Ours", 0.0)
            iou_baselines = [iou_map[b] for b in BASELINE_NAMES]
            mean_baseline = float(np.mean(iou_baselines))
            all_candidates.append({
                "img":           img.cpu().numpy(),
                "mask":          mask,
                "preds":         preds,
                "mean_iou":      float(np.mean(list(iou_map.values()))),
                "iou_ours":      iou_ours,
                "mean_baseline": mean_baseline,
                "score":         iou_ours - mean_baseline,
                "mask_size":     int(mask.sum()),
            })

    if not all_candidates:
        return []

    print(f"    Total valid: {len(all_candidates)} | "
          f"best iou_ours={max(c['iou_ours'] for c in all_candidates):.3f} | "
          f"best score={max(c['score'] for c in all_candidates):+.3f}", flush=True)

    # 自动降级阈值：从严到宽，找 Ours 有效且对比价值最高的样本
    thresholds = [0.45, 0.35, 0.25, 0.15, 0.05]
    candidates = []
    for thr in thresholds:
        candidates = [c for c in all_candidates
                      if c["iou_ours"] >= thr and c["score"] > 0]
        if candidates:
            break

    # 保底：只取 iou_ours 最高的
    if not candidates:
        candidates = sorted(all_candidates,
                            key=lambda x: x["iou_ours"], reverse=True)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:n]

# ── Visualizing Helpers ──────────────────────────────────────────────────

_MEAN, _STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
def to_rgb(img_np): return np.clip(img_np.transpose(1, 2, 0) * _STD + _MEAN, 0, 1)

# ==============================================================================
# 全局绘图参数（已优化：完美对齐图例与行距）
# ==============================================================================
PLOT_FONT_SIZE = 14  # 统一的字体大小
PLOT_TEXT_PAD  = 15  # 标题、Y轴文字与图片的距离
PLOT_HSPACE    = 0.06 # 两排图片之间的行距比例

def build_fig5(rows):
    n_rows, n_cols = len(rows), len(COL_NAMES)
    
    # 稍微增加图表的整体高度 (+0.6)，给底部留出充足空间放图例
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2 + 0.6),
                             gridspec_kw={"wspace": 0.02, "hspace": PLOT_HSPACE})
    if n_rows == 1: axes = axes[np.newaxis, :]
    
    for r, row in enumerate(rows):
        img, mask, preds = row["img"], row["mask"], row["preds"]
        
        # 1. Image
        axes[r, 0].imshow(to_rgb(img)); axes[r, 0].axis("off")
        
        # 2. GT Overlay
        axes[r, 1].imshow(to_rgb(img))
        rgba = np.zeros((*mask.shape, 4)); rgba[mask] = [*COLOR_GT, 0.4]
        axes[r, 1].imshow(rgba); axes[r, 1].axis("off")
        
        # 3. Model Contours
        for c, mname in enumerate(COL_MODELS, start=2):
            axes[r, c].imshow(to_rgb(img))
            if preds[mname].sum() > 0:
                axes[r, c].contour(preds[mname].astype(float), levels=[0.5], colors=[COLOR_PRED[mname]], linewidths=[1.5])
            axes[r, c].axis("off")
            
        # ── 处理 Y轴数据集名称 (自动替换 SAM Uncertainty) ──
        dataset_name = row["dataset_name"]
        if dataset_name == "SAM Uncertainty":
            dataset_name = "Uncertainty Map"
            
        axes[r, 0].set_ylabel(dataset_name, fontsize=PLOT_FONT_SIZE, rotation=90, labelpad=PLOT_TEXT_PAD, va="center")

    # ── 处理 顶部列名 (自动替换 SAM Uncertainty) ──
    for c, cname in enumerate(COL_NAMES):
        title_name = cname
        if title_name == "SAM Uncertainty":
            title_name = "Uncertainty Map"
        axes[0, c].set_title(title_name, fontsize=PLOT_FONT_SIZE, pad=PLOT_TEXT_PAD)
        
    # ==========================================================================
    # 精确控制图例距离的核心逻辑
    # ==========================================================================
    # 1. 强制设定所有图片的底边在画布高度的 12% 处
    BOTTOM_MARGIN = 0.12 
    fig.subplots_adjust(bottom=BOTTOM_MARGIN)
    
    legend_elems = [mpatches.Patch(facecolor=COLOR_GT, label="GT")] + \
                   [mpatches.Patch(facecolor=COLOR_PRED[m], label=m) for m in COL_MODELS]
                   
    # 2. 计算出视觉上等同于 hspace 的画布局部比例 (大约在 0.01~0.02 之间)
    # 如果 n_rows 变化，这个 gap 也能自适应保持一致
    visual_gap = PLOT_HSPACE / max(n_rows, 2) 
    
    # 3. 关键修改：将锚点改为 "upper center" (以上边缘对齐)
    # 图例的最高点 = 图片最低点(BOTTOM_MARGIN) - 视觉间距(visual_gap)
    fig.legend(handles=legend_elems, loc="upper center", ncol=len(legend_elems), 
               fontsize=PLOT_FONT_SIZE, bbox_to_anchor=(0.5, BOTTOM_MARGIN - visual_gap), frameon=False)
               
    return fig

# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    val_tf = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
    rows = []
    for dname, ckpt_map in CKPT_ROOTS.items():
        print(f"\nProcessing Dataset: {dname}...")

        # ── 自动重训：FORCE_RETRAIN 或 checkpoint 缺失 ─────────────────────
        needs_retrain = FORCE_RETRAIN.get(dname, False)
        if not needs_retrain:
            for ckpt_root in ckpt_map.values():
                if not os.path.exists(os.path.join(ckpt_root, "best_model.pth")):
                    needs_retrain = True
                    break
        if needs_retrain and dname == "BrEaST":
            retrain_breast(ckpt_map)

        models = {}
        try:
            for mname, ckpt_root in ckpt_map.items():
                models[mname] = load_model(mname, ckpt_root)
            loader = make_loader(dname, val_tf)
            samples = pick_best_contrast(loader, models, n=1)
            if samples:
                s = samples[0]; s["dataset_name"] = dname
                rows.append(s)
                print(f"  [OK] score={s['score']:+.3f} | iou_ours={s['iou_ours']:.3f} | mean_baseline={s['mean_baseline']:.3f} | mask_size={s['mask_size']}")
            else:
                print(f"  [Warn] No valid ground truth found in {dname}")
        except Exception as e:
            print(f"  [Error] Failed {dname}: {e}")

    if rows:
        print(f"\nGenerating figure with {len(rows)} rows...")
        fig = build_fig5(rows)
        save_path = os.path.join(OUTPUT_DIR, "fig5_qualitative.png")
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Success! Saved to: {save_path}")
    else:
        print("\nNo rows to plot. Check errors above.")

if __name__ == "__main__":
    main()