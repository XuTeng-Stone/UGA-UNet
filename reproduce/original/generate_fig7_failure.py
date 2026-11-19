"""
generate_fig7_failure.py
────────────────────────
生成论文 Fig 7：失败案例图
列：Image | GT | Best Baseline | Ours | Uncertainty Map

选取逻辑：
  - 遍历 BUSI 测试集全部样本
  - 对每张图，找当前表现最好的 baseline（IoU 最高）
  - 计算 gap = iou_best_baseline - iou_ours
  - 选 gap 最大的 N_CASES 个案例（Ours 相对最差的）
  - 额外约束：iou_best_baseline >= 0.4（baseline 确实预测得好）
              iou_ours <= 0.6（Ours 确实表现差）
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
from PIL import Image as PILImage

from small_unet_ablation import MobileUVitUNet_Ablation
from models.AnatoNet      import UResNetWithAttention
from models.CMUNeXt       import cmunext_l
from step2_dataset_busi   import BUSIDataset

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════
DEVICE     = "cuda:2"
SEED       = 42
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_BUSI_ROOT  = "/data/xuten/Breast Ultrasound Datasets/BUSI"
_TEST_CSV   = os.path.join(_BUSI_ROOT, "test_split.csv")
_WEIGHT_DIR = os.path.join(_BUSI_ROOT, "uncertainty_weights")

# 所有 baseline 权重路径（动态选当前案例最好的那个）
BASELINE_CKPTS = {
    "AnatoNet":   "BUSI/BUSI_Anchor_results/results_AnatoNet_busi_proposed_Dice1.0_CE1.0_BG0.25",
    "DeepLabV3+": "BUSI/BUSI_Anchor_results/results_DeepLabV3_busi_proposed_Dice1.0_CE1.0_BG0.25",
    "CMUNeXt-L":  "BUSI/BUSI_Anchor_results/results_CMUNeXt_busi_proposed_Dice1.0_CE1.0_BG0.25",
}
OURS_CKPT = "BUSI/BUSI_Anchor_results/results_depth_ablation_busi_proposed/fusion_2blocks_BUSI_Proposed/best_model.pth"

N_CASES = 2   # 展示 2 个失败案例

# 颜色
COLOR_GT      = (0.18, 0.80, 0.44)
COLOR_BASELINE = (0.95, 0.60, 0.07)   # amber（baseline，名称动态）
COLOR_OURS    = (0.85, 0.15, 0.25)   # red

COL_NAMES = ["Image", "GT", "Best Baseline", "Ours", "Uncertainty Map"]

# ══════════════════════════════════════════════════════════════════════════
#  Wrappers（与训练脚本一致）
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
            feats = F.interpolate(feats, size=logits.shape[-2:],
                                  mode='bilinear', align_corners=False)
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
            feats = F.interpolate(feats, size=outs.shape[-2:],
                                  mode='bilinear', align_corners=False)
        logits = torch.cat([-outs, outs], dim=1) if outs.shape[1] == 1 else outs
        return logits, feats

import torchvision.models as tv_models
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
        logits   = F.interpolate(logits, size=input_shape,
                                 mode='bilinear', align_corners=False)
        feats    = F.interpolate(feats, size=input_shape,
                                 mode='bilinear', align_corners=False)
        return logits, feats

# ══════════════════════════════════════════════════════════════════════════
#  Model loading
# ══════════════════════════════════════════════════════════════════════════
def build_baseline(name):
    if name == "AnatoNet":
        base = UResNetWithAttention(n_classes=2, beta=3.5, gamma=1.0)
        return UniversalAnchorWrapper(base)
    elif name == "DeepLabV3+":
        return DeepLabV3AnchorWrapper(n_classes=2)
    elif name == "CMUNeXt-L":
        try:    base = cmunext_l(num_classes=2)
        except: base = cmunext_l()
        return CMUNeXtAnchorWrapper(base)
    raise ValueError(f"Unknown baseline: {name}")

def load_baselines():
    models = {}
    for name, ckpt_root in BASELINE_CKPTS.items():
        ckpt = os.path.join(ckpt_root, "best_model.pth")
        if not os.path.exists(ckpt):
            print(f"  [WARN] Checkpoint not found for {name}: {ckpt}")
            continue
        m = build_baseline(name)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        m.to(DEVICE).eval()
        models[name] = m
        print(f"  [OK] Loaded {name}")
    return models

def load_ours():
    m = MobileUVitUNet_Ablation(mode="fusion", in_channels=3,
                                n_classes=2, num_bottleneck_blocks=2)
    m.load_state_dict(torch.load(OURS_CKPT, map_location=DEVICE, weights_only=True))
    m.to(DEVICE).eval()
    return m

# ══════════════════════════════════════════════════════════════════════════
#  Inference & uncertainty map
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict(model, img_tensor):
    x   = img_tensor.unsqueeze(0).to(DEVICE)
    out = model(x)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(bool)

def iou_score(p, t):
    return float(np.logical_and(p, t).sum() /
                 (np.logical_or(p, t).sum() + 1e-6))

def load_uncertainty(real_img_path):
    """
    real_img_path: 图像的完整路径，如 .../Images/benign (1).png
    weight 文件命名规则（来自 step2_dataset_busi.py）:
      os.path.splitext(img_base)[0] + "_weight.png"
    """
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
def umap_failure_overlap(umap, pred_ours, mask):
    """
    计算 uncertainty map map 与 Ours 失败区域的重叠程度。
    失败区域定义为：mask XOR pred_ours（预测错的像素）。
    返回失败区域内的平均 uncertainty 值 ∈ [0,1]。
    值越高说明 模型越能"预见"到这里模型会出问题。
    """
    if umap is None:
        return 0.0
    # 确保尺寸一致
    if umap.shape != mask.shape:
        import PIL.Image as _PILImg
        umap = np.array(_PILImg.fromarray((umap * 255).astype(np.uint8))
                        .resize((mask.shape[1], mask.shape[0]),
                                _PILImg.BILINEAR), dtype=np.float32) / 255.0
    failure_region = mask ^ pred_ours   # XOR: FP + FN
    if failure_region.sum() == 0:
        return 0.0
    return float(umap[failure_region].mean())


def pick_failure_cases(baselines, model_ours, loader, dataset, n=2):
    """
    遍历全部测试样本，选以下综合得分最高的案例：
      score = gap * umap_overlap
    其中：
      gap          = iou_best_baseline - iou_ours  （Ours 失败程度）
      umap_overlap = 失败区域内的平均 uncertainty map （Uncertainty Map 的解释力）

    这样选出来的案例：Ours 确实失败 AND uncertainty map 正好覆盖了失败区域，
    即 uncertainty map 能直观解释为何模型在此处出错。
    """
    all_candidates = []
    global_idx = 0   # 跨 batch 的全局样本索引

    for batch in loader:
        imgs, msks = batch[0], batch[1]

        for i in range(imgs.shape[0]):
            img  = imgs[i]
            mask = msks[i].squeeze().numpy().astype(bool)
            if mask.sum() < 100:
                global_idx += 1
                continue

            # 用数据集内部的 df + resolve_path 拿真实图像路径
            csv_img      = dataset.df.iloc[global_idx]["image"]
            real_img_path = dataset.resolve_path(csv_img, dataset.image_index)
            umap         = load_uncertainty(real_img_path) if real_img_path else None
            global_idx  += 1

            # Ours prediction
            pred_ours = predict(model_ours, img)
            iou_ours  = iou_score(pred_ours, mask)

            # All baselines — find best
            best_name, best_pred, best_iou = None, None, -1.0
            for bname, bmodel in baselines.items():
                p   = predict(bmodel, img)
                iou = iou_score(p, mask)
                if iou > best_iou:
                    best_iou  = iou
                    best_pred = p
                    best_name = bname

            gap     = best_iou - iou_ours
            overlap = umap_failure_overlap(umap, pred_ours, mask)
            # 综合得分：gap 和 Uncertainty Map overlap 同时大才得高分
            score   = gap * overlap

            all_candidates.append({
                "img":        img.cpu().numpy(),
                "mask":       mask,
                "pred_ours":  pred_ours,
                "pred_best":  best_pred,
                "best_name":  best_name,
                "iou_ours":   iou_ours,
                "iou_best":   best_iou,
                "gap":        gap,
                "overlap":    overlap,
                "score":      score,
                "umap":       umap,
                "img_name":   real_img_path or "",
            })

    if not all_candidates:
        return []

    scores = [c["score"] for c in all_candidates]
    print(f"  Gap stats: max={max(c['gap'] for c in all_candidates):.3f} | "
          f"cases with gap>0.3: {sum(1 for c in all_candidates if c['gap']>0.3)}",
          flush=True)

    # 硬性约束：baseline 预测好 + Ours 预测差 + 差距明显
    # 这三条同时满足才算真正的失败案例
    valid = [c for c in all_candidates
             if c["iou_best"] >= 0.60   # baseline 确实预测准
             and c["iou_ours"] <= 0.35  # Ours 确实预测差
             and c["gap"] >= 0.25]      # 绝对差距 ≥ 25%

    # 降级1：放宽 Ours IoU 要求
    if not valid:
        valid = [c for c in all_candidates
                 if c["iou_best"] >= 0.55 and c["gap"] >= 0.20]

    # 降级2：只看 gap
    if not valid:
        valid = [c for c in all_candidates if c["gap"] >= 0.15]

    # 保底
    if not valid:
        valid = all_candidates

    # 主排序：gap 降序（视觉差距最大优先）；次排序：overlap 降序（Uncertainty Map 解释力）
    valid.sort(key=lambda x: (x["gap"], x["overlap"]), reverse=True)
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

def show_uncertainty(ax, img_np, umap):
    """显示 uncertainty map map，resize 到与图像相同尺寸后全幅展示"""
    H, W = img_np.shape[1], img_np.shape[2]   # img_np: (3,H,W)
    if umap is not None:
        # resize 到与图像一致
        umap_resized = np.array(
            PILImage.fromarray((umap * 255).astype(np.uint8))
            .resize((W, H), PILImage.BILINEAR),
            dtype=np.float32
        ) / 255.0
        ax.imshow(umap_resized, cmap="hot", vmin=0, vmax=1)
    else:
        # 无 uncertainty map 时显示全灰占位
        ax.imshow(np.zeros((H, W)), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")


def build_fig7(cases):
    n_rows = len(cases)
    n_cols = len(COL_NAMES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.2, n_rows * 2.2),
        gridspec_kw={"wspace": 0.02, "hspace": 0.08},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, c in enumerate(cases):
        img       = c["img"]
        mask      = c["mask"]
        pred_best = c["pred_best"]
        pred_ours = c["pred_ours"]
        umap      = c.get("umap")
        bname     = c["best_name"]

        show_image(axes[r, 0], img)
        show_overlay(axes[r, 1], img, mask, COLOR_GT)
        show_contour(axes[r, 2], img, pred_best, COLOR_BASELINE)
        show_contour(axes[r, 3], img, pred_ours, COLOR_OURS)
        show_uncertainty(axes[r, 4], img, umap)   # img 留着给函数取 H/W 用，不变

        axes[r, 0].set_ylabel(
            f"Best: {bname}\n"
            f"IoU {bname[:4]}: {c['iou_best']:.2f} | Ours: {c['iou_ours']:.2f}",
            fontsize=11, rotation=90, labelpad=10, va="center"
        )

    # Dynamic col titles: "Best Baseline" shows actual model name only in row label
    for col, cname in enumerate(COL_NAMES):
        axes[0, col].set_title(cname, fontsize=11, pad=10)

    legend_elems = [
        mpatches.Patch(facecolor=COLOR_GT,       label="GT"),
        mpatches.Patch(facecolor=COLOR_BASELINE, label="Best baseline"),
        mpatches.Patch(facecolor=COLOR_OURS,     label="Ours"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, 0), frameon=False)
    return fig

# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════
def seed_everything(seed=42):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.backends.cudnn.deterministic = True

def main():
    seed_everything(SEED)

    print("Loading models ...", flush=True)
    baselines  = load_baselines()
    model_ours = load_ours()

    print("\nScanning BUSI test set ...", flush=True)
    val_tf = A.Compose([A.Resize(256,256), A.Normalize(), ToTensorV2()])
    val_ds = BUSIDataset(_TEST_CSV, _BUSI_ROOT, _WEIGHT_DIR,
                         transform=val_tf, only_tumor=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True)

    cases = pick_failure_cases(baselines, model_ours, val_loader, val_ds, n=N_CASES)
    if not cases:
        print("No failure cases found.")
        return

    # Report selected cases (umap already loaded inside pick_failure_cases)
    for c in cases:
        status = "loaded" if c["umap"] is not None else "not found"
        print(f"  [{c['best_name']}] gap={c['gap']:.3f} | overlap={c['overlap']:.3f} | "
              f"score={c['score']:.3f} | iou_best={c['iou_best']:.3f} | "
              f"iou_ours={c['iou_ours']:.3f} | umap={status}")

    print(f"\nRendering Fig 7 ({len(cases)} rows) ...", flush=True)
    fig = build_fig7(cases)
    pdf = os.path.join(OUTPUT_DIR, "fig7_failure.pdf")
    png = os.path.join(OUTPUT_DIR, "fig7_failure.png")
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved:\n  {pdf}\n  {png}")

if __name__ == "__main__":
    main()