"""
train_and_visualize_pipeline.py
===============================
端到端流水线：
1. 训练阶段：从头训练 MobileUVitUNet (使用 CosineAnnealingLR，150 Epochs 确保收敛锐度)
2. 可视化阶段：训练完成后，自动加载刚生成的 best_model.pth，并执行完整的可视化流程：
   - 包含 01_input_image
   - 包含 02_ground_truth
   - 包含 03_pred_k (12 次 MedSAM 扰动生成的掩膜)
   - 包含 04_uncertainty_U
   - 包含 05_uncertainty_Ubar_with_legend
   - 包含 06_segmentation_Z
"""

from __future__ import annotations
import os, sys, re, random, argparse
from pathlib import Path

# --- 训练相关的库 ---
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score

# --- 自定义模块 (需在同级目录下) ---
from step2_dataset_busi import BUSIDataset
from step2_loss_anchor import LossCEDice, anchor_loss_from_maps
from small_unet_ablation import MobileUVitUNet_Ablation
from metrics_utils import cal_params_flops, calculate_advanced_metrics

# --- 可视化相关的库 ---
import cv2
from PIL import Image as PILImage
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from transformers import SamModel, SamProcessor

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
SEED = 42

def seed_everything(seed=42): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)


# ==========================================================================
# PART 1: 训练模块
# ==========================================================================
@torch.no_grad()
def evaluate_dataset(model, loader, device="cuda:0", desc="Eval"):
    model.eval()
    ious, f1s, assds, hd95s = [], [], [], []
    for imgs, msks, _ in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(device)
        msks = torch.clamp(msks, 0, 1).long().to(device)
        if msks.dim() == 4 and msks.shape[1] == 1: msks = msks.squeeze(1)
        
        outs, _ = model(imgs)
        pred, target = torch.argmax(outs, dim=1).cpu().numpy(), msks.cpu().numpy()
        
        for i in range(pred.shape[0]):
            p, t = pred[i].flatten(), target[i].flatten()
            if np.sum(t) > 0: 
                inter, union = np.logical_and(t, p).sum(), np.logical_or(t, p).sum()
                ious.append(inter / (union + 1e-6))
                f1s.append(f1_score(t, p, average='binary', zero_division=1))
        
        assd, hd95 = calculate_advanced_metrics(outs, msks)
        if assd > 0: assds.append(assd)
        if hd95 > 0: hd95s.append(hd95)
        
    return (np.mean(ious) if ious else 0.0, 
            np.mean(f1s) if f1s else 0.0, 
            np.mean(assds) if assds else 0.0, 
            np.mean(hd95s) if hd95s else 0.0)

def run_training_pipeline(args):
    print(f"\n{'='*30}")
    print(f" [PHASE 1] STARTING TRAINING")
    print(f" {'='*30}")
    
    train_csv = os.path.join(args.busi_root, "train_split.csv")
    test_csv = os.path.join(args.busi_root, "test_split.csv")
    weight_dir = os.path.join(args.busi_root, "uncertainty_weights")
    
    save_dir = os.path.join(args.save_dir_root, f"fusion_{args.train_blocks}blocks_BUSI_Proposed")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")
    
    train_tf = A.Compose([
        A.Resize(256, 256), 
        A.HorizontalFlip(p=0.5), A.Rotate(limit=30, p=0.5), 
        A.Normalize(), ToTensorV2()
    ])
    val_tf = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
    
    train_ds = BUSIDataset(train_csv, args.busi_root, weight_dir, transform=train_tf)
    val_ds = BUSIDataset(test_csv, args.busi_root, weight_dir, transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True) 

    model = MobileUVitUNet_Ablation(mode='fusion', in_channels=3, n_classes=2, num_bottleneck_blocks=args.train_blocks).to(args.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion_seg = LossCEDice(w_ce=1.0, w_dice=1.0).to(args.device)
    
    best_iou, best_alpha, warmup_epochs = 0.0, 0.0, int(0.1 * args.epochs)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        alpha = 0.0 if epoch < warmup_epochs else 0.1 * ((epoch - warmup_epochs) / (args.epochs - warmup_epochs))
            
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}", leave=False)
        for imgs, msks, wgts in loop:
            imgs = imgs.to(args.device)
            msks = torch.clamp(msks, 0, 1).long().to(args.device)
            if msks.dim() == 4 and msks.shape[1] == 1: msks = msks.squeeze(1)
            wgts = wgts.float().squeeze(1).to(args.device) if wgts.dim() == 4 else wgts.float().to(args.device)
            
            optimizer.zero_grad()
            with autocast('cuda'):
                logits, feats = model(imgs)
                l_seg = criterion_seg(logits, msks)
                l_anchor = anchor_loss_from_maps(
                    feat=feats.float(), gt=msks, u_fg=wgts, w_fg=None, 
                    margin=0.3, lam_bg=0.25, min_pts=10
                ) if alpha > 1e-6 else torch.tensor(0.0, device=args.device)
                
                loss = l_seg + alpha * l_anchor
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch > (args.epochs // 2):
            val_iou, _, _, _ = evaluate_dataset(model, val_loader, device=args.device)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Ep {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | Val mIoU: {val_iou:.4f} | LR: {current_lr:.6f} | Alpha: {alpha:.4f}")
            
            if val_iou > best_iou: 
                best_iou, best_alpha = val_iou, alpha
                torch.save(model.state_dict(), best_model_path)
                print(f"  --> 🔥 New Best Model Saved! (mIoU: {best_iou:.4f})")
                
    print(f"\n[PHASE 1 COMPLETE] Best model saved to: {best_model_path}\n")
    return best_model_path


# ==========================================================================
# PART 2: 样本扫描与打分 (完全对齐 diagram)
# ==========================================================================
def boundary_uncertainty_score(mask_bin, umap):
    dilated    = ndimage.binary_dilation(mask_bin, iterations=6)
    eroded     = ndimage.binary_erosion(mask_bin,  iterations=6)
    boundary   = dilated & ~eroded
    interior   = eroded
    background = ~dilated
    if boundary.sum() == 0: return 0.0
    return (float(umap[boundary].mean())
            - 0.4 * float(umap[interior].mean()   if interior.sum()    > 0 else 0)
            - 0.4 * float(umap[background].mean() if background.sum() > 0 else 0))

def scan_top_samples(busi_root, top_n=3, force_image_name=None):
    weights_dir = os.path.join(busi_root, "uncertainty_weights")
    candidates = []
    forced_candidate = None
    
    print(f"[info] Scanning BUSI dataset for visualization...")
    for root, dirs, files in os.walk(busi_root):
        root_l = root.lower()
        if any(k in root_l for k in ["mask","label","weight"]): continue
        for f in sorted(files):
            if not f.endswith(".png") or f.startswith("."): continue
            
            is_forced = (force_image_name is not None and f.lower() == force_image_name.lower())
            if not is_forced and "normal" in f.lower(): continue
            
            img_path  = os.path.join(root, f)
            stem      = os.path.splitext(f)[0]
            mask_path = None
            
            for mroot, _, mfiles in os.walk(busi_root):
                if "mask" not in mroot.lower() and "label" not in mroot.lower(): continue
                for mf in mfiles:
                    if stem.lower() in mf.lower():
                        mask_path = os.path.join(mroot, mf); break
                if mask_path: break
            
            umap_path = os.path.join(weights_dir, stem + "_weight.png")
            if mask_path is None or not os.path.exists(umap_path): continue
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: continue
            mask_r   = cv2.resize(mask, (256,256))
            mask_bin = (mask_r > 0)
            ratio    = float(mask_bin.mean())
            
            if not is_forced and (ratio < 0.06 or ratio > 0.40): continue
            
            umap = np.array(PILImage.open(umap_path).convert("L"), dtype=np.float32)
            umap_r = np.array(PILImage.fromarray(umap.astype(np.uint8)).resize((256,256), PILImage.LANCZOS), dtype=np.float32)
            umap_sm = cv2.GaussianBlur(umap_r, (7,7), 2.0)
            umap_sm = (umap_sm - umap_sm.min()) / (umap_sm.max() - umap_sm.min() + 1e-6)
            
            size_score = max(0.0, 1.0 - abs(ratio - 0.20) / 0.20)
            bu_score   = boundary_uncertainty_score(mask_bin, umap_sm)
            peak_score = float(umap_sm.max())
            combined   = size_score * max(0.0, bu_score) * peak_score
            
            cand_info = {
                "score": combined, "img": img_path, "mask": mask_path, 
                "umap": umap_path, "ratio": ratio, "bu": bu_score, "peak": peak_score
            }
            
            if is_forced:
                cand_info["score"] = float('inf')
                forced_candidate = cand_info
            else:
                candidates.append(cand_info)
            
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"\n[info] Top {min(10, len(candidates))} general candidates found:")
    for i, c in enumerate(candidates[:10]):
        print(f"  [{i+1}] score={c['score']:.4f} ratio={c['ratio']:.3f} bu={c['bu']:.3f} peak={c['peak']:.3f}  {os.path.basename(c['img'])}")
    
    chosen = []
    if forced_candidate:
        print(f"\n[info] 🎯 FORCED INCLUSION triggered for: {os.path.basename(forced_candidate['img'])}")
        chosen.append(forced_candidate)
        
    needed = top_n - len(chosen)
    if needed > 0: chosen.extend(candidates[:needed])
        
    return [(c["img"], c["mask"]) for c in chosen]

# ==========================================================================
# PART 3: MedSAM 与生成整套组图逻辑 (包含 01 到 06 的所有图片)
# ==========================================================================
class MedSAMRunner:
    def __init__(self, model_name="wanglab/medsam-vit-base", device="cuda:0"):
        from transformers import SamModel, SamProcessor
        print(f"[info] loading {model_name} -> {device}")
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model     = SamModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device    = device

    @staticmethod
    def get_core_points(mask_bin):
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > 20]
        if not valid: return None
        cores = []
        for c in valid:
            M = cv2.moments(c)
            if M["m00"] != 0: cores.append([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])])
            else: cores.append(c[0][0].tolist())
        return cores

    @staticmethod
    def make_K_prompts(cores, K, perturb=5):
        rng = np.random.RandomState(SEED)
        prompts = []
        for _ in range(K):
            pts, lbs = [], []
            for cx, cy in cores:
                ox, oy = rng.randint(-perturb, perturb + 1), rng.randint(-perturb, perturb + 1)
                pts.append([max(0, cx + ox), max(0, cy + oy)]); lbs.append(1)
            prompts.append((pts, lbs))
        return prompts

    @torch.no_grad()
    def run_K(self, img_rgb, mask_bin, K=12, perturb=5):
        cores = self.get_core_points(mask_bin)
        if cores is None: raise RuntimeError("No valid lesion contour found.")
        prompts_K = self.make_K_prompts(cores, K=K, perturb=perturb)
        if img_rgb.ndim == 2: img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        preds = []
        for pts, lbs in prompts_K:
            inputs = self.processor(img_rgb, input_points=[pts], input_labels=[lbs], return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            mask_out = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )[0][0][0].numpy()
            preds.append(mask_out.astype(np.uint8))
        return np.stack(preds, axis=0), prompts_K

def compute_uncertainty_step1_style(preds_K):
    var_map = np.var(preds_K.astype(np.float32), axis=0)
    U = cv2.GaussianBlur(var_map, (3, 3), 0)
    U_bar = cv2.GaussianBlur(var_map, (7, 7), 2.0)
    U_bar = (U_bar - U_bar.min()) / (U_bar.max() - U_bar.min() + 1e-6)
    return U, U_bar

def load_saved_weight(busi_root, img_path):
    p = os.path.join(busi_root, "uncertainty_weights", os.path.splitext(os.path.basename(img_path))[0] + "_weight.png")
    if not os.path.exists(p): return None
    return np.array(PILImage.open(p).convert("L"), dtype=np.float32) / 255.0

def load_seg_model(seg_ckpt_path, device, blocks=2):
    import inspect
    sys.path.insert(0, ".")
    state = torch.load(seg_ckpt_path, map_location=device, weights_only=True)
    state = state.get("state_dict", state)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    n_classes = 2
    model = MobileUVitUNet_Ablation(mode='fusion', in_channels=3, n_classes=n_classes, num_bottleneck_blocks=blocks)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    model._n_classes = n_classes
    return model

@torch.no_grad()
def run_seg(seg_model, img_rgb, target_size=256, device="cuda:0"):
    H, W = img_rgb.shape[:2]
    inp = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.
    t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
    out = seg_model(t)
    if isinstance(out, (list, tuple)): out = out[0]
    prob = torch.softmax(out, dim=1)[0, 1].cpu().numpy()
    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    return prob

def save_clean(arr, path, cmap=None, vmin=None, vmax=None):
    h, w = arr.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    if arr.ndim == 3: ax.imshow(arr)
    else: ax.imshow(arr, cmap=cmap or "gray", vmin=vmin, vmax=vmax)
    fig.savefig(path, dpi=200, bbox_inches=None, pad_inches=0)
    plt.close(fig)

def save_with_colorbar(arr, path, title, cmap="hot", vmin=0.0, vmax=1.0, cb_label=r"$\bar{U}(x)$", simple_legend=True):
    h, w = arr.shape[:2]
    aspect = w / max(h, 1)
    fig = plt.figure(figsize=(6.0, 6.0 / aspect), dpi=180, facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#BA7517"); sp.set_linewidth(1.8)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(im, cax=cax)
    if simple_legend:
        cb.set_ticks([vmin, vmax]); cb.ax.set_yticklabels(["Low", "High"], fontsize=10); cb.ax.tick_params(length=0) 
    else:
        cb.set_label(cb_label, fontsize=10); cb.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        cb.ax.set_yticklabels([f"{vmin:.1f}\n(Low)", f"{(vmin+vmax)/2:.2f}", f"{vmax:.1f}\n(High)"])
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def save_set(out_dir, img_rgb, gt_mask, preds_K, U, U_bar, U_bar_saved, Z, cmap="hot"):
    os.makedirs(out_dir, exist_ok=True)
    
    # 01_input_image
    save_clean(img_rgb, os.path.join(out_dir, "01_input_image.png"))
    
    # 02_ground_truth
    gt_vis = (gt_mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(out_dir, "02_ground_truth.png"), gt_vis)

    # 03_pred_K: 保存 12 次所有的扰动预测掩膜图
    for k in range(preds_K.shape[0]):
        cv2.imwrite(os.path.join(out_dir, f"03_pred_k{k+1:02d}.png"), (preds_K[k].astype(np.uint8) * 255))

    # 04_uncertainty_U: 纯方差计算图
    np.save(os.path.join(out_dir, "04_uncertainty_U.npy"), U.astype(np.float32))
    cv2.imwrite(os.path.join(out_dir, "04_uncertainty_U.png"), ((U / (float(U.max()) + 1e-6)) * 255).astype(np.uint8))

    # 05_uncertainty_Ubar_with_legend: 平滑+归一化的美化热图
    save_with_colorbar(U_bar, os.path.join(out_dir, "05_uncertainty_Ubar_with_legend.png"), title=r"Uncertainty map  $U(x)$" "\n" r"High = ambiguous boundary region", cmap=cmap)
    np.save(os.path.join(out_dir, "05_uncertainty_Ubar.npy"), U_bar.astype(np.float32))

    # 05b: 如果设置了 --use_saved_weight，则读旧数据
    if U_bar_saved is not None:
        save_with_colorbar(U_bar_saved, os.path.join(out_dir, "05b_uncertainty_Ubar_from_disk.png"), title=r"$\bar{U}$ from uncertainty_weights/*_weight.png", cmap=cmap)

    # 06_segmentation_Z: 概率热图
    if Z is not None:
        save_with_colorbar(Z, os.path.join(out_dir, "06_segmentation_Z.png"), title=r"Segmentation logits $Z$  (foreground probability)", cmap=cmap)
        cv2.imwrite(os.path.join(out_dir, "06b_segmentation_Z_binary.png"), ((Z > 0.5).astype(np.uint8) * 255))
        np.save(os.path.join(out_dir, "06_segmentation_Z.npy"), Z.astype(np.float32))

def run_visualization_pipeline(args, best_model_path):
    print(f"\n{'='*30}")
    print(f" [PHASE 2] STARTING VISUALIZATION")
    print(f" {'='*30}")
    
    seg_model = load_seg_model(best_model_path, args.device, blocks=args.train_blocks)
    samples = scan_top_samples(busi_root=args.busi_root, top_n=args.num_samples, force_image_name=args.force_image)

    if not samples:
        print("[error] No valid samples found.")
        return

    medsam = MedSAMRunner(model_name=args.medsam_name, device=args.device)

    for idx, (img_p, mask_p) in enumerate(samples):
        print(f"\n[Processing Sample {idx+1}/{len(samples)}]: {os.path.basename(img_p)}")
        img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask > 0).astype(np.uint8)

        # 运行 12 次 MedSAM
        preds_K, prompts_K = medsam.run_K(img, mask_bin, K=args.K, perturb=args.perturb_px)
        
        # 运行 U 计算
        U, U_bar = compute_uncertainty_step1_style(preds_K)

        U_bar_saved = None
        if args.use_saved_weight:
            U_bar_saved = load_saved_weight(args.busi_root, img_p)

        # 运行刚出炉的最新分割模型
        Z = run_seg(seg_model, img, device=args.device)

        # 生成所有文件
        sid = Path(img_p).stem
        set_dir = os.path.join(args.output_dir, f"set_{idx+1:02d}_{sid}")
        save_set(set_dir, img, mask_bin, preds_K, U, U_bar, U_bar_saved, Z, cmap=args.cmap)
        print(f"  -> Generated full set 01 to 06 successfully at: {set_dir}/")

    print("\n[done] All visualizations generated successfully!")


# ==========================================================================
# 流程入口
# ==========================================================================
def main():
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument("--busi_root", type=str, default="/data/xuten/Breast Ultrasound Datasets/BUSI")
    parser.add_argument("--save_dir_root", type=str, default="results_depth_ablation_busi_proposed")
    parser.add_argument("--output_dir", type=str, default="figures_sets")
    # 训练参数
    parser.add_argument("--skip_train", action="store_true", help="跳过训练，直接使用已有权重进行可视化")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_blocks", type=int, default=2, help="bottleneck blocks 的数量")
    # 可视化参数
    parser.add_argument("--medsam_name", type=str, default="wanglab/medsam-vit-base")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--K", type=int, default=12)
    parser.add_argument("--perturb_px", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_saved_weight", action="store_true")
    parser.add_argument("--cmap", type=str, default="hot")
    parser.add_argument("--force_image", type=str, default="benign (342).png")
    
    args = parser.parse_args()

    # 1. 训练阶段
    if not args.skip_train:
        best_ckpt = run_training_pipeline(args)
    else:
        best_ckpt = os.path.join(args.save_dir_root, f"fusion_{args.train_blocks}blocks_BUSI_Proposed", "best_model.pth")
        if not os.path.exists(best_ckpt):
            print(f"[ERROR] You chose --skip_train, but {best_ckpt} does not exist!")
            sys.exit(1)
            
    # 2. 可视化阶段
    run_visualization_pipeline(args, best_ckpt)


if __name__ == "__main__":
    main()