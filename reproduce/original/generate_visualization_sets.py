"""
generate_visualization_sets.py
==============================
更新版：
  06_segmentation_Z.png      -- 概率热图 + colorbar (smooth gradient)
  06b_final_segmentation.png -- 最终二值分割结果 (黑底白前景, 与 GT 同格式)
"""

from __future__ import annotations
import os, sys, re, random, argparse
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image as PILImage
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ==========================================================================
# 1. 挑样本
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
    seen_stems = set()

    print(f"[info] Scanning BUSI dataset for unique samples...")
    for root, dirs, files in os.walk(busi_root):
        root_l = root.lower()
        if any(k in root_l for k in ["mask", "label", "weight"]): continue
        for f in sorted(files):
            if not f.endswith(".png") or f.startswith("."): continue
            stem = os.path.splitext(f)[0]
            if stem in seen_stems: continue
            is_forced = (force_image_name is not None
                         and f.lower() == force_image_name.lower())
            if not is_forced and "normal" in f.lower(): continue

            img_path = os.path.join(root, f)
            mask_path = None
            for mroot, _, mfiles in os.walk(busi_root):
                if "mask" not in mroot.lower() and "label" not in mroot.lower():
                    continue
                for mf in mfiles:
                    if stem.lower() in mf.lower():
                        mask_path = os.path.join(mroot, mf); break
                if mask_path: break

            umap_path = os.path.join(weights_dir, stem + "_weight.png")
            if mask_path is None or not os.path.exists(umap_path): continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: continue
            mask_r = cv2.resize(mask, (256, 256))
            mask_bin = (mask_r > 0)
            ratio = float(mask_bin.mean())
            if not is_forced and (ratio < 0.06 or ratio > 0.40): continue

            umap = np.array(PILImage.open(umap_path).convert("L"), dtype=np.float32)
            umap_r = np.array(PILImage.fromarray(umap.astype(np.uint8))
                              .resize((256, 256), PILImage.LANCZOS),
                              dtype=np.float32)
            umap_sm = cv2.GaussianBlur(umap_r, (7, 7), 2.0)
            umap_sm = (umap_sm - umap_sm.min()) / (umap_sm.max() - umap_sm.min() + 1e-6)

            size_score = max(0.0, 1.0 - abs(ratio - 0.20) / 0.20)
            bu_score = boundary_uncertainty_score(mask_bin, umap_sm)
            peak_score = float(umap_sm.max())
            combined = size_score * max(0.0, bu_score) * peak_score

            cand_info = {"score": combined, "img": img_path, "mask": mask_path,
                         "ratio": ratio, "bu": bu_score, "peak": peak_score, "stem": stem}
            if is_forced:
                cand_info["score"] = float("inf"); forced_candidate = cand_info
            else:
                candidates.append(cand_info)
            seen_stems.add(stem)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    chosen = []
    if forced_candidate:
        print(f"[info] Forced Inclusion: {os.path.basename(forced_candidate['img'])}")
        chosen.append(forced_candidate)
    needed = top_n - len(chosen)
    if needed > 0: chosen.extend(candidates[:needed])
    print(f"\n[info] Final picked {len(chosen)} unique samples.")
    return [(c["img"], c["mask"]) for c in chosen]


# ==========================================================================
# 2.  Inference (ImageNet 标准化)
# ==========================================================================
@torch.no_grad()
def run_seg(seg_model, img_rgb, target_size=256, device="cuda:0"):
    H, W = img_rgb.shape[:2]
    inp = cv2.resize(img_rgb, (target_size, target_size),
                     interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
    out = seg_model(t)
    if isinstance(out, (list, tuple)): out = out[0]
    prob = torch.softmax(out, dim=1)[0, 1].cpu().numpy()
    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    return prob


# ==========================================================================
# 3. MedSAM, U 计算, ckpt 加载
# ==========================================================================
class MedSAMRunner:
    def __init__(self, model_name="wanglab/medsam-vit-base", device="cuda:0"):
        from transformers import SamModel, SamProcessor
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    @staticmethod
    def get_core_points(mask_bin):
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > 20]
        if not valid: return None
        cores = []
        for c in valid:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cores.append([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])])
            else:
                cores.append(c[0][0].tolist())
        return cores

    @torch.no_grad()
    def run_K(self, img_rgb, mask_bin, K=12, perturb=5):
        cores = self.get_core_points(mask_bin)
        rng = np.random.RandomState(SEED)
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        preds = []
        for _ in range(K):
            pts, lbs = [], []
            for cx, cy in cores:
                pts.append([max(0, cx + rng.randint(-perturb, perturb+1)),
                            max(0, cy + rng.randint(-perturb, perturb+1))])
                lbs.append(1)
            inputs = self.processor(img_rgb, input_points=[pts],
                                     input_labels=[lbs], return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            mask_out = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0][0][0].numpy()
            preds.append(mask_out.astype(np.uint8))
        return np.stack(preds, axis=0)


def compute_uncertainty_step1_style(preds_K):
    var_map = np.var(preds_K.astype(np.float32), axis=0)
    U = cv2.GaussianBlur(var_map, (3, 3), 0)
    U_bar = cv2.GaussianBlur(var_map, (7, 7), 2.0)
    U_bar = (U_bar - U_bar.min()) / (U_bar.max() - U_bar.min() + 1e-6)
    return U, U_bar


def load_seg_model(seg_ckpt_path, device):
    state = torch.load(seg_ckpt_path, map_location=device, weights_only=True)
    state = state.get("state_dict", state)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    from small_unet_ablation import MobileUVitUNet_Ablation
    model = MobileUVitUNet_Ablation(mode="fusion", in_channels=3,
                                     n_classes=2, num_bottleneck_blocks=2).to(device)
    model.load_state_dict(state, strict=False)
    model.eval(); model._n_classes = 2
    return model


# ==========================================================================
# 4.  画图工具
# ==========================================================================
def save_with_colorbar(arr, path, title, cmap="hot"):
    h, w = arr.shape[:2]
    fig = plt.figure(figsize=(6.0, 6.0 / (w/h)), dpi=180, facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#BA7517"); sp.set_linewidth(1.8)
    divider = make_axes_locatable(ax)
    cb = fig.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.1))
    cb.set_ticks([0, 1]); cb.ax.set_yticklabels(["Low", "High"], fontsize=9)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def save_segmentation_binary(pred_bin, path):
    """
    最终分割结果: 纯黑白二值 mask, 黑底白前景.
    跟 02_ground_truth.png 同样的格式, 方便对比.
    """
    out = (pred_bin > 0).astype(np.uint8) * 255
    cv2.imwrite(path, out)


# ==========================================================================
# 5.  Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--busi_root", type=str,
                        default="/data/xuten/Breast Ultrasound Datasets/BUSI")
    parser.add_argument("--seg_ckpt", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="figures_sets")
    parser.add_argument("--force_image", type=str, default="benign (342).png")
    parser.add_argument("--seg_threshold", type=float, default=0.5,
                        help="二值化 Z 的阈值, 用于 06b/06c")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    samples = scan_top_samples(args.busi_root, top_n=args.num_samples,
                                force_image_name=args.force_image)

    seg_model = load_seg_model(args.seg_ckpt, device)
    medsam = MedSAMRunner(device=device)

    for idx, (img_p, mask_p) in enumerate(samples):
        img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        mask_bin = (cv2.imread(mask_p, 0) > 0).astype(np.uint8)

        preds_K = medsam.run_K(img, mask_bin)
        U, U_bar = compute_uncertainty_step1_style(preds_K)
        Z = run_seg(seg_model, img, device=device)
        Z_bin = (Z > args.seg_threshold).astype(np.uint8)

        out_dir = Path(args.output_dir) / f"set_{idx+1:02d}_{Path(img_p).stem}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 01 原图
        cv2.imwrite(str(out_dir/"01_input_image.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # 02 GT
        cv2.imwrite(str(out_dir/"02_ground_truth.png"), mask_bin * 255)
        # 03 12 个 prompt-based 预测
        for k in range(preds_K.shape[0]):
            cv2.imwrite(str(out_dir/f"03_pred_k{k+1:02d}.png"), preds_K[k]*255)
        # 04 U raw
        cv2.imwrite(str(out_dir/"04_uncertainty_U.png"),
                    (U / (U.max()+1e-6) * 255).astype(np.uint8))
        # 05 U_bar
        save_with_colorbar(U_bar, str(out_dir/"05_uncertainty_Ubar_with_legend.png"),
                            r"Uncertainty map  $\bar{U}$")
        # 06 Z 概率热图
        save_with_colorbar(Z, str(out_dir/"06_segmentation_Z.png"),
                            r"Segmentation logits  $Z$")
        # 06b 最终分割结果 (纯黑白二值, 跟 02_ground_truth.png 同格式)
        save_segmentation_binary(Z_bin, str(out_dir/"06b_final_segmentation.png"))

        np.save(out_dir/"05_uncertainty_Ubar.npy", U_bar.astype(np.float32))
        np.save(out_dir/"06_segmentation_Z.npy",   Z.astype(np.float32))
        print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()