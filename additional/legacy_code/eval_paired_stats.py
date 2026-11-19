# eval_paired_stats.py
# -----------------------------------------------------------
# Paired evaluation of two segmentation models (ResNet vs SwinLayer4):
# - Per-image IoU/F1, HD95, ASSD (symmetric)
# - Mean ± std across 3–5 seeds
# - Paired Wilcoxon signed-rank test on per-image IoU/HD95/ASSD
# - (optional) Paired t-test if normal
# - Bootstrap CI (10k) for mean ΔMetric
# - Histogram of ΔIoU and counts of improve/decline/tie
# -----------------------------------------------------------

import os, glob, argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# Project imports
from dataset_new import BreastCancerSegmentation
from models.uresnet import UResNet
# from models.uresnet_resenc_swin_layer4 import UResNet_ResEnc_SwinLayer4
from models.uresnet_resenc_dualswin_enc45_512 import UResNet_ResEnc_DualSwinEnc45_512
# Optional stats dependencies
try:
    from scipy.stats import wilcoxon, ttest_rel, shapiro
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    SCIPY_ND_OK = True
except Exception:
    SCIPY_ND_OK = False


# -----------------------------
# Metric helpers
# -----------------------------
def torch_argmax2d_logits(logits_2ch):
    # logits: (B,2,H,W) -> (B,H,W) in {0,1}
    return torch.argmax(logits_2ch, dim=1)

def compute_confusion(pred_bin, gt_bin):
    tp = (pred_bin & gt_bin).sum().item()
    fp = (pred_bin & (~gt_bin)).sum().item()
    fn = ((~pred_bin) & gt_bin).sum().item()
    tn = ((~pred_bin) & (~gt_bin)).sum().item()
    return tp, fp, fn, tn

def iou_from_conf(tp, fp, fn, eps=1e-6):
    denom = tp + fp + fn
    return (tp / (denom + eps)) if denom > 0 else 1.0  # empty-empty -> IoU=1

def f1_from_conf(tp, fp, fn, eps=1e-6):
    denom = 2*tp + fp + fn
    return (2*tp / (denom + eps)) if denom > 0 else 1.0

# ---- Surface distances (needs SciPy) ----
def boundary_mask(bin_mask):
    er = binary_erosion(bin_mask, structure=np.ones((3,3), dtype=bool), border_value=0)
    return bin_mask ^ er

def sym_surface_distances(bin_a, bin_b, spacing=(1.0,1.0)):
    if not SCIPY_ND_OK:
        return None
    a = bin_a.astype(bool)
    b = bin_b.astype(bool)
    surf_a = boundary_mask(a)
    surf_b = boundary_mask(b)
    if not surf_a.any() and not surf_b.any():
        return np.array([0.0])  # identical empty surfaces
    dt_b = distance_transform_edt(~surf_b, sampling=spacing)
    dt_a = distance_transform_edt(~surf_a, sampling=spacing)
    dists_a_to_b = dt_b[surf_a]
    dists_b_to_a = dt_a[surf_b]
    if dists_a_to_b.size == 0: dists_a_to_b = np.array([np.inf])
    if dists_b_to_a.size == 0: dists_b_to_a = np.array([np.inf])
    return np.concatenate([dists_a_to_b, dists_b_to_a])

def hd95_and_assd(bin_pred, bin_gt, spacing=(1.0,1.0)):
    if not SCIPY_ND_OK:
        return (np.nan, np.nan)
    dists = sym_surface_distances(bin_pred, bin_gt, spacing=spacing)
    if dists is None or not np.isfinite(dists).any():
        return (np.nan, np.nan)
    finite = np.isfinite(dists)
    hd95 = np.percentile(dists[finite], 95) if finite.any() else np.nan
    assd = np.mean(dists[finite]) if finite.any() else np.nan
    return (hd95, assd)


# -----------------------------
# Evaluation per model
# -----------------------------
@torch.no_grad()
def eval_one_model(model, loader, device, compute_hd_assd=True, spacing=(1.0,1.0)):
    model.eval()
    iou_list, f1_list, hd95_list, assd_list = [], [], [], []

    for inputs, masks in tqdm(loader, desc="Eval"):
        inputs = inputs.to(device)
        masks  = masks.to(device)  # (B,H,W) in {0,1}

        logits = model(inputs)     # (B,2,H,W)
        pred = torch_argmax2d_logits(logits)  # (B,H,W)

        for b in range(pred.shape[0]):
            p = pred[b].bool().cpu()
            g = masks[b].bool().cpu()

            tp, fp, fn, _ = compute_confusion(p, g)
            iou_list.append(iou_from_conf(tp, fp, fn))
            f1_list.append(f1_from_conf(tp, fp, fn))

            if compute_hd_assd:
                hd95, assd = hd95_and_assd(p.numpy(), g.numpy(), spacing=spacing)
                hd95_list.append(hd95)
                assd_list.append(assd)

    out = {"iou": np.array(iou_list, dtype=float), "f1": np.array(f1_list, dtype=float)}
    if compute_hd_assd:
        out["hd95"] = np.array(hd95_list, dtype=float)
        out["assd"] = np.array(assd_list, dtype=float)
    return out


# -----------------------------
# Bootstrap CI
# -----------------------------
def bootstrap_ci_mean(delta, B=10000, ci=(2.5, 97.5), seed=0):
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, n)
        means[b] = delta[idx].mean()
    lo, hi = np.percentile(means, ci)
    return delta.mean(), (lo, hi)


# -----------------------------
# Pretty paired stats printer
# -----------------------------
def paired_stats_block(name, a, b, lower_is_better=False):
    """
    name: metric name
    a: np.ndarray for ResNet
    b: np.ndarray for Swin
    lower_is_better: for distances (HD95/ASSD) set True; for IoU/F1 set False
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size == 0:
        print(f"\n=== {name} ===\nNo finite values found.")
        return

    d = b - a  # Swin - ResNet
    direction = "lower" if lower_is_better else "higher"
    print(f"\n=== {name} across all (image,seed) pairs ===")
    print(f"ResNet: mean ± std = {a.mean():.4f} ± {a.std(ddof=1):.4f}")
    print(f"Swin  : mean ± std = {b.mean():.4f} ± {b.std(ddof=1):.4f}")
    print(f"Δ{name} (Swin-ResNet): mean = {d.mean():.6f}  "
          f"(negative is better for {name} if {direction} is better)")

    if SCIPY_OK:
        try:
            W,p = shapiro(d)
            print(f"Shapiro-Wilk on Δ{name}: W={W:.4f}, p={p:.4g}")
        except Exception:
            pass
        stat,p = wilcoxon(d, zero_method='wilcox')
        print(f"Wilcoxon signed-rank on Δ{name}: stat={stat:.4f}, p={p:.4g}")
        t,p = ttest_rel(b, a)
        print(f"Paired t-test on {name}: t={t:.4f}, p={p:.4g}")
    else:
        print("SciPy not available → skipping normality/Wilcoxon/t-test.")

    md, (lo, hi) = bootstrap_ci_mean(d, B=10000, ci=(2.5, 97.5), seed=0)
    print(f"Bootstrap 95% CI for mean Δ{name}: {md:.6f} [{lo:.6f}, {hi:.6f}]")
    if lower_is_better:
        print(f"Improve (Δ<0): {(d<0).sum()}, Decline (Δ>0): {(d>0).sum()}, "
              f"Tie: {(d==0).sum()} (out of {d.size})")
    else:
        print(f"Improve (Δ>0): {(d>0).sum()}, Decline (Δ<0): {(d<0).sum()}, "
              f"Tie: {(d==0).sum()} (out of {d.size})")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--img_h", type=int, default=256)
    ap.add_argument("--img_w", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--resnet_ckpts", type=str, required=True, help='Glob for ResNet checkpoints')
    ap.add_argument("--swin_ckpts", type=str, required=True, help='Glob for Swin checkpoints')
    ap.add_argument("--spacing_y", type=float, default=1.0, help="Pixel spacing in mm (height)")
    ap.add_argument("--spacing_x", type=float, default=1.0, help="Pixel spacing in mm (width)")
    ap.add_argument("--no_surface", action="store_true", help="Disable HD95/ASSD if set")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_csv", type=str, default="paired_metrics.csv")
    ap.add_argument("--out_fig", type=str, default="delta_iou_hist.png")
    args = ap.parse_args()

    print(f"[DEBUG] no_surface flag: {args.no_surface}")
    print(f"[DEBUG] SCIPY_OK={SCIPY_OK}, SCIPY_ND_OK={SCIPY_ND_OK}")
    compute_surface = (not args.no_surface) and SCIPY_ND_OK
    print(f"[DEBUG] Will compute HD95/ASSD: {compute_surface}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print("Using device:", device)

    # Dataset / loader
    test_pairs_df = pd.read_csv(args.test_csv)
    test_pairs = list(zip(
        [os.path.join(args.data_root, "Images", p) for p in test_pairs_df["Image"].tolist()],
        [os.path.join(args.data_root, "Labels", p) for p in test_pairs_df["Mask"].tolist()],
    ))
    test_dataset = BreastCancerSegmentation(test_pairs, args.img_h, args.img_w, is_augment=False)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Checkpoints
    resnet_paths = sorted(glob.glob(args.resnet_ckpts))
    swin_paths   = sorted(glob.glob(args.swin_ckpts))
    assert len(resnet_paths) == len(swin_paths) and len(resnet_paths) > 0, \
        f"Expected same nonzero count for ckpts, got {len(resnet_paths)} vs {len(swin_paths)}"
    print("ResNet ckpts:", resnet_paths)
    print("Swin   ckpts:", swin_paths)
    n_seeds = len(resnet_paths)

    # Containers
    all_rows = []
    per_image_iou_resnet, per_image_iou_swin = [], []

    # Evaluate each seed pair
    for s_idx, (rpth, spth) in enumerate(zip(resnet_paths, swin_paths)):
        print(f"\n=== Seed {s_idx+1}/{n_seeds} ===")

        m_res = UResNet(n_classes=2, pretrained=False).to(device)
        m_swn = UResNet_ResEnc_DualSwinEnc45_512(
            n_classes=2, input_size=(args.img_h, args.img_w),
            ws=4, heads=16, mlp_ratio=4.0,
            dropout=0.0, attn_dropout=0.0, drop_path_prob=0.0,
            dec_norm='gn'
        ).to(device)

        m_res.load_state_dict(torch.load(rpth, map_location=device))
        m_swn.load_state_dict(torch.load(spth, map_location=device))

        metrics_res = eval_one_model(m_res, test_loader, device,
                                     compute_hd_assd=compute_surface,
                                     spacing=(args.spacing_y, args.spacing_x))
        metrics_swn = eval_one_model(m_swn, test_loader, device,
                                     compute_hd_assd=compute_surface,
                                     spacing=(args.spacing_y, args.spacing_x))

        n_imgs = len(metrics_res["iou"])
        for i in range(n_imgs):
            row = {
                "seed_idx": s_idx,
                "img_idx": i,
                "iou_resnet": float(metrics_res["iou"][i]),
                "iou_swin":   float(metrics_swn["iou"][i]),
                "f1_resnet":  float(metrics_res["f1"][i]),
                "f1_swin":    float(metrics_swn["f1"][i]),
            }
            if compute_surface and "hd95" in metrics_res and "hd95" in metrics_swn:
                row.update({
                    "hd95_resnet": float(metrics_res["hd95"][i]),
                    "hd95_swin":   float(metrics_swn["hd95"][i]),
                    "assd_resnet": float(metrics_res["assd"][i]),
                    "assd_swin":   float(metrics_swn["assd"][i]),
                })
            all_rows.append(row)

        per_image_iou_resnet.append(metrics_res["iou"])
        per_image_iou_swin.append(metrics_swn["iou"])

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved per-image paired metrics to: {args.out_csv}")

    # Aggregate IoU
    iou_res = np.concatenate(per_image_iou_resnet, axis=0)
    iou_swn = np.concatenate(per_image_iou_swin,   axis=0)
    delta   = iou_swn - iou_res

    mean_res = iou_res.mean(); std_res = iou_res.std(ddof=1)
    mean_swn = iou_swn.mean(); std_swn = iou_swn.std(ddof=1)

    print("\n=== IoU across all (image,seed) pairs ===")
    print(f"ResNet: mean ± std = {mean_res:.4f} ± {std_res:.4f}")
    print(f"Swin  : mean ± std = {mean_swn:.4f} ± {std_swn:.4f}")
    print(f"ΔIoU (Swin-ResNet): mean = {delta.mean():.4f}")

    if SCIPY_OK:
        try:
            w_stat, w_p = shapiro(delta)
            print(f"Shapiro-Wilk on ΔIoU: W={w_stat:.4f}, p={w_p:.4g} (p<0.05 => non-normal)")
        except Exception:
            pass
        stat_w, p_w = wilcoxon(delta, alternative='two-sided', zero_method='wilcox')
        print(f"Wilcoxon signed-rank on ΔIoU: stat={stat_w:.4f}, p={p_w:.4g}")
        stat_t, p_t = ttest_rel(iou_swn, iou_res)
        print(f"Paired t-test on IoU: t={stat_t:.4f}, p={p_t:.4g}")
    else:
        print("SciPy not available -> skipping Shapiro/Wilcoxon/t-test. Install: pip install scipy")

    mean_d, (ci_lo, ci_hi) = bootstrap_ci_mean(delta, B=10000, ci=(2.5, 97.5), seed=0)
    print(f"Bootstrap 95% CI for mean ΔIoU: {mean_d:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    n_improve = (delta > 0).sum(); n_decline = (delta < 0).sum(); n_tie = (delta == 0).sum()
    print(f"Improve: {n_improve}, Decline: {n_decline}, Tie: {n_tie} (out of {delta.size})")

    # Histogram
    plt.figure(figsize=(7,5))
    plt.hist(delta, bins=40)
    plt.xlabel("ΔIoU = IoU(Swin) − IoU(ResNet)")
    plt.ylabel("Count")
    plt.title("Histogram of ΔIoU across (image, seed)")
    plt.axvline(0.0, linestyle="--")
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=150)
    print(f"Saved histogram to: {args.out_fig}")

    # ---- Print HD95 / ASSD paired stats if present ----
    if compute_surface:
        cols = set(df.columns)
        if {"hd95_resnet","hd95_swin"} <= cols:
            paired_stats_block("HD95 (mm)", df["hd95_resnet"].values, df["hd95_swin"].values,
                               lower_is_better=True)
        else:
            print("HD95 columns not found in CSV; skipping HD95 summary.")
        if {"assd_resnet","assd_swin"} <= cols:
            paired_stats_block("ASSD (mm)", df["assd_resnet"].values, df["assd_swin"].values,
                               lower_is_better=True)
        else:
            print("ASSD columns not found in CSV; skipping ASSD summary.")


if __name__ == "__main__":
    main()
