"""
generate_fig_uncertainty_diagram.py — v5
三步流程：[原图 + 提示点] → [MedSAM] → [Uncertainty Map]
"""

import os, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from scipy import ndimage
import cv2
from PIL import Image as PILImage

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_BUSI_ROOT  = "/data/xuten/Breast Ultrasound Datasets/BUSI"
_WEIGHT_DIR = os.path.join(_BUSI_ROOT, "uncertainty_weights")

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# 改这个数字可以换样本（1=最佳，2=次佳，以此类推）
SAMPLE_RANK = 7

# ══════════════════════════════════════════════════════════════════════════
#  Sample selection: large lesion + boundary-focused uncertainty
# ══════════════════════════════════════════════════════════════════════════
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

def scan_best_sample():
    best = {"score": -1, "img": None, "mask": None, "umap": None}
    for root, dirs, files in os.walk(_BUSI_ROOT):
        root_l = root.lower()
        if any(k in root_l for k in ["mask","label","weight"]): continue
        for f in sorted(files):
            if not f.endswith(".png") or f.startswith("."): continue
            if "normal" in f.lower(): continue
            img_path  = os.path.join(root, f)
            stem      = os.path.splitext(f)[0]
            mask_path = None
            for mroot, _, mfiles in os.walk(_BUSI_ROOT):
                if "mask" not in mroot.lower() and "label" not in mroot.lower(): continue
                for mf in mfiles:
                    if stem.lower() in mf.lower():
                        mask_path = os.path.join(mroot, mf); break
                if mask_path: break
            umap_path = os.path.join(_WEIGHT_DIR, stem + "_weight.png")
            if mask_path is None or not os.path.exists(umap_path): continue
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: continue
            mask_r   = cv2.resize(mask, (256,256))
            mask_bin = (mask_r > 0)
            ratio    = float(mask_bin.mean())
            if ratio < 0.06 or ratio > 0.40: continue
            # load umap with high-quality resize
            umap = np.array(PILImage.open(umap_path).convert("L"), dtype=np.float32)
            umap_r = np.array(
                PILImage.fromarray(umap.astype(np.uint8))
                .resize((256,256), PILImage.LANCZOS), dtype=np.float32)
            umap_r = (umap_r - umap_r.min()) / (umap_r.max() - umap_r.min() + 1e-6)
            size_score = max(0.0, 1.0 - abs(ratio - 0.20) / 0.20)
            bu_score   = boundary_uncertainty_score(mask_bin, umap_r)
            combined   = size_score * max(0.0, bu_score)
            if combined > best["score"]:
                best = {"score": combined, "img": img_path,
                        "mask": mask_path, "umap": umap_path,
                        "ratio": ratio}
    return best

def scan_top_samples(top_n=10):
    """收集得分最高的 top_n 个样本，打印排名后返回列表"""
    candidates = []
    for root, dirs, files in os.walk(_BUSI_ROOT):
        root_l = root.lower()
        if any(k in root_l for k in ["mask","label","weight"]): continue
        for f in sorted(files):
            if not f.endswith(".png") or f.startswith("."): continue
            if "normal" in f.lower(): continue
            img_path  = os.path.join(root, f)
            stem      = os.path.splitext(f)[0]
            mask_path = None
            for mroot, _, mfiles in os.walk(_BUSI_ROOT):
                if "mask" not in mroot.lower() and "label" not in mroot.lower(): continue
                for mf in mfiles:
                    if stem.lower() in mf.lower():
                        mask_path = os.path.join(mroot, mf); break
                if mask_path: break
            umap_path = os.path.join(_WEIGHT_DIR, stem + "_weight.png")
            if mask_path is None or not os.path.exists(umap_path): continue
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: continue
            mask_r   = cv2.resize(mask, (256,256))
            mask_bin = (mask_r > 0)
            ratio    = float(mask_bin.mean())
            if ratio < 0.06 or ratio > 0.40: continue
            umap = np.array(PILImage.open(umap_path).convert("L"), dtype=np.float32)
            umap_r = np.array(
                PILImage.fromarray(umap.astype(np.uint8))
                .resize((256,256), PILImage.LANCZOS), dtype=np.float32)
            # extra smoothing to evaluate quality
            umap_sm = cv2.GaussianBlur(umap_r, (7,7), 2.0)
            umap_sm = (umap_sm - umap_sm.min()) / (umap_sm.max() - umap_sm.min() + 1e-6)
            size_score = max(0.0, 1.0 - abs(ratio - 0.20) / 0.20)
            bu_score   = boundary_uncertainty_score(mask_bin, umap_sm)
            # also reward high peak uncertainty (vivid heatmap)
            peak_score = float(umap_sm.max())
            combined   = size_score * max(0.0, bu_score) * peak_score
            candidates.append({
                "score": combined, "img": img_path,
                "mask": mask_path, "umap": umap_path,
                "ratio": ratio, "bu": bu_score, "peak": peak_score
            })
    candidates.sort(key=lambda x: x["score"], reverse=True)
    print(f"\nTop {min(top_n, len(candidates))} candidates:")
    for i, c in enumerate(candidates[:top_n]):
        print(f"  [{i+1}] score={c['score']:.4f} ratio={c['ratio']:.3f} "
              f"bu={c['bu']:.3f} peak={c['peak']:.3f}  {os.path.basename(c['img'])}")
    return candidates

# ══════════════════════════════════════════════════════════════════════════
#  Main draw — 3-panel: Image+points | SAM box | Uncertainty map
# ══════════════════════════════════════════════════════════════════════════
def draw():
    candidates = scan_top_samples(top_n=10)
    if not candidates:
        print("ERROR: no suitable sample found"); return
    idx = min(SAMPLE_RANK - 1, len(candidates) - 1)
    sample = candidates[idx]
    print(f"\nSelected rank {SAMPLE_RANK}: {sample['img']}")

    # load with LANCZOS for quality
    img  = cv2.cvtColor(cv2.imread(sample["img"]), cv2.COLOR_BGR2RGB)
    img  = np.array(PILImage.fromarray(img).resize((256,256), PILImage.LANCZOS))
    mask = cv2.imread(sample["mask"], cv2.IMREAD_GRAYSCALE)
    mask_bin = (cv2.resize(mask,(256,256)) > 0)
    umap_raw = np.array(PILImage.open(sample["umap"]).convert("L"), dtype=np.float32)
    umap = np.array(
        PILImage.fromarray(umap_raw.astype(np.uint8))
        .resize((256,256), PILImage.LANCZOS), dtype=np.float32)
    # additional smoothing for display
    umap = cv2.GaussianBlur(umap, (5,5), 1.5)
    umap = (umap - umap.min()) / (umap.max() - umap.min() + 1e-6)

    # centroid + K=5 perturbed points arranged visibly
    ys, xs = np.where(mask_bin)
    cy, cx = int(ys.mean()), int(xs.mean())

    K = 12
    np.random.seed(SEED+10)
    # K=12 prompt points, evenly spread around centroid for visual clarity
    # actual code: centroid ±5px; here we use ±15px radius so dots are distinguishable
    angles = np.linspace(0, 2*np.pi, K, endpoint=False) + 0.3
    radii  = np.random.randint(20, 36, K)
    prompt_pts = []
    for a, r in zip(angles, radii):
        px = int(np.clip(cx + r*np.cos(a), 4, 251))
        py = int(np.clip(cy + r*np.sin(a), 4, 251))
        if not mask_bin[py, px]:
            px = int((px + cx) / 2)
            py = int((py + cy) / 2)
        prompt_pts.append((px, py))

    pt_colors = [
        "#E8961A","#2090D0","#C02828","#28A840","#9B30C0",
        "#D4C010","#10B8B8","#E06090","#5060E0","#A06030",
        "#30D060","#C040D0"
    ]

    # ── figure: 7 cols [img | arr | SAM | arr | masks | arr | umap] ────────
    fig = plt.figure(figsize=(15, 4.0), facecolor="white")
    gs = fig.add_gridspec(
        2, 7,
        width_ratios=[3.0, 0.85, 2.2, 0.52, 3.2, 1.10, 3.0],
        height_ratios=[1, 1],
        wspace=0.05, hspace=0.08,
        left=0.01, right=0.99, top=0.90, bottom=0.05,
    )

    # ── panel 0: image + points ──────────────────────────────────────────
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img)
    # centroid: white ring
    ax_img.scatter(cx, cy, s=160, c="white", marker="o",
                   edgecolors="#222", linewidths=1.8, zorder=6)
    ax_img.scatter(cx, cy, s=30, c="#222", marker="o", zorder=7)
    # perturbed points + dashed spokes
    for k,(px,py) in enumerate(prompt_pts):
        ax_img.plot([cx,px],[cy,py], color=pt_colors[k],
                    lw=0.9, ls="--", alpha=0.65, zorder=4)
        ax_img.scatter(px, py, s=55, c=pt_colors[k], marker="o",
                       edgecolors="white", linewidths=0.9, zorder=5)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_edgecolor("#555"); sp.set_linewidth(1.2)
    ax_img.set_title("Input image  +  K prompt points\n"
                     r"(centroid $\pm$ 5 px  per pass)",
                     fontsize=8.5, pad=4)
    # compact legend
    handles = [mpatches.Patch(color="none", label="● Centroid (true)")]
    for k in range(4):
        handles.append(mpatches.Patch(color=pt_colors[k],
                                      label=f"Pass {k+1}"))
    handles.append(mpatches.Patch(color="none", label=f"···  ({K} total)"))
    ax_img.legend(handles=handles, fontsize=5.8, loc="lower right",
                  framealpha=0.82, edgecolor="none", handlelength=1.0,
                  handletextpad=0.4, borderpad=0.5)

    # ── panel 1: arrow img→SAM ───────────────────────────────────────────
    def arrow_ax(cell):
        ax = fig.add_subplot(cell)
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
        ax.annotate("", xy=(0.93,0.5), xytext=(0.07,0.5),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#444",
                                    lw=1.8, mutation_scale=15))
        return ax

    ax_a1 = arrow_ax(gs[:, 1])
    ax_a1.text(0.50, 0.54, f"K={K} passes", ha="center", va="bottom",
               fontsize=7.0, color="#333", transform=ax_a1.transAxes)

    # ── panel 2: MedSAM box ──────────────────────────────────────────────
    ax_sam = fig.add_subplot(gs[:, 2])
    ax_sam.set_xlim(0,1); ax_sam.set_ylim(0,1); ax_sam.axis("off")
    box = FancyBboxPatch((0.06,0.08), 0.88, 0.84,
                         boxstyle="round,pad=0.05",
                         lw=1.8, edgecolor="#534AB7", facecolor="#F2F1FD")
    ax_sam.add_patch(box)
    # Title
    ax_sam.text(0.50, 0.88, "MedSAM", ha="center", va="center",
                fontsize=11, fontweight="bold", color="#3C3489")
    ax_sam.text(0.50, 0.76, "ViT-B backbone  ·  frozen weights",
                ha="center", va="center", fontsize=6.8,
                color="#6358C0", style="italic")
    # Divider
    ax_sam.plot([0.12,0.88],[0.69,0.69], color="#C8C4EE", lw=0.9)
    # Four key points with bullet symbols, evenly spaced
    items = [
        ("K = 12", "stochastic forward passes"),
        ("Prompt", "lesion centroid  ±5 px"),
        (r"$\hat{Y}^{(j)}$", "binary mask at pass  $j$"),
        ("∇ = 0", "no gradient, inference only"),
    ]
    y_start = 0.60
    dy = 0.135
    for i, (key, val) in enumerate(items):
        y = y_start - i * dy
        ax_sam.text(0.18, y, key, ha="center", va="center",
                    fontsize=7.2, fontweight="bold", color="#3C3489")
        ax_sam.text(0.58, y, val, ha="center", va="center",
                    fontsize=6.5, color="#333")

    # ── panel 3: arrow SAM→masks ─────────────────────────────────────────
    ax_a2 = arrow_ax(gs[:, 3])
    ax_a2.text(0.50, 0.54, f"K masks", ha="center", va="bottom",
               fontsize=7.0, color="#333", transform=ax_a2.transAxes)

    # ── panel 4: K mask 2×2 grid ─────────────────────────────────────────
    from scipy import ndimage as _ndi
    def _perturb(mb, mode):
        if mode == 0:   m = _ndi.binary_dilation(mb, iterations=np.random.randint(3,7))
        elif mode == 1: m = _ndi.binary_erosion(mb,  iterations=np.random.randint(2,5))
        elif mode == 2:
            m = _ndi.binary_dilation(mb, iterations=2)
            sx,sy = np.random.randint(-10,11), np.random.randint(-10,11)
            m = _ndi.shift(m.astype(float),(sy,sx),mode="constant",cval=0)>0.5
        else:
            m = mb.copy(); noise = np.random.rand(*m.shape)<0.03; m = m^noise
        sx,sy = np.random.randint(-6,7), np.random.randint(-6,7)
        return _ndi.shift(m.astype(float),(sy,sx),mode="constant",cval=0)>0.5

    np.random.seed(SEED+2)
    masks_k = [_perturb(mask_bin, k) for k in range(4)]
    mask_face = ["#F0A060","#6AAEE0","#E07070","#70C870"]

    ax_col4 = fig.add_subplot(gs[:, 4])
    ax_col4.set_xlim(0,1); ax_col4.set_ylim(0,1); ax_col4.axis("off")
    ax_col4.set_title(f"Binary predictions  (4 of K={K} shown)",
                      fontsize=8.5, pad=4, color="#222")

    pos4 = [(0.03,0.52,0.45,0.44),(0.52,0.52,0.45,0.44),
            (0.03,0.04,0.45,0.44),(0.52,0.04,0.45,0.44)]
    for k,(x0,y0,w,h) in enumerate(pos4):
        ins = ax_col4.inset_axes([x0,y0,w,h])
        ins.imshow(img, alpha=0.65)
        rgba = np.zeros((*masks_k[k].shape, 4))
        rgba[masks_k[k]] = [*mcolors.to_rgb(mask_face[k]), 0.58]
        ins.imshow(rgba)
        px,py = prompt_pts[k % len(prompt_pts)]
        ins.scatter(px, py, s=35, c=pt_colors[k % len(pt_colors)],
                    marker="o", edgecolors="white", linewidths=0.7, zorder=5)
        ins.set_xticks([]); ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_edgecolor(pt_colors[k % len(pt_colors)]); sp.set_linewidth(1.4)
        ins.set_title(f"Pass {k+1}", fontsize=6.5, pad=2, color="#333")
        if k == 3:   # after last shown mask, add ellipsis below figure
            ax_col4.text(0.50, -0.04, f"···  ({K-4} more passes not shown)",
                         ha="center", va="top", fontsize=7.0, color="#888",
                         transform=ax_col4.transAxes)

    # ── panel 5: arrow masks→umap ─────────────────────────────────────────
    ax_a3 = arrow_ax(gs[:, 5])
    # formula above arrow (two lines for readability)
    ax_a3.text(0.50, 0.54,
               r"$U=\mathrm{Var}(\hat{Y}^{(1)},\ldots,\hat{Y}^{(K)})$",
               ha="center", va="bottom", fontsize=5.8, color="#222",
               transform=ax_a3.transAxes)
    ax_a3.text(0.50, 0.46,
               "smooth  +  normalise",
               ha="center", va="top", fontsize=5.6, color="#555",
               transform=ax_a3.transAxes)

    # ── panel 6: uncertainty map ─────────────────────────────────────────
    ax_umap = fig.add_subplot(gs[:, 6])
    im = ax_umap.imshow(umap, cmap="hot", vmin=0, vmax=1, interpolation="bilinear")
    ax_umap.set_xticks([]); ax_umap.set_yticks([])
    for sp in ax_umap.spines.values():
        sp.set_edgecolor("#BA7517"); sp.set_linewidth(1.8)
    ax_umap.set_title("Uncertainty map  $U(x)$\n"
                      "High = ambiguous boundary region",
                      fontsize=8.5, pad=4)
    cb = fig.colorbar(im, ax=ax_umap, fraction=0.046, pad=0.03)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"], fontsize=8)

    pdf = os.path.join(OUTPUT_DIR, "fig_uncertainty_diagram.pdf")
    png = os.path.join(OUTPUT_DIR, "fig_uncertainty_diagram.png")
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved:\n  {pdf}\n  {png}")

if __name__ == "__main__":
    draw()