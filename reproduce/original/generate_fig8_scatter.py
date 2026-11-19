"""
generate_fig8_scatter.py  — redesigned
对数横轴 + 精调标注 + 更清晰的视觉风格
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 数据（来自 Table 2）────────────────────────────────────────────────────
#  name          params  BUSI   BrEaST  QaTa-v2  QaTa-v1  lightweight
MODELS = [
    ("MedT",        1.4,  0.608, 0.781, 0.754, 0.710, True),
    ("U-KAN",       4.2,  0.657, 0.842, 0.781, 0.751, True),
    ("CMUNeXt-L",   8.3,  0.762, 0.833, 0.785, 0.754, True),
    ("UNet++",     26.9,  0.772, 0.821, 0.782, 0.751, False),
    ("U-Net",      34.5,  0.727, 0.846, 0.795, 0.766, False),
    ("DeepLabV3+", 39.6,  0.776, 0.532, 0.798, 0.778, False),
    ("TransUNet",  93.2,  0.773, 0.793, 0.796, 0.766, False),
    ("AnatoNet",  116.4,  0.791, 0.818, 0.781, 0.743, False),
    ("Ours",        3.9,  0.775, 0.842, 0.795, 0.768, True),
]

data = [(n, p, np.mean([d1,d2,d3,d4]), lw)
        for n, p, d1, d2, d3, d4, lw in MODELS]

# ── 颜色 & 样式 ───────────────────────────────────────────────────────────
C_LW   = "#E07B45"   # orange — lightweight
C_HW   = "#4A7C3F"   # dark green — heavyweight
C_OURS = "#4A3AAA"   # indigo — ours

# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F8F8")
ax.grid(True, which="both", color="white", linewidth=0.8, zorder=0)

# 对数横轴
ax.set_xscale("log")

# 轻量级分界线
ax.axvline(x=10, color="#999999", linestyle="--", linewidth=1.0,
           alpha=0.7, zorder=1)
ax.text(10.5, 0.697, "10M", fontsize=7.5, color="#888888", va="bottom")

# 绘制点
for name, params, mean_dice, lw in data:
    is_ours = (name == "Ours")
    c  = C_OURS if is_ours else (C_LW if lw else C_HW)
    mk = "*"  if is_ours else ("o" if lw else "s")
    sz = 220  if is_ours else 80
    zo = 5    if is_ours else 3
    ax.scatter(params, mean_dice, s=sz, c=c, marker=mk,
               zorder=zo, edgecolors="white", linewidths=0.8)

# ── 标注（手动精调，避免重叠）────────────────────────────────────────────
# (dx 是 log 轴倍数偏移，dy 是 Dice 偏移)
label_cfg = {
    #  name           ha       va       dx_log   dy
    "MedT":       ("center", "top",    1.0,   -0.010),
    "U-KAN":      ("left",   "bottom", 1.25,   0.004),
    "CMUNeXt-L":  ("left",   "bottom", 1.20,   0.003),
    "UNet++":     ("left",   "top",    1.20,  -0.007),
    "U-Net":      ("left",   "bottom", 1.20,   0.003),
    "DeepLabV3+": ("left",   "bottom", 1.20,   0.003),
    "TransUNet":  ("right",  "bottom", 0.80,   0.003),
    "AnatoNet":   ("left",   "top",    1.15,  -0.007),
    "Ours":       ("right",  "bottom", 0.72,   0.004),
}

for name, params, mean_dice, lw in data:
    ha, va, dx_log, dy = label_cfg[name]
    is_ours = (name == "Ours")
    ax.annotate(
        name,
        xy=(params, mean_dice),
        xytext=(params * dx_log, mean_dice + dy),
        fontsize=8.0,
        ha=ha, va=va,
        color=C_OURS if is_ours else "#222222",
        fontweight="bold" if is_ours else "normal",
        arrowprops=dict(arrowstyle="-", color="#aaaaaa",
                        lw=0.5) if abs(dx_log-1.0)>0.05 or abs(dy)>0.006 else None,
    )

# ── Axes ──────────────────────────────────────────────────────────────────
ax.set_xlabel("Number of parameters (M)", fontsize=10)
ax.set_ylabel("Mean Dice (avg. over 4 datasets)", fontsize=10)
ax.set_xlim(0.8, 200)
ax.set_ylim(0.693, 0.815)

# 自定义 x 刻度
ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"], fontsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")

# ── Legend ────────────────────────────────────────────────────────────────
legend_elems = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_LW,
           markersize=8, label="Lightweight (<10M)"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=C_HW,
           markersize=8, label="Heavyweight (≥10M)"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor=C_OURS,
           markersize=12, label="Ours"),
]
ax.legend(handles=legend_elems, fontsize=8.5, frameon=True,
          framealpha=0.9, edgecolor="#cccccc",
          loc="lower right", bbox_to_anchor=(0.99, 0.02))

plt.tight_layout()
pdf = os.path.join(OUTPUT_DIR, "fig8_scatter.pdf")
png = os.path.join(OUTPUT_DIR, "fig8_scatter.png")
fig.savefig(pdf, bbox_inches="tight", dpi=300)
fig.savefig(png, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved:\n  {pdf}\n  {png}")

# summary
print("\nModel summary (sorted by params):")
print(f"{'Model':<14} {'Params':>8} {'MeanDice':>10}")
print("-" * 34)
for name, params, mean_dice, lw in sorted(data, key=lambda x: x[1]):
    print(f"{'†' if lw else ' '}{name:<13} {params:>7.1f}M {mean_dice:>10.3f}")