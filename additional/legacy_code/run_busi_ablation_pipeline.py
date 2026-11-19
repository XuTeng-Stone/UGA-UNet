import os
import subprocess
import pandas as pd
import sys
import numpy as np

# ================= 配置区域 =================
TRAIN_SCRIPT = "step3_train_busi_anchor_metrics.py" 
SUMMARY_CSV = "/data/xuten/Code/results_busi_metrics/all_experiments_summary.csv"

# 固定参数
W_DICE = 1.0
W_CE = 1.0
EPOCHS = 100

# ================= 工具函数 =================
def run_command(alpha, lam_bg, margin, group_name):
    if not os.path.exists(TRAIN_SCRIPT):
        print(f"\n[FATAL ERROR] 找不到训练脚本: {TRAIN_SCRIPT}")
        sys.exit(1)

    print(f"\n>>> Running {group_name}: Alpha={alpha}, LamBG={lam_bg}, Margin={margin}")
    
    cmd = [
        "python", "-u", TRAIN_SCRIPT,
        "--alpha", str(alpha),
        "--lam_bg", str(lam_bg),
        "--margin", str(margin),
        "--w_dice", str(W_DICE),
        "--w_ce", str(W_CE),
        "--epochs", str(EPOCHS)
    ]
    
    log_file = f"log_busi_{group_name}_A{alpha}_L{lam_bg}_M{margin}.txt"
    
    try:
        with open(log_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f"    Finished. Log saved to {log_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] 训练脚本执行失败！退出代码: {e.returncode}")
        print(f"请立即查看日志文件: {log_file}")
        sys.exit(1) 

def get_best_param(param_name, fixed_conditions):
    if not os.path.exists(SUMMARY_CSV):
        print(f"\n[Error] 找不到结果汇总表: {SUMMARY_CSV}")
        sys.exit(1)
        
    df = pd.read_csv(SUMMARY_CSV)
    
    # 筛选
    filtered = df.copy()
    for col, val in fixed_conditions.items():
        filtered = filtered[np.isclose(filtered[col], val)]
        
    if filtered.empty:
        print(f"[Error] 找不到符合条件的数据: {fixed_conditions}")
        sys.exit(1)
        
    # [关键修改] 列名改为 mIoU
    best_row = filtered.loc[filtered['mIoU'].idxmax()]
    best_val = best_row[param_name]
    best_iou = best_row['mIoU']
    
    print(f"    [Analysis] Found Best {param_name} = {best_val} (mIoU: {best_iou:.4f})")
    return best_val

# ================= 主流程 =================
def main():
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    
    print("=== [BUSI] Stage 1: Tuning Alpha Max ===")
    # Group 1-4: alpha in [0.0, 0.1, 0.2, 0.5]
    stage1_configs = [0.0, 0.1, 0.2, 0.5]
    for i, alpha in enumerate(stage1_configs):
        run_command(alpha=alpha, lam_bg=0.25, margin=0.2, group_name=f"Group{i+1}")
        
    print(">>> Analyzing Stage 1 Results...")
    best_alpha = get_best_param("Alpha", {"Lam_BG": 0.25, "Margin": 0.2})
    
    print(f"\n=== [BUSI] Stage 2: Tuning LAM_BG (Fixed Alpha={best_alpha}) ===")
    # Group 5-7: lam_bg in [0.0, 0.25, 0.5]
    stage2_configs = [0.0, 0.25, 0.5]
    for i, lam in enumerate(stage2_configs):
        run_command(alpha=best_alpha, lam_bg=lam, margin=0.2, group_name=f"Group{i+5}")
        
    print(">>> Analyzing Stage 2 Results...")
    best_lam_bg = get_best_param("Lam_BG", {"Alpha": best_alpha, "Margin": 0.2})
    
    print(f"\n=== [BUSI] Stage 3: Tuning Margin (Fixed Alpha={best_alpha}, LamBG={best_lam_bg}) ===")
    # Group 8-10: margin in [0.1, 0.2, 0.3]
    # [关键] 包含 0.3，对应 Group 10
    stage3_configs = [0.1, 0.2, 0.3]
    for i, margin in enumerate(stage3_configs):
        run_command(alpha=best_alpha, lam_bg=best_lam_bg, margin=margin, group_name=f"Group{i+8}")
        
    print("\n=== All 10 Groups Finished! ===")
    print(f"Final Best Config: Alpha={best_alpha}, LamBG={best_lam_bg}")
    print(f"Check summary at: {SUMMARY_CSV}")

if __name__ == "__main__":
    main()