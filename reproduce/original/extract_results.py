import os
import re
import pandas as pd

def extract_from_logs():
    results_summary = []
    # 扫描当前目录下所有的训练日志
    log_files = [f for f in os.listdir('.') if f.startswith('train_log_') and f.endswith('.txt')]
    
    print(f"🔍 正在扫描 {len(log_files)} 个实验日志...")

    for log_name in log_files:
        # 1. 检查是否跑完 100 Epoch
        is_finished = False
        with open(log_name, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if "Ep 100" in content:
                is_finished = True
        
        # 解析模型名、数据集和种子 (从文件名 train_log_step3_train_unet_qata_v1_seed123.txt 提取)
        parts = log_name.replace('train_log_step3_train_', '').replace('.txt', '').split('_seed')
        model_setting = parts[0]
        seed = parts[1] if len(parts) > 1 else "42"
        
        # 2. 确定结果文件夹路径
        # 结果文件夹通常是 results_模型名_数据集_seedXXX
        # 注意：这里需要匹配你脚本中 RESULTS_DIR 的命名习惯
        results_dir = None
        for d in os.listdir('.'):
            if os.path.isdir(d) and model_setting in d.lower() and f"seed{seed}" in d:
                results_dir = d
                break
        
        data = {
            "Experiment": model_setting,
            "Seed": seed,
            "Status": "✅ Finished" if is_finished else "❌ Incomplete"
        }

        # 3. 如果跑完了，提取具体数值
        if is_finished and results_dir:
            csv_path = None
            # 优先找 CSV 文件
            for f in os.listdir(results_dir):
                if f.endswith('.csv'):
                    csv_path = os.path.join(results_dir, f)
                    break
            
            if csv_path:
                try:
                    df_res = pd.read_csv(csv_path)
                    # 提取指标并格式化为 6 位小数
                    for col in ['mIoU', 'mF1', 'ASSD', 'HD95']:
                        if col in df_res.columns:
                            val = float(df_res[col].iloc[0])
                            data[col] = f"{val:.6f}"
                except:
                    pass
            
            # 如果 CSV 没读到，尝试读 final_report.txt
            report_path = os.path.join(results_dir, "final_report.txt")
            if 'mIoU' not in data and os.path.exists(report_path):
                with open(report_path, 'r') as rf:
                    r_content = rf.read()
                    for col in ['mIoU', 'mF1', 'ASSD', 'HD95']:
                        match = re.search(f"{col}:\\s*([0-9.]+)", r_content)
                        if match:
                            data[col] = f"{float(match.group(1)):.6f}"

        results_summary.append(data)

    # 保存总表
    final_df = pd.DataFrame(results_summary)
    final_df.to_csv("overall_experiment_summary.csv", index=False)
    print(f"\n✨ 提取完成！总表已保存至: overall_experiment_summary.csv")
    
    # 在控制台打印已完成的预览
    finished_only = final_df[final_df['Status'] == "✅ Finished"]
    print(f"📊 已完成实验预览 (共 {len(finished_only)} 个):")
    print(finished_only.head(20))

if __name__ == "__main__":
    extract_from_logs()
    