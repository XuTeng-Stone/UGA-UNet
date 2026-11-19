import os
import subprocess

# ================= 配置区域 =================
# 这里填入 UCLM 数据集对应的所有训练脚本
scripts_to_run = [
    "step3_train_deeplabv3_uclm_proposed.py",
    "step3_train_anato_uclm_proposed.py",
    "step3_train_cmunext_uclm_proposed.py",
    "step3_train_medt_uclm_proposed.py",
    "step3_train_transunet_uclm_proposed.py",
    "step3_train_unet_uclm_proposed.py",
    "step3_train_unetplusplus_uclm_proposed.py",
    "step3_train_ukan_uclm_proposed.py",
    "step3_train_depth_uclm.py"  # UCLM 的深度消融实验
]

def main():
    print("🚀 开始在一张显卡上串行运行所有 UCLM 模型...")
    
    for script in scripts_to_run:
        if not os.path.exists(script):
            print(f"⚠️ 警告: 找不到脚本 {script}，已跳过。请检查文件名是否拼写正确。")
            continue
            
        print(f"\n{'='*60}")
        print(f"▶ 正在运行: {script}")
        print(f"{'='*60}")
        
        # 为每个脚本单独生成一个明确的日志文件
        log_file = f"train_log_{script.replace('.py', '')}.txt"
        
        # 构造执行命令
        cmd = f"python -u {script} > {log_file} 2>&1"
        
        try:
            # shell=True 允许执行 bash 命令，check=True 保证出错时抛出异常
            # subprocess.run 是阻塞的，上一个没跑完绝对不会执行下一个
            subprocess.run(cmd, shell=True, check=True)
            print(f"✅ {script} 运行完成！日志已保存至: {log_file}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {script} 运行失败 (崩溃或显存溢出)！请检查日志: {log_file}")
            print("👉 将继续尝试运行队列中的下一个脚本...\n")
            # 如果您希望遇到报错就立刻停止整个流水线，请取消注释下面这行：
            # break 

    print("\n🎉 所有 UCLM 队列任务执行完毕！")

if __name__ == "__main__":
    main()