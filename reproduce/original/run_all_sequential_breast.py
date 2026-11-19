import os
import subprocess

# ================= 配置区域 =================
scripts_to_run = [
    "step3_train_deeplabv3_breast_proposed.py",
    "step3_train_anato_breast_proposed.py",
    "step3_train_cmunext_breast_proposed.py",
    "step3_train_medt_breast_proposed.py",
    "step3_train_transunet_breast_proposed.py",
    "step3_train_unet_breast_proposed.py",
    "step3_train_unetplusplus_breast_proposed.py",
    "step3_train_ukan_breast_proposed.py",
    "step3_train_depth_breast.py"  # BrEaST 的深度消融实验
]

def main():
    print("🚀 开始在一张显卡上串行运行所有 BrEaST 模型...")
    
    for script in scripts_to_run:
        if not os.path.exists(script):
            print(f"⚠️ 警告: 找不到脚本 {script}，已跳过。")
            continue
            
        print(f"\n{'='*60}")
        print(f"▶ 正在运行: {script}")
        print(f"{'='*60}")
        
        log_file = f"train_log_{script.replace('.py', '')}.txt"
        cmd = f"python -u {script} > {log_file} 2>&1"
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"✅ {script} 运行完成！日志已保存至: {log_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ {script} 运行失败！请检查日志: {log_file}")
            print("👉 将继续尝试运行队列中的下一个脚本...\n")

    print("\n🎉 所有 BrEaST 队列任务执行完毕！")

if __name__ == "__main__":
    main()