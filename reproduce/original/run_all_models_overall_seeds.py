import os
import subprocess
import re  # 引入正则模块，用于精准替换

# ================= 配置区域 =================
# 9 大核心对比模型 
models = [
    #"unet", 
    #"unetplusplus", 
    #"deeplabv3", 
    #"cmunext", 
    "anato", 
    #"medt", 
    #"transunet", 
    #"ukan",
    #"depth"    
]

# 涵盖你所有可能的实验设定的脚本后缀
settings = [
    #"_baseline.py",          # BUSI 纯基线
    #"_busi.py",              # BUSI 纯基线备用
    #"_busi_proposed.py",     # BUSI (Anchor 我们的方法)
    #"_breast_proposed.py",   # BrEaST (Anchor)
    #"_breast.py",
    "_qata_v2.py",           # QaTa_v2 (Anchor)
    #"_qata_v1.py"            # QaTa_v1 (Anchor)
]

# 需要追加测试的随机种子
seeds = [999]

def main():
    print("🚀 开始自动生成矩阵：9大模型 × 全场景设定 × 2个Seed...")
    
    # 自动组合出所有的脚本文件名
    base_scripts = []
    for model in models:
        for setting in settings:
            base_scripts.append(f"step3_train_{model}{setting}")
            
    # 智能过滤：只保留当前目录下真实存在的文件
    valid_scripts = [s for s in base_scripts if os.path.exists(s)]
    print(f"✅ 扫描完毕！在当前目录下共找到 {len(valid_scripts)} 个匹配的训练脚本，即将开始排队训练。")
    
    for script in valid_scripts:
        with open(script, 'r', encoding='utf-8') as f:
            original_code = f.read()
            
        for seed in seeds:
            print(f"\n{'='*70}")
            print(f"▶ 正在生成并运行: [模型/设定] {script} | [Seed] {seed}")
            print(f"{'='*70}")
            
            # ----------------- 核心安全注入逻辑 -----------------
            modified_code = original_code
            
            # 1. [修复显卡报错] 强制将代码中的 DEVICE = "cuda:X" 替换为 DEVICE = "cuda:0"
            # 这样配合外部的 CUDA_VISIBLE_DEVICES=X 使用时，内部永远只认 0 号卡，就不会报错了
            modified_code = re.sub(r'DEVICE\s*=\s*"cuda:\d+"', 'DEVICE = "cuda:0"', modified_code)
            
            # 2. 自动替换主函数中的随机种子
            modified_code = modified_code.replace("seed_everything(42)", f"seed_everything({seed})")
            
            # 3. 动态修改保存路径，绝对防覆盖机制
            target_mkdir_normal = "if not os.path.exists(RESULTS_DIR):"
            injected_mkdir_normal = f'RESULTS_DIR = RESULTS_DIR + "_seed{seed}"\nif not os.path.exists(RESULTS_DIR):'
            
            target_mkdir_depth = "if not os.path.exists(SAVE_DIR_ROOT):"
            injected_mkdir_depth = f'SAVE_DIR_ROOT = SAVE_DIR_ROOT + "_seed{seed}"\nif not os.path.exists(SAVE_DIR_ROOT):'
            
            if target_mkdir_normal in modified_code:
                modified_code = modified_code.replace(target_mkdir_normal, injected_mkdir_normal)
            elif target_mkdir_depth in modified_code:
                modified_code = modified_code.replace(target_mkdir_depth, injected_mkdir_depth)
            else:
                print(f"⚠️ 警告：在 {script} 中无法定位目录创建逻辑，可能会覆盖原结果！请检查脚本变量名。")
            # ----------------------------------------------------
            
            # 将修改后的代码生成为一个临时的执行脚本
            temp_script_name = script.replace(".py", f"_seed{seed}.py")
            with open(temp_script_name, 'w', encoding='utf-8') as f:
                f.write(modified_code)
                
            log_file = f"train_log_{temp_script_name.replace('.py', '')}.txt"
            cmd = f"python -u {temp_script_name} > {log_file} 2>&1"
            
            try:
                print(f"⏳ 正在后台训练，输出重定向至: {log_file}")
                subprocess.run(cmd, shell=True, check=True)
                print(f"✅ 运行完成！结果已安全保存在结尾带有 _seed{seed} 的独立文件夹中。")
            except subprocess.CalledProcessError:
                print(f"❌ 运行崩溃！请检查日志: {log_file}")
                
            # 清理临时生成的脚本文件
            if os.path.exists(temp_script_name):
                os.remove(temp_script_name)

    print("\n🎉 所有模型的多种子(Seed) Overall 实验已全部执行完毕！")

if __name__ == "__main__":
    main()