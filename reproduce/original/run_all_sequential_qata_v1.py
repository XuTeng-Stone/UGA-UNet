import os
import subprocess

scripts_to_run = [
    "step3_train_unet_qata_v1.py",
    "step3_train_unetplusplus_qata_v1.py",
    "step3_train_deeplabv3_qata_v1.py",
    "step3_train_cmunext_qata_v1.py",
    "step3_train_anato_qata_v1.py",
    "step3_train_medt_qata_v1.py",
    "step3_train_transunet_qata_v1.py",
    "step3_train_ukan_qata_v1.py",
    "step3_train_depth_qata_v1.py"
]

def main():
    print("Starting sequential execution for QaTa-COV19 v1 models...")
    for script in scripts_to_run:
        if not os.path.exists(script):
            print(f"Warning: Script {script} not found. Skipping.")
            continue
        print(f"\n{'='*60}\nRunning: {script}\n{'='*60}")
        log_file = f"train_log_{script.replace('.py', '')}.txt"
        cmd = f"python -u {script} > {log_file} 2>&1"
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Success: {script}. Log saved to: {log_file}")
        except subprocess.CalledProcessError:
            print(f"Failed: {script}. Check log: {log_file}")
    print("\nAll V1 tasks completed.")

if __name__ == "__main__":
    main()