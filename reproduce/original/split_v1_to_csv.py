import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# ================= 配置区域 =================
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/QaTa-COV19"
V1_IMG_DIR = os.path.join(DATA_ROOT, "QaTa-COV19-v1", "Images")
V1_GT_DIR = os.path.join(DATA_ROOT, "QaTa-COV19-v1", "Ground-truths")

TRAIN_CSV_OUT = os.path.join(DATA_ROOT, "train_v1.csv")
TEST_CSV_OUT = os.path.join(DATA_ROOT, "test_v1.csv")

def main():
    print(f"开始扫描 V1 数据集: {V1_GT_DIR}")
    
    all_samples = []
    
    # [关键安全机制] 使用 sorted 强制字母序，保证跨设备、跨运行时的绝对一致性
    mask_paths = sorted(glob.glob(os.path.join(V1_GT_DIR, "mask_*.png")))
    
    for msk_path in mask_paths:
        base_name = os.path.basename(msk_path)
        img_name = base_name.replace("mask_", "")
        img_path = os.path.join(V1_IMG_DIR, img_name)
        
        # 确保图和掩码成对存在
        if os.path.exists(img_path):
            all_samples.append({
                'image_path': img_path,
                'mask_path': msk_path,
                'img_name': img_name
            })
            
    total_valid = len(all_samples)
    print(f"成功匹配成对的有效样本数量: {total_valid}")
    
    if total_valid == 0:
        print("未找到任何数据，请检查路径！")
        return

    # [核心划分] 80/20 划分，锁定 random_state=42
    train_samples, test_samples = train_test_split(all_samples, test_size=0.2, random_state=42)
    
    # 转换为 DataFrame 并保存
    df_train = pd.DataFrame(train_samples)
    df_test = pd.DataFrame(test_samples)
    
    df_train.to_csv(TRAIN_CSV_OUT, index=False)
    df_test.to_csv(TEST_CSV_OUT, index=False)
    
    print(f"数据切分固化完成！")
    print(f"-> 训练集 ({len(df_train)} 个样本) 已保存至: {TRAIN_CSV_OUT}")
    print(f"-> 测试集 ({len(df_test)} 个样本) 已保存至: {TEST_CSV_OUT}")

if __name__ == "__main__":
    main()