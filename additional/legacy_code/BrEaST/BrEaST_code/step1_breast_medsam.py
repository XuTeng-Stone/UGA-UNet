import os
import cv2
import numpy as np
import pandas as pd
import torch
from transformers import SamModel, SamProcessor
from tqdm import tqdm

# ================= 配置区域 =================
# [关键] BrEaST 数据集根目录
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/BrEaST"
# CSV 路径 (根据你的截图，csv 就在根目录下，或者你自己指定)
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")

# 输出目录
OUT_DIR = os.path.join(DATA_ROOT, "uncertainty_weights")

MODEL_NAME = "wanglab/medsam-vit-base"
DEVICE = "cuda:1"
K_RUNS = 12        # 采样次数 (用于计算方差/不确定性)

# ================= 逻辑类 =================
class UncertaintyGenerator:
    def __init__(self):
        print(f"[Step 1] Loading MedSAM: {MODEL_NAME} -> {DEVICE}...")
        self.processor = SamProcessor.from_pretrained(MODEL_NAME)
        self.model = SamModel.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.eval()

    def get_prompts_multi_tumor(self, mask_gt):
        """
        针对多病灶优化：找到所有连通域，为每个连通域生成提示点
        """
        # 1. 找轮廓
        contours, _ = cv2.findContours(mask_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None

        # 2. 筛选有效病灶 (面积 > 20 像素，防止噪点)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
        if not valid_contours: return None

        # 3. 基础正向点 (每个病灶的重心)
        core_points = []
        for c in valid_contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                core_points.append([cx, cy])
            else:
                core_points.append(c[0][0].tolist())

        # 4. 生成 K 组扰动提示
        # 策略：每次基于 Core Points 做一点随机偏移，模拟点击误差
        all_prompts = []
        
        for _ in range(K_RUNS):
            current_run_points = []
            current_run_labels = []
            
            # 为每个病灶生成 1 个正向点
            for pt in core_points:
                # 随机偏移 +/- 5 像素
                offset_x = np.random.randint(-5, 6)
                offset_y = np.random.randint(-5, 6)
                new_pt = [max(0, pt[0] + offset_x), max(0, pt[1] + offset_y)]
                
                current_run_points.append(new_pt)
                current_run_labels.append(1) # 1 = 前景
            
            all_prompts.append((current_run_points, current_run_labels))
            
        return all_prompts

    @torch.no_grad()
    def process_image(self, image, mask):
        prompts = self.get_prompts_multi_tumor(mask)
        if prompts is None: 
            # 空 Mask (可能是 Normal)，返回全黑 (无不确定性) 或 全白 (全不确定)?
            # 通常 Normal 样本不需要 Uncertainty Map 参与 Anchor Loss，返回全1权重即可
            return np.ones(mask.shape, dtype=np.float32)

        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        preds = []
        
        # 遍历 K 次采样
        for points, labels in prompts:
            # MedSAM 推理
            inputs = self.processor(
                image, 
                input_points=[points], 
                input_labels=[labels], 
                return_tensors="pt"
            ).to(DEVICE)
            
            outputs = self.model(**inputs)
            
            # Post-process
            mask_out = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), 
                inputs["original_sizes"].cpu(), 
                inputs["reshaped_input_sizes"].cpu()
            )[0][0][0].numpy() # Batch 0, Mask 0
            
            preds.append(mask_out)

        # 计算方差 -> 不确定性
        # preds stack: (K, H, W) -> var map
        var_map = np.var(np.array(preds), axis=0)
        
        # 平滑 & 归一化
        var_map = cv2.GaussianBlur(var_map, (3, 3), 0)
        max_val = var_map.max() + 1e-6
        uncertainty_norm = var_map / max_val
        
        return uncertainty_norm.astype(np.float32)

def main():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    
    # 1. 读取 CSV
    dfs = []
    if os.path.exists(TRAIN_CSV): dfs.append(pd.read_csv(TRAIN_CSV))
    if os.path.exists(TEST_CSV): dfs.append(pd.read_csv(TEST_CSV))
    
    if not dfs:
        print("[Error] CSV files not found!")
        return

    df = pd.concat(dfs, ignore_index=True)
    # 去重 (防止 Train/Test 有重复图片名)
    df = df.drop_duplicates(subset=['image'])
    
    gen = UncertaintyGenerator()
    
    print(f"Processing {len(df)} images for BrEaST dataset...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 构建绝对路径
        # CSV 里是 images/caseXXX.png, 我们需要拼上 DATA_ROOT
        # 注意：有些 CSV 可能带 'images/' 前缀，有些不带，这里统一处理
        rel_img = str(row['image']).strip()
        rel_msk = str(row['label']).strip() # 注意 CSV 表头是 label 不是 mask
        
        # 如果路径以 / 开头，说明是绝对路径？不太可能，通常是相对路径
        # BrEaST 数据集通常结构：DATA_ROOT/images/caseXXX.png
        
        # 简单清洗路径 (移除可能存在的 windows 风格反斜杠)
        rel_img = rel_img.replace('\\', '/')
        rel_msk = rel_msk.replace('\\', '/')
        
        # 拼接
        img_path = os.path.join(DATA_ROOT, rel_img)
        msk_path = os.path.join(DATA_ROOT, rel_msk)
        
        # 检查文件是否存在
        if not os.path.exists(img_path):
            # 尝试另一种拼接：如果 CSV 只有文件名
            # img_path = os.path.join(DATA_ROOT, 'images', os.path.basename(rel_img))
            pass 

        if not os.path.exists(img_path) or not os.path.exists(msk_path):
            # print(f"[Skip] File not found: {rel_img}")
            continue
        
        # 输出文件名：caseXXX_weight.png
        base_name = os.path.basename(img_path)
        save_name = os.path.splitext(base_name)[0] + "_weight.png"
        save_path = os.path.join(OUT_DIR, save_name)
        
        # 断点续传
        if os.path.exists(save_path): continue
        
        # 读取
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if msk is None: continue
        
        # 二值化
        msk = (msk > 0).astype(np.uint8) 
        
        try:
            u_map = gen.process_image(img, msk)
            # 保存为 0-255 png
            cv2.imwrite(save_path, (u_map * 255).astype(np.uint8))
        except Exception as e:
            print(f"Error processing {base_name}: {e}")

if __name__ == "__main__":
    main()