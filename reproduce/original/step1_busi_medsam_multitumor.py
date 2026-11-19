import os
import cv2
import numpy as np
import pandas as pd
import torch
import re
from transformers import SamModel, SamProcessor
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/BUSI"
TRAIN_CSV = os.path.join(DATA_ROOT, "train_split.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test_split.csv")
OUT_DIR = os.path.join(DATA_ROOT, "uncertainty_weights")

MODEL_NAME = "wanglab/medsam-vit-base"
DEVICE = "cuda:0"
K_RUNS = 12        # 采样次数
TAU = 0.2          # 温度系数

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
                # 如果算不出重心，取轮廓上的一个点
                core_points.append(c[0][0].tolist())

        # 4. 生成 K 组扰动提示
        # 策略：每次基于 Core Points 做一点随机偏移，模拟点击误差
        # MedSAM 接受 input_points 形状为 (Batch, N_points, 2)
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
            
            # 添加 1 个随机背景点 (负样本)，帮助定界
            # 简单策略：在 Mask 外部随机找一点
            # y_bg, x_bg = np.where(mask_gt == 0)
            # if len(x_bg) > 0:
            #     idx = np.random.choice(len(x_bg))
            #     current_run_points.append([x_bg[idx], y_bg[idx]])
            #     current_run_labels.append(0) # 0 = 背景

            all_prompts.append((current_run_points, current_run_labels))
            
        return all_prompts

    @torch.no_grad()
    def process_image(self, image, mask):
        prompts = self.get_prompts_multi_tumor(mask)
        if prompts is None: 
            # 可能是 Normal 或空 Mask，返回全1权重 (信任背景)
            return np.ones(mask.shape, dtype=np.float32)

        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        preds = []
        
        # 遍历 K 次采样
        for points, labels in prompts:
            # points: [[x1,y1], [x2,y2], ...]
            # labels: [1, 1, ...]
            
            inputs = self.processor(
                image, 
                input_points=[points], 
                input_labels=[labels], 
                return_tensors="pt"
            ).to(DEVICE)
            
            outputs = self.model(**inputs)
            
            # MedSAM 输出 (1, 1, 256, 256)
            mask_out = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), 
                inputs["original_sizes"].cpu(), 
                inputs["reshaped_input_sizes"].cpu()
            )[0][0][0].numpy() # Batch 0, Mask 0 (Best match)
            
            preds.append(mask_out)

        # 计算方差 -> 不确定性
        # preds stack: (K, H, W)
        var_map = np.var(np.array(preds), axis=0)
        
        # 平滑
        var_map = cv2.GaussianBlur(var_map, (3, 3), 0)
      
        # 简单的 min-max 归一化，防止 outlier
        max_val = var_map.max() + 1e-6
        uncertainty_norm = var_map / max_val
        
        return uncertainty_norm.astype(np.float32)

# ================= 辅助：Dataset 扫描逻辑 (复用 step2 的 robust 逻辑) =================
def scan_files(root_dir):
    img_map = {}
    msk_map = {}
    for root, dirs, files in os.walk(root_dir):
        root_lower = root.lower()
        is_mask_folder = 'label' in root_lower or 'mask' in root_lower
        is_img_folder = 'image' in root_lower
        
        for f in files:
            if f.startswith('.'): continue
            simple_key = re.sub(r'[^a-z0-9]', '', f.lower())
            path = os.path.join(root, f)
            if is_mask_folder: msk_map[simple_key] = path
            elif is_img_folder: img_map[simple_key] = path
    return img_map, msk_map

def resolve(csv_path, index):
    if pd.isna(csv_path): return None
    key = re.sub(r'[^a-z0-9]', '', os.path.basename(str(csv_path)).lower())
    
    if key in index: return index[key]
    if 'mask' in key:
        k2 = key.replace('mask', '')
        if k2 in index: return index[k2]
    else:
        k2 = key + 'mask'
        if k2 in index: return index[k2]
    return None

def main():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    
    img_map, msk_map = scan_files(DATA_ROOT)
    
    dfs = []
    if os.path.exists(TRAIN_CSV): dfs.append(pd.read_csv(TRAIN_CSV))
    if os.path.exists(TEST_CSV): dfs.append(pd.read_csv(TEST_CSV))
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['image'])
    
    # Filter Normal
    df = df[~df['image'].astype(str).str.lower().str.contains('normal')]
    
    gen = UncertaintyGenerator()
    
    print(f"Processing {len(df)} images...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = resolve(row['image'], img_map)
        msk_path = resolve(row['mask'], msk_map)
        
        if not img_path or not msk_path: continue
        
        base_name = os.path.basename(img_path)
        save_name = os.path.splitext(base_name)[0] + "_weight.png"
        save_path = os.path.join(OUT_DIR, save_name)
        
        if os.path.exists(save_path): continue
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        # [关键] 保持和 Dataset 一致的二值化逻辑
        msk = (msk > 0).astype(np.uint8) 
        
        try:
            u_map = gen.process_image(img, msk)
            # 保存为 0-255
            cv2.imwrite(save_path, (u_map * 255).astype(np.uint8))
        except Exception as e:
            print(f"Error {base_name}: {e}")

if __name__ == "__main__":
    main()