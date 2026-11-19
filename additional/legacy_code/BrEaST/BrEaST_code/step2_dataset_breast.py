import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class BrEaSTDataset(Dataset):
    # [修改点 1] 增加 only_tumor=True 和 **kwargs 兼容现有训练脚本的传参
    def __init__(self, csv_file, root_dir, weight_dir=None, transform=None, only_tumor=True, **kwargs):
        """
        Dataset for BrEaST (CSV header: image, label)
        """
        self.root_dir = root_dir
        self.weight_dir = weight_dir
        self.transform = transform

        # 读取 CSV
        try:
            self.df = pd.read_csv(csv_file)
        except Exception as e:
            raise RuntimeError(f"CSV Read Failed: {e}")
        
        # [修改点 2] 如果传入了 only_tumor=True，过滤掉路径中包含 'normal' 的样本
        if only_tumor:
            original_len = len(self.df)
            self.df = self.df[~self.df['image'].astype(str).str.lower().str.contains('normal')]
            filtered_len = len(self.df)
            if original_len != filtered_len:
                print(f"[Dataset] Filtered Normal -> {filtered_len} samples remain.")
        
        # 简单清洗路径 (移除可能存在的 windows 反斜杠)
        # BrEaST csv 里的路径通常是 relative path: "images/caseXXX.png"
        print(f"[Dataset] Loaded {len(self.df)} samples from {os.path.basename(csv_file)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        row = self.df.iloc[idx]
        rel_img = str(row['image']).strip().replace('\\', '/')
        rel_msk = str(row['label']).strip().replace('\\', '/') # BrEaST uses 'label'
        
        img_path = os.path.join(self.root_dir, rel_img)
        msk_path = os.path.join(self.root_dir, rel_msk)
        
        # 1. 读取图片
        if not os.path.exists(img_path):
             raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. 读取 Mask
        if not os.path.exists(msk_path):
             raise FileNotFoundError(f"Mask not found: {msk_path}")
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        
        # 二值化 (确保是 0/1)
        mask = (mask > 0).astype(np.float32)

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 3. 读取权重 (如果需要)
        weight_map = np.ones_like(mask)
        if self.weight_dir:
            # Step 1 生成的文件名通常是: caseXXX_weight.png
            # 从 rel_img ("images/case123.png") 提取文件名 "case123"
            file_name = os.path.basename(rel_img) # case123.png
            base_name = os.path.splitext(file_name)[0] # case123
            wgt_name = f"{base_name}_weight.png"
            wgt_path = os.path.join(self.weight_dir, wgt_name)
            
            if os.path.exists(wgt_path):
                w = cv2.imread(wgt_path, cv2.IMREAD_GRAYSCALE)
                if w is not None: 
                    if w.shape[:2] != image.shape[:2]:
                        w = cv2.resize(w, (image.shape[1], image.shape[0]))
                    weight_map = w.astype(np.float32) / 255.0
            else:
                # 如果找不到权重图，可能是 Step 1 没生成（比如完全没病灶），默认给全1
                pass

        if self.transform:
            augmented = self.transform(image=image, masks=[mask, weight_map])
            image = augmented['image']
            mask = augmented['masks'][0]
            weight_map = augmented['masks'][1]

        if not isinstance(mask, torch.Tensor): mask = torch.from_numpy(mask)
        if not isinstance(weight_map, torch.Tensor): weight_map = torch.from_numpy(weight_map)
            
        return image, mask.long(), weight_map.float()