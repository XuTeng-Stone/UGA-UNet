import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class QaTaV1CSVDataset(Dataset):
    def __init__(self, csv_file, weight_dir=None, weight_prefix="v1_", transform=None):
        """
        基于固化 CSV 的 QaTa-COV19-v1 数据集加载器
        :param csv_file: 固化好的 train_v1.csv 或 test_v1.csv 路径
        :param weight_dir: 不确定性权重图文件夹路径
        :param weight_prefix: 权重文件名前缀 (例如 "v1_")
        """
        self.weight_dir = weight_dir
        self.weight_prefix = weight_prefix
        self.transform = transform
        
        # 1. 读取固化的 CSV 文件
        try:
            self.df = pd.read_csv(csv_file)
            print(f"[Dataset] Successfully loaded {len(self.df)} samples from {os.path.basename(csv_file)}.")
        except Exception as e:
            print(f"[Error] Failed to read CSV: {csv_file}")
            raise e

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        
        row = self.df.iloc[idx]
        img_path = row['image_path']
        msk_path = row['mask_path']
        img_name = row['img_name']

        # --- Image ---
        image = cv2.imread(img_path)
        if image is None: raise FileNotFoundError(f"CV2 Read Failed: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Mask ---
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            mask = mask.astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)

        # --- Weight (Uncertainty Map) ---
        weight_map = np.ones_like(mask)
        if self.weight_dir:
            img_base = os.path.splitext(img_name)[0]
            wgt_name = self.weight_prefix + img_base + "_weight.png"
            wgt_path = os.path.join(self.weight_dir, wgt_name)
            
            if os.path.exists(wgt_path):
                w = cv2.imread(wgt_path, cv2.IMREAD_GRAYSCALE)
                if w is not None: 
                    weight_map = w.astype(np.float32) / 255.0

        # --- Augmentation ---
        if self.transform:
            augmented = self.transform(image=image, masks=[mask, weight_map])
            image = augmented['image']
            mask = augmented['masks'][0]
            weight_map = augmented['masks'][1]

        # --- To Tensor ---
        if not isinstance(mask, torch.Tensor): mask = torch.from_numpy(mask)
        if not isinstance(weight_map, torch.Tensor): weight_map = torch.from_numpy(weight_map)
            
        mask = mask.long()
        weight_map = weight_map.float()

        return image, mask, weight_map