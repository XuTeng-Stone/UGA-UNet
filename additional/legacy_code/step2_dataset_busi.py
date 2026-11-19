import os
import cv2
import torch
import numpy as np
import pandas as pd
import re
from torch.utils.data import Dataset

class BUSIDataset(Dataset):
    def __init__(self, csv_file, root_dir, weight_dir=None, transform=None, only_tumor=True):
        self.root_dir = root_dir
        self.weight_dir = weight_dir
        self.transform = transform

        # -----------------------------------------------------------
        # 1. 建立索引 (文件名 -> 路径)
        # -----------------------------------------------------------
        print(f"[Dataset] Scanning BUSI files in {root_dir}...")
        self.image_index = {} 
        self.mask_index = {}  
        
        for root, dirs, files in os.walk(root_dir):
            root_lower = root.lower().replace('\\', '/')
            is_mask_folder = 'label' in root_lower or 'mask' in root_lower
            is_img_folder = 'image' in root_lower

            for f in files:
                if f.startswith('.'): continue
                # 生成指纹: benign (195).png -> benign195
                simple_key = re.sub(r'[^a-z0-9]', '', f.lower())
                full_path = os.path.join(root, f)
                
                if is_mask_folder:
                    self.mask_index[simple_key] = full_path
                elif is_img_folder:
                    self.image_index[simple_key] = full_path

        print(f"[Dataset] Indexed {len(self.image_index)} images and {len(self.mask_index)} masks.")
        
        # -----------------------------------------------------------
        # 2. 读取 CSV
        # -----------------------------------------------------------
        try:
            self.df = pd.read_csv(csv_file)
        except Exception as e:
            raise RuntimeError(f"CSV Read Failed: {e}")

        # -----------------------------------------------------------
        # 3. 过滤 Normal
        # -----------------------------------------------------------
        if only_tumor:
            if 'image' in self.df.columns:
                self.df = self.df[~self.df['image'].astype(str).str.lower().str.contains('normal')]
            if 'mask' in self.df.columns:
                self.df = self.df[self.df['mask'].notna()]
            self.df = self.df.reset_index(drop=True)
            print(f"[Dataset] Filtered Normal -> {len(self.df)} samples")

    def resolve_path(self, csv_path, index_dict):
        if pd.isna(csv_path) or str(csv_path) == '': return None
        filename = os.path.basename(str(csv_path))
        simple_key = re.sub(r'[^a-z0-9]', '', filename.lower())
        
        if simple_key in index_dict: return index_dict[simple_key]
        if 'mask' in simple_key:
            no_mask_key = simple_key.replace('mask', '')
            if no_mask_key in index_dict: return index_dict[no_mask_key]
        
        mask_key = simple_key + 'mask'
        if mask_key in index_dict: return index_dict[mask_key]

        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        csv_img = self.df.iloc[idx]['image']
        csv_msk = self.df.iloc[idx]['mask']
        
        # 1. 查找路径
        real_img_path = self.resolve_path(csv_img, self.image_index)
        if real_img_path is None:
             raise FileNotFoundError(f"Image not found: {csv_img}")

        real_msk_path = self.resolve_path(csv_msk, self.mask_index)
        if real_msk_path is None:
             raise FileNotFoundError(f"Mask not found: {csv_msk}")

        # 2. 读取图片
        image = cv2.imread(real_img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. 读取 Mask
        mask = cv2.imread(real_msk_path, cv2.IMREAD_GRAYSCALE)
        
        # [关键修改] 鲁棒的二值化处理
        # 不管原图是 0-1 还是 0-255，只要大于 0 就变成 1.0
        mask = (mask > 0).astype(np.float32)

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 4. 读取权重
        weight_map = np.ones_like(mask)
        if self.weight_dir:
            img_base = os.path.basename(real_img_path)
            wgt_name = os.path.splitext(img_base)[0] + "_weight.png"
            wgt_path = os.path.join(self.weight_dir, wgt_name)
            
            if os.path.exists(wgt_path):
                w = cv2.imread(wgt_path, cv2.IMREAD_GRAYSCALE)
                if w is not None: 
                    if w.shape[:2] != image.shape[:2]:
                        w = cv2.resize(w, (image.shape[1], image.shape[0]))
                    weight_map = w.astype(np.float32) / 255.0

        if self.transform:
            augmented = self.transform(image=image, masks=[mask, weight_map])
            image = augmented['image']
            mask = augmented['masks'][0]
            weight_map = augmented['masks'][1]

        if not isinstance(mask, torch.Tensor): mask = torch.from_numpy(mask)
        if not isinstance(weight_map, torch.Tensor): weight_map = torch.from_numpy(weight_map)
            
        return image, mask.long(), weight_map.float()