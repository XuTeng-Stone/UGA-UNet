import os
from random import triangular
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class BUSUCLMDataset(Dataset):
    def __init__(self, csv_file, root_dir, weight_dir=None, transform=None, only_tumor=True):
        """
        BUS-UCLM Dataset (Fix: Separate indexes for Images and Masks to avoid collision)
        """
        self.root_dir = root_dir
        self.weight_dir = weight_dir
        self.transform = transform

        # -----------------------------------------------------------
        # 1. 建立双重文件索引 (关键修复！)
        # -----------------------------------------------------------
        print(f"[Dataset] Indexing files in {root_dir}...")
        self.image_index = {} # 专存图片
        self.mask_index = {}  # 专存Mask
        
        for root, dirs, files in os.walk(root_dir):
            parent_dir = os.path.basename(root).lower()
            
            # 启发式判断当前文件夹类型
            is_mask_folder = any(x in parent_dir for x in ['label', 'mask', 'gt', 'segmentation'])
            is_image_folder = any(x in parent_dir for x in ['image', 'img', 'original'])
            
            for f in files:
                if f.startswith('.'): continue
                key = f.lower()
                full_path = os.path.join(root, f)
                
                # 分类存入索引
                if is_mask_folder:
                    self.mask_index[key] = full_path
                elif is_image_folder:
                    self.image_index[key] = full_path
                else:
                    # 如果文件夹名看不出来 (比如 root)，根据文件名猜
                    if 'mask' in key:
                        self.mask_index[key] = full_path
                    else:
                        # 默认当作图片，或者同时放入两者(如果不冲突)
                        # 为了安全，未分类的优先当作图片
                        if key not in self.image_index:
                            self.image_index[key] = full_path

        print(f"[Dataset] Indexed {len(self.image_index)} images and {len(self.mask_index)} masks.")

        # -----------------------------------------------------------
        # 2. 读取 CSV
        # -----------------------------------------------------------
        try:
            self.df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"[Error] Failed to read CSV: {csv_file}")
            raise e

        # 列名清洗
        self.df.columns = [str(c).strip().lower() for c in self.df.columns]
        curr_cols = self.df.columns.tolist()
        
        # 映射列名
        if 'image_path' not in curr_cols:
            for cand in ['image', 'img', 'filename', 'file']:
                if cand in curr_cols: self.df.rename(columns={cand: 'image_path'}, inplace=True); break
        
        if 'mask_path' not in curr_cols:
            for cand in ['mask', 'mask_file', 'segmentation']:
                if cand in curr_cols: self.df.rename(columns={cand: 'mask_path'}, inplace=True); break
        
        if 'class' not in curr_cols:
            for cand in ['label', 'class_name', 'category']:
                if cand in curr_cols: self.df.rename(columns={cand: 'class'}, inplace=True); break

        # -----------------------------------------------------------
        # 3. 过滤 Normal
        # -----------------------------------------------------------
        if only_tumor:
            initial_len = len(self.df)
            if 'class' in self.df.columns:
                self.df = self.df[self.df['class'].astype(str).str.lower().str.contains('normal') == False]
            
            if 'mask_path' in self.df.columns:
                self.df = self.df[self.df['mask_path'].notna()]
                self.df = self.df[self.df['mask_path'].astype(str).str.strip() != '']
                self.df = self.df[self.df['mask_path'].astype(str).str.lower() != 'nan']

            self.df = self.df.reset_index(drop=True)
            print(f"[Dataset] {os.path.basename(csv_file)}: Filtered Normal -> {len(self.df)} samples")
        else:
            print(f"[Dataset] {os.path.basename(csv_file)}: Loaded {len(self.df)} samples.")

    def resolve_path(self, filename, index_dict):
        """从指定的索引字典中查找文件"""
        if pd.isna(filename) or str(filename) == '': return None
        fname = str(filename).strip()
        key = fname.lower()
        
        if key in index_dict: return index_dict[key]
        
        # 尝试换后缀
        base, ext = os.path.splitext(key)
        if ext == '.png':
            trial = base + '.jpg'
            if trial in index_dict: return index_dict[trial]
        elif ext == '.jpg':
            trial = base + '.png'
            if trial in index_dict: return index_dict[trial]
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        img_name = self.df.iloc[idx]['image_path']
        
        # [关键] 强制在图片索引里找图片
        real_img_path = self.resolve_path(img_name, self.image_index)
        
        if real_img_path is None:
             # 最后的容错：如果图片没在 image_index 里，试着在 mask_index 找找？(不，这样会重蹈覆辙)
             raise FileNotFoundError(f"Image not found in Image folders: {img_name}")

        image = cv2.imread(real_img_path)
        if image is None: raise FileNotFoundError(f"CV2 Read Failed: {real_img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Mask ---
        mask_path_raw = None
        if 'mask_path' in self.df.columns:
            mask_path_raw = self.df.iloc[idx]['mask_path']
        
        # [关键] 强制在 Mask 索引里找 Mask
        real_mask_path = self.resolve_path(mask_path_raw, self.mask_index)
        
        if real_mask_path:
            mask = cv2.imread(real_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            else:
                mask = mask.astype(np.float32) / 255.0
                mask = (mask > 0.5).astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # --- Weight ---
        weight_map = np.ones_like(mask)
        # (Weights 通常和 Mask 对齐，或者是另外的)
        if self.weight_dir:
            # 简单处理：假设 weight 在 weight_dir
            img_base = os.path.basename(real_img_path)
            wgt_name = os.path.splitext(img_base)[0] + "_weight.png"
            wgt_path = os.path.join(self.weight_dir, wgt_name)
            if os.path.exists(wgt_path):
                w = cv2.imread(wgt_path, cv2.IMREAD_GRAYSCALE)
                if w is not None: weight_map = w.astype(np.float32) / 255.0

        # Augmentation
        if self.transform:
            augmented = self.transform(image=image, masks=[mask, weight_map])
            image = augmented['image']
            mask = augmented['masks'][0]
            weight_map = augmented['masks'][1]

        # To Tensor
        if not isinstance(mask, torch.Tensor): mask = torch.from_numpy(mask)
        if not isinstance(weight_map, torch.Tensor): weight_map = torch.from_numpy(weight_map)
            
        # [Fix] CE Loss requires Long
        mask = mask.long()
        weight_map = weight_map.float()

        return image, mask, weight_map