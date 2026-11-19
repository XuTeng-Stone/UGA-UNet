import os
import cv2
import torch
import numpy as np
import glob
from torch.utils.data import Dataset

class QaTaDataset(Dataset):
    def __init__(self, img_dir, gt_dir, weight_dir=None, weight_prefix="", transform=None):
        """
        QaTa-COV19 专属数据集加载器
        :param img_dir: 图像文件夹路径 (Images)
        :param gt_dir: 掩码文件夹路径 (Ground-truths)
        :param weight_dir: 不确定性权重图文件夹路径
        :param weight_prefix: 权重文件名前缀 (例如 "v2_train_")
        """
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.weight_dir = weight_dir
        self.weight_prefix = weight_prefix
        self.transform = transform
        
        # 1. 以 Ground-truths 为核心建立索引
        print(f"[Dataset] Indexing files in {gt_dir}...")
        mask_paths = glob.glob(os.path.join(gt_dir, "mask_*.png"))
        
        self.samples = []
        for msk_path in mask_paths:
            base_name = os.path.basename(msk_path)
            img_name = base_name.replace("mask_", "")
            img_path = os.path.join(img_dir, img_name)
            
            # 确保原始图像确实存在
            if os.path.exists(img_path):
                self.samples.append({
                    'image': img_path,
                    'mask': msk_path,
                    'img_name': img_name
                })
                
        print(f"[Dataset] Successfully indexed {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # --- Image ---
        image = cv2.imread(sample['image'])
        if image is None: raise FileNotFoundError(f"CV2 Read Failed: {sample['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Mask ---
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            mask = mask.astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)

        # --- Weight (Uncertainty Map) ---
        weight_map = np.ones_like(mask)
        if self.weight_dir:
            img_base = os.path.splitext(sample['img_name'])[0]
            # 根据 step1 的命名规则拼凑
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