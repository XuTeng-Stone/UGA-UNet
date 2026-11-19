import os
import cv2
import numpy as np
import torch
import glob
from transformers import SamModel, SamProcessor
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

BASE_DIR = "/data/xuten/Breast Ultrasound Datasets/QaTa-COV19"
OUT_DIR = os.path.join(BASE_DIR, "uncertainty_weights")

MODEL_NAME = "wanglab/medsam-vit-base"
DEVICE = "cuda:4"
K_RUNS = 12
TAU = 0.2

class UncertaintyGenerator:
    def __init__(self):
        print(f"[Step 1] Loading MedSAM: {MODEL_NAME} -> {DEVICE}...")
        self.processor = SamProcessor.from_pretrained(MODEL_NAME)
        self.model = SamModel.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.eval()

    def get_prompts_bilateral(self, mask_gt):
        H, W = mask_gt.shape
        mid_w = W // 2
        
        mask_left = mask_gt.copy()
        mask_left[:, mid_w:] = 0
        
        mask_right = mask_gt.copy()
        mask_right[:, :mid_w] = 0
        
        core_points = []
        
        contours_l, _ = cv2.findContours(mask_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_l:
            if cv2.contourArea(c) > 20:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    core_points.append([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])])
                else:
                    core_points.append(c[0][0].tolist())
                    
        contours_r, _ = cv2.findContours(mask_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_r:
            if cv2.contourArea(c) > 20:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    core_points.append([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])])
                else:
                    core_points.append(c[0][0].tolist())

        if not core_points: 
            return None

        all_prompts = []
        for _ in range(K_RUNS):
            current_run_points = []
            current_run_labels = []
            
            for pt in core_points:
                offset_x = np.random.randint(-5, 6)
                offset_y = np.random.randint(-5, 6)
                new_pt = [max(0, min(pt[0] + offset_x, W - 1)), 
                          max(0, min(pt[1] + offset_y, H - 1))]
                
                current_run_points.append(new_pt)
                current_run_labels.append(1)
                
            all_prompts.append((current_run_points, current_run_labels))
            
        return all_prompts

    @torch.no_grad()
    def process_image(self, image, mask):
        prompts = self.get_prompts_bilateral(mask)
        if prompts is None: 
            return np.ones(mask.shape, dtype=np.float32)

        if len(image.shape) == 2: 
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        preds = []
        
        for points, labels in prompts:
            inputs = self.processor(
                image, 
                input_points=[points], 
                input_labels=[labels], 
                return_tensors="pt"
            ).to(DEVICE)
            
            outputs = self.model(**inputs)
            
            mask_out = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), 
                inputs["original_sizes"].cpu(), 
                inputs["reshaped_input_sizes"].cpu()
            )[0][0][0].numpy()
            
            preds.append(mask_out)

        var_map = np.var(np.array(preds), axis=0)
        var_map = cv2.GaussianBlur(var_map, (3, 3), 0)
        
        max_val = var_map.max() + 1e-6
        uncertainty_norm = var_map / max_val
        
        return uncertainty_norm.astype(np.float32)

def process_qata_dataset(img_dir, gt_dir, generator, prefix=""):
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    mask_paths = glob.glob(os.path.join(gt_dir, "mask_*.png"))
    print(f"Found {len(mask_paths)} masks in {gt_dir}")
    
    for msk_path in tqdm(mask_paths):
        base_name = os.path.basename(msk_path)
        img_name = base_name.replace("mask_", "")
        img_path = os.path.join(img_dir, img_name)
        
        if not os.path.exists(img_path):
            continue
            
        save_name = prefix + os.path.splitext(img_name)[0] + "_weight.png"
        save_path = os.path.join(OUT_DIR, save_name)
        
        if os.path.exists(save_path): 
            continue
            
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        msk = (msk > 0).astype(np.uint8) 
        
        try:
            u_map = generator.process_image(img, msk)
            cv2.imwrite(save_path, (u_map * 255).astype(np.uint8))
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

def main():
    gen = UncertaintyGenerator()
    
    print("\n--- Processing QaTa-COV19-v1 ---")
    v1_img_dir = os.path.join(BASE_DIR, "QaTa-COV19-v1", "Images")
    v1_gt_dir = os.path.join(BASE_DIR, "QaTa-COV19-v1", "Ground-truths")
    if os.path.exists(v1_gt_dir):
        process_qata_dataset(v1_img_dir, v1_gt_dir, gen, prefix="v1_")
    
    print("\n--- Processing QaTa-COV19-v2 (Train Set) ---")
    v2_train_img = os.path.join(BASE_DIR, "QaTa-COV19-v2", "Train Set", "Images")
    v2_train_gt = os.path.join(BASE_DIR, "QaTa-COV19-v2", "Train Set", "Ground-truths")
    if os.path.exists(v2_train_gt):
        process_qata_dataset(v2_train_img, v2_train_gt, gen, prefix="v2_train_")
        
    print("\n--- Processing QaTa-COV19-v2 (Test Set) ---")
    v2_test_img = os.path.join(BASE_DIR, "QaTa-COV19-v2", "Test Set", "Images")
    v2_test_gt = os.path.join(BASE_DIR, "QaTa-COV19-v2", "Test Set", "Ground-truths")
    if os.path.exists(v2_test_gt):
        process_qata_dataset(v2_test_img, v2_test_gt, gen, prefix="v2_test_")

if __name__ == "__main__":
    main()