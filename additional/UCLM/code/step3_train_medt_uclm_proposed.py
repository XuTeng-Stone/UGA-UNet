import os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
import pandas as pd

from step2_dataset_bus_uclm import BUSUCLMDataset
from step2_loss_anchor import LossCEDice, anchor_loss_from_maps
from models.axialnet import MedT
from metrics_utils import cal_params_flops, calculate_advanced_metrics

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/BUS-UCLM"
TRAIN_CSV, TEST_CSV = os.path.join(DATA_ROOT, "train.csv"), os.path.join(DATA_ROOT, "test.csv")
WEIGHT_DIR = os.path.join(DATA_ROOT, "uncertainty_weights")

MODEL_NAME = "MedT"
WEIGHT_DICE, WEIGHT_CE, LAMBDA_BG = 1.0, 1.0, 0.25
DIR_NAME = f"results_{MODEL_NAME}_uclm_proposed_Dice{WEIGHT_DICE}_CE{WEIGHT_CE}_BG{LAMBDA_BG}"
RESULTS_DIR = DIR_NAME
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE = "cuda:0", 8, 100, 1e-4
MAX_ALPHA, ANCHOR_MARGIN, WARMUP_RATIO = 0.1, 0.3, 0.1

def seed_everything(seed=42):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.backends.cudnn.deterministic = True

class UniversalAnchorWrapper(nn.Module):
    def __init__(self, base_model, num_classes=2):
        super().__init__()
        self.model = base_model; self.extracted_features = None
        def hook_fn(module, inputs): self.extracted_features = inputs[0]
        target_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d): target_conv = module 
        if target_conv is not None: target_conv.register_forward_pre_hook(hook_fn)
    def forward(self, x):
        self.extracted_features = None
        logits = self.model(x)
        if isinstance(logits, (tuple, list)): logits = logits[0]
        feats = self.extracted_features 
        if feats is None or feats.shape[1] < 2: feats = logits
        elif feats.shape[-2:] != logits.shape[-2:]: feats = F.interpolate(feats, size=logits.shape[-2:], mode='bilinear', align_corners=False)
        return logits, feats

@torch.no_grad()
def evaluate_dataset(model, loader, desc="Eval"):
    model.eval()
    ious, f1s, assds, hd95s = [], [], [], []
    for imgs, msks, _ in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(DEVICE); msks = torch.clamp(msks, 0, 1).long().to(DEVICE)
        if msks.dim() == 4 and msks.shape[1] == 1: msks = msks.squeeze(1)
        outs, _ = model(imgs); pred, target = torch.argmax(outs, dim=1).cpu().numpy(), msks.cpu().numpy()
        for i in range(pred.shape[0]):
            p, t = pred[i].flatten(), target[i].flatten()
            if np.sum(t) > 0: 
                inter, union = np.logical_and(t, p).sum(), np.logical_or(t, p).sum()
                ious.append(inter / (union + 1e-6)); f1s.append(f1_score(t, p, average='binary', zero_division=1))
        assd, hd95 = calculate_advanced_metrics(outs, msks)
        if assd > 0: assds.append(assd)
        if hd95 > 0: hd95s.append(hd95)
    return (np.mean(ious) if ious else 0.0, np.mean(f1s) if f1s else 0.0, np.mean(assds) if assds else 0.0, np.mean(hd95s) if hd95s else 0.0)

def main():
    seed_everything(42)
    train_tf = A.Compose([A.Resize(256, 256), A.HorizontalFlip(p=0.5), A.Rotate(limit=30, p=0.5), A.Normalize(), ToTensorV2()])
    val_tf = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
    train_ds = BUSUCLMDataset(TRAIN_CSV, DATA_ROOT, WEIGHT_DIR, transform=train_tf, only_tumor=True)
    val_ds = BUSUCLMDataset(TEST_CSV, DATA_ROOT, WEIGHT_DIR, transform=val_tf, only_tumor=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    base_model = MedT(img_size=256, imgchan=3, num_classes=2).to(DEVICE)
    model = UniversalAnchorWrapper(base_model, num_classes=2).to(DEVICE)
    try: params, flops = cal_params_flops(base_model, input_size=(1, 3, 256, 256), device=DEVICE)
    except: params, flops = 0, 0

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler('cuda')
    criterion_seg = LossCEDice(w_ce=WEIGHT_CE, w_dice=WEIGHT_DICE).to(DEVICE)

    best_iou, best_alpha = 0.0, 0.0
    warmup_epochs = int(WARMUP_RATIO * NUM_EPOCHS)
    
    for epoch in range(NUM_EPOCHS):
        model.train(); epoch_loss = 0
        if epoch < warmup_epochs: alpha, stage = 0.0, "Warmup"
        else: alpha, stage = MAX_ALPHA * ((epoch - warmup_epochs) / (NUM_EPOCHS - warmup_epochs)), "Ramp"
        loop = tqdm(train_loader, desc=f"Ep {epoch+1} [{stage}]", leave=False)
        for imgs, msks, wgts in loop:
            imgs = imgs.to(DEVICE); msks = torch.clamp(msks, 0, 1).long().to(DEVICE)
            if msks.dim() == 4 and msks.shape[1] == 1: msks = msks.squeeze(1)
            wgts = wgts.float().squeeze(1).to(DEVICE) if wgts.dim() == 4 else wgts.float().to(DEVICE)

            optimizer.zero_grad()
            with autocast('cuda'):
                logits, feats = model(imgs)
                l_seg = criterion_seg(logits, msks)
                if alpha > 1e-6:
                    l_anchor = anchor_loss_from_maps(feat=feats.float(), gt=msks, u_fg=wgts, w_fg=None, margin=ANCHOR_MARGIN, lam_bg=LAMBDA_BG, min_pts=10)
                else:
                    l_anchor = torch.tensor(0.0, device=DEVICE)
                loss = l_seg + alpha * l_anchor
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            epoch_loss += loss.item(); loop.set_postfix(loss=loss.item())

        val_iou, _, _, _ = evaluate_dataset(model, val_loader)
        print(f"Ep {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | Val mIoU: {val_iou:.4f} | Alpha: {alpha:.4f}")
        if val_iou > best_iou: best_iou, best_alpha = val_iou, alpha; torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pth"))

    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_model.pth")))
    iou, f1, assd, hd95 = evaluate_dataset(model, val_loader, desc="Report")
    open(os.path.join(RESULTS_DIR, "final_report.txt"), "w").write(f"mIoU: {iou:.4f}\nmF1: {f1:.4f}\nASSD: {assd:.4f}\nHD95: {hd95:.4f}\nBest_Alpha: {best_alpha:.4f}")
    pd.DataFrame([{'Model': MODEL_NAME, 'Params': params, 'FLOPs': flops, 'mIoU': iou, 'mF1': f1, 'ASSD': assd, 'HD95': hd95, 'Best_Alpha': best_alpha}]).to_csv(os.path.join(RESULTS_DIR, f"{MODEL_NAME.lower()}_uclm_results.csv"), index=False)

if __name__ == "__main__": main()