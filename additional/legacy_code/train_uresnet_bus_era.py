import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import random
from dataset_new import BreastCancerSegmentation
import torch.nn as nn
import torch.optim as optim
from models.uresnet import UResNet
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import f1_score


def calculate_f1_score(pred, target):
    pred = torch.argmax(pred, dim=1).cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    return f1_score(target, pred, average='binary')

def calculate_iou(pred, target):
    pred = torch.argmax(pred, dim=1)
    pred_tumor = (pred == 1).float()
    target_tumor = (target == 1).float()
    intersection = (pred_tumor * target_tumor).sum(dim=(1, 2))
    union = pred_tumor.sum(dim=(1, 2)) + target_tumor.sum(dim=(1, 2)) - intersection
    iou = intersection / (union + 1e-6)
    iou[union == 0] = 1
    return iou.mean()

def train_model(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0

    for inputs, masks in tqdm(train_loader, desc="Training Batches"):
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        batch_iou = calculate_iou(outputs, masks).item()
        batch_f1 = calculate_f1_score(outputs, masks)

        running_loss += loss.item() * inputs.size(0)
        running_iou += batch_iou * inputs.size(0)
        running_f1 += batch_f1 * inputs.size(0)

        print(f'Batch Loss: {loss.item():.4f}, Batch IoU: {batch_iou:.4f}, Batch F1: {batch_f1:.4f}')

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_iou = running_iou / len(train_loader.dataset)
    epoch_f1 = running_f1 / len(train_loader.dataset)
    scheduler.step()
    return epoch_loss, epoch_iou, epoch_f1

def validate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0

    with torch.no_grad():
        for inputs, masks in tqdm(test_loader, desc="Validation Batches"):
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            batch_iou = calculate_iou(outputs, masks).item()
            batch_f1 = calculate_f1_score(outputs, masks)
            running_loss += loss.item() * inputs.size(0)
            running_iou += batch_iou * inputs.size(0)
            running_f1 += batch_f1 * inputs.size(0)
            print(f'Batch Loss: {loss.item():.4f}, Batch IoU: {batch_iou:.4f}, Batch F1: {batch_f1:.4f}')

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_iou = running_iou / len(test_loader.dataset)
    epoch_f1 = running_f1 / len(test_loader.dataset)
    return epoch_loss, epoch_iou, epoch_f1
def load_pairs_from_csv(csv_file, root):
    df = pd.read_csv(csv_file)
    pairs = []
    norm_root = os.path.normpath(root)
    for _, row in df.iterrows():
        img_path = row["Image"]
        mask_path = row["Mask"]
        # if img_path.startswith(norm_root):
        img_path = root + 'Images/' + img_path
        # if mask_path.startswith(norm_root):
        mask_path = root + 'Labels/' + mask_path
        # img_full = os.path.join(root, img_path)
        # mask_full = os.path.join(root, mask_path)
        pairs.append((img_path, mask_path))
    return pairs

if __name__ == "__main__":
    seed_value = 42
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_root = "../BUS-BRA/"
    train_csv = os.path.join(dataset_root, "train.csv")
    test_csv = os.path.join(dataset_root, "test.csv")
    train_pairs = load_pairs_from_csv(train_csv, dataset_root)
    test_pairs = load_pairs_from_csv(test_csv, dataset_root)

    train_dataset = BreastCancerSegmentation(train_pairs, 256, 256, is_augment=True)
    test_dataset = BreastCancerSegmentation(test_pairs, 256, 256, is_augment=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    print('Training samples #: ', len(train_dataset))
    print('Test samples #: ', len(test_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize model, criterion, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UResNet(n_classes=2, pretrained=True).to(device)
    # Print the trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}")

    # Total number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Directory to save checkpoints
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    num_epochs = 50

    for epoch in range(num_epochs):
        train_loss, train_iou, train_f1 = train_model(model, train_loader, criterion, optimizer, scheduler, device)
        test_loss, test_iou, test_f1 = validate_model(model, test_loader, criterion, device)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Train F1: {train_f1:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}, Test F1: {test_f1:.4f}')

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    final_model_path = os.path.join(checkpoint_dir, "resnet50_model_bus_bra.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")




