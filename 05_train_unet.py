"""
Phase 3, Step 3-4: U-Net model + training loop
Russell Glacier / Paakitsoq, SW Greenland

Input: 6-channel patches (B2, B3, B4, B8, B11, NDWI), 256x256
Output: binary lake segmentation mask

Uses a lightweight U-Net (few filters, since lakes have a strong, learnable
spectral signature — no need for a huge model) and Dice loss (handles the
class imbalance between lake/non-lake pixels much better than plain BCE).

Prereqs:
    pip install torch torchvision numpy pandas tqdm matplotlib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Config ---
PATCHES_DIR = Path('data/patches')
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
N_CHANNELS = 6  # B2, B3, B4, B8, B11, NDWI

# --- Device selection: MPS (Apple Silicon GPU) > CUDA > CPU ---
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')


# --- 1. Dataset ---
class LakePatchDataset(Dataset):
    def __init__(self, patches_dir, split):
        self.images_dir = patches_dir / split / 'images'
        self.masks_dir = patches_dir / split / 'masks'
        self.patch_ids = [f.stem for f in sorted(self.images_dir.glob('*.npy'))]

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        image = np.load(self.images_dir / f'{patch_id}.npy')  # (6, 256, 256)
        mask = np.load(self.masks_dir / f'{patch_id}.npy')    # (256, 256)
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float().unsqueeze(0)


# --- 2. U-Net architecture (lightweight) ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, base_filters=32):
        super().__init__()
        f = base_filters
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.enc4 = DoubleConv(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(f * 8, f * 16)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = DoubleConv(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)

        self.out_conv = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)  # raw logits


# --- 3. Dice loss ---
def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (probs_flat * targets_flat).sum(dim=1)
    dice = (2. * intersection + smooth) / (probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth)
    return 1 - dice.mean()


def combined_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return bce + dice


# --- 4. Metrics ---
@torch.no_grad()
def compute_iou(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# --- 5. Training loop ---
def train():
    train_ds = LakePatchDataset(PATCHES_DIR, 'train')
    val_ds = LakePatchDataset(PATCHES_DIR, 'val')
    print(f'Train patches: {len(train_ds)}, Val patches: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet(in_channels=N_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [train]'):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = combined_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = combined_loss(logits, masks)
                val_loss += loss.item() * images.size(0)
                val_iou += compute_iou(logits, masks) * images.size(0)
        val_loss /= len(val_ds)
        val_iou /= len(val_ds)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)

        print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_IoU={val_iou:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_unet_model.pt')
            print(f'  -> Saved new best model (val_loss={val_loss:.4f})')

    # --- Plot training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].set_title('Loss')
    axes[1].plot(history['val_iou'], label='Val IoU', color='green')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('IoU'); axes[1].legend(); axes[1].set_title('Validation IoU')
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print('\nSaved: training_curves.png')
    print(f'Best val_loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    train()