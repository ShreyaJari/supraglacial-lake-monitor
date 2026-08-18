"""
Phase 3, Step 5: Evaluate trained U-Net on held-out test set
Russell Glacier / Paakitsoq, SW Greenland

Loads best_unet_model.pt (selected by validation loss during training) and
evaluates on the 6 test dates (951 patches) never seen during training or
validation. Reports IoU, Dice, precision, recall, and saves qualitative
comparison figures (input / ground truth / prediction) for the portfolio.

Prereqs (already installed): torch, numpy, pandas, matplotlib
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

PATCHES_DIR = Path('data/patches')
N_CHANNELS = 6

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')


# --- Same Dataset and UNet classes as training script ---
class LakePatchDataset(Dataset):
    def __init__(self, patches_dir, split):
        self.images_dir = patches_dir / split / 'images'
        self.masks_dir = patches_dir / split / 'masks'
        self.patch_ids = [f.stem for f in sorted(self.images_dir.glob('*.npy'))]

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        image = np.load(self.images_dir / f'{patch_id}.npy')
        mask = np.load(self.masks_dir / f'{patch_id}.npy')
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float().unsqueeze(0), patch_id


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
        return self.out_conv(d1)


# --- Metrics ---
@torch.no_grad()
def compute_metrics(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))

    intersection = tp
    union = tp + fp + fn
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (2 * tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)

    return iou, dice, precision, recall


def evaluate():
    test_ds = LakePatchDataset(PATCHES_DIR, 'test')
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    print(f'Test patches: {len(test_ds)}')

    model = UNet(in_channels=N_CHANNELS).to(device)
    model.load_state_dict(torch.load('best_unet_model.pt', map_location=device))
    model.eval()

    all_iou, all_dice, all_precision, all_recall = [], [], [], []

    with torch.no_grad():
        for images, masks, patch_ids in test_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            iou, dice, precision, recall = compute_metrics(logits, masks)
            all_iou.extend(iou.cpu().numpy())
            all_dice.extend(dice.cpu().numpy())
            all_precision.extend(precision.cpu().numpy())
            all_recall.extend(recall.cpu().numpy())

    print('\n=== Test Set Results (951 patches, 6 held-out dates) ===')
    print(f'Mean IoU:       {np.mean(all_iou):.4f}')
    print(f'Mean Dice:      {np.mean(all_dice):.4f}')
    print(f'Mean Precision: {np.mean(all_precision):.4f}')
    print(f'Mean Recall:    {np.mean(all_recall):.4f}')

    # --- Qualitative comparison: 6 example patches with at least some lake ---
    model.eval()
    fig, axes = plt.subplots(6, 3, figsize=(9, 18))

    shown = 0
    with torch.no_grad():
        for images, masks, patch_ids in test_loader:
            if shown >= 6:
                break
            images_dev = images.to(device)
            logits = model(images_dev)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu()

            for i in range(images.size(0)):
                if shown >= 6:
                    break
                if masks[i].sum() < 10:  # skip near-empty patches for clearer examples
                    continue

                rgb = images[i, [2, 1, 0]].permute(1, 2, 0).numpy()  # B4,B3,B2 -> RGB
                rgb = np.clip(rgb / 0.3, 0, 1)

                axes[shown, 0].imshow(rgb)
                axes[shown, 0].set_title(f'{patch_ids[i]}\nInput RGB' if shown == 0 else patch_ids[i])
                axes[shown, 1].imshow(masks[i, 0], cmap='Blues', vmin=0, vmax=1)
                axes[shown, 1].set_title('Ground Truth' if shown == 0 else '')
                axes[shown, 2].imshow(preds[i, 0], cmap='Blues', vmin=0, vmax=1)
                axes[shown, 2].set_title('Prediction' if shown == 0 else '')

                for ax in axes[shown]:
                    ax.axis('off')
                shown += 1

    plt.tight_layout()
    plt.savefig('test_predictions_qualitative.png', dpi=150)
    print('\nSaved: test_predictions_qualitative.png')


if __name__ == '__main__':
    evaluate()