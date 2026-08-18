"""
Phase 4, Step 1: Full-scene inference + lake instance extraction
Russell Glacier / Paakitsoq, SW Greenland — all 31 clean dates

Runs the trained U-Net across each full-scene GeoTIFF (tiled inference,
stitched back together), then uses connected-component labeling to break
the binary lake mask into individual lake instances with area, centroid,
and bounding box — the per-date building blocks for cross-date tracking.

Prereqs:
    pip install torch rasterio scipy numpy pandas
"""

import torch
import torch.nn as nn
import numpy as np
import rasterio
from scipy import ndimage
import pandas as pd
from pathlib import Path

RAW_DIR = Path('data/raw')
OUTPUT_DIR = Path('data/lake_instances')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PATCH_SIZE = 256
N_CHANNELS = 6
MIN_LAKE_PIXELS = 4  # drop instances smaller than ~400 m^2 (4 pixels @ 10m) as noise

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')


# --- U-Net (same architecture as training) ---
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


model = UNet(in_channels=N_CHANNELS).to(device)
model.load_state_dict(torch.load('best_unet_model.pt', map_location=device))
model.eval()


def run_full_scene_inference(tif_path):
    """Tile the full scene, run the model on each tile, stitch predictions back."""
    with rasterio.open(tif_path) as src:
        data = src.read()  # (bands, H, W)
        transform = src.transform
        crs = src.crs

    n_bands, height, width = data.shape
    image_bands = data[:6]  # B2, B3, B4, B8, B11, NDWI

    pred_mask = np.zeros((height, width), dtype=np.float32)

    with torch.no_grad():
        for row in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):
            for col in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):
                tile = image_bands[:, row:row + PATCH_SIZE, col:col + PATCH_SIZE]
                tile = np.where(np.isnan(tile), 0, tile)
                tile_tensor = torch.from_numpy(tile).float().unsqueeze(0).to(device)

                logits = model(tile_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
                pred_mask[row:row + PATCH_SIZE, col:col + PATCH_SIZE] = (probs > 0.5).astype(np.float32)

    return pred_mask, transform, crs


def extract_lake_instances(pred_mask, transform, date_str):
    """Connected-component labeling: turn the binary mask into individual lake instances."""
    labeled_array, n_features = ndimage.label(pred_mask)

    instances = []
    for lake_id in range(1, n_features + 1):
        lake_pixels = (labeled_array == lake_id)
        n_pixels = lake_pixels.sum()

        if n_pixels < MIN_LAKE_PIXELS:
            continue

        rows, cols = np.where(lake_pixels)
        centroid_row, centroid_col = rows.mean(), cols.mean()
        # Convert pixel centroid to real-world coordinates using the raster transform
        centroid_x, centroid_y = transform * (centroid_col, centroid_row)

        area_km2 = (n_pixels * 100) / 1e6  # 100 = 10m x 10m pixel area

        instances.append({
            'date': date_str,
            'instance_id_local': lake_id,
            'area_km2': round(area_km2, 5),
            'n_pixels': int(n_pixels),
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'row_min': int(rows.min()), 'row_max': int(rows.max()),
            'col_min': int(cols.min()), 'col_max': int(cols.max()),
        })

    return instances


# --- Process all dates ---
tif_files = sorted(RAW_DIR.glob('sglm_*.tif'))
all_instances = []

print(f'Processing {len(tif_files)} scenes...')
for tif_path in tif_files:
    date_str = tif_path.stem.replace('sglm_', '')
    pred_mask, transform, crs = run_full_scene_inference(tif_path)
    instances = extract_lake_instances(pred_mask, transform, date_str)
    all_instances.extend(instances)
    print(f'  {date_str}: {len(instances)} lake instances, total area {sum(i["area_km2"] for i in instances):.2f} km²')

    # Save the full predicted mask too, for later visual QC if needed
    np.save(OUTPUT_DIR / f'{date_str}_pred_mask.npy', pred_mask)

df = pd.DataFrame(all_instances)
df.to_csv(OUTPUT_DIR / 'lake_instances_all_dates.csv', index=False)
print(f'\nTotal lake instances across all dates: {len(df)}')
print(f'Saved: {OUTPUT_DIR / "lake_instances_all_dates.csv"}')