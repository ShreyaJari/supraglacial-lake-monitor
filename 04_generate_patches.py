"""
Phase 3, Step 2: Generate training patches from exported GeoTIFFs
Russell Glacier / Paakitsoq, SW Greenland — 31 clean dates

Reads each full-scene GeoTIFF (bands: B2, B3, B4, B8, B11, NDWI, lake_mask),
tiles into 256x256 patches, filters out patches with too much cloud-masked
(nodata) area, and splits into train/val/test BY DATE (not by random patch)
to avoid near-duplicate patches leaking across splits.

Note: splitting by date still means the same spatial location appears across
multiple dates within a split (fixed AOI, tiled identically every date) —
this is a known limitation of single-region/single-season training data.
Worth flagging in the portfolio writeup as a direction for future work
(multi-region training would give a better test of generalization).

Prereqs:
    pip install rasterio numpy pandas
"""

import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import random

# --- Config ---
RAW_DIR = Path('data/raw')
PATCHES_DIR = Path('data/patches')
PATCH_SIZE = 256
MAX_NODATA_FRACTION = 0.2   # skip patches with >20% cloud-masked/nodata pixels
RANDOM_SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # remainder (0.15) goes to test

BAND_NAMES = ['B2', 'B3', 'B4', 'B8', 'B11', 'NDWI', 'lake_mask']

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- 1. Set up output directories ---
for split in ['train', 'val', 'test']:
    (PATCHES_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
    (PATCHES_DIR / split / 'masks').mkdir(parents=True, exist_ok=True)

# --- 2. Find all exported GeoTIFFs and split dates into train/val/test ---
tif_files = sorted(RAW_DIR.glob('sglm_*.tif'))
dates = [f.stem.replace('sglm_', '') for f in tif_files]
print(f'Found {len(tif_files)} GeoTIFFs')

shuffled_dates = dates.copy()
random.shuffle(shuffled_dates)
n_train = int(len(shuffled_dates) * TRAIN_FRAC)
n_val = int(len(shuffled_dates) * VAL_FRAC)

date_split = {}
for d in shuffled_dates[:n_train]:
    date_split[d] = 'train'
for d in shuffled_dates[n_train:n_train + n_val]:
    date_split[d] = 'val'
for d in shuffled_dates[n_train + n_val:]:
    date_split[d] = 'test'

print(f"Split: {list(date_split.values()).count('train')} train dates, "
      f"{list(date_split.values()).count('val')} val dates, "
      f"{list(date_split.values()).count('test')} test dates")

# --- 3. Tile each GeoTIFF into patches ---
manifest = []
patch_counter = 0

for tif_path, date_str in zip(tif_files, dates):
    split = date_split[date_str]

    with rasterio.open(tif_path) as src:
        data = src.read()  # shape: (bands, H, W)
        nodata_val = src.nodata

    n_bands, height, width = data.shape

    # Identify nodata mask (True = missing/cloud-masked)
    if nodata_val is not None:
        invalid = np.any(data == nodata_val, axis=0)
    else:
        invalid = np.any(np.isnan(data), axis=0)

    n_patches_this_date = 0

    for row in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):
        for col in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):
            patch = data[:, row:row + PATCH_SIZE, col:col + PATCH_SIZE]
            patch_invalid = invalid[row:row + PATCH_SIZE, col:col + PATCH_SIZE]

            nodata_fraction = patch_invalid.mean()
            if nodata_fraction > MAX_NODATA_FRACTION:
                continue  # skip: too much cloud/missing data

            image_patch = patch[:6]   # B2, B3, B4, B8, B11, NDWI
            mask_patch = patch[6]     # lake_mask

            # Replace any remaining nodata pixels with 0 (image) — small
            # residual gaps under our 20% threshold are fine to zero-fill
            image_patch = np.where(np.isnan(image_patch), 0, image_patch)
            mask_patch = np.where(np.isnan(mask_patch), 0, mask_patch)

            lake_fraction = mask_patch.mean()

            patch_id = f'{date_str}_r{row}_c{col}'
            np.save(PATCHES_DIR / split / 'images' / f'{patch_id}.npy', image_patch.astype(np.float32))
            np.save(PATCHES_DIR / split / 'masks' / f'{patch_id}.npy', mask_patch.astype(np.float32))

            manifest.append({
                'patch_id': patch_id,
                'date': date_str,
                'split': split,
                'nodata_fraction': round(nodata_fraction, 3),
                'lake_pixel_fraction': round(lake_fraction, 4)
            })
            n_patches_this_date += 1
            patch_counter += 1

    print(f'  {date_str} ({split}): {n_patches_this_date} patches')

# --- 4. Save manifest ---
manifest_df = pd.DataFrame(manifest)
manifest_df.to_csv(PATCHES_DIR / 'manifest.csv', index=False)

print(f'\nTotal patches generated: {patch_counter}')
print(manifest_df.groupby('split').size())
print(f'\nPatches with at least some lake pixels: {(manifest_df["lake_pixel_fraction"] > 0).sum()} '
      f'({(manifest_df["lake_pixel_fraction"] > 0).mean()*100:.1f}%)')
print(f'Saved manifest: {PATCHES_DIR / "manifest.csv"}')