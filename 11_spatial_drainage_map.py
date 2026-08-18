"""
Phase 7 (viz): Spatial map of detected drainage events
Russell Glacier / Paakitsoq — overlays drainage event locations on a
representative mid-season Sentinel-2 scene.

Prereqs (already installed): rasterio, pandas, numpy, matplotlib
"""

import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RAW_DIR = Path('data/raw')
BASEMAP_DATE = '2019-07-20'  # mid-season, low cloud, good visual reference

# --- 1. Load a representative scene as the visual basemap ---
with rasterio.open(RAW_DIR / f'sglm_{BASEMAP_DATE}.tif') as src:
    data = src.read()

rgb = data[[2, 1, 0]]  # B4, B3, B2 -> RGB
rgb = np.transpose(rgb, (1, 2, 0))
rgb = np.clip(rgb / 0.3, 0, 1)
rgb = np.nan_to_num(rgb, nan=1.0)  # nodata -> white

# --- 2. Load lake tracks (has per-date pixel centroids) and drainage events ---
tracks_df = pd.read_csv('data/lake_instances/lake_tracks_persistent.csv')
events_df = pd.read_csv('drainage_events_v2.csv')

# --- 3. For each event, get the lake's centroid at the date JUST BEFORE it drained ---
events_with_location = events_df.merge(
    tracks_df[['lake_id', 'date', 'centroid_row', 'centroid_col']],
    left_on=['lake_id', 'date_before'],
    right_on=['lake_id', 'date'],
    how='left'
)

print(f'Events with matched location: {events_with_location["centroid_row"].notna().sum()} / {len(events_with_location)}')

# --- 4. Plot ---
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(rgb)

high_conf = events_with_location[events_with_location['confidence'] == 'HIGH']
med_conf = events_with_location[events_with_location['confidence'] != 'HIGH']

ax.scatter(med_conf['centroid_col'], med_conf['centroid_row'],
           s=25, c='orange', alpha=0.5, label=f'Medium confidence ({len(med_conf)})', edgecolors='none')
ax.scatter(high_conf['centroid_col'], high_conf['centroid_row'],
           s=45, c='crimson', alpha=0.85, label=f'High confidence ({len(high_conf)})', edgecolors='white', linewidths=0.5)

ax.set_title(f'Detected supraglacial lake drainage events, 2019 melt season\n(locations overlaid on {BASEMAP_DATE} Sentinel-2 scene, Russell Glacier / Paakitsoq)')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlabel('Pixel column (10m resolution)')
ax.set_ylabel('Pixel row (10m resolution)')

plt.tight_layout()
plt.savefig('figures/spatial_drainage_map.png', dpi=150)
print('Saved: figures/spatial_drainage_map.png')