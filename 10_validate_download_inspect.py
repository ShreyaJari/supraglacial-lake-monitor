"""
Phase 6, Step 1: Download + inspect Qiu & Ran (2023) validation dataset
Source: Zenodo record 13924069 (Greenland-wide supraglacial lake area,
2017-2022, CC BY 4.0)

This step ONLY downloads and inspects the data structure — we don't yet
know the exact column names/schema, so rather than guess and write
analysis code blind, we look first, then build the real comparison in
step 2 based on what's actually there.

Prereqs:
    pip install requests geopandas
"""

import requests
import zipfile
from pathlib import Path
import geopandas as gpd
import pandas as pd

VALIDATION_DIR = Path('data/validation')
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

ZENODO_RECORD_ID = '13924069'

# --- 1. Query Zenodo API for the actual file list (avoids guessing filenames) ---
api_url = f'https://zenodo.org/api/records/{ZENODO_RECORD_ID}'
response = requests.get(api_url)
response.raise_for_status()
record = response.json()

print('Files available in this Zenodo record:')
for f in record['files']:
    print(f"  {f['key']}  ({f['size'] / 1e6:.1f} MB)")
    print(f"    -> {f['links']['self']}")

# --- 2. Download each file (skip if already present) ---
for f in record['files']:
    filename = f['key']
    filepath = VALIDATION_DIR / filename
    if filepath.exists():
        print(f'Already downloaded: {filename}')
        continue
    print(f'Downloading {filename}...')
    download_url = f['links']['self']
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, 'wb') as out:
            for chunk in r.iter_content(chunk_size=8192):
                out.write(chunk)
    print(f'  Saved: {filepath}')

# --- 3. Unzip everything ---
for zip_path in VALIDATION_DIR.glob('*.zip'):
    extract_dir = VALIDATION_DIR / zip_path.stem
    if not extract_dir.exists():
        print(f'Extracting {zip_path.name}...')
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)

# --- 4. Inspect whatever we got: shapefile(s) and CSV(s) ---
print('\n=== Inspecting extracted contents ===')
for shp_path in VALIDATION_DIR.rglob('*.shp'):
    print(f'\nShapefile: {shp_path}')
    gdf = gpd.read_file(shp_path)
    print(f'  Shape: {gdf.shape}')
    print(f'  Columns: {list(gdf.columns)}')
    print(f'  CRS: {gdf.crs}')
    print(gdf.head(3))

for csv_path in VALIDATION_DIR.rglob('*.csv'):
    print(f'\nCSV: {csv_path}')
    df = pd.read_csv(csv_path, nrows=5)
    print(f'  Columns: {list(df.columns)}')
    print(df.head(3))