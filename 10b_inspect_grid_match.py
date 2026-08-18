"""
Phase 6, Step 2: Find overlapping grid cell(s) + inspect file format
Russell Glacier / Paakitsoq AOI vs. Qiu & Ran (2023) 10km grid

Prereqs (already installed): geopandas, pandas
"""

import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from pathlib import Path

VALIDATION_DIR = Path('data/validation')
SHP_PATH = VALIDATION_DIR / 'SGL_occurrence_grids_shapefile/SGL_occurrence_grids_shapefile/GrIS_Grid_10km_Periphry_3006.shp'
TS_DIR = VALIDATION_DIR / 'SGL_area_time_series_with_uncertainty_assessment/SGL_area_time_series_with_uncertainty_assessment'

# --- 1. Load the grid and our AOI, reproject AOI into the grid's CRS ---
grid = gpd.read_file(SHP_PATH)
print(f'Grid CRS: {grid.crs}')

aoi_wgs84 = gpd.GeoDataFrame(
    {'geometry': [box(-50.35, 67.00, -49.55, 67.30)]}, crs='EPSG:4326'
)
aoi_reprojected = aoi_wgs84.to_crs(grid.crs)

# --- 2. Find overlapping grid cells ---
overlapping = gpd.sjoin(grid, aoi_reprojected, how='inner', predicate='intersects')
print(f'\nOverlapping grid cell(s): {len(overlapping)}')
print(overlapping[['Grid_id', 'Area']])

# --- 3. Inspect the file format for one overlapping grid cell ---
if len(overlapping) > 0:
    sample_grid_id = overlapping.iloc[0]['Grid_id']
    sample_file = TS_DIR / f'Area_Grid{sample_grid_id}.txt'
    print(f'\n=== Inspecting {sample_file.name} ===')
    if sample_file.exists():
        with open(sample_file) as f:
            lines = f.readlines()
        print(f'Total lines: {len(lines)}')
        print('First 10 lines (raw):')
        for line in lines[:10]:
            print(repr(line))
    else:
        print('File not found — check Grid_id matches filename convention exactly.')