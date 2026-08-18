"""
Phase 6, Step 5: Area-weighted validation comparison
Scales each grid cell's reported area by its actual overlap fraction with
our AOI, rather than using the full nominal cell value — the correct way
to do this comparison, confirming whether weighting narrows or widens the
gap found in step 3.

Prereqs (already installed): geopandas, pandas, shapely, matplotlib
"""

import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SHP_PATH = 'data/validation/SGL_occurrence_grids_shapefile/SGL_occurrence_grids_shapefile/GrIS_Grid_10km_Periphry_3006.shp'
TS_DIR = Path('data/validation/SGL_area_time_series_with_uncertainty_assessment/SGL_area_time_series_with_uncertainty_assessment')
OVERLAPPING_GRID_IDS = [168, 1568, 1716, 1913, 1916, 1978, 2478, 2869, 2896, 3132, 3628, 3639]

grid = gpd.read_file(SHP_PATH)
aoi_wgs84 = gpd.GeoDataFrame({'geometry': [box(-50.35, 67.00, -49.55, 67.30)]}, crs='EPSG:4326')
aoi_geom = aoi_wgs84.to_crs(grid.crs).geometry.iloc[0]

overlap_fractions = {}
for _, row in grid[grid['Grid_id'].isin(OVERLAPPING_GRID_IDS)].iterrows():
    intersection_km2 = row['geometry'].intersection(aoi_geom).area / 1e6
    overlap_fractions[int(row['Grid_id'])] = intersection_km2 / row['Area']

print('Overlap fractions used for weighting:', overlap_fractions)

all_records = []
for grid_id in OVERLAPPING_GRID_IDS:
    filepath = TS_DIR / f'Area_Grid{grid_id}.txt'
    df = pd.read_csv(filepath, sep='\t', header=None,
                      names=['decimal_year', 'date', 'area_km2', 'uncertainty_km2'])
    df['weighted_area_km2'] = df['area_km2'] * overlap_fractions[grid_id]
    all_records.append(df)

qiu_ran_df = pd.concat(all_records, ignore_index=True)
qiu_ran_df['date'] = pd.to_datetime(qiu_ran_df['date'])
qiu_ran_2019 = qiu_ran_df[(qiu_ran_df['date'] >= '2019-06-01') & (qiu_ran_df['date'] <= '2019-09-15')]

weighted_daily = qiu_ran_2019.groupby('date')['weighted_area_km2'].sum().reset_index().sort_values('date')
print('\nArea-weighted Qiu & Ran totals:')
print(weighted_daily.to_string(index=False))
print(f'\nWeighted mean: {weighted_daily["weighted_area_km2"].mean():.2f} km²')
print(f'(Unweighted mean was 3.60 km² — for comparison)')
print(f'Our mean: 22.26 km²')