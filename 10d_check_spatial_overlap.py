"""
Phase 6, Step 4: Check actual spatial overlap between our AOI and the
12 "intersecting" Qiu & Ran grid cells — before trusting the area comparison

Prereqs (already installed): geopandas, shapely
"""

import geopandas as gpd
from shapely.geometry import box

SHP_PATH = 'data/validation/SGL_occurrence_grids_shapefile/SGL_occurrence_grids_shapefile/GrIS_Grid_10km_Periphry_3006.shp'
OVERLAPPING_GRID_IDS = [168, 1568, 1716, 1913, 1916, 1978, 2478, 2869, 2896, 3132, 3628, 3639]

grid = gpd.read_file(SHP_PATH)
aoi_wgs84 = gpd.GeoDataFrame({'geometry': [box(-50.35, 67.00, -49.55, 67.30)]}, crs='EPSG:4326')
aoi_reprojected = aoi_wgs84.to_crs(grid.crs)
aoi_geom = aoi_reprojected.geometry.iloc[0]

overlapping_cells = grid[grid['Grid_id'].isin(OVERLAPPING_GRID_IDS)]

print('Per-cell overlap with our AOI:')
total_overlap_km2 = 0
for _, row in overlapping_cells.iterrows():
    intersection = row['geometry'].intersection(aoi_geom)
    overlap_km2 = intersection.area / 1e6  # m^2 -> km^2
    overlap_fraction = overlap_km2 / row['Area']  # Area column is already in km^2 (100 = full cell)
    total_overlap_km2 += overlap_km2
    print(f"  Grid_id {int(row['Grid_id'])}: {overlap_km2:.1f} km² overlap ({overlap_fraction*100:.0f}% of the cell)")

print(f'\nOur AOI total area: {aoi_geom.area / 1e6:.1f} km²')
print(f'Sum of 12 grid cells (full, nominal): {12 * 100} km²')
print(f'Actual overlap area (grid cells ∩ our AOI): {total_overlap_km2:.1f} km²')
print(f'Overlap as % of our AOI: {total_overlap_km2 / (aoi_geom.area/1e6) * 100:.0f}%')