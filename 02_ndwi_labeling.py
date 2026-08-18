"""
Phase 2 (final): Full-season lake area time series with coverage tracking
Russell Glacier / Paakitsoq, SW Greenland — 2019 melt season (35 scenes)

Labeling rule (validated via histogram + visual inspection across v1-v3):
    lake_mask = (NDWI > 0.15) AND (B4/red < 0.15)

Rationale:
- Plain NDWI alone gives dry snow a false-positive water signal (~0.2-0.3),
  since snow's green reflectance is slightly higher than NIR even when dry.
- SWIR (B11) does NOT fix this, because dry snow and liquid water are both
  SWIR-absorbing — SWIR only separates snow+water from rock/cloud.
- Red band (B4) is the fix: liquid water absorbs red strongly (dark),
  snow/ice reflects it strongly (bright) — a clean, well-separated bimodal
  signal (standard in the supraglacial lake literature, e.g. Selmes et al.,
  Yang & Smith).

Also tracks per-date AOI coverage (% non-cloud-masked pixels) to catch dates
where cloud sits specifically over the melt zone — a whole-scene
CLOUDY_PIXEL_PERCENTAGE filter doesn't catch this, since it's a scene-wide
average, not AOI-specific. Confirmed via mask-overlay visualization that 4
dates (06-26, 07-25, 08-05, 08-30) have cloud concentrated directly over the
lake-rich zone; these are flagged and excluded from the clean output.

Prereqs:
    pip install earthengine-api pandas
    earthengine authenticate   # one-time
"""

import ee
import pandas as pd
import datetime

# --- 1. Initialize Earth Engine ---
ee.Initialize(project='supraglacial-lake-monitor')

# --- 2. Define Area of Interest (Russell Glacier / Paakitsoq, SW Greenland) ---
aoi = ee.Geometry.Rectangle([-50.35, 67.00, -49.55, 67.30])

# --- 3. Cloud masking function (Sentinel-2 SCL band) ---
def mask_s2_clouds(image):
    scl = image.select('SCL')
    # Mask: cloud shadow(3), cloud medium(8), cloud high(9), cirrus(10)
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask).divide(10000).copyProperties(image, ['system:time_start'])

# --- 4. Build the Sentinel-2 L2A collection ---
collection = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate('2019-06-01', '2019-09-15')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    .map(mask_s2_clouds)
    .select(['B3', 'B4', 'B8'])
)

# --- 5. NDWI + lake mask ---
NDWI_THRESHOLD = 0.15
RED_THRESHOLD = 0.15

def add_ndwi(image):
    return image.addBands(image.normalizedDifference(['B3', 'B8']).rename('NDWI'))

def add_lake_mask(image):
    lake_mask = (
        image.select('NDWI').gt(NDWI_THRESHOLD)
        .And(image.select('B4').lt(RED_THRESHOLD))
    ).rename('lake_mask')
    return image.addBands(lake_mask)

collection = collection.map(add_ndwi).map(add_lake_mask)

# --- 6. Total AOI pixel count at 10m, for coverage % calculation ---
total_pixels = ee.Image.pixelArea().divide(100).reduceRegion(  # 100 = 10m x 10m
    reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e9
).get('area').getInfo()

# --- 7. Process every scene: lake area + coverage % ---
dates_ms = collection.aggregate_array('system:time_start').getInfo()
dates_sorted = sorted(dates_ms)

results = []
print(f'Processing {len(dates_sorted)} scenes...')

for i, ms in enumerate(dates_sorted):
    date_str = datetime.datetime.utcfromtimestamp(ms / 1000).strftime('%Y-%m-%d')
    image = collection.filterDate(date_str, ee.Date(date_str).advance(1, 'day')).first()

    # Lake area (km²)
    lake_stats = image.select('lake_mask').multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e9
    ).get('lake_mask')
    lake_area_m2 = lake_stats.getInfo()
    lake_area_km2 = (lake_area_m2 / 1e6) if lake_area_m2 is not None else 0.0

    # Valid (non-cloud-masked) pixel count as a coverage proxy
    valid_count = image.select('NDWI').reduceRegion(
        reducer=ee.Reducer.count(), geometry=aoi, scale=10, maxPixels=1e9
    ).get('NDWI').getInfo()
    coverage_pct = (valid_count / total_pixels) * 100 if valid_count else 0.0

    results.append({
        'date': date_str,
        'lake_area_km2': round(lake_area_km2, 3),
        'coverage_pct': round(coverage_pct, 1)
    })

    flag = '  <-- LOW COVERAGE' if coverage_pct < 80 else ''
    print(f'  [{i+1}/{len(dates_sorted)}] {date_str}: {lake_area_km2:.2f} km²  |  coverage: {coverage_pct:.1f}%{flag}')

# --- 8. Save full (unfiltered) results ---
df = pd.DataFrame(results)
df.to_csv('lake_area_timeseries_2019.csv', index=False)
print('\nSaved: lake_area_timeseries_2019.csv (all 35 dates)')

# --- 9. Exclude dates confirmed as cloud-contaminated via visual inspection ---
# (coverage_pct alone under-catches this — see scoping doc Phase 2 notes)
EXCLUDED_DATES = ['2019-06-26', '2019-07-25', '2019-08-05', '2019-08-30']
df_clean = df[~df['date'].isin(EXCLUDED_DATES)].reset_index(drop=True)
df_clean.to_csv('lake_area_timeseries_2019_clean.csv', index=False)

print(f'\nClean series: {len(df_clean)} dates (excluded {len(EXCLUDED_DATES)} cloud-contaminated)')
print('Saved: lake_area_timeseries_2019_clean.csv')
print(df_clean['lake_area_km2'].describe())