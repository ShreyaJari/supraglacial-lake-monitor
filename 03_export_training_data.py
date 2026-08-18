"""
Phase 3, Step 1: Export training imagery + labels to Google Drive
Russell Glacier / Paakitsoq, SW Greenland — 31 clean dates from Phase 2

Submits one Earth Engine batch export task per date. Each export is a
multi-band GeoTIFF containing:
    B2, B3, B4, B8, B11  — reflectance bands (input features)
    NDWI                  — computed water index (input feature)
    lake_mask             — our validated label (target for training)

Exports run asynchronously on Google's servers — this script only submits
the tasks, it does not wait for them to finish. Monitor progress at:
    https://code.earthengine.google.com/tasks

Prereqs (already installed): earthengine-api
"""

import ee

ee.Initialize(project='supraglacial-lake-monitor')

aoi = ee.Geometry.Rectangle([-50.35, 67.00, -49.55, 67.30])

def mask_s2_clouds(image):
    scl = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask).divide(10000).copyProperties(image, ['system:time_start'])

collection = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate('2019-06-01', '2019-09-15')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    .map(mask_s2_clouds)
    .select(['B2', 'B3', 'B4', 'B8', 'B11'])
)

NDWI_THRESHOLD = 0.15
RED_THRESHOLD = 0.15

def add_bands(image):
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    lake_mask = (
        ndwi.gt(NDWI_THRESHOLD).And(image.select('B4').lt(RED_THRESHOLD))
    ).rename('lake_mask').toFloat()   # <-- added .toFloat() here
    return image.addBands(ndwi).addBands(lake_mask)

collection = collection.map(add_bands)

# --- Clean dates from Phase 2 (cloud-contaminated dates already excluded) ---
CLEAN_DATES = [
    '2019-06-01', '2019-06-05', '2019-06-08', '2019-06-10', '2019-06-11',
    '2019-06-15', '2019-06-16', '2019-06-18', '2019-06-20',
    '2019-07-03', '2019-07-06', '2019-07-08', '2019-07-10', '2019-07-11',
    '2019-07-15', '2019-07-16', '2019-07-20', '2019-07-21', '2019-07-23',
    '2019-07-31',
    '2019-08-02', '2019-08-04', '2019-08-09', '2019-08-10', '2019-08-12',
    '2019-08-20', '2019-08-24', '2019-08-25', '2019-08-29',
    '2019-09-01', '2019-09-09'
]

print(f'Submitting {len(CLEAN_DATES)} export tasks...')

tasks = []
for date_str in CLEAN_DATES:
    image = collection.filterDate(date_str, ee.Date(date_str).advance(1, 'day')).first()
    image = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'NDWI', 'lake_mask'])

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f'sglm_{date_str}',
        folder='sglm_training_data',
        fileNamePrefix=f'sglm_{date_str}',
        region=aoi,
        scale=10,
        crs='EPSG:32622',  # UTM zone 22N — appropriate for this longitude in Greenland
        maxPixels=1e10,
        fileFormat='GeoTIFF'
    )
    task.start()
    tasks.append((date_str, task))
    print(f'  Submitted: {date_str}')

print('\nAll tasks submitted.')
print('Monitor progress at: https://code.earthengine.google.com/tasks')
print('Each file will appear in your Google Drive under a folder named "sglm_training_data" once complete.')