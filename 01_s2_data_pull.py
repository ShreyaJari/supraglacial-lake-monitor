"""
Phase 1: Sentinel-2 data acquisition — Russell Glacier / Paakitsoq, SW Greenland
Melt season: 2019 (June 1 - Sept 15)

Prereqs:
    pip install earthengine-api geemap matplotlib requests pillow
    earthengine authenticate   # run once in terminal, follow browser auth flow
"""

import ee
import datetime
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# --- 1. Initialize Earth Engine ---
ee.Initialize(project='supraglacial-lake-monitor')

# --- 2. Define Area of Interest (Russell Glacier / Paakitsoq, SW Greenland) ---
# Rough bounding box — refine once we inspect real coverage
aoi = ee.Geometry.Rectangle([-50.35, 67.00, -49.55, 67.30])

# --- 3. Cloud masking function (Sentinel-2 SCL band) ---
def mask_s2_clouds(image):
    scl = image.select('SCL')
    # Keep: vegetation(4), bare soil(5), water(6), snow/ice(11)
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
    .select(['B2', 'B3', 'B4', 'B8'])  # Blue, Green, Red, NIR — enough for NDWI + RGB
)

print('Number of scenes found:', collection.size().getInfo())

# --- 5. List scene dates so we know what we're working with ---
dates = collection.aggregate_array('system:time_start').getInfo()
for d in sorted(dates):
    print(datetime.datetime.utcfromtimestamp(d / 1000).strftime('%Y-%m-%d'))

# --- 6. Visual sanity check: export first scene as a static RGB thumbnail ---
first_image = collection.first()
vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}

url = first_image.getThumbURL({**vis_params, 'region': aoi, 'dimensions': 800, 'format': 'png'})
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img.save('first_scene_rgb.png')

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.title('First scene RGB — Russell Glacier / Paakitsoq AOI')
plt.axis('off')
plt.tight_layout()
plt.savefig('aoi_check.png', dpi=150)
print('Saved: aoi_check.png — open this in VS Code to confirm AOI coverage')