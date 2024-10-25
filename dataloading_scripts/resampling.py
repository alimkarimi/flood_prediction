import xarray as xr
from dataloading_scripts.open_era5 import nc_era5
from dataloading_scripts.open_dtm import open_dtm

import rasterio
from rasterio.enums import Resampling

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image



# steps to reproject DTM to common grid.

# 1. check that DTM and climate data exist in common projection
# nc_era5 is in lat / long
dtm = open_dtm('../dtm_data/MeanElevation_Kenya.tif', verbose=False)
print(dtm.crs) # this is EPSG: 4326

# 2. if not, reproject to common projection
# in a common projection
# 3. trim DTM to bounds of climate data
# get extents of DTM: 

left = dtm.bounds[0]
bottom = dtm.bounds[1]

right = dtm.bounds[2]
top = dtm.bounds[3]

print(left, bottom, right, top)

# get extents of nc_era5 spatial data:
print(nc_era5.variables['latitude'][:].min())
left_nce = nc_era5.variables['longitude'][:].min()
bottom_nce = nc_era5.variables['latitude'][:].min()

right_nce = nc_era5.variables['longitude'][:].max()
top_nce = nc_era5.variables['latitude'][:].max()
print(left_nce, bottom_nce, right_nce, top_nce)

#print(dir(dtm))
print(dtm.statistics(1), dtm.units, dtm.scales, dtm.res, dtm.crs,)

print(dtm.index(left, bottom))
print(dtm.index(right, top))
print(dtm.index(left, top))

print(dtm.width, dtm.height, dtm.shape)

# Access the latitude and longitude variables
latitudes = nc_era5.variables['latitude'][:]
longitudes = nc_era5.variables['longitude'][:]

# Calculate the resolution as the difference between consecutive points
lat_res_era5 = abs(latitudes[1] - latitudes[0])
lon_res_era5 = abs(longitudes[1] - longitudes[0])

print(f"Latitude resolution: {lat_res_era5} degrees")
print(f"Longitude resolution: {lon_res_era5} degrees")
# 4 - try to resample with a factor of source / target (source is the dtm, target is the climate ERA5 data)

scale_factor = dtm.res[0] / lat_res_era5
with rasterio.open("./dtm_data/MeanElevation_Kenya.tif") as dataset:

    # resample data to target shape
    data = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height * scale_factor),
            int(dataset.width * scale_factor)
        ),
        resampling=Resampling.bilinear
    )

    # scale image transform
    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data.shape[-1]),
        (dataset.height / data.shape[-2])
    ) 
    print(data)
    print(type(data))
    print(data.shape)
    print(np.max(data), np.min(data), np.std(data))

        # Define new metadata for the output file
    new_meta = dataset.meta.copy()
    new_meta.update({
        "height": data.shape[1],
        "width": data.shape[2],
        "transform": transform
    })

    # Write resampled data to a new file
    with rasterio.open("./dtm_data/resampled_example.tif", "w", **new_meta) as dest:
            dest.write(data)

# open resampled_example.tif:

dtm_resampled = open_dtm('../dtm_data/resampled_example.tif', verbose=False)
print(dtm_resampled.crs)
print(dtm_resampled.statistics(1), dtm_resampled.units, dtm_resampled.scales, dtm_resampled.res, dtm_resampled.crs,)

print(dtm_resampled.index(left, bottom))
print(dtm_resampled.index(right, top))
print(dtm_resampled.index(left, top))

arr = Image.open('./dtm_data/resampled_example.tif')
# Convert to numpy array for processing
arr_np = np.array(arr)

# Optionally, you can define vmin and vmax based on your data
vmin = np.min(arr_np)  # Minimum elevation value
vmax = np.max(arr_np)  # Maximum elevation value

# Create a figure and axis
plt.figure(figsize=(8, 6))
# Display the image with a suitable colormap for elevation
img_display = plt.imshow(arr_np, cmap='terrain', vmin=vmin, vmax=vmax)

# Add a color bar
cbar = plt.colorbar(img_display)
cbar.set_label('Elevation (meters)')  # Set the label for the color bar

# Save the figure
plt.savefig('./dtm_data/resampled_displayed.jpg')

# Show the plot (optional)
plt.show()