import xarray as xr
import os
"""
What: This script merges all of the .nc files from 1996 - 2018 together. 

Why: The problem is that .nc files are individual year based making it difficult to build features or target cubes without opening
and closing individual files. Having one file across all years makes this work faster because of less I/O.  
"""
# Define the path to your files
file_path_template = "./data/features/download_{}.nc"

# List to store all datasets
datasets = []

# Loop through the years and load each file
for year in range(1996, 2019):
    file_path = file_path_template.format(year)
    if os.path.exists(file_path):
        ds = xr.open_dataset(file_path, engine="netcdf4")
        datasets.append(ds)
    else:
        print(f"File {file_path} not found.")

# Concatenate datasets along the 'time' dimension (which already exists)
merged_ds = xr.concat(datasets, dim='time')

# Save the merged dataset into a new NetCDF file
merged_ds.to_netcdf("./data/features/merged_years.nc")

print("Merged file saved as 'merged_data.nc'")