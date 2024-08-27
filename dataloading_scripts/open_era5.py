from netCDF4 import Dataset
import itertools
import numpy as np


"""
This script will open up a netCDF4 file from ERA5 with the variables specificed in the download from the 
download_era5.py file.
"""

nc_era5 = Dataset('/Users/alim/Documents/ccai_floods/floods_in_kenya/download.nc')

lon_era5 = nc_era5.variables['longitude'][:]

lat_era5 = nc_era5.variables['latitude'][:]

time_era5 = nc_era5.variables['time'][:]

time = nc_era5.variables['time'] # this looks like it includes metadata, so it is different from time_era5 variable.

cartesian_product_lon_lat = np.array(list(itertools.product(lat_era5, lon_era5))) # this variable is a list of all the 
# coordinates present in the ERA5 dataset. As a reminder, this can be done by taking the cartesian product of the list
# of longs and list of lats. (A x B , where x is the cartesian product, will generate the set of all ordered pairs
#  (a,b) where a is in set A and b is in set B). A throw back to set theory and algorithms ;) 


if __name__ == "__main__":
    print(f"num longitude (x or cols) steps: {len(lon_era5)}")
    print(f"num latitude (y or row) steps: {len(lat_era5)}")
    print(f"num time steps: {len(time_era5)}")
    print(f"shape of cartesian product of x X y: {cartesian_product_lon_lat.shape}")