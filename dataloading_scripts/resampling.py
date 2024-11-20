from dataloading_scripts.open_era5 import nc_era5, cartesian_product_lon_lat
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt


""""
This program will read all dtm .tif files that have SRTM data, merge them into one
contiguous 2d array. Then, resample using a 300 x 300 px window (corresponding to 0.25 x 0.25)
degrees.

This program also cleans up some spurious values of the SRTM data.
"""

def resample_dtm(plot=True):
    # Create list of SRTM dtm files:
    srtm_files_root = '../flood_prediction/data/srtm/'
    srtm_file_list = os.listdir(srtm_files_root)

    # merge DTM tifs into one 2d array:
    e_list = []
    n_s_list = []

    super_arr = np.zeros((12 * 300 * 4, 10 * 300 * 4)) # dimensions make sense for Kenya
    empty_arr = np.zeros((12,10)) # dimensions make sense for Kenya

    for fn in srtm_file_list:
        
        # logic to figure out where each srtm tif file belongs in the larger grid
        e_list.append(int(fn[6:8]))
        
        if fn[0] == 'n':
            n_s_list.append(int(fn[1:3]))

        if fn[0] == 's':
            n_s_list.append(-int(fn[1:3]))

        #print(f"last appended row: {n_s_list[-1]} while last appended col is {e_list[-1]}")
        #print(f"mapped row should be {(n_s_list[-1] - 5 ) * -1} and col should be {e_list[-1] - 33}")
        mapped_ns = (n_s_list[-1] - 5 ) * -1 # row
        mapped_e = e_list[-1] -33           # column
        empty_arr[mapped_ns, mapped_e] = 1

        # now convert mapped_ns and mapped_e to 1200 x 1200 box.
        mapped_ns_super = mapped_ns * 300 * 4
        mapped_e_super = mapped_e * 300 * 4
        #print(mapped_ns_super, mapped_ns_super+(1200), mapped_e_super, mapped_e_super+(1200))

        # open dtm and place dtm data into the right part of the super_arr
        with rasterio.open(srtm_files_root + fn) as dataset:
            
            extent = (dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top)
            #print(extent)
        
            extent_left = np.round(dataset.bounds.left)
            extent_right = np.round(dataset.bounds.right)
            extent_top = np.round(dataset.bounds.top)
            extent_bottom = np.round(dataset.bounds.bottom)

            # Define the bounds for the current 1 x 1-degree window
            lat1, lat2 = extent_top, extent_bottom # ex: 2, 1.75
            lon1, lon2 = extent_left, extent_right # ex: 40, 40.25

            # Convert the lat/lon bounds to pixel indices
            row_min, col_min = dataset.index(lon1, lat1)  # Upper-left corner
            row_max, col_max = dataset.index(lon2, lat2)  # Lower-right corner

            window = ((row_min, row_max), (col_min, col_max))
            windowed_data = dataset.read(1, window=window)  # Reads only the specified window in the 1st band
            #print(windowed_data.shape)

            # insert windowed data into correct location of super_arr:
            super_arr[mapped_ns_super : mapped_ns_super + 1200, mapped_e_super : mapped_e_super + 1200] = windowed_data


    # clean up spurious values in DTM:
    # for some reason, 0.03% of DTM values are -32767.0. Rewrite those to 0
    super_arr[super_arr == -32767.0] = 0 # this can be improved - some of these illogical elevations are in mountainous areas.

    # Now, resample super_arr to match grid resolution of climate data from netcdf
    resampled_arr = np.zeros((41,35))

    resampled_row_counter = 0
    resampled_col_counter = 0
    for c, coord in enumerate(cartesian_product_lon_lat):
        # print(f"The latitude is {coord[0]} and the longitude is {coord[1]}")
        # print(f"This means we want to get pixels between latitudes 5 - 0.125 and 5 + 0.125, or {coord[0] - 0.125} and {coord[0] + .125}")
        # print(f"This also means we want to get pixels between longitudes {coord[1] - 0.125} and {coord[1] + 0.125}")

        # 300 pixels is one degree. 150 pixels is half a degree. 75 pixels is a quarter of a degree.

        # to get the pixel centered at exactly 5 lat, 34 long, subtract current lat (5) from max lat (6) (this is 1). 
        # also, subtract min long from current long (34 - 33) (this is 1)
        # This means that the coordinate 5 lat, 34 long is 1 degree by 1 degree in to the image/dtm. 
        # To convert this to pixels, multiply this 1 degree by 1200. This is because our dtm data is natively in 3 arcseconds and there are 
        # 3600 arc seconds in a degree. 
        # So, getting the center pixel at 5 lat, 34 long means indexing into super_arr[1 * 1200, 1 * 1200].
        # To get the surround pixels and take the average, we can perform an operation like: 
        # np.mean(super_arr[1 * 1200 - 150 : 1 * 1200 + 150, 1 * 1200 - 150 : 1 * 1200 + 150])
        # save that result of np.mean() to to the correct index of resampled_arr
        
        diff_lat = 6 - coord[0] # 6 - coord[0] because 6 is the latitude that corresponds to the top row of data
        diff_lon = coord[1] - 33 # coord[1] - 33 because 33 is the longitude that corresponds to the first column of the data
        # print(diff_lat, diff_lon)

        center_px_lat = int(diff_lat * 1200)
        center_px_lon = int(diff_lon * 1200)
        # print(f"Center Pixel is {center_px_lat, center_px_lon}")

        
        # grab window around center pixel. Window size is 300 by 300. Consider adding filter if poor results in model
        window = super_arr[center_px_lat - 150 : center_px_lat + 150, center_px_lon - 150 : center_px_lon + 150]
        window_mean = np.mean(window)

        # write to resampled_arr:
        # print(f"Writing to index {resampled_row_counter, resampled_col_counter}")
        resampled_arr[resampled_row_counter, resampled_col_counter] = window_mean
        
        if resampled_col_counter != 35:
            resampled_col_counter += 1
        if resampled_col_counter == 35:
            resampled_col_counter = 0
            resampled_row_counter += 1
    
    if plot:
        plt.imshow(resampled_arr, cmap = 'terrain', extent=(34, 42.5, 5, -5))
        plt.colorbar(label='Elevation (m)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Digital Terrain Model - Downsampled')
        plt.show()

    return resampled_arr

if __name__ == "__main__":
    resampled_dtm = resample_dtm()