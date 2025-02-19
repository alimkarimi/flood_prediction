import numpy as np
from .open_era5 import lon_era5, lat_era5, time_era5, time, cartesian_product_lon_lat
from .open_flood_obs import gt_kenya
import cftime
from netCDF4 import Dataset, num2date, date2num
import pandas as pd




def find_closest_vector(source, target):
    # Calculate the Euclidean distance between each row in the source matrix and the target vector
    distances = np.linalg.norm(source - target, axis=1)
    # Find the index of the minimum distance
    min_index = np.argmin(distances)
    
    # Return the closest vector and its index
    return source[min_index], min_index

def convert_coordinates_to_2d_index(coordinates, range_lon, range_lat, verbose=False):
    """
    Given a range of latitudes and longitudes with a given spacing, this function takes coordinates in the (long, lat) (note 
    that long, lat is the same as row, col order) and returns a 2d index (0 based indexing) so that downstream code can plot 
    something based on this indexing system. 
    """
    idx_col = np.argmin(np.abs(range_lon - coordinates[1]))

    idx_row = np.argmin(np.abs(range_lat - coordinates[0]))
    if verbose:
        print(idx_row, idx_col)
        print(range_lon, range_lat)

    return (idx_row, idx_col)

def append_flood_observations(target_time_cube, ground_truth_df, verbose=False):
    ground_truth_df['Began'] = pd.to_datetime(gt_kenya['Began'], format='%Y%m%d')
    ground_truth_df['Ended'] = pd.to_datetime(gt_kenya['Ended'], format='%Y%m%d')

    for (t_start, t_end, lat, long) in zip(ground_truth_df['Began'], ground_truth_df['Ended'], ground_truth_df['lat'], ground_truth_df['long']):
        numdate_began = date2num(t_start, units=time.units, has_year_zero=False, calendar='gregorian')
        numdate_ended = date2num(t_end, units=time.units, has_year_zero=False, calendar='gregorian')
    
        # get the nearest time indices in cube
        # first: the begining time index:
        start_time_idx = np.argmin(np.abs(numdate_began - time_era5))
    
        # second: the ending time index
        end_time_idx = np.argmin(np.abs(numdate_ended - time_era5))
        if verbose:
            print('this is start and end', start_time_idx, end_time_idx)
    
        # now, get the lat and long of the flood observation:
        temp_obs_coord = np.array([lat, long])

        # now, retrieve the nearest lat and long coordinate in the gridding system:
        closest_vector, _ = find_closest_vector(source=cartesian_product_lon_lat, target=temp_obs_coord)
        if verbose:
            print(f"Closest vector: {closest_vector}")
    
        # now, convert the nearest lat and long vector to a 2d index so the time cube can be filled in.
        idx_row, idx_col = convert_coordinates_to_2d_index(coordinates = closest_vector, range_lon = lon_era5, range_lat = lat_era5)    

        target_time_cube[start_time_idx : end_time_idx, idx_row, idx_col] += 1

    return target_time_cube

if __name__ == "__main__":
    target_cube = np.zeros((len(time_era5), len(lat_era5), len(lon_era5)))
    target_cube = append_flood_observations(target_cube, ground_truth_df=gt_kenya)
    print(f"Target cube has {int(np.sum(target_cube))} positive flood observations in total")
    print(f"Target cube has dimensions {target_cube.shape}")

