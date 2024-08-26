import numpy as np
from open_era5 import lon_era5, lat_era5, cartesian_product_lon_lat
from open_flood_obs import gt_kenya


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

def append_flood_observation(target_cube, ground_truth_df, verbose=False):
    # get event longs, lats from ground_truth_df:
    temp_lon, temp_lat = ground_truth_df['long'], ground_truth_df['lat']

    # for each long, lat, use the find_closest_vector_2d function to find the closest long/lat in the ERA5 grid system
    for obs_coordinates in zip(temp_lat, temp_lon):
        # use find_closest_vector function
        closest_vector, index = find_closest_vector(cartesian_product_lon_lat, obs_coordinates)
        if verbose:
            print(f"Closest vector:{closest_vector}")
            print(f"Index of closest vector: {index}")
        # convert that closest vector to a 2d zero based indexing system. 
        temp_raster_idx = convert_coordinates_to_2d_index(closest_vector, range_lon = lon_era5, range_lat = lat_era5)
        temp_row_idx, temp_col_idx = temp_raster_idx[0], temp_raster_idx[1] # unpack temp_raster_idx
        
        target_cube[temp_row_idx, temp_col_idx]+= 1 # add 1 to the target_cube

    return target_cube

if __name__ == "__main__":
    target_cube = np.zeros((len(lat_era5), len(lon_era5)))
    target_cube = append_flood_observation(target_cube, ground_truth_df=gt_kenya)
    print(f"Target cube has {int(np.sum(target_cube))} observations in total")
    

