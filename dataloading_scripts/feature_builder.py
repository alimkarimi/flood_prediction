import numpy as np
from dataloading_scripts.open_era5 import nc_era5
from dataloading_scripts.target_cube_builder import *
from dataloading_scripts.resampling import resample_dtm

from dataloading_scripts.init_db import createBaseDataFrame, appendObservationalData

"""
This script will get feature vectors (6d - from ERA5) for all true positives and a sample of true negatives. 

This scrip will also create a 7th dimension, the elevation, from resample_dtm

Currently, the number of negative samples is controlled by the num_neg_samples argument to get_features().

There is only an option to sample all of the positive samples. One cannot sample a subset currently.

The argument pos_feature_extraction is set to True. If True, the function returns a dataframe with positive 
samples. If False, the function returns a dataframe with negative samples. In both cases, the function returns a
dataframe with the label of the sampled feature.

The features are sampled at one time point and one space point. The spatiotemporal nature of the sample is something
that can be extended in the future.
"""

predictor_vars = ['slt', 't2m', 'dl', 'lai_hv', 'lai_lv', 'tp']

# print(nc_era5.variables['slt'].long_name)
# print(nc_era5.variables['t2m'].long_name)
# print(nc_era5.variables['dl'].long_name)
# print(nc_era5.variables['lai_hv'].long_name)
# print(nc_era5.variables['lai_lv'].long_name)
# print(nc_era5.variables['tp'].long_name)
#print(nc_era5)

# get true positive indices (for the .nc file specified in open_era5.py)
target_cube = np.zeros((len(time_era5), len(lat_era5), len(lon_era5)))
target_cube = append_flood_observations(target_cube, ground_truth_df=gt_kenya)
print(f"Target cube has {int(np.sum(target_cube))} observations in total")
print(f"Target cube has dimensions {target_cube.shape}")

# create dataframe to store observations, indices, and labels
df_pos = createBaseDataFrame()
df_neg = createBaseDataFrame()

np.random.seed(0) # set seed for reproducibility - especially to ensure the same negative flood example indices are repeated.


def getFeatures(df,target_cube = target_cube, predictor_vars = predictor_vars, append_dtm = True, num_neg_samples = 1000000, pos_feature_extraction=True,
                lag = 3, spatial_window = None):
    """
    To do - incorporate spatial window.

    This function currently gets true positive event features. The features are stored in nc_era5 and the 
    ground truth data is stored in target_cube. Positive samples have a 1 in target_cube whereas negative samples have 
    a zero.

    The variable "lag" represents the number of timesteps we look back in order to build a feature. For example, if lag is 1, we 
    only look back one timestep and the feature for the given climate variable is a scalar. If lag is greater than 1, we 
    look back multiple timesteps (t_idx - 1) and get the current index too. So, if the flood happened at timestep 500, we index from 498 : 501, which really just
    gets us the observation at 498 and 499. 

    """

    if append_dtm:
        # run resampling code and build resampled dtm
        resampled_dtm = resample_dtm()

    if not pos_feature_extraction: # entering this condition implies the program will aim to get features for negative samples
        # get indices for true neg:
        
        true_neg_indices = np.where(target_cube == 0) # this is a tuple of arrays. The first array is the time index, the second is the row index, and the third is the column index.
        # above, we index into target_cube[lag:] because we want to make sure that when we look back for lagged features, they 
        # are not out of bounds.
        print('neg samples', len(true_neg_indices))
        print(true_neg_indices[0].shape, true_neg_indices[1].shape)
        print(type(true_neg_indices))
        print('true_neg_indices', true_neg_indices)

        # choose negative samples from true_neg_indices. We need samples because we do not want to 
        # choose every negative sample - there will be too many.
        num_neg_indices = true_neg_indices[0].shape[0]
        random_sample_idx = np.random.randint(low=0, high=num_neg_indices, size=num_neg_samples, dtype=int)
        print('random sample idx', random_sample_idx.shape, random_sample_idx)

        # create empty matrix to hold the time, row, and column indices of the true negatives:
        idx_holder_neg = np.zeros((random_sample_idx.shape[0], 3))

        # fill in rows of the idx_holder_neg with time, row, and column indices of true negatives
        for k, idx in enumerate(random_sample_idx):
            temp_t = int(true_neg_indices[0][idx]) # the [0] indexes into the first array in the tuple called true_neg_indices and 
            # the [idx] indexes into the the idx-th element of that array.
            temp_row = int(true_neg_indices[1][idx]) # [1] indexes into the second array in the tuple called true_neg_indices and 
            # the [idx] indexes into the the idx-th element of that array.
            temp_col = int(true_neg_indices[2][idx]) # the [2] indexes into the third array in the tuple called true_neg_indices and 
            # the [idx] indexes into the the idx-th element of that array.
            idx_holder_neg[k] = [temp_t, temp_row, temp_col] # fill in the k-th row of the k x 3 matrix idx_holder_neg
        print('idx holder', idx_holder_neg, idx_holder_neg.shape)

        # get indices where lag would go out of the data cube:
        bool_mask_out_of_bounds = idx_holder_neg[:, 0] < lag # bool_mask holds true/false values. True
        # means that the index is out of bounds. False means that the index is in bounds.
        num_out_of_bounds = np.sum(bool_mask_out_of_bounds) # compute number of sampled true negatives that are out of bounds.
        print(f" removed {num_out_of_bounds} out of bounds samples")
        print(idx_holder_neg[:, 0] < lag)

        # we want to adjust the random_sample_idx to only include those indices that are in bounds.
        random_sample_idx = random_sample_idx[~bool_mask_out_of_bounds] # this will give us the indices of the true negatives that are in bounds.
        # we want to adjust the idx_holder_neg to only include those indices that are in bounds.
        idx_holder_neg = idx_holder_neg[~bool_mask_out_of_bounds] # this will give us the indices of the true negatives that are in bounds.
        print(f"number of samples in idx_holder_neg {idx_holder_neg.shape[0]}")


        df['time_idx'] = idx_holder_neg[:, 0] # write index to df
        df['row_idx'] =  idx_holder_neg[:, 1] # write index to df
        df['col_idx'] =  idx_holder_neg[:, 2] # write index to df
        df['label'] = 'tn' # write label to df (tn = true negative, i.e, not a flood)
        
        # create empty matrix of observations that will eventually hold features
        obs_matrix_true_neg = np.zeros((idx_holder_neg.shape[0], lag * len(predictor_vars) + lag * int(append_dtm))) # if append_dtm is true, add 1.
        # shape of obs_matrix_true_neg is k x (6 * lag) or k x (7 * lag), depending on if append_dtm is true or false. 

        # iterate through idx_holder_neg matrix rows which has indices of the negative observations.
        # use those indcies to get the negative features from the ERA5 dataset and save those features row 
        # wise into the obs_matrix_true_neg matrix.

        for n, idx in enumerate(range(idx_holder_neg.shape[0])):
            t_idx, row_idx, col_idx = idx_holder_neg[n]
            for m, predictor in enumerate(predictor_vars):
                # get each ERA5 variable's value.
                if m == 0:
                    temp_feature_vector = np.zeros((lag, int(obs_matrix_true_neg.shape[1]/ lag))) # create lag x 6 * lag or lag x 7 * lag empty vector
                    # get the 0th feature for the temp_feature_vector
                #print('t_idx - lag', int(t_idx) - int(lag))
                #print('lagged predictor:', nc_era5.variables[predictor][int(t_idx) - lag : int(t_idx) + 1, row_idx, col_idx])
                #print('t_idx', t_idx, 'row_idx', row_idx, 'col_idx', col_idx)
                temp_feature_vector[:, m] = nc_era5.variables[predictor][int(t_idx) - lag : int(t_idx), row_idx, col_idx] # fill in the m-th column of the 1 x 6 or 1 x 7 vector
                
                if m == len(predictor_vars) - 1:
                    # grab dtm data for row_idx, col_idx and add to last column of temp_feature_vector if func specifies to do so:
                    if append_dtm:
                        temp_feature_vector[:, -1] = resampled_dtm[int(row_idx), int(col_idx)]
                    
                    obs_matrix_true_neg[n] = temp_feature_vector.reshape(-1) # reshape so entire observation fits in one row. 
                    #df.at[n, 'feature_vector'] = temp_feature_vector.squeeze()
                    df.at[n, 'feature_vector'] = temp_feature_vector.reshape(-1)
                    

        return obs_matrix_true_neg, df

    if pos_feature_extraction:
        # get indices for true pos:
        true_pos_indices = np.nonzero(target_cube)

        # create empty matrix to hold the time, row, and column indices of the true positives
        idx_holder = np.zeros((true_pos_indices[0].shape[0], 3))

        print(idx_holder.shape)   

        # fill in rows of the idx_holder matrix with time, row, and column indices of true positives
        for i in range(idx_holder.shape[0]):
            # get the time, row, and col index of each true positive index and read it into the n x 3 matrix
            idx_holder[i] = [true_pos_indices[0][i], true_pos_indices[1][i], true_pos_indices[2][i]]

        df['time_idx'] = true_pos_indices[0]
        df['row_idx'] = true_pos_indices[1]
        df['col_idx'] = true_pos_indices[2]
        df['label'] = 'tp'

        # create empty matrix of observations that will eventually hold features
        obs_matrix_true_pos = np.zeros((idx_holder.shape[0], lag * len(predictor_vars) + lag * int(append_dtm)))
        print('shape of true pos obs matrix', obs_matrix_true_pos.shape)


        # iterate through idx_holder matrix which now has indices of positive observations. Extract relevant features
        # from the ERA5 dataset at the corresponding indices. Save those features into the obs_matrix_true_pos 
        for n, idx in enumerate(range(idx_holder.shape[0])):
            # iterate through time, row, and col indexes together
            t_idx, row_idx, col_idx = idx_holder[n]
            for x, predictor in enumerate(predictor_vars):
                # get each ERA5 variable's value.
                if x == 0:
                    temp_feature_vector = np.zeros(( lag, int(obs_matrix_true_pos.shape[1] / lag) )) # lag (1 or 2 now) x 6 or 7
                    # get the 0th feature for the temp_feature_vector

                temp_feature_vector[:, x] = nc_era5.variables[predictor][int(t_idx - lag): int(t_idx), row_idx, col_idx] # get lag 
                # timesteps before flood event.
                        
                if x == len(predictor_vars) - 1:
                    # append dtm data if function specifies to do so:
                    if append_dtm:
                        temp_feature_vector[:, -1] = resampled_dtm[int(row_idx), int(col_idx)]
                    obs_matrix_true_pos[n] = temp_feature_vector.reshape(-1) # reshape so entire observation fits in one row. 
                    #df.at[n, 'feature_vector'] = temp_feature_vector.squeeze()
                    df.at[n, 'feature_vector'] = temp_feature_vector.reshape(-1)
        
        return obs_matrix_true_pos, df

obs_matrix_true_pos, df_pos = getFeatures(df_pos, pos_feature_extraction=True)
obs_matrix_true_neg, df_neg = getFeatures(df_neg, pos_feature_extraction=False)

obs_matrix_all = np.concatenate((obs_matrix_true_pos, obs_matrix_true_neg), axis = 0)

# concatenate dataframes:
df_pos_neg = pd.concat([df_pos, df_neg], ignore_index=True)
print(df_pos_neg)
print(df_pos_neg['feature_vector'][0].shape)
print(obs_matrix_all.shape)

# get 24 true negatives by sampling the 
if __name__ == "__main__":
    print('In Main.')
    



