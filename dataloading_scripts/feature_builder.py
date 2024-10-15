import numpy as np
from dataloading_scripts.open_era5 import nc_era5
from dataloading_scripts.target_cube_builder import *

from dataloading_scripts.init_db import createBaseDataFrame, appendObservationalData

# get the features for 1996
"""
This script will get feature vectors (6d) for true positives and true negatives. The goal is to 
understand if TP and TNs cluster nicely in some sort of dim reduction scheme like PCA, UMAP, etc.

If they do, then, I think this validates that there is a nice decision boundary that a model like 
SVM can find/learn.
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

np.random.seed(0)

def getFeatures(df,target_cube = target_cube, predictor_vars = predictor_vars, num_neg_samples = 1000000, pos_feature_extraction=True,
                ):
    """

    This function currently gets true positive event features. The features are stored in nc_era5 and the 
    ground truth data is stored in target_cube. Positive samples have a 1 in target_cube whereas negative samples have 
    a zero.

    """
    if not pos_feature_extraction:
        # get indices for true neg:
        true_neg_indices = np.where(target_cube == 0)
        print('neg samples', len(true_neg_indices))
        print(true_neg_indices[0].shape, true_neg_indices[1].shape)

        # choose negative samples from true_neg_indices. We need samples because we do not want to 
        # choose every negative sample - there will be too many.
        num_neg_indices = true_neg_indices[0].shape[0]
        random_sample_idx = np.random.randint(low=0, high=num_neg_indices, size=num_neg_samples, dtype=int)
        #print('random sample idx', random_sample_idx.shape, random_sample_idx)

        # create empty matrix to hold the time, row, and column indices of the true negatives:
        idx_holder_neg = np.zeros((random_sample_idx.shape[0], 3))

        # fill in rows of the idx_holder_neg with time, row, and column indices of true negatives
        for k, idx in enumerate(random_sample_idx):
            temp_t = int(true_neg_indices[0][idx])
            temp_row = int(true_neg_indices[1][idx])
            temp_col = int(true_neg_indices[2][idx])
            idx_holder_neg[k] = [temp_t, temp_row, temp_col]

        df['time_idx'] = true_neg_indices[0][random_sample_idx]
        df['row_idx'] = true_neg_indices[1][random_sample_idx]
        df['col_idx'] = true_neg_indices[2][random_sample_idx]
        df['label'] = 'tn'
        
        # create empty matrix of observations that will eventually hold features
        obs_matrix_true_neg = np.zeros((idx_holder_neg.shape[0], len(predictor_vars)))

        # iterate through idx_holder_neg matrix rows which has indices of the negative observations.
        # use those indcies to get the negative features from the ERA5 dataset and save those features row 
        # wise into the obs_matrix_true_neg matrix.

        for n, idx in enumerate(range(idx_holder_neg.shape[0])):
            t_idx, row_idx, col_idx = idx_holder_neg[n]
            for m, predictor in enumerate(predictor_vars):
                # get each ERA5 variable's value.
                if m == 0:
                    temp_feature_vector = np.zeros((1, len(predictor_vars)))
                    # get the 0th feature for the temp_feature_vector
                temp_feature_vector[0, m] = nc_era5.variables[predictor][t_idx, row_idx, col_idx]
                
                
                # print(predictor, 'at idx', t_idx, row_idx, col_idx, nc_era5.variables[predictor][t_idx, row_idx, col_idx])
                
                if m == len(predictor_vars) - 1:
                    #print(temp_feature_vector)
                    obs_matrix_true_neg[n] = temp_feature_vector
                    df.at[n, 'feature_vector'] = temp_feature_vector.squeeze()
                    

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
        print(df)

        # create empty matrix of observations that will eventually hold features
        obs_matrix_true_pos = np.zeros((idx_holder.shape[0], len(predictor_vars)))
        print('shape of obs matrix', obs_matrix_true_pos.shape)


        # iterate through idx_holder matrix which now has indices of positive observations. Extract relevant features
        # from the ERA5 dataset at the corresponding indices. Save those features into the obs_matrix_true_pos 
        for n, idx in enumerate(range(idx_holder.shape[0])):
            # iterate through time, row, and col indexes together
            t_idx, row_idx, col_idx = idx_holder[n]
            for x, predictor in enumerate(predictor_vars):
                # get each ERA5 variable's value.
                if x == 0:
                    temp_feature_vector = np.zeros((1, len(predictor_vars)))
                    # get the 0th feature for the temp_feature_vector
                temp_feature_vector[0, x] = nc_era5.variables[predictor][t_idx, row_idx, col_idx]
                
                
                # print(predictor, 'at idx', t_idx, row_idx, col_idx, nc_era5.variables[predictor][t_idx, row_idx, col_idx])
                
                if x == len(predictor_vars) - 1:
                    obs_matrix_true_pos[n] = temp_feature_vector
                    df.at[n, 'feature_vector'] = temp_feature_vector.squeeze()
        
        return obs_matrix_true_pos, df

obs_matrix_true_pos, df_pos = getFeatures(df_pos, pos_feature_extraction=True)
obs_matrix_true_neg, df_neg = getFeatures(df_neg, pos_feature_extraction=False)

obs_matrix_all = np.concatenate((obs_matrix_true_pos, obs_matrix_true_neg), axis = 0)

# concatenate dataframes:
df_pos_neg = pd.concat([df_pos, df_neg], ignore_index=True)
print(df_pos_neg)
print(obs_matrix_all.shape)

# get 24 true negatives by sampling the 
if __name__ == "__main__":
    print('In Main.')


