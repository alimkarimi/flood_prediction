import dask.dataframe as dd
import pandas as pd
import numpy as np

"""
This file will hold functions to spin up persistent databases that can be used across other tasks
in the repo
"""

def createBaseDataFrame():
    # To create a daask dataframe, first create a pandas dataframe:

    df = pd.DataFrame()
    df['time_idx'] = df['row_idx'] = df['col_idx'] = df['feature_vector'] = df['label'] = None
    

    #dask_df = dd.from_pandas(df, npartitions=1)

    return df

def appendObservationalData(data):
    """
    Pass in data in the order time_idx, row_idx, col_idx, feature_vector, label
    """
    return None
    

if __name__ == "__main__":
    df = createBaseDataFrame()
    new_row = pd.DataFrame({'time_idx' : 0, 'row_idx' : 1, 'col_idx' : 2, 'feature_vector' : 0})
    df = dd.concat([df, new_row])
    print(new_row)