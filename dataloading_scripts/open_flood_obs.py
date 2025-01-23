import numpy as np
import pandas as pd
import os

current_directory = os.getcwd() # get the current working directory

gt = pd.read_csv(current_directory + '/data/gt/FloodArchive.txt', encoding= 'unicode_escape', on_bad_lines='skip', 
                sep= '\t')

gt_kenya = gt[gt['Country'] == "Kenya"]
gt_kenya = gt_kenya.reset_index(drop=True)