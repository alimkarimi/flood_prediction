import numpy as np
import pandas as pd

gt = pd.read_csv('/Users/alim/Documents/ccai_floods/gt/FloodArchive.txt', encoding= 'unicode_escape', on_bad_lines='skip', 
                sep= '\t')

gt_kenya = gt[gt['Country'] == "Kenya"]
gt_kenya = gt_kenya.reset_index(drop=True)