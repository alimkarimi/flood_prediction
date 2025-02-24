import os
import requests
import cdsapi

"""
This script is to get ERA5 data. This will be used as predictors for generating flood predictions
"""

# Define the path where you want to create the new folder
current_directory = os.getcwd()
new_folder_path = current_directory + "/flood_prediction/data/features"

# Create the folder (including any missing parent directories)
os.makedirs(new_folder_path, exist_ok=True)

# Change to a specific directory
os.chdir(new_folder_path)

years = [str(year) for year in range(1996, 2019)]

c = cdsapi.Client()

for year in years:

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '2m_temperature', 'lake_depth', 'leaf_area_index_high_vegetation',
                'leaf_area_index_low_vegetation', 'soil_type', 'total_precipitation',
            ],
            'year': year,
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '03:00', '06:00',
                '09:00', '12:00', '15:00',
                '18:00', '21:00',
            ],
            'area': [
                5, 34, -5,
                42.5,
            ],
        },
        'download_' + year + '.nc')
