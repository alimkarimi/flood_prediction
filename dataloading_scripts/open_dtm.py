import numpy as np
from PIL import Image
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import rasterio 

file_list = os.listdir('./dtm_data')

def open_dtm(file):
    
    dtm_opened = Image.open('./dtm_data/' + file)
    #print(dtm_opened.height, dtm_opened.width)
    arr = np.array(dtm_opened)
    print(arr.shape, 'when opened')
    
    return arr
    
def plot():
    dtm_list = []
    for file in file_list:
        dtm = open_dtm(file)
        dtm_list.append(dtm)
        print('appeneded')
        
    # Create a figure with 2 rows and 3 columns of subplots
    fig, axes = plt.subplots(nrows=2, ncols=3) 

    # Plot data on each subplot
    axes[0, 0].imshow(dtm_list[0])
    axes[0, 0].set_title(file_list[0])

    axes[0, 1].imshow(dtm_list[1])
    axes[0, 1].set_title(file_list[1])

    axes[0, 2].imshow(dtm_list[2])
    axes[0, 2].set_title(file_list[2])

    axes[1, 0].imshow(dtm_list[3])
    axes[1, 0].set_title(file_list[3])


    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Increase these values for more space
    
    fig.savefig('./dtm_data/DTMs_visualized.jpg')



# Open the TIFF file
with rasterio.open("./dtm_data/maxElevation_Kenya.tif") as dataset:
    # Access metadata
    print("File Name:", dataset.name)
    print("Width:", dataset.width)
    print("Height:", dataset.height)
    print("Number of Bands:", dataset.count)
    print("Coordinate Reference System:", dataset.crs)
    print("Bounds:", dataset.bounds)
    print("Driver:", dataset.driver)
    print("Data Type:", dataset.dtypes)
    print("No Data Value:", dataset.nodatavals)
    pixel_col = np.array([0, 1, 2])
    pixel_row = np.array([0, 1, 2])
    lon, lat = dataset.transform * (pixel_col, pixel_row)
    print("corresponding lat long is:", lon, lat)

if __name__ == "__main__":
    plot()