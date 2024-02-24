import os
import numpy as np  
import h5py as h5
from skimage.feature import peak_local_max
from collections import namedtuple
import classes as cl
import functions as fn

def main():
    """
    reads the water_background.h5 file and processes the peak images in the peak_images_dir
    """
    paths = cl.PathManager()
    threshold_value = 1000  # Adjust as needed

    water_background = cl.ImageProcessor.load_h5_image(paths.__get_path__('water_background_h5'))
    processor = cl.ImageProcessor(water_background)

    conf = input(f"Are you sure you want to process ... \n\n the peak images: {paths.__get_path__('peak_images_dir')} \n\n ...  and apply (water_background.h5) {paths.__get_path__('water_background_h5')} \n\n ... and output processed: {paths.__get_path__('processed_images_dir')}, \n\n ... and output labels: {paths.__get_path__('label_images_dir')} \n\n (y/n): ")
    if conf.lower() == 'y':
        processor.process_directory((paths.__get_path__('peak_images_dir'), paths.__get_path__('processed_images_dir'), paths.__get_path__('label_images_dir')), threshold_value)
    else:
        print("Operation cancelled.")
        
if __name__ == '__main__':
    main()
