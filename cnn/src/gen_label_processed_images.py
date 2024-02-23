import os
import numpy as np  
import h5py as h5
from skimage.feature import peak_local_max
from collections import namedtuple
from dataprep import PathManager

class ImageProcessor:
    def __init__(self, water_background_array):
        """
        Initialize the ImageProcessor class.

        Parameters:
        - water_background_array: numpy array
            The water background array to be applied to the image.
        """
        self.water_background_array = water_background_array
    
    @staticmethod
    def load_h5_image(file_path):
        """
        Load an HDF5 image.

        Parameters:
        - file_path: str
            The path to the HDF5 image file.

        Returns:
        - numpy array
            The loaded image as a numpy array.
        """
        with h5.File(file_path, 'r') as f:
            return np.array(f['entry/data/data'])
        
    def apply_water_background(self, peak_image_array):
        """
        Apply the water background to the image.

        Parameters:
        - peak_image_array: numpy array
            The image array to which the water background will be applied.

        Returns:
        - numpy array
            The image array with the water background applied.
        """
        return peak_image_array + self.water_background_array
    
    def find_peaks_and_label(self, peak_image_array, threshold_value=0, min_distance=3):
        """
        Find peaks in the image and generate a labeled image.

        Parameters:
        - peak_image_array: numpy array
            The image array in which peaks will be found.
        - threshold_value: int, optional
            The threshold value for peak detection. Default is 0.
        - min_distance: int, optional
            The minimum distance between peaks. Default is 3.

        Returns:
        - namedtuple
            A named tuple containing the coordinates of the peaks and the labeled image array.
        """
        Output = namedtuple('Out', ['coordinates', 'labeled_array'])
        coordinates = peak_local_max(peak_image_array, min_distance=min_distance, threshold_abs=threshold_value)
        labeled_array = np.zeros(peak_image_array.shape)
        for y, x in coordinates:
            labeled_array[y, x] = 1

        return Output(coordinates, labeled_array)
    
    def process_directory(self, paths, threshold_value): 
        """
        Process all HDF5 images in a directory.

        Parameters:
        - paths: tuple
            A tuple containing the paths to the peak images directory, processed images directory, and label output directory.
        - threshold_value: int
            The threshold value for peak detection.

        Returns:
        None
        """
        # unpack paths
        peak_images_path, processed_images_path, label_output_path = paths
    
        for file in os.listdir(peak_images_path):
            if file.endswith('.h5') and file != 'water_background.h5':
                full_path = os.path.join(peak_images_path, file)
                print(f"Processing {file}...")
                image = self.load_h5_image(full_path)
                processed_image = self.apply_water_background(image)
                
                Out = self.find_peaks_and_label(processed_image, threshold_value)
                labeled_image = Out.labeled_array
                coordinates = Out.coordinates
                
                processed_save_path = os.path.join(processed_images_path, f"processed_{file}")
                labeled_save_path = os.path.join(label_output_path, f"labeled_{file}")
            
            with h5.File(processed_save_path, 'w') as pf:
                pf.create_dataset('entry/data/data', data=processed_image)
            with h5.File(labeled_save_path, 'w') as lf:
                lf.create_dataset('entry/data/data', data=labeled_image)
                
            print(f'Saved processed image to {processed_save_path}')
            print(f'Saved labeled image to {labeled_save_path}')
    
def main():
    """
    The main function of the script.
    """
    paths = PathManager()
    threshold_value = 1000  # Adjust as needed

    water_background = ImageProcessor.load_h5_image(paths.__get_path__('water_background_h5'))
    processor = ImageProcessor(water_background)

    conf = input(f"Are you sure you want to process ... \n\n the peak images: {paths.__get_path__('peak_images_dir')} \n\n ...  and apply (water_background.h5) {paths.__get_path__('water_background_h5')} \n\n ... and output processed: {paths.__get_path__('processed_images_dir')}, \n\n ... and output labels: {paths.__get_path__('label_images_dir')} \n\n (y/n): ")
    if conf.lower() == 'y':
        processor.process_directory((paths.__get_path__('peak_images_dir'), paths.__get_path__('processed_images_dir'), paths.__get_path__('label_images_dir')), threshold_value)
    else:
        print("Operation cancelled.")
        
if __name__ == '__main__':
    main()
