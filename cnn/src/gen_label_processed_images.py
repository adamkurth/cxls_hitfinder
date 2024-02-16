import os
import numpy as np  
import h5py as h5
from skimage.feature import peak_local_max
from collections import namedtuple

class ImageProcessor:
    def __init__(self, water_background_array):
        self.water_background_array = water_background_array
    
    @staticmethod
    def load_h5_image(file_path):
        """Load an HDF5 image."""
        with h5.File(file_path, 'r') as f:
            return np.array(f['entry/data/data'])
        
    def apply_water_background(self, peak_image_array):
        """Apply the water background to the image."""
        return peak_image_array + self.water_background_array
    
    def find_peaks_and_label(self, peak_image_array, threshold_value=0, min_distance=3):
        """Find peaks in the image and generate a labeled image."""
        Output = namedtuple('Out', ['coordinates', 'labeled_array'])
        coordinates = peak_local_max(peak_image_array, min_distance=min_distance, threshold_abs=threshold_value)
        labeled_array = np.zeros(peak_image_array.shape)
        for y, x in coordinates:
            labeled_array[y, x] = 1

        return Output(coordinates, labeled_array)
    
    def process_directory(self, paths, threshold_value):
        """Process all HDF5 images in a directory."""
        
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
    # given generate_label_h5.py is in src, this is the root path of the repo
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    background_h5_path = os.path.join(root_path, 'sim', 'water_background.h5')
    # repo specific (below)
    peak_images_path = os.path.join(root_path, 'images', 'peaks')
    processed_images_path = os.path.join(root_path, 'images', 'data')
    label_output_path = os.path.join(root_path, 'images', 'labels')
    
    # Paths = namedtuple('Paths', ['peak_images_path', 'processed_images_path', 'label_output_path'])
    paths = (peak_images_path, processed_images_path, label_output_path)
    threshold_value = 1000  # Adjust as needed

    water_background = ImageProcessor.load_h5_image(background_h5_path)
    processor = ImageProcessor(water_background)

    conf = input(f"Are you sure you want to process ... \n\n the peak images: {peak_images_path} \n\n ...  and apply (water_background.h5) {background_h5_path} \n\n ... and output prorcesed: {processed_images_path}, \n\n ... and output labels: {label_output_path} \n\n (y/n): ")
    if conf.lower() == 'y':
        processor.process_directory(paths, threshold_value)
    else:
        print("Operation cancelled.")

if __name__ == '__main__':
    main()
