import os
import numpy as np
import h5py as h5
import torch
import shutil
import re
import h5py as h5
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import namedtuple
from skimage.feature import peak_local_max

class PathManager:
    def __init__(self):
        # grabs peaks and processed images from images directory /
        self.root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.images_dir = os.path.join(self.root, 'images') #/images
        self.sim_dir = os.path.join(self.root, 'sim')  #/sim
        self.sim_specs_dir = os.path.join(self.sim_dir, 'sim_specs') # sim/sim_specs
        # KEY
        self.peak_images_dir = os.path.join(self.images_dir, 'peaks') # images/peaks
        self.water_images_dir = os.path.join(self.images_dir, 'data') # images/data
        # 
        # built just in case further development is needed
        self.processed_images_dir = os.path.join(self.images_dir, 'processed_images') # images/processed_images
        self.preprocessed_images_dir = os.path.join(self.images_dir, 'preprocessed_images') # images/preprocessed_images
        self.label_images_dir = os.path.join(self.images_dir, 'labels') # images/labels
        self.pdb_dir = os.path.join(self.sim_dir, 'pdb') # /sim/pdb
        self.sh_dir = os.path.join(self.sim_dir, 'sh') # /sim/sh
        self.water_background_h5 = os.path.join(self.sim_dir, 'water_background.h5') # /sim/water_background.h5
    
    def __get_path__(self, path_name):
        # returns the path of the path_name
        paths_dict = {
            'root': self.root,
            'images_dir': self.images_dir,
            'sim_dir': self.sim_dir,
            'sim_specs_dir': self.sim_specs_dir,
            'peak_images_dir': self.peak_images_dir,
            'water_images_dir': self.water_images_dir,
            'processed_images_dir': self.processed_images_dir,
            'preprocessed_images_dir': self.preprocessed_images_dir,
            'label_images_dir': self.label_images_dir,
            'pdb_dir': self.pdb_dir,
            'sh_dir': self.sh_dir,
            'water_background_h5': self.water_background_h5,
        }
        return paths_dict.get(path_name, None)

    def __get_all_paths__(self):
        # returns a namedtuple of all paths for easier access
        PathsOutput = namedtuple('PathsOutput', ['root', 'images_dir', 'sim_dir', 'peak_images_dir', 'water_images_dir', 'processed_images_dir', 'label_images_dir', 'pdb_dir', 'sh_dir', 'water_background_h5'])
        return PathsOutput(self.root, self.images_dir, self.sim_dir, self.peak_images_dir, self.processed_images_dir, self.label_images_dir, self.pdb_dir, self.sh_dir, self.water_background_h5)
    
    def __get_peak_images_paths__(self):
        # searches the peak images directory for .h5 files and returns a list of paths
        return [os.path.join(self.peak_images_dir, f) for f in os.listdir(self.peak_images_dir) if f.endswith('.h5')]    
    
    def __get_water_images_paths__(self):
        # searches the water images directory (images/data)for .h5 files and returns a list of paths
        return [os.path.join(self.water_images_dir, f) for f in os.listdir(self.water_images_dir) if f.endswith('.h5')]
    
    def __get_processed_images_paths__(self):
        # searches the processed images directory for .h5 files and returns a list of paths
        return [os.path.join(self.processed_images_dir, f) for f in os.listdir(self.processed_images_dir) if f.startswith('processed')]
    
    def __get_label_images_paths__(self):
        # searches the label images directory for .h5 files and returns a list of paths
        return [os.path.join(self.label_images_dir, f) for f in os.listdir(self.label_images_dir) if f.startswith('label')]
    
    def __get_pdb_path__(self, pdb_file):
        # returns the .pdb file path of the file name in the pdb directory
        return os.path.join(self.pdb_dir, pdb_file)
    
    def __get_sh_path__(self, sh_file):
        # returns the .sh file path of the file name in the sh directory
        return os.path.join(self.sh_dir, sh_file)

    def clean_sim(self):
        file_types = ['.err', '.out', '.sh']
        files_moved = False  # flag to check if any files were moved

        # manage the .err, .out, .sh files into sim_specs
        for file_type in file_types:
            for f in os.listdir(self.sim_dir):
                if f.endswith(file_type):
                    source_path = os.path.join(self.sim_dir, f)
                    dest_path = os.path.join(self.sim_specs_dir, f)
                    
                    # checks to ensure that the file is not already in the sim_specs directory
                    if not os.path.exists(dest_path):
                        print(f"Moving {source_path} to {dest_path}\n")
                        shutil.move(source_path, dest_path)
                        
                        # copy the .sh file to the sim directory
                        if file_type == '.sh':
                            sh_copy_path = os.path.join(self.sim_dir, f)
                            print(f"Copying {dest_path} to {sh_copy_path}\n")
                            shutil.copy(dest_path, sh_copy_path)
                        
                        files_moved = True
                        
        if not files_moved:
            print("clean_sim did not move any files\n\n")
            
class PeakImageDataset(Dataset):
    def __init__(self, paths, transform=None, augment=False):
        self.peak_image_paths = paths.__get_peak_images_paths__()
        self.water_image_paths = paths.__get_water_images_paths__()
        assert len(self.peak_image_paths) == len(self.water_image_paths), "The number of peak images must match the number of water images."
        self.transform = transform or self.default_transform()
        self.augment = augment
        
    def __len__(self):
        # returns number of images in the dataset
        return len(self.peak_image_paths)

    def __get_item__(self, idx):
        # gets peak/water image and returns numpy array and applies transform
        peak_image = self.__load_h5__(self.peak_image_paths[idx])
        water_image = self.__load_h5__(self.water_image_paths[idx])
        
        if self.augment:
            peak_image = self.augment_image(peak_image)
            water_image = self.augment_image(water_image)
        
        if self.transform:
            peak_image = self.transform(peak_image)
            water_image = self.transform(water_image)
    
        return peak_image, water_image 
        
    def __load_h5__(self, image_path):
        with H5h5.File(image_path, 'r') as f:
            return np.array(f['entry/data/data'])

    def __to_pil__(self, image):
        image = self.__load_h5__(image)
        return transforms.ToPILImage()(image)
    
    def augment_image(self, image):
        pil_image = transforms.ToPILImage()(image)
        rotated_image = pil_image.rotate(90)
        return transforms.ToTensor()(rotated_image)
    
    def default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float()), #ensure float type
            # transforms.Normalize(mean=[0.5], std=[0.5]) #normalize to [-1, 1]
        ])
    
    def __preview__(self, idx, image_type='peak'):
        image_path = self.peak_image_paths[idx] if image_type == 'peak' else self.water_image_paths[idx]
        image = self.__load_h5__(image_path)
        # visualize outliers 
        plt.imshow(image, cmap='viridis')
        plt.colorbar()
        plt.title(f'{image_type.capitalize()} Image at Index {idx}')
        plt.axis('off')
        plt.show()
        
class DataPreparation:
    def __init__(self, paths, batch_size=32):
        self.paths = paths
        self.batch_size = batch_size
        
    def prep_data(self):        
        # ensure matching number of paths of peak and water images         
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float()), # Ensure float type
            transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize to [-1, 1]
        ])     
           
        dataset = PeakImageDataset(self.paths, transform=transform)
        num_items = len(dataset)
        num_train = int(0.8 * num_items)
        num_test = num_items - num_train     
           
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])
            
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        print("Data prepared.")
        print(f"Train size: {len(train_dataset)}")
        print(f"Test size: {len(test_dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches: {len(train_loader)}")
        
        return train_loader, test_loader

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
    
