from typing import List, Tuple
import os
import h5py as h5
import numpy as np
import torch
from skimage.feature import peak_local_max
from pkg import *

       
class Processor:
    def __init__(self, paths, dataset: str) -> None:
        self.paths = paths  # Expecting an instance of PathManager
        self.length = None          # .15, .25, .35 m
        self.photon_energy = None   #  6, 7.5, 8 keV 
        self.threshold = 1 # used for peak images (pure signal)
        self.dataset = dataset
        self.dataset_init()
        self.water_background = f.load_h5(self.paths.get_water_background(dataset))

    def dataset_init(self) -> None:
        clen_values, photon_energy_values = [1.5, 2.5, 3.5], [6000, 7000, 8000]
        self.dataset_dict = {
            '01': [clen_values[0], photon_energy_values[0]],
            '02': [clen_values[0], photon_energy_values[1]],
            '03': [clen_values[0], photon_energy_values[2]],
            '04': [clen_values[1], photon_energy_values[0]],
            '05': [clen_values[1], photon_energy_values[1]],
            '06': [clen_values[1], photon_energy_values[2]],
            '07': [clen_values[2], photon_energy_values[0]],
            '08': [clen_values[2], photon_energy_values[1]],
            '09': [clen_values[2], photon_energy_values[2]],
        }
        return self.dataset_dict
  
    def get_parameters(self) -> tuple:
        self.parameters = self.dataset_dict.get(self.dataset, None)        
        return self.parameters #list of clen, photon_energy
    
    @staticmethod
    def new_attibute(file_path:str, new:tuple) -> None:
        name, val = new
        with h5.File(file_path, 'a') as f:
            f.attrs[name] = val
        print(f"Attribute '{name}' assigned to {file_path}")
    
    @staticmethod
    def update_attributes(file_path: str, **kwargs) -> None:
        """
        Update attributes for an HDF5 file with arbitrary key-value pairs.

        Args:
        file_path (str): Path to the HDF5 file.
        **kwargs: Arbitrary keyword arguments representing attribute names and values.
        """
        # Check file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        try:
            with h5.File(file_path, 'a') as f:
                for key, value in kwargs.items():
                    f.attrs[key] = value
            attr_names = ', '.join(kwargs.keys())
            print(f"Attributes {attr_names} updated for {file_path}")
        
        # Handle exceptions
        except IOError as io_err:
            # Handle I/O errors separately, e.g., file permissions or lock issues
            print(f"Failed to open {file_path} due to an I/O error: {io_err}")
        except KeyError as key_err:
            # Handle errors related to accessing non-existent keys in HDF5 structure
            print(f"Failed to update attributes for {file_path} because of a key error: {key_err}")
        except Exception as e:
            # Catch-all for other exceptions, including issues within the HDF5 library
            print(f"Unexpected error updating attributes for {file_path}: {e}")
            
    def apply_water_background(self, peak_image_array: np.ndarray) -> np.ndarray:
        return peak_image_array + self.water_background

    def heatmap(self, peak_image_array: np.ndarray, min_distance: int = 3) -> np.ndarray:
        coordinates = peak_local_max(peak_image_array, min_distance=min_distance, threshold_abs=self.threshold)
        labeled_array = np.zeros(peak_image_array.shape, dtype=np.float32)
        labeled_array[tuple(coordinates.T)] = 1
        return labeled_array

    def heatmap_tensor(self, peak_image_array: np.ndarray, min_distance: int = 3) -> torch.Tensor:
        # call to retrieve labeled array of same peak image
        labeled_array = self.heatmap(peak_image_array, min_distance)
        # Assuming to_tensor is a function converting numpy arrays to PyTorch tensors
        heatmap_tensor = torch.tensor(labeled_array).unsqueeze(0)  # Add a channel dimension at the beginning
        return heatmap_tensor  # Ensure it matches expected dimensions [C, H, W]        

    def confirm_value(self, value:float, value_name:str) -> bool:
        # used in process_directory()
        confirm = input(f"Confirm '{value_name}' value: {value} (type 'y' to confirm): ")
        if confirm.lower() == 'y':
            print(f"'{value_name}' value {value} confirmed successfully.")
            return True
        else:
            print(f"'{value_name}' confirmation failed at attempt\n")
            return False

    def confirm_action(self, action_discription:str) -> bool:
        # used in process_empty()
        confirm = input(f"Confirm action: {action_discription} (type 'y' to confirm): ")
        if confirm.lower() == 'y':
            print(f"Action '{action_discription}' confirmed successfully.")
            return True
        else:
            print(f"Action '{action_discription}' confirmation failed at attempt\n")
            return False
    
    def process_directory(self, dataset:str, clen:float, photon_energy:int) -> None: 
        """
        Processes all peak images in the directory, applies the water background,
        generates label heatmaps for detecting Bragg peaks, and updates the HDF5 files
        with 'clen' and 'photon_energy' attributes for peak images, water images, and label images.
        """
        # Confirmations for clen and photon_energy

        dataset_confirmed = self.confirm_value(dataset, 'dataset')
        clen_confirmed = self.confirm_value(clen, 'clen')
        photon_energy_confirmed = self.confirm_value(photon_energy, 'photon_energy')
        water_background_confirmed = self.confirm_value(self.paths.get_water_background(dataset), 'water_background')
        
        if not (clen_confirmed and photon_energy_confirmed and dataset_confirmed and water_background_confirmed):
            print("Operation cancelled due to confirmation failure.")
            return
        
        # get peak images
        peak_image_paths = self.paths.get_peak_image_paths(dataset=dataset)
        # get water background image 
        water_background = self.water_background
        
        for peak_path in peak_image_paths:
            # derive a unique basename for each peak image
            basename = os.path.basename(peak_path)
            
            # unique filenames for the overlay and label images based on the current peak image's basename
            out_overlay_path = os.path.join(self.paths.peaks_water_overlay_dir, dataset, f'overlay_{basename}')
            out_label_path = os.path.join(self.paths.labels_dir, dataset, f'label_{basename}')
            
            peak_image = f.load_h5(file_path=peak_path)
            labeled_array = self.heatmap(peak_image_array=peak_image)
            peak_water_overlay_image = self.apply_water_background(peak_image_array=peak_image)
            
           # Save the processed peak image and labeled image to their respective unique paths
            f.save_h5(file_path=out_overlay_path, data=peak_water_overlay_image, save_parameters=True, params=[clen, photon_energy])
            f.save_h5(file_path=out_label_path, data=labeled_array, save_parameters=True, params=[clen, photon_energy])
            
            # Update attributes for each file with clen and photon energy
            self.update_attributes(file_path=peak_path, clen=clen, photon_energy=photon_energy, peak=True)
            self.update_attributes(file_path=out_overlay_path, clen=clen, photon_energy=photon_energy, peak=True)
            self.update_attributes(file_path=out_label_path, clen=clen, photon_energy=photon_energy, peak=True)
            self.update_attributes(file_path=self.paths.get_water_background(dataset=dataset),
                                    clen=clen, photon_energy=photon_energy, peak=False)
            
            # self.assign_attributes(file_path=peak_path, clen=clen, photon_energy=photon_energy, peak=True)
            # self.assign_attributes(file_path=out_overlay_path, clen=clen, photon_energy=photon_energy, peak=True)
            # self.assign_attributes(file_path=out_label_path, clen=clen, photon_energy=photon_energy, peak=True)
            # self.assign_attributes(file_path=self.paths.get_water_background(dataset=dataset), clen=clen, photon_energy=photon_energy, peak=False)
            print(f"Processed and labeled images for {basename} saved.")
        print(f"\nProcessing complete for dataset {dataset}.\n")
            
    def cleanup(self) -> None:
        """
        Scans through specified directories and removes files containing the keyword 'empty' in their filename.
        Also, physically removes the files from the filesystem and prints the count of deleted files per directory.
        """
        print("Starting cleanup of empty images...")
        directories = [self.paths.peaks_dir, self.paths.labels_dir, self.paths.peaks_water_overlay_dir]

        for dir_path in directories:
            deleted = 0  # Reset deleted counter for each directory
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if 'empty' in file:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            print(f"Successfully removed: {file_path}")
                            deleted += 1
                        except OSError as e:
                            print(f"Failed to remove {file_path}: {e}")
            print(f"Total 'empty' files deleted in {dir_path}: {deleted}")
    
        print("Cleanup of empty images complete.")
        self.cleanup_authenticator()
        
    def cleanup_authenticator(self) -> None:
        """
        Scans through specified dataset directories and removes files containing the keyword 'empty' in their filename.
        Also, physically removes the files from the filesystem and prints the count of deleted files per directory.
        """
        print("Starting cleanup of empty images...")
        # Specify directories specific to the dataset
        directories = [
            os.path.join(self.paths.peaks_dir, self.dataset),
            os.path.join(self.paths.labels_dir, self.dataset),
            os.path.join(self.paths.peaks_water_overlay_dir, self.dataset),
        ]

        for dir_path in directories:
            deleted = 0  # Reset deleted counter for each directory
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if 'empty' in file:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            print(f"Successfully removed: {file_path}")
                            deleted += 1
                        except OSError as e:
                            print(f"Failed to remove {file_path}: {e}")
            print(f"Total 'empty' files deleted in {dir_path}: {deleted}")

        print("Cleanup of empty images complete.")
    
    def process_empty(self, percent_empty:float=0.3) -> None:
        self.percent_empty = percent_empty  
        print('\nStarting Step 2...')
        print(f"Processing empty images for dataset {self.dataset}...")
        
        print(f"Percent empty: {percent_empty}")
        if not self.confirm_action("This will generate empty images based on the specified percentage."):
            print("Operation canceled by the user.")
            return 

        # calculate number of empty images to add based on percentage
        num_exist_images = len(self.paths.get_peak_image_paths(self.dataset)) # number of duplicated empty images
        num_empty_images = int(percent_empty * num_exist_images) # number of duplicated water images 
        
        # generate and save empty images, update PathManager accordingly
        print(f'Will generate {num_empty_images} empty images for dataset {self.dataset}...')
        
        for i in range(num_empty_images):
            formatted = f'{i+1:05d}' # starts at 00001
            empty_label_path = os.path.join(self.paths.labels_dir, self.dataset, f'empty_label_{self.dataset}_{formatted}.h5')
            empty_peak_path = os.path.join(self.paths.peaks_dir, self.dataset, f'empty_peak_{self.dataset}_{formatted}.h5')
            empty_overlay_path = os.path.join(self.paths.peaks_water_overlay_dir, self.dataset, f'empty_overlay_{self.dataset}_{formatted}.h5')
            
            # data 
            empty = np.zeros_like(self.water_background, dtype=np.float32) # empty image no water_background 
            water_background = self.water_background # water background image
            
            # save h5 files 
            # labels -> empty, peaks -> empty, overlay -> water background
            parameters = self.get_parameters()
            f.save_h5(file_path=empty_label_path, data=empty, save_parameters=True, params=parameters) # empty label
            clen, photon_energy = parameters[0], parameters[1]
            f.save_h5(file_path=empty_peak_path, data=empty, save_parameters=True, params=parameters) # empty peak
            f.save_h5(file_path=empty_overlay_path, data=water_background, save_parameters=True, params=parameters) # water background
            
            # save paths to PathManager
            self.paths.update_path(file_path=empty_label_path, dir_type='label')
            self.paths.update_path(empty_label_path, dir_type='peak')
            self.paths.update_path(empty_overlay_path, dir_type='overlay')
            
            empty = False
            f.assign_attributes(file_path=empty_label_path, clen=clen, photon_energy=photon_energy, peak=empty)
            f.assign_attributes(empty_peak_path, clen=clen, photon_energy=photon_energy, peak=empty)
            f.assign_attributes(empty_overlay_path, sclen=clen, photon_energy=photon_energy, peak=empty)
            
            print(f"\nEmpty label file created: {empty_label_path}")
            print(f"Empty peak file created: {empty_peak_path}")
            print(f"Empty overlay file mimicking water background created: {empty_overlay_path}\n")

        print(f"Empty images added: {num_empty_images}")
        print(f"Processing complete for dataset {self.dataset}.\n")
        
        print("Step 2 complete.\n")
        print(f"Number of peak files: {len(self.paths.get_peak_image_paths(self.dataset))}")
        print(f"Number of label files: {len(self.paths.get_label_images_paths(self.dataset))}")
        print(f"Number of overlay files: {len(self.paths.get_peaks_water_overlay_image_paths(self.dataset))}")