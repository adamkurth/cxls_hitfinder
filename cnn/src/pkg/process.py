from typing import List, Tuple
import os
import h5py as h5
import numpy as np
import torch
from skimage.feature import peak_local_max
from typing import List
from pkg import *

class Processor:
    def __init__(self, paths, datasets: List[int]) -> None:
        self.paths = paths  # Expecting an instance of PathManager
        self.threshold = 1 # used for peak images (pure signal)
        self.datasets = [str(num).zfill(2) for num in datasets] # list of datasets
        self.dataset_dict = self.dataset_init()
        self.parameters_dict = self.selected_datasets() 
        self.water_backgrounds = self.init_water_background()
        self.water_background_dict = self.convert_water_backgrounds_to_dict()

    def init_water_background(self) -> list:
        self.water_backgrounds = self.paths.total_paths.water_background
        print(f"Water backgrounds initialized: {len(self.water_backgrounds)}")
        return self.water_backgrounds        
    
    def convert_water_backgrounds_to_dict(self) -> dict:
        """
        Converts the list of water background paths into a dictionary keyed by dataset identifiers.
        
        Returns:
            dict: A dictionary where keys are dataset identifiers and values are water background paths.
        """
        water_dict = {}
        for path in self.water_backgrounds:
            # Extract dataset identifier from the path
            try: # this works for mac/linux
                dataset_id = path.split('/')[-2]
            except: # this works for windows
                dataset_id = path.split('\\')[-2]

            water_dict[dataset_id] = path
            # print(water_dict)
        return water_dict
            
    def selected_datasets(self) -> dict:
        """Returns a dictionary of selected datasets with their corresponding parameters.

        Returns:
            dict: A dictionary where the keys are the dataset names and the values are dictionaries
                  containing the parameters 'clen' and 'photon_energy' for each dataset.
        """
        parameters_dict = {}
        for dataset in self.datasets:
            params = self.dataset_dict.get(dataset, None)
            if params is not None:
                clen, photon_energy = params
                parameters_dict[dataset] = {'clen': clen, 'photon_energy': photon_energy}
        return parameters_dict

    def dataset_init(self) -> None:
        """
        Initializes the dataset dictionary with predefined values for clen and photon energy.

        Returns:
            None
        """
        clen_values, photon_energy_values = [0.15, 0.25, 0.35], [6000, 7000, 8000]
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
  
    def get_parameters(self) -> dict:
        return self.selected_datasets()
        
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
    
    def process_directory(self) -> None:
        """Processes directories for all selected datasets, applying water background and generating labels."""
        # Iterate through each dataset and its parameters
        for key_dataset, parameters in self.parameters_dict.items():
            clen, photon_energy = parameters['clen'], parameters['photon_energy']
            # confirm dataset and parameters 
            dataset_confirmed = self.confirm_value(key_dataset, 'dataset')
            clen_confirmed = self.confirm_value(clen, 'clen')
            photon_energy_confirmed = self.confirm_value(photon_energy, 'photon_energy')
            water_background_path = self.water_background_dict.get(key_dataset)
            water_background_confirmed = self.confirm_value(water_background_path, 'water_background')
                
        
            if not (dataset_confirmed and clen_confirmed and photon_energy_confirmed and water_background_confirmed):
                print(f"Skipping dataset {key_dataset} due to unconfirmed parameters.")
                continue
            
            # Process each dataset independently
            self.process_single_dataset(dataset=key_dataset, clen=clen, photon_energy=photon_energy) 
            print(f"Completed processing for dataset {key_dataset}.")

    def process_single_dataset(self, dataset: str, clen: float, photon_energy: int) -> None:
        """Processes a single dataset by applying water background and generating label images."""
        peak_paths = self.paths.get_peak_image_paths(dataset=dataset)
        water_background_path = self.water_background_dict.get(dataset)
        self.water_background = f.load_h5(file_path=water_background_path)
        for peak_path in peak_paths:
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
        for dataset in self.datasets:
            directories = [
                os.path.join(self.paths.peaks_dir, dataset),
                os.path.join(self.paths.labels_dir, dataset),
                os.path.join(self.paths.peaks_water_overlay_dir, dataset),
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

        print(f"Cleanup of empty images complete for {self.datasets}.")
    
        
    def process_empty(self, percent_empty:float=0.3) -> None:
        self.percent_empty = percent_empty  
        
        print('\nStarting Step 2...')
        
        for dataset in self.datasets:
            print(f"\nProcessing empty images for dataset {dataset}...")
            
            # Confirm action for the current dataset
            action_description = f"This will generate empty images based on {percent_empty*100}% of the existing images for dataset {dataset}."
            if self.confirm_action(action_description):
                self.process_empty_single_dataset(dataset=dataset, percent_empty=percent_empty)
            else:
                print(f"Operation canceled by the user for dataset {dataset}.")
            
    def process_empty_single_dataset(self, dataset:str, percent_empty:float) -> None:
        """Processes a single dataset to generate and save empty images."""
        print(f"\nGenerating empty images for dataset {dataset} with {percent_empty*100}% empty.")
        self.dataset = dataset

        # Fetch paths and calculate number of empty images to add based on percentage
        peak_image_paths = self.paths.get_peak_image_paths(dataset)
        num_exist_images = len(peak_image_paths)
        num_empty_images = int(percent_empty * num_exist_images)
        # generate and save empty images, update PathManager accordingly
        print(f'Will generate {num_empty_images} empty images for dataset {dataset}...')
        
        for i in range(num_empty_images):
            formatted = f'{i+1:05d}' # starts at 00001
            empty_label_path = os.path.join(self.paths.labels_dir, dataset, f'empty_label_{dataset}_{formatted}.h5')
            empty_peak_path = os.path.join(self.paths.peaks_dir, dataset, f'empty_peak_{dataset}_{formatted}.h5')
            empty_overlay_path = os.path.join(self.paths.peaks_water_overlay_dir, dataset, f'empty_overlay_{dataset}_{formatted}.h5')
            
            # data 
            water_background_path = self.water_background_dict.get(f.convert2str_single(dataset))
            water_background = f.load_h5(water_background_path) # water background
            empty = np.zeros_like(water_background, dtype=np.float32) # empty image no water_background 
                        
            # save h5 files 
            # labels -> empty, peaks -> empty, overlay -> water background
            parameters = self.parameters_dict.get(dataset)
            parameters = [parameters['clen'], parameters['photon_energy']]
            
            f.save_h5(file_path=empty_label_path, data=empty, save_parameters=True, params=parameters) # empty label
            clen, photon_energy = parameters[0], parameters[1]
            f.save_h5(file_path=empty_peak_path, data=empty, save_parameters=True, params=parameters) # empty peak
            f.save_h5(file_path=empty_overlay_path, data=water_background, save_parameters=True, params=parameters) # water background
            
            # save paths to PathManager
            self.paths.update_path(file_path=empty_label_path, dir_type='label')
            self.paths.update_path(file_path=empty_label_path, dir_type='peak')
            self.paths.update_path(file_path=empty_overlay_path, dir_type='overlay')
            
            empty = False
            f.assign_attributes(file_path=empty_label_path, clen=clen, photon_energy=photon_energy, peak=empty)
            f.assign_attributes(empty_peak_path, clen=clen, photon_energy=photon_energy, peak=empty)
            f.assign_attributes(empty_overlay_path, clen=clen, photon_energy=photon_energy, peak=empty)
            
            print(f"\nEmpty label file created: {empty_label_path}")
            print(f"Empty peak file created: {empty_peak_path}")
            print(f"Empty overlay file mimicking water background created: {empty_overlay_path}\n")

        print(f"Empty images added: {num_empty_images}")
        print(f"Processing complete for dataset {dataset}.\n")
        
        print("Step 2 complete.\n")
        print(f"Number of peak files: {len(self.paths.get_peak_image_paths(dataset))}")
        print(f"Number of label files: {len(self.paths.get_label_images_paths(dataset))}")
        print(f"Number of overlay files: {len(self.paths.get_peaks_water_overlay_image_paths(dataset))}")        
        