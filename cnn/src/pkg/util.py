import os 
import re
import h5py as h5 
import numpy as np
import torch
from functools import lru_cache
from skimage.feature import peak_local_max
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import logging 
from glob import glob
import numpy as np

def load_h5(file_path:str) -> np.ndarray:
    with h5.File(file_path, 'r') as file:
        return np.array(file['entry/data/data'])

def save_h5(file_path:str, data:np.ndarray, save_attributes:bool, parameters:tuple) -> None:
    with h5.File(file_path, 'w') as file:
        file.create_dataset('entry/data/data', data=data)
    if save_attributes:
        assign_attributes(file_path, parameters)
    print(f"File saved: {file_path}")

def parameter_matrix(clen_values: list, photon_energy_values: list) -> None:
    # limited to 2d for now 
    dtype = [('clen', float), ('photon_energy', float)]
    matrix = np.zeros((len(clen_values), len(photon_energy_values)), dtype=dtype)
    for i, clen in enumerate(clen_values):
        for j, photon_energy in enumerate(photon_energy_values):
            matrix[i, j] = (clen, photon_energy)
    return matrix

def assign_attributes(file_path: str, parameters: tuple): 
    clen, photon_energy = parameters
    with h5.File(file_path, 'a') as f:
        f.attrs['clen'] = clen
        f.attrs['photon_energy'] = photon_energy
    print(f"Attributes 'clen' and 'photon_energy' assigned to {file_path}")
  
def retrieve_attributes(file_path: str) -> tuple:
    """
    Retrieves 'clen' and 'photon_energy' attributes from an HDF5 file.
    """
    with h5.File(file_path, 'r') as file:
        clen = file.attrs.get('clen')
        photon_energy = file.attrs.get('photon_energy')
    return clen, photon_energy

def check_attributes(paths, dataset: str, type: str) -> bool:
    """
    Checks that 'clen' and 'photon_energy' attributes for all files in a specified type within a dataset
    match expected values derived from the parameter matrix.
    """
    clen_values, photon_energy_values = [1.5, 2.5, 3.5], [6000, 7000, 8000]
    matrix = parameter_matrix(clen_values, photon_energy_values)
    dataset_index = int(dataset) - 1  # Assumes '01' corresponds to index 0, '02' to 1, etc.
    exp_clen, exp_photon_energy = matrix[dataset_index % len(clen_values), dataset_index // len(clen_values)]
    
    if type == 'peak':
        paths = paths.get_peak_image_paths(dataset)
    elif type == 'label':
        paths = paths.get_label_images_paths(dataset)
    elif type == 'overlay':
        paths = paths.get_peaks_water_overlay_image_paths(dataset)
    elif type == 'background': # do not know if i need this for 'background' type
        paths = [paths.get_water_background(dataset)]  # Make it a list to use in iteration 
    else:
        raise ValueError("Invalid type specified.")
    
    for path in paths:
        clen, photon_energy = retrieve_attributes(path)
        if clen != exp_clen or photon_energy != exp_photon_energy:
            print(f"File {path} has mismatching attributes: clen={clen}, photon_energy={photon_energy}")
            return False  # Mismatch found
    
    print(f"All files in dataset {dataset} of type '{type}' have matching attributes.")
    return True

def get_counts(path_manager):
    """
    Counts and reports the number of 'normal' and 'empty' images in the specified directories
    for the selected dataset, using the path_manager to access directory paths.
    """
    # Refresh the lists in PathManager to ensure they are up-to-date
    path_manager.refresh_all()

    # Directories to check
    directory_types = ['peaks', 'labels', 'peaks_water_overlay']
    dataset = path_manager.dataset

    # Loop through each directory type and count files
    for directory_type in directory_types:
        directory_path = os.path.join(path_manager.images_dir, directory_type, dataset)
        all_files = glob(os.path.join(directory_path, '*.h5'))  # Corrected usage here
        normal_files = [file for file in all_files if 'empty' not in os.path.basename(file)]
        empty_files = [file for file in all_files if 'empty' in os.path.basename(file)]

        # Reporting the counts
        print(f"Directory: {directory_type}/{dataset}")
        print(f"  Total files: {len(all_files)}")
        print(f"  Normal images: {len(normal_files)}")
        print(f"  Empty images: {len(empty_files)}")

def prepare(data_manager:Dataset, batch_size:int=32) -> tuple:
    """
    Prepares and splits the data into training and testing datasets.
    Applies transformations and loads them into DataLoader objects.

    :param data_manager: An instance of DatasetManager, which is a subclass of torch.utils.data.Dataset.
    :param batch_size: The size of each data batch.
    :return: A tuple containing train_loader and test_loader.
    """
    # Split the dataset into training and testing sets.
    num_items = len(data_manager)
    num_train = int(0.8 * num_items)
    num_test = num_items - num_train
    train_dataset, test_dataset = torch.utils.data.random_split(data_manager, [num_train, num_test])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    
        
    print("\nData prepared.")
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches in train_loader: {len(train_loader)} \n")
    
    return train_loader, test_loader # returns train/test tensor data loaders
    
class PathManager:
    def __init__(self, dataset:str, root_dir=None) -> None:
        if root_dir is None:
            self.current_path = os.getcwd()  # Using the current working directory as a fallback
            self.root = self.re_root(self.current_path)
        else:
            self.root = root_dir  # Directly using the provided root directory
        self.dataset = dataset
        self.setup_directories()
        
    def setup_directories(self) -> None:
        # Setup directories based on the root
        self.images_dir = os.path.join(self.root, 'images')
        self.peaks_dir = os.path.join(self.images_dir, 'peaks') 
        self.labels_dir = os.path.join(self.images_dir, 'labels')
        self.peaks_water_overlay_dir = os.path.join(self.images_dir, 'peaks_water_overlay')
        self.water_background_dir = os.path.join(self.images_dir, 'water')
        self.temp = os.path.join(self.images_dir, 'temp')
        self.select_dataset(dataset=self.dataset) # calls init_lists()
        
    def init_lists(self, dataset:str) -> None:
        self.peak_list = self.get_peak_image_paths(self.dataset)
        self.water_peak_list = self.get_peaks_water_overlay_image_paths(self.dataset)
        self.label_list = self.get_label_images_paths(self.dataset)
        self.water_background_list = [self.get_water_background(self.dataset)] # expecting 1 image
        return self.peak_list, self.water_peak_list, self.label_list, self.water_background_list
    
    def refresh_all(self) -> tuple:
        """
        Refreshes the internal lists of file paths to reflect current directory state.
        """
        # Clears the cache for each method to force re-computation
        self.get_peak_image_paths.cache_clear()
        self.get_peaks_water_overlay_image_paths.cache_clear()
        self.get_label_images_paths.cache_clear()
        self.get_water_background.cache_clear()

        # Reinitialize the lists with current directory contents
        self.peak_list, self.water_peak_list, self.label_list, self.water_background_list = self.select_dataset(self.dataset)
        
        print(f"Paths refreshed for dataset {self.dataset}.")
    
    def re_root(self, current_path: str) -> str:
        match = re.search("cxls_hitfinder", current_path)
        if match:
            root = current_path[:match.end()]
            return root
        else:
            raise Exception("Could not find the root directory. (cxls_hitfinder)\n", "Current working dir:", self.current_path)
        
    def select_dataset(self, dataset:str) -> tuple:    
        self.dataset = dataset # select dataset 01 through 09
        return self.init_lists(self.dataset)  #peak_paths, water_peak_paths, labels_paths, water_background
    
    def get_path(self, path_name:str) -> str:
            paths_dict = {
                'root': self.root,
                'images_dir': self.images_dir,
                'peaks_dir': self.peaks_dir,
                'labels_dir': self.labels_dir,
                'peaks_water_overlay_dir': self.peaks_water_overlay_dir,
                'water_background_dir': self.water_background_dir,
                'temp': self.temp,
            }
            return paths_dict.get(path_name, None)

    # following functions return list of images of dataset 01 through 09 
    
    @lru_cache(maxsize=32)
    def get_peak_image_paths(self, dataset:str) -> list:
        # returns all peak images of dataset 01 through 09
        dataset_dir = os.path.join(self.peaks_dir, dataset)
        return [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.h5')]
    
    @lru_cache(maxsize=32)
    def get_peaks_water_overlay_image_paths(self, dataset:str) -> list:
        dataset_dir = os.path.join(self.peaks_water_overlay_dir, dataset)
        return [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.h5')]    
    
    @lru_cache(maxsize=32)
    def get_label_images_paths(self, dataset:str) -> list:
        dataset_dir = os.path.join(self.labels_dir, dataset)
        return [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.startswith('label')]
    
    @lru_cache(maxsize=32)
    def get_water_background(self, dataset:str) -> str:
        dataset_dir = os.path.join(self.water_background_dir, dataset)
        water_images = [f for f in os.listdir(dataset_dir) if f.startswith('water') and f.endswith('.h5')]
        if len(water_images) == 1:
            # print(f"Found water background image: {water_images[0]}")
            return os.path.join(dataset_dir, water_images[0]) # expecting 1 image, output:str
        elif len(water_images) > 1:
            raise Exception("Multiple water images found in the specified dataset directory.")
        else:
            raise Exception("Could not find water image in the specified dataset directory.")
    
    def update_path(self, file_path:str, dir_type:str) -> None:
        """
        Updates the internal lists of file paths after a new file has been added.

        Parameters:
        - file_path: The full path of the newly added file.
        - dir_type: The type of the file added ('peak', 'overlay', 'label', or 'background').
        """
        target_list = None
        if dir_type == 'peak':
            target_list = self.peak_list
        elif dir_type == 'overlay':
            target_list = self.water_peak_list
        elif dir_type == 'label':
            target_list = self.label_list
        elif dir_type == 'background':
            target_list = self.water_background_list
        
        if target_list is not None:
            target_list.append(file_path)
            print(f"Path appended to {dir_type}: {file_path}")
        else:
            raise ValueError("Invalid type specified for updating paths.")
        
        print(f"Paths updated for {dir_type}. New file added: {file_path}")

    def remove_path(self, file_path: str, dir_type: str) -> None:
        """
        Removes the specified file path from the internal tracking list based on the directory type.
        """
        target_list = getattr(self, f"{dir_type}_list", None)
        
        if target_list is not None and file_path in target_list:
            target_list.remove(file_path)
            print(f"Path removed from {dir_type}: {file_path}")
        else:
            print(f"File path not found in {dir_type} list or invalid type specified.")    
        

class Processor:
    def __init__(self, paths, dataset: str) -> None:
        self.paths = paths  # Expecting an instance of PathManager
        self.length = None          # .15, .25, .35 m
        self.photon_energy = None   #  6, 7.5, 8 keV 
        self.threshold = 1 # used for peak images (pure signal)
        self.dataset = dataset
        self.dataset_init()
        self.water_background = load_h5(self.paths.get_water_background(dataset))

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
        clen, photon_energy = self.get_parameters()
        return self.dataset_dict
  
    def get_parameters(self) -> tuple:
        clen, photon_energy = self.dataset_dict[self.dataset] # get parameters 
        self.parameters = (clen, photon_energy)
        return clen, photon_energy
    
    @staticmethod
    def update_attributes(file_path:str, clen:float, photon_energy:float) -> None:
        # check file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        try:
            with h5.File(file_path, 'a') as f:
                f.attrs['clen'] = clen
                f.attrs['photon_energy'] = photon_energy
            print(f"Attributes 'clen' and 'photon_energy' updated for {file_path}")
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
        
        parameters = self.get_parameters()

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
            
            peak_image = load_h5(file_path=peak_path)
            labeled_array = self.heatmap(peak_image_array=peak_image)
            peak_water_overlay_image = self.apply_water_background(peak_image_array=peak_image)
            
           # Save the processed peak image and labeled image to their respective unique paths
            save_h5(file_path=out_overlay_path, data=peak_water_overlay_image, save_attributes=True, parameters=parameters)
            save_h5(file_path=out_label_path, data=labeled_array, save_attributes=True, parameters=parameters)
            
            # Update attributes for each file with clen and photon energy
            self.update_attributes(file_path=peak_path, clen=clen, photon_energy=photon_energy)
            self.update_attributes(file_path=out_overlay_path, clen=clen, photon_energy=photon_energy)
            self.update_attributes(file_path=out_label_path, clen=clen, photon_energy=photon_energy)
            
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
            save_h5(empty_label_path, empty, save_attributes=True, parameters=self.parameters) # empty label
            save_h5(empty_peak_path, empty, save_attributes=True, parameters=self.parameters) # empty peak
            save_h5(empty_overlay_path, water_background, save_attributes=True, parameters=self.parameters) # water background
            
            # save paths to PathManager
            self.paths.update_path(file_path=empty_label_path, dir_type='label')
            self.paths.update_path(empty_label_path, dir_type='peak')
            self.paths.update_path(empty_overlay_path, dir_type='overlay')
            
            assign_attributes(empty_label_path, self.parameters)
            assign_attributes(empty_peak_path, self.parameters)
            assign_attributes(empty_overlay_path, self.parameters)
            
            print(f"\nEmpty label file created: {empty_label_path}")
            print(f"Empty peak file created: {empty_peak_path}")
            print(f"Empty overlay file mimicking water background created: {empty_overlay_path}\n")

        print(f"Empty images added: {num_empty_images}")
        print(f"Processing complete for dataset {self.dataset}.\n")
        
        print("Step 2 complete.\n")
        print(f"Number of peak files: {len(self.paths.get_peak_image_paths(self.dataset))}")
        print(f"Number of label files: {len(self.paths.get_label_images_paths(self.dataset))}")
        print(f"Number of overlay files: {len(self.paths.get_peaks_water_overlay_image_paths(self.dataset))}")
        
class DatasetManager(Dataset):
    # for PyTorch DataLoader
    #NOTE: self.water_background is a string path
    #   self.water_count is the number of water images to be added to the dataset
    #   self.include_water_background is a boolean flag to include water background images in the dataset
    #   self.percent_water_repeat is the percentage of water images to be added to the dataset
    
    def __init__(self, paths, dataset:str, parameters:tuple, transform=False) -> None:
        self.paths = paths
        self.dataset = dataset
        self.parameters = parameters
        self.transform = transform if transform is not None else TransformToTensor()
        self.setup_directories(dataset=dataset)
        
    def setup_directories(self, dataset:str) -> None:
        get_counts(self.paths)
        self.paths.refresh_all()
        self.peak_paths, self.water_peak_paths, self.labels_paths, self.water_background = self.paths.select_dataset(dataset)
        self.authenticate_dataset()

    def __len__(self) -> int:
        return len(self.peak_paths)
    
    def __getitem__(self, idx:int) -> tuple:
        peak_image = load_h5(self.peak_paths[idx])
        water_image = load_h5(self.water_peak_paths[idx])
        label_image = load_h5(self.labels_paths[idx])
            
        if self.transform:
            peak_image = self.transform(peak_image) 
            water_image = self.transform(water_image) # dimensions: C x H x W
            label_image = self.transform(label_image)
            
        return (peak_image, water_image), label_image
    
    def authenticate_dataset(self) -> None:
        """
        Authenticates the dataset by verifying the parameters of a sample from each path category.
        """
        self.count_empty_images()
        unif_peaks = check_attributes(self.paths, self.dataset, 'peak')
        unif_label = check_attributes(self.paths, self.dataset, 'label')
        unif_overlay = check_attributes(self.paths, self.dataset, 'overlay')
        unif_water = check_attributes(self.paths, self.dataset, 'background')
        if not (unif_peaks and unif_label and unif_overlay and unif_water):
            raise ValueError(f"Dataset {self.dataset} failed authentication.")
        else:        
            print(f"Dataset {self.dataset} authenticated.\n")
    
    def count_empty_images(self) -> float:
        """
        Counts the percentage of empty images across different directories within the dataset.
        """
        empty_counts = {
            'peaks': sum(1 for path in self.peak_paths if 'empty' in os.path.basename(path)),
            'water_overlays': sum(1 for path in self.water_peak_paths if 'empty' in os.path.basename(path)),
            'labels': sum(1 for path in self.labels_paths if 'empty' in os.path.basename(path))
        }
        total_images_count = len(self.peak_paths) + len(self.water_peak_paths) + len(self.labels_paths)
        total_empty_images = sum(empty_counts.values())
        actual_percent_empty = (total_empty_images / total_images_count) * 100 if total_images_count > 0 else 0

        print(f"Actual percentage of empty images: {actual_percent_empty}% across peaks, water_overlays, and labels directories.\n")
    

class TransformToTensor:
    def __call__(self, image:np.ndarray) -> torch.Tensor:
        """
        Converts a numpy array to a PyTorch tensor while ensuring:
        - The output is in B x C x H x W format for a single image.
        - The tensor is of dtype float32.
        - For grayscale images, a dummy channel is added to ensure compatibility.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            torch.Tensor: The transformed image tensor.
        """
        if image.ndim == 2:  # For grayscale images
            image = np.expand_dims(image, axis=0)  # Add channel dimension 
        else : 
            raise ValueError(f"Image has invalid dimensions: {image.shape}")
        image_tensor = torch.from_numpy(image).float().to(dtype=torch.float32)  # Convert to tensor with dtype=torch.float32
        return image_tensor # dimensions: C x H x W 
    

class TrainTestModels:
    """ 
    This class trains, tests, and plots the loss, accuracy, and confusion matrix of a model.
    There are two methods for training: test_model_no_freeze and test_model_freeze.
    """
    def __init__(self, model, loader: list, criterion, optimizer, device, cfg: dict) -> None:
        """ 
        Takes the arguments for training and testing and makes them available to the class.

        Args:
            model: PyTorch model
            loader: list of torch.utils.data.DataLoader where loader[0] is the training set and loader[1] is the testing set
            criterion: PyTorch loss function
            optimizer: PyTorch optimizer
            device: torch.device which is either 'cuda' or 'cpu'
            cfg: dict which holds the configuration parameters num_epochs, batch_size, and num_classes
        """
        self.model = model
        self.loader = loader
        self.train_loader, self.test_loader = loader[0], loader[1]
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = cfg['num_epochs']
        self.device = device
        self.batch = cfg['batch_size']
        self.classes = cfg['num_classes']
        self.plot_train_accuracy = np.zeros(self.epochs)
        self.plot_train_loss = np.zeros(self.epochs)
        self.plot_test_accuracy = np.zeros(self.epochs)
        self.plot_test_loss = np.zeros(self.epochs)
        self.logger = logging.getLogger(__name__)

    def train(self) -> None:
        """
        This function trains the model without freezing the parameters in the case of transfer learning.
        This will print the loss and accuracy of the training sets per epoch.
        """
        self.logger.info(f'Model training: {self.model.__class__.__name__}')
        
        for epoch in range(self.epochs):
            self.logger.info('-- epoch '+str(epoch)) 
            running_loss_train = accuracy_train = predictions = total_predictions = 0.0

            self.model.train()
            for inputs, labels in self.loader[0]: 
                peak_images, overlay_images = inputs
                peak_images, overlay_images, labels = peak_images.to(self.device), overlay_images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                score = self.model(peak_images)
                loss = self.criterion(score, labels)

                loss.backward()
                self.optimizer.step()
                running_loss_train += loss.item()  
                predictions = (torch.sigmoid(score) > 0.5).long()  
                accuracy_train += (predictions == labels).float().sum()
                total_predictions += np.prod(labels.shape)
                
            loss_train = running_loss_train / self.batch
            self.plot_train_loss[epoch] = loss_train
            
            self.logger.info(f'Train loss: {loss_train}')

            # If you want to uncomment these lines, make sure the calculation of accuracy_train is corrected as follows:
            accuracy_train /= total_predictions
            self.plot_train_accuracy[epoch] = accuracy_train
            self.logger.info(f'Train accuracy: {accuracy_train}')
            
    def test_freeze(self) -> None:
        """ 
        This function trains the model with freezing the parameters of in the case of transfer learning.
        This will print the loss and accuracy of the testing sets per epoch.
        WIP
        """
        pass
        
    def test(self) -> None:
        """ 
        This function test the model and prints the loss and accuracy of the testing sets per epoch.
        """
        self.logger.info(f'Model testing: {self.model.__class__.__name__}')
        
        for epoch in range(self.epochs):
            self.logger.info('-- epoch '+str(epoch)) 
            
            running_loss_test = accuracy_test = predicted = total = 0.0
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.loader[1]:
                    peak_images, _ = inputs
                    peak_images = peak_images.to(self.device)
                    labels = labels.to(self.device)

                    score = self.model(peak_images)
                    loss = self.criterion(score, labels)
                    running_loss_test += loss.item()  # Convert to Python number with .item()
                    predicted = (torch.sigmoid(score) > 0.5).long()  # Assuming 'score' is the output of your model
                    accuracy_test += (predicted == labels).float().sum()
                    total += np.prod(labels.shape)

            loss_test = running_loss_test/self.batch[1]
            self.plot_test_loss[epoch] = loss_test

            accuracy_test /= total
            self.plot_test_accuracy[epoch] = accuracy_test

            self.logger.info(f'Test loss: {loss_test}')
            self.logger.info(f'Test accuracy: {accuracy_test}')
        
    def plot_loss_accuracy(self) -> None:
        """ 
        This function plots the loss and accuracy of the training and testing sets per epoch.
        """
        plt.plot(range(self.epochs), self.plot_train_accuracy, marker='o', color='red')
        plt.plot(range(self.epochs), self.plot_test_accuracy, marker='o', color='orange', linestyle='dashed')
        plt.plot(range(self.epochs), self.plot_train_loss ,marker='o',color='blue')
        plt.plot(range(self.epochs), self.plot_test_loss ,marker='o',color='teal',linestyle='dashed')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss/accuracy')
        plt.legend(['accuracy train','accuracy test','loss train','loss test'])
        plt.show()
    
    def plot_confusion_matrix(self) -> None:
        """ 
        This function plots the confusion matrix of the testing set.
        """
        cm_test = np.zeros((self.classes,self.classes), dtype=int)
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, label in self.loader[1]:
                peak_images, _ = inputs
                peak_images = peak_images.to(self.device)
                label = label.to(self.device)

                score = self.model(peak_images).squeeze()

                predictions = (torch.sigmoid(score) > 0.5).long()

                all_labels.extend(label.cpu().numpy().flatten()) 
                all_predictions.extend(predictions.cpu().numpy().flatten())

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        cm_test = confusion_matrix(all_labels, all_predictions)

        plt.matshow(cm_test,cmap="Blues")
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    