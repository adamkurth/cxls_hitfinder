import os 
import re
import h5py as h5 
import numpy as np
import torch
from functools import lru_cache
from skimage.feature import peak_local_max
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import numpy as np

def load_h5(file_path:str) -> np.ndarray:
    with h5.File(file_path, 'r') as file:
        return np.array(file['entry/data/data'])

def save_h5(file_path:str, data:np.ndarray) -> None:
    with h5.File(file_path, 'w') as file:
        file.create_dataset('entry/data/data', data=data)

def parameter_matrix(clen_values: list, photon_energy_values: list) -> None:
    # limited to 2d for now 
    print("\ntrue parameter matrix...\n")
    dtype = [('clen', float), ('photon_energy', float)]
    matrix = np.zeros((len(clen_values), len(photon_energy_values)), dtype=dtype)
    for i, clen in enumerate(clen_values):
        for j, photon_energy in enumerate(photon_energy_values):
            matrix[i, j] = (clen, photon_energy)
    print(matrix)
    
class PathManager:
    def __init__(self, root_dir=None) -> None:
        if root_dir is None:
            self.current_path = os.getcwd()  # Using the current working directory as a fallback
            self.root = self.re_root(self.current_path)
        else:
            self.root = root_dir  # Directly using the provided root directory
        self.dataset = None
        self.setup_directories()
        
    def setup_directories(self) -> None:
        # Setup directories based on the root
        self.images_dir = os.path.join(self.root, 'images')
        self.peaks_dir = os.path.join(self.images_dir, 'peaks') 
        self.labels_dir = os.path.join(self.images_dir, 'labels')
        self.peaks_water_overlay_dir = os.path.join(self.images_dir, 'peaks_water_overlay')
        self.water_background_dir = os.path.join(self.images_dir, 'water')
        self.temp = os.path.join(self.images_dir, 'temp')
    
    def select_dataset(self, dataset:str) -> tuple:
        self.dataset = dataset # select dataset 01 through 09
        peak_paths = self.get_peak_image_paths(dataset)
        water_peak_paths = self.get_peaks_water_overlay_image_paths(dataset)
        labels = self.get_label_images_paths(dataset)
        water_background = self.get_water_background(dataset)
        return peak_paths, water_peak_paths, labels, water_background
         
    def re_root(self, current_path: str) -> str:
        match = re.search("cxls_hitfinder", current_path)
        if match:
            root = current_path[:match.end()]
            return root
        else:
            raise Exception("Could not find the root directory. (cxls_hitfinder)\n", "Current working dir:", self.current_path)

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
        water_image = [f for f in os.listdir(dataset_dir) if f.startswith('water') and f.endswith('.h5')]
        if len(water_image) > 0:
            return os.path.join(dataset_dir, water_image[0]) # expecting 1 image
        else:
            raise Exception("Could not find water image in the specified dataset directory.")
        
class Processor:
    def __init__(self, paths, dataset: str) -> None:
        self.paths = paths  # Expecting an instance of PathManager
        self.length = None     # .15, .25, .35 m
        self.photon_energy = None # 6,7.5, 8 keV 
        self.threshold = 1 # used for peak images (pure signal)
        # self.protein = '1IC6'
        self.water_background = load_h5(self.paths.get_water_background(dataset))

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
        confirm = input(f"Confirm '{value_name}' value: {value} (type 'y' to confirm): ")
        if confirm.lower() == 'y':
            print(f"'{value_name}' value {value} confirmed successfully.")
            return True
        else:
            print(f"'{value_name}' confirmation failed at attempt {i}.")
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
        
        if not (clen_confirmed and photon_energy_confirmed and dataset_confirmed):
            print("Operation cancelled due to confirmation failure.")
            return
        
        # get peak images
        peak_image_paths = self.paths.get_peak_image_paths(dataset)
        # get water background image 
        water_background = self.water_background
        
        for peak_path in peak_image_paths:
            # derive a unique basename for each peak image
            basename = os.path.basename(peak_path)
            # unique filenames for the overlay and label images based on the current peak image's basename
            out_overlay_path = os.path.join(self.paths.peaks_water_overlay_dir, dataset, f'overlay_{basename}')
            out_label_path = os.path.join(self.paths.labels_dir, dataset, f'label_{basename}')
            
            peak_image = load_h5(peak_path)
            labeled_array = self.heatmap(peak_image)
            peak_water_overlay_image = self.apply_water_background(peak_image)
            
           # Save the processed peak image and labeled image to their respective unique paths
            save_h5(out_overlay_path, peak_water_overlay_image)
            save_h5(out_label_path, labeled_array)
            
            # Update attributes for each file with clen and photon energy
            self.update_attributes(peak_path, clen, photon_energy)
            self.update_attributes(out_overlay_path, clen, photon_energy)
            self.update_attributes(out_label_path, clen, photon_energy)
            
            print(f"Processed and labeled images for {basename} saved.")
    
        

# class DatasetManager(Dataset):
#     # for PyTorch DataLoader
#     def __init__(self, paths:PathManager, dataset:str, transform=False) -> None:
#         self.paths = paths
#         self.dataset = dataset
#         self.peak_paths, self.water_peak_paths, self.labels = self.paths.select_dataset(dataset)
#         # self.transform = transform if transform is not None else TransformToTensor()
    
#     def __len__(self) -> int:
#         return len(self.peak_paths)
    
#     def __getitem__(self, idx:int) -> tuple:
#         peak_image = self.load_h5(self.peak_image_paths[idx])
#         water_image = self.load_h5(self.water_image_paths[idx])
#         label_image = self.load_h5(self.label_image_paths[idx])

#         # if self.transform:
#         #     peak_image = self.transform(peak_image) 
#         #     water_image = self.transform(water_image) # dimensions: C x H x W
#         #     label_image = self.transform(label_image)
#         return (peak_image, water_image), label_image

