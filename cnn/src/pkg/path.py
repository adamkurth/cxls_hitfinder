import os
import re
from functools import lru_cache
from glob import glob
from typing import Union, List
from collections import namedtuple
from pkg import functions as f
from pkg import process as p
class PathManager:
    def __init__(self, datasets:List[int], root_dir=None) -> None:
        if root_dir is None:
            self.current_path = os.getcwd()  # Using the current working directory as a fallback
            self.root = self.re_root(self.current_path)
        else:
            self.root = root_dir  # Directly using the provided root directory
        self.datasets = [str(num).zfill(2) for num in datasets]
        print(self.datasets)
        self.setup_directories()
        
    def setup_directories(self) -> None:
        # self.datasets = [str(num).zfill(2) for num in self.datasets]
        self.images_dir = os.path.join(self.root, 'images')
        self.peaks_dir = os.path.join(self.images_dir, 'peaks')
        self.labels_dir = os.path.join(self.images_dir, 'labels')
        self.peaks_water_overlay_dir = os.path.join(self.images_dir, 'peaks_water_overlay')
        self.water_background_dir = os.path.join(self.images_dir, 'water')
        self.temp = os.path.join(self.images_dir, 'temp')
        self.total_paths = self.selected_datasets()
        
    def selected_datasets(self):
        Paths = namedtuple('Paths', ['peaks', 'overlays', 'labels', 'water_background'])
        peaks, overlays,  labels, water_background = [], [], [], []
        
        for dataset in self.datasets:
            # Ensure correct paths are appended to their respective lists
            peaks += glob(os.path.join(self.peaks_dir, dataset, '*.h5'))
            overlays += glob(os.path.join(self.peaks_water_overlay_dir, dataset, '*.h5'))
            labels += glob(os.path.join(self.labels_dir, dataset, '*.h5'))
            water_background += glob(os.path.join(self.water_background_dir, dataset, '*.h5'))
            
            
        return Paths(peaks=peaks, overlays=overlays, labels=labels, water_background=water_background)

    def init_lists(self, dataset:str) -> list: 
        self.peak_list = self.get_peak_image_paths(self.dataset)
        self.water_peak_list = self.get_peaks_water_overlay_image_paths(self.dataset)
        self.label_list = self.get_label_images_paths(self.dataset)
        self.water_background_list = [self.get_water_background(self.dataset)] # expecting 1 image
        return self.peak_list, self.water_peak_list, self.label_list, self.water_background_list

    def fetch_paths_by_type(self, dataset:str, dir_type:str) -> list:
        """
        Fetches and returns a list of file paths based on the specified directory type.
        """
        if dir_type == 'peak':
            return self.get_peak_image_paths(dataset=dataset)
        elif dir_type == 'overlay':
            return self.get_peaks_water_overlay_image_paths(dataset=dataset)
        elif dir_type == 'label':
            return self.get_label_images_paths(dataset=dataset)
        elif dir_type == 'background':
            return [self.get_water_background(dataset=dataset)]
        else:
            raise ValueError("Invalid directory type specified.")
    
    def refresh_all(self) -> tuple:
        """
        Refreshes the internal lists of file paths to reflect current directory state.
        """
        # Clears the cache for each method to force re-computation
        self.get_peak_image_paths.cache_clear()
        self.get_peaks_water_overlay_image_paths.cache_clear()
        self.get_label_images_paths.cache_clear()
        self.get_water_background.cache_clear()
        self.total_paths = self.selected_datasets()
        # # Reinitialize the lists with current directory contents
        # self.peak_list, self.water_peak_list, self.label_list, self.water_background_list = self.init_lists(self.dataset)
        
        print(f"Paths refreshed for dataset {self.datasets}.")
    
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
        return [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.startswith('label') or f.startswith('empty') and f.endswith('.h5')]

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
            target_list = self.total_paths.peaks
        elif dir_type == 'overlay':
            target_list = self.total_paths.overlays
        elif dir_type == 'label':
            target_list = self.total_paths.labels
        elif dir_type == 'background':
            target_list = self.total_paths.water_background
        
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
        