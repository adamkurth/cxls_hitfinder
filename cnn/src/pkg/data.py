import os
import h5py as h5
import numpy as np
from typing import Any
from glob import glob
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Union
from pkg.functions import load_h5, get_counts, get_params, convert2int, convert2str

class DatasetManager(Dataset):
    # for PyTorch DataLoader
    #NOTE: self.water_background is a string path
    #   self.water_count is the number of water images to be added to the dataset
    #   self.include_water_background is a boolean flag to include water background images in the dataset
    #   self.percent_water_repeat is the percentage of water images to be added to the dataset
    
    def __init__(self, paths, datasets:List[int], transform=None) -> None:
        self.paths = paths
        self.datasets = convert2str(datasets=datasets)
        self.parameters = get_params(datasets=self.datasets)
        self.setup_datasets()
        # self.transform = transform if transform is not None else TransformToTensor()
        self.count_empty_images()
        print(f"Final dataset sizes - Peaks: {len(self.peak_paths)}, Labels: {len(self.label_paths)}, Overlays: {len(self.water_peak_paths)}")
    
    def setup_datasets(self):
        self.total_paths = self.paths.selected_datasets()
        self.peak_paths = self.total_paths.peaks
        self.water_peak_paths = self.total_paths.overlays
        self.label_paths = self.total_paths.labels
        self.water_background = self.total_paths.water_background
        
    def __len__(self) -> int:
        return len(self.peak_paths)
    
    def __getitem__(self, idx:int) -> tuple:
        water_image = load_h5(self.water_peak_paths[idx])
        label_image = load_h5(self.label_paths[idx])
            
        if self.transform:
            water_image = self.transform(water_image) # dimensions: C x H x W
            label_image = self.transform(label_image)
            
        return water_image, label_image
    
    # def authenticate_attributes(self) -> None:
    #     """
    #     Authenticates the attributes of the dataset by verifying the parameters of a sample from each path category.
    #     """
    #     for dataset in self.datasets:
    #         params = self.parameters.get(dataset)
    #         if not params:
    #             raise ValueError(f"Found no parameters found for dataset {dataset}.")
            
    #         exp_clen, exp_photon_energy = params
    #         # paths, dataset: str, dir_type: str, **expected_attrs
    #         unif_peaks = check_attributes(paths=self.paths, dataset=dataset, dir_type='peak', clen=exp_clen, photon_energy=exp_photon_energy)
    #         unif_label = check_attributes(paths=self.paths, dataset=dataset, dir_type='label', clen=exp_clen, photon_energy=exp_photon_energy)
    #         unif_overlay = check_attributes(paths=self.paths, dataset=dataset, dir_type='overlay', clen=exp_clen, photon_energy=exp_photon_energy)
    #         unif_water = check_attributes(paths=self.paths, dataset=dataset, dir_type='background', clen=exp_clen, photon_energy=exp_photon_energy)
            
    #         if not (unif_peaks and unif_label and unif_overlay and unif_water):
    #             raise ValueError(f"Dataset {dataset} failed authentication.")
    #         else:
    #             print(f"Dataset {dataset} authenticated successfully.\n")
    
    def count_empty_images(self) -> float:
        """
        Counts the percentage of empty images across different directories within the dataset.
        """
        empty_counts = {
            'peaks': sum(1 for path in self.peak_paths if 'empty' in os.path.basename(path)),
            'water_overlays': sum(1 for path in self.water_peak_paths if 'empty' in os.path.basename(path)),
            'labels': sum(1 for path in self.label_paths if 'empty' in os.path.basename(path))
        }
        total_images_count = len(self.peak_paths) + len(self.water_peak_paths) + len(self.label_paths)
        total_empty_images = sum(empty_counts.values())
        actual_percent_empty = (total_empty_images / total_images_count) * 100 if total_images_count > 0 else 0

        print(f"Actual percentage of empty images: {actual_percent_empty}% across peaks, water_overlays, and labels directories.\n")

            