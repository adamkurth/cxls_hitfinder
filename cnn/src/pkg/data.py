import os
import h5py as h5
import numpy as np
from typing import Any
from glob import glob
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pkg.functions import load_h5, check_attributes, get_counts
from pkg.transform import TransformToTensor

class DatasetManager(Dataset):
    # for PyTorch DataLoader
    #NOTE: self.water_background is a string path
    #   self.water_count is the number of water images to be added to the dataset
    #   self.include_water_background is a boolean flag to include water background images in the dataset
    #   self.percent_water_repeat is the percentage of water images to be added to the dataset
    
    def __init__(self, paths, dataset:str, parameters:list, transform=False) -> None:
        self.paths = paths
        self.dataset = dataset
        self.parameters = parameters # list
        self.transform = transform if transform is not None else TransformToTensor()
        self.setup(dataset=dataset)
        print(f"Final dataset sizes - Peaks: {len(self.peak_paths)}, Labels: {len(self.label_paths)}, Overlays: {len(self.water_peak_paths)}")
   
    def setup(self, dataset:str) -> None:
        get_counts(self.paths)
        self.paths.refresh_all()
        self.peak_paths, self.water_peak_paths, self.label_paths, self.water_background = self.paths.select_dataset(dataset)
        self.count_empty_images()
        self.authenticate_attributes()
        
    def __len__(self) -> int:
        return len(self.peak_paths)
    
    def __getitem__(self, idx:int) -> tuple:
        # print(idx, len(self.peak_paths), len(self.water_peak_paths), len(self.labels_paths)) # debug print for idx out of range error
        peak_image = load_h5(self.peak_paths[idx])
        water_image = load_h5(self.water_peak_paths[idx])
        label_image = load_h5(self.label_paths[idx])
            
        if self.transform:
            peak_image = self.transform(peak_image) 
            water_image = self.transform(water_image) # dimensions: C x H x W
            label_image = self.transform(label_image)
            
        return (peak_image, water_image), label_image
    
    def authenticate_attributes(self) -> None:
        """
        Authenticates the attributes of the dataset by verifying the parameters of a sample from each path category.
        """
        clen, photon_energy = self.parameters[0], self.parameters[1]
        # paths, dataset: str, type: str, **expected_attrs
        unif_peaks = check_attributes(paths=self.paths, dataset=self.dataset, type='peak', clen=clen, photon_energy=photon_energy)
        unif_label = check_attributes(paths=self.paths, dataset=self.dataset, type='label', clen=clen, photon_energy=photon_energy)
        unif_overlay = check_attributes(paths=self.paths, dataset=self.dataset, type='overlay', clen=clen, photon_energy=photon_energy)
        unif_water = check_attributes(paths=self.paths, dataset=self.dataset, type='background', clen=clen, photon_energy=photon_energy)
        
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
            'labels': sum(1 for path in self.label_paths if 'empty' in os.path.basename(path))
        }
        total_images_count = len(self.peak_paths) + len(self.water_peak_paths) + len(self.label_paths)
        total_empty_images = sum(empty_counts.values())
        actual_percent_empty = (total_empty_images / total_images_count) * 100 if total_images_count > 0 else 0

        print(f"Actual percentage of empty images: {actual_percent_empty}% across peaks, water_overlays, and labels directories.\n")
    
    def validate_dataset_sizes(self):
        """
        Validates that the lengths of peak_paths, label_paths, and water_peak_paths are consistent.
        Throws an error if there's a mismatch.
        """
        if not (len(self.peak_paths) == len(self.label_paths) == len(self.water_peak_paths)):
            raise ValueError("Dataset size mismatch detected. Please ensure each peak image has a corresponding label and overlay.")