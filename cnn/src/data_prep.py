import os
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import namedtuple

class PeakImageDataset(Dataset):
    def __init__(self, image_paths, water_images, transform=None):
        self.image_paths = image_paths
        # self.label_paths = label_paths # add label_paths arg
        self.water_images = water_images
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        h5_path = 'entry/data/data'
        with h5py.File(self.image_paths[idx], 'r') as f:
            peak_image = np.array(f[h5_path])
        with h5py.File(self.label_paths[idx], 'r') as f:
            water_image = np.array(f[h5_path])
        # with h5py.File(self.label_paths[idx], 'r') as f:
        #     label_image = np.array(f[h5_path])
                
        if self.transform:
            peak_image = self.transform(peak_image)
            water_image = self.transform(water_image)
            # label_image = self.transform(label_image)
        
        return peak_image, water_image
  
  
def prep_data(peak_image_paths, water_image_path):
    found_peak_image_paths = [os.path.join(peak_image_paths, f) for f in os.listdir(peak_image_paths) if f.endswith('.h5')] 
    found_water_image_path = [os.path.join(water_image_path, f) for f in os.listdir(water_image_path) if f.startswith('processed')]
    
    print(found_peak_image_paths)
    print(found_water_image_path)
    
    # join the data with water/peak images, then split 80/20
    train_image_paths, test_image_paths, train_water_paths, test_water_paths = train_test_split(found_peak_image_paths, found_water_image_path, test_size=0.2, random_state=42)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()) #ensure float type
    ])
    
    train_dataset = PeakImageDataset(train_image_paths, train_water_paths, transform=transform)
    test_dataset = PeakImageDataset(test_image_paths, test_water_paths, transform=transform)
    
    print(len(train_dataset), len(test_dataset))
    
    return train_dataset, test_dataset
    
def get_relative_paths():
    # want to be universal, repo specific not local specific
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    background_h5_path = os.path.join(root_path, 'sim', 'water_background.h5')
    peak_images_path = os.path.join(root_path, 'images', 'peaks')
    processed_images_path = os.path.join(root_path, 'images', 'data')
    label_output_path = os.path.join(root_path, 'images', 'labels')
    # namedtuple 
    Paths = namedtuple('Paths', ['peak_images', 'processed_images', 'label_output', 'background_h5', 'root'])
    Paths = Paths(peak_images_path, processed_images_path, label_output_path, background_h5_path, root_path)
    
    return Paths
    
Paths = get_relative_paths()
peak_images_path = Paths.peak_images
water_images_path = Paths.processed_images
label_images_path = Paths.label_output
background_h5_path = Paths.background_h5
root_path = Paths.root

train_dataset, test_dataset = prep_data(peak_images_path, water_images_path)
