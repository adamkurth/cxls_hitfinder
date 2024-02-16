import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import namedtuple

class PeakImageDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths 
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        with h5py.File(self.image_paths[idx], 'r') as f:
            image = np.array(f['entry/data/data'])
        with h5py.File(self.label_paths[idx], 'r') as f:
            label = np.array(f['entry/data/data'])
                
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image, label
  