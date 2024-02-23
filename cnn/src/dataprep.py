import os
import numpy as np
import h5py
import torch
import shutil
import re
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import namedtuple
import classes as cl
import functions as fn

# this python file is for testing the classes and functions in the classes.py and functions.py files
# supposed to load the data in preparation for the CNN model  

def main():
    # instances
    paths = cl.PathManager()
    dataset = cl.PeakImageDataset(paths, transform=transforms.ToTensor(), augment=False)
    
    print(type(dataset), dataset)
    num_items = len(dataset)
    
    data_preparation = cl.DataPreparation(paths, batch_size=1)
    paths.clean_sim() # moves all .err, .out, .sh files sim_specs 
    train_loader, test_loader = data_preparation.prep_data()

    sim_dict = fn.sim_parameters(paths)
    print(sim_dict)

if __name__ == "__main__":
    main()