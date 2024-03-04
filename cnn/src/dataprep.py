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

from pkg import c, m, f

# this python file is for testing the classes and functions in the classes.py and functions.py files
# supposed to load the data in preparation for the CNN model

def main():
    # instances
    paths = c.PathManager()
    dataset = c.PeakImageDataset(paths, transform=transforms.ToTensor(), augment=True)

    print(type(dataset), dataset)
    num_items = len(dataset)
    print(f"Number of items: {num_items}")

    data_preparation = c.DataPreparation(paths, batch_size=10)
    paths.clean_sim() # moves all .err, .out, .sh files sim_specs

    # train and test data loaders
    train_loader, test_loader = data_preparation.prep_data()

    sim_dict = f.sim_parameters(paths)
    print(sim_dict)



if __name__ == "__main__":
    main()
