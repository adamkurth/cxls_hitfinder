from importlib import reload
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import subprocess
from glob import glob
from pkg import *
from pkg.functions import convert2int, convert2str, get_params
from typing import List, Dict, Any
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision.transforms import v2
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from skimage.filters import gaussian, sobel
from matplotlib.colors import SymLogNorm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

datasets = [1,4,5,6,7,8]

myPaths = path.PathManager(datasets=datasets)
myProcessor = process.Processor(paths=myPaths, datasets=datasets)
params = myProcessor.get_parameters()
f.get_counts(paths=myPaths, datasets=datasets)

f.check_attributes(paths=myPaths, datasets=f.convert2str(datasets), dir_type='peak')

transform = None
myDataManager = data.DatasetManager(paths=myPaths, datasets=datasets, transform=transform)
train_loader, test_loader = f.prepare(data_manager=myDataManager, batch_size=20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scheduler = ReduceLROnPlateau

cfg = {
    "loader": [train_loader, test_loader],
    'batch_size': train_loader.batch_size,
    'device': device,
    'scheduler': scheduler
    }


peak_config = class_configs.Peak_Detection_Configuration(myPaths, datasets, device, save_path='../models/peak_model.pt')
print(f'weights for peak : {peak_config.get_loss_weights()}')

a = train.TrainTestModels(cfg, peak_config)
a.epoch_loop()
a.plot_loss_accuracy('../models/loss_accuracy_plot.png')
a.evaluate_model()
a.plot_confusion_matrix('../models/confusion_matrix.png')
a.make_classification_report()
a.save_model()