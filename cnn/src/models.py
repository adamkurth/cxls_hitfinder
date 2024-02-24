import os
import torch 
import numpy as np
import h5py as h5
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models.resnet import ResNet50_Weights
from sklearn.model_selection import train_test_split
from collections import namedtuple

# MODEL: 
#   1. RESNET -> RESNET + + Attention Mechanisms 
#   2. TRANSFORMER BASED MODEL FOR VISION (VIT) -> VIT + Attention Mechanisms ?? 
# FURTHER DEVELOPMENT: multi-task learning model predicts clen and pdb at the same time

# to detect the subleties in the data, the deeper the model like ResNet-50 or -101 would be ideal. 
# 50 and 101 models balance between depth and computational efficiency, and are widely used in practice.

# RESNET-50
class CustomResNet50(nn.Module):
    def __init__(self, num_proteins=3, num_camlengths=3):
        # num_classes = num_proteins, num_camlengths
        super(CustomResNet50, self).__init__()
        # load the pre-trained model
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        # replace the final fully connected layer
        self.resnet.fc = nn.Linear(num_ftrs, num_proteins + num_camlengths) # proteins + camlengths
        
    def forward(self, x):
        return self.resnet(x)