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
#   1. RESNET -> RESNET + Attention Mechanisms
#   2. TRANSFORMER BASED MODEL FOR VISION (VIT) -> VIT + Attention Mechanisms ??
# FURTHER DEVELOPMENT: multi-task learning model predicts clen and pdb at the same time

# to detect the subleties in the data, the deeper the model like ResNet-50 or -101 would be ideal.
# 50 and 101 models balance between depth and computational efficiency, and are widely used in practice.

# RESNET-50
class CustomResNet50(nn.Module):
    # single channel ResNet-50
    def __init__(self, num_proteins=3, num_camlengths=3):
        # num_classes = num_proteins, num_camlengths
        super(CustomResNet50, self).__init__()
        # Assuming both peak and water images are grayscale, initialize two identical ResNet backbones
        self.peak_resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.water_resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Adjust the first convolutional layer of each ResNet to accept 1-channel input
        self.peak_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.water_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Assuming both backbones have the same feature size at the end
        num_ftrs = self.peak_resnet.fc.in_features

        # Replace the final layer with a combined one for each task
        self.fc_proteins = nn.Linear(num_ftrs * 2, num_proteins)  # *2 for concatenated features
        self.fc_camlengths = nn.Linear(num_ftrs * 2, num_camlengths)

    def forward(self, x):
        peak_images, water_images = x

        # Forward pass through each backbone
        peak_features = self.forward_backbone(self.peak_resnet, peak_images)
        water_features = self.forward_backbone(self.water_resnet, water_images)

        # Concatenate features
        combined_features = torch.cat((peak_features, water_features), dim=1)

        # Predictions for each task
        protein_preds = self.fc_proteins(combined_features)
        camlength_preds = self.fc_camlengths(combined_features)

        return protein_preds, camlength_preds

    def forward_backbone(self, backbone, x):
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)

        x = backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return x
