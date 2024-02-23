import os
import torch 
import numpy as np
import h5py as h5
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from collections import namedtuple
from dataprep import PeakImageDataset, PathManager, DataPreparation, sim_parameters

# MODEL: 
#   1. RESNET -> RESNET + + Attention Mechanisms 
#   2. TRANSFORMER BASED MODEL FOR VISION (VIT) -> VIT + Attention Mechanisms ?? 
# FURTHER DEVELOPMENT: multi-task learning model predicts clen and pdb at the same time

# to detect the subleties in the data, the deeper the model like ResNet-50 or -101 would be ideal. 
# 50 and 101 models balance between depth and computational efficiency, and are widely used in practice.

# RESNET-50
class CustomResNet50(nn.Module):
    def __init__(self, num_proteins=3, num_camlengths=3):
        super(CustomResNet50, self).__init__()
        # load the pre-trained model
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        # replace the final fully connected layer
        self.resnet.fc = nn.Linear(num_ftrs, num_proteins + num_camlengths) # proteins + camlengths
        
    def forward(self, x):
        return self.resnet(x)

# instances
paths = PathManager()
data_preparation = DataPreparation(paths, batch_size=32)
paths.clean_sim() # moves all .err, .out, .sh files sim_specs 

# train and test data
train_loader, test_loader = data_preparation.prep_data()

# model
model = CustomResNet50(num_classes=10, num_lengths=3)
print(model)

# loss and optimizer
criteron = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train 
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criteron(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    


        
        



