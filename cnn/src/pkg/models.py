import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.densenet import DenseNet121_Weights
import os
import torch.nn.functional as F
from scipy.signal import find_peaks

class Binary_Classification(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, input_size=(2163, 2069)):
        super(Binary_Classification, self).__init__()
  
        self.kernel_size1 = 10
        self.stride1 = 1
        self.padding1 = 1
        self.kernel_size2 = 3  
        self.stride2 = 1
        self.padding2 = 1
        num_groups1 = 4  
        num_groups2 = 4  

        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1)
        self.gn1 = nn.GroupNorm(num_groups=num_groups1, num_channels=8)
        self.pool1 = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(8, 16, kernel_size=self.kernel_size2, stride=self.stride2, padding=self.padding2)
        self.gn2 = nn.GroupNorm(num_groups=num_groups2, num_channels=16)

        out_height1 = self.calculate_output_dimension(input_size[0], self.kernel_size1, self.stride1, self.padding1)
        out_width1 = self.calculate_output_dimension(input_size[1], self.kernel_size1, self.stride1, self.padding1)
        out_height2 = self.calculate_output_dimension(out_height1 // 2, self.kernel_size2, self.stride2, self.padding2)  # Pooling reduces size
        out_width2 = self.calculate_output_dimension(out_width1 // 2, self.kernel_size2, self.stride2, self.padding2)
        
        self.fc_size = 16 * out_height2 * out_width2
        self.fc = nn.Linear(self.fc_size, output_channels)

    def calculate_output_dimension(self, input_dim, kernel_size, stride, padding):
        return ((input_dim + 2 * padding - kernel_size) // stride) + 1

    def forward(self, x):
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        x = F.relu(self.gn2(self.conv2(x)))
        x = x.view(-1, self.fc_size) 
        x = self.fc(x)
        return x
    
    
    
class Binary_Classification_With_Parameters(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, input_size=(2163, 2069)):
        super(Binary_Classification_With_Parameters, self).__init__()
        
        self.kernel_size1 = 10
        self.stride1 = 1
        self.padding1 = 1
        self.kernel_size2 = 3 
        self.stride2 = 1
        self.padding2 = 1
        num_groups1 = 4  
        num_groups2 = 4  

        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1)
        self.gn1 = nn.GroupNorm(num_groups=num_groups1, num_channels=8)
        self.pool1 = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(8, 16, kernel_size=self.kernel_size2, stride=self.stride2, padding=self.padding2)
        self.gn2 = nn.GroupNorm(num_groups=num_groups2, num_channels=16)

        out_height1 = self.calculate_output_dimension(input_size[0], self.kernel_size1, self.stride1, self.padding1)
        out_width1 = self.calculate_output_dimension(input_size[1], self.kernel_size1, self.stride1, self.padding1)
        out_height2 = self.calculate_output_dimension(out_height1 // 2, self.kernel_size2, self.stride2, self.padding2) 
        out_width2 = self.calculate_output_dimension(out_width1 // 2, self.kernel_size2, self.stride2, self.padding2)
        
        self.fc_size_1 = 16 * out_height2 * out_width2
        self.fc_size_2 = (out_height2 * out_width2) // 4418
        
        self.fc1 = nn.Linear(self.fc_size_1, self.fc_size_2)
        self.fc2 = nn.Linear(self.fc_size_2 + 2, output_channels)

    def calculate_output_dimension(self, input_dim, kernel_size, stride, padding):
        return ((input_dim + 2 * padding - kernel_size) // stride) + 1

    def forward(self, x, camera_length, photon_energy):
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        x = F.relu(self.gn2(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        params = torch.stack((camera_length, photon_energy), dim=1)
        x = torch.cat((x, params), dim=1)
        x = self.fc2(x)
        return x


class Linear(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, input_size=(2163, 2069)):
        super(Linear, self).__init__()

        self.fc_size = input_size[0] * input_size[1]
        self.fc = nn.Linear(self.fc_size, 3)
    
    def forward(self, x):

        x = x.view(-1, self.fc_size)  
        x = self.fc(x)
        
        return x

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Separate pathways for avg and max pooling
        self.fc_avg = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )
        
        self.fc_max = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc_avg(self.avg_pool(x).view(b, c))
        max_out = self.fc_max(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.view(b, c, 1, 1)
    
class SpatialAttention(nn.Module):
    def __init__(self, num_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, num_channels, kernel_size=7, padding=3, dilation=2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_out, avg_out], dim=1)
        x = self.sigmoid(self.conv1(x))
        return x * x


class HeatmapCNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, heatmap_size=(2163, 2069)):
        super(HeatmapCNN, self).__init__()
        
        self.heatmap_size = heatmap_size
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.ca = ChannelAttention(32)
        self.sa = SpatialAttention(32)  # Assuming this is defined elsewhere
        self.heatmap_conv = nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(size=heatmap_size, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.ca(x)
        x = self.sa(x)
        x = self.heatmap_conv(x)
        x = self.upsample(x)
        return x