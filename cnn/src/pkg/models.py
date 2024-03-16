import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
import torch.nn.functional as F

# MODEL:
#   1. RESNET -> RESNET + Attention Mechanisms
#   2. TRANSFORMER BASED MODEL FOR VISION (VIT) -> VIT + Attention Mechanisms ??
# FURTHER DEVELOPMENT: multi-task learning model predicts clen and pdb at the same time

# to detect the subleties in the data, the deeper the model like ResNet-50 or -101 would be ideal.
# 50 and 101 models balance between depth and computational efficiency, and are widely used in practice.

# RESNET-50

class ResNet50BraggPeakClassifier(nn.Module):
    """
    Simplified model for detecting Bragg peaks in crystallography images using ResNet.
    This model focuses solely on the peak detection task.
    """
    def __init__(self, input_channels=1, output_channels=1, heatmap_size=(2163, 2069)):
        super(ResNet50BraggPeakClassifier, self).__init__()
        # use pretrained resnet50, but modify this to work with grayscale images (1 channel)
        # and output the haetmap for Bragg peak locations
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Adjust the first convolutional layer for 1-channel grayscale images
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Use adaptive average pooling to reduce to a fixed size output while maintaining spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((heatmap_size[0] // 32, heatmap_size[1] // 32))
        
        # Remove fully connected layers
        self.resnet.fc = nn.Identity()
        
        # Extra convolutional layer to generate heatmap with one channel
        self.heatmap_conv = nn.Conv2d(2048, output_channels, kernel_size=1)
        
        # Upsampling layer to match the desired output size
        self.upsample = nn.Upsample(size=heatmap_size, mode='bilinear', align_corners=True)
            
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.adaptive_pool(x)  # Adaptive Pooling to reduce spatial size
        x = self.heatmap_conv(x)  # Convolution to get heatmap
        x = self.upsample(x)  # Upsample to original image size

        return x




if __name__ == "__main__":
    model = ResNet50BraggPeakClassifier()
    img_np = np.random.rand(2163, 2069)
    # add function in classes to handle this
    batch_size = 4 
    img_np_new = np.expand_dims(img_np, axis=0) # channel dim
    img_np_new = np.expand_dims(img_np_new, axis=0) # batch dim
    img_np_new = np.repeat(img_np_new, batch_size, axis=0) # 4D
    
    # convert to tensor
    img_tensor = torch.tensor(img_np_new, dtype=torch.float32)
    
    output = model(img_tensor)
    print(output.size()) # torch.Size([4, 1, 2163, 2069]) -> 4 images, 1 channel, 2163x2069

    
    
    
    