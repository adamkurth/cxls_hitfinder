import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.densenet import DenseNet121_Weights
import os
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




# if __name__ == "__main__":
#     model = ResNet50BraggPeakClassifier()
#     img_np = np.random.rand(2163, 2069)
#     # add function in classes to handle this
#     batch_size = 4 
#     img_np_new = np.expand_dims(img_np, axis=0) # channel dim
#     img_np_new = np.expand_dims(img_np_new, axis=0) # batch dim
#     img_np_new = np.repeat(img_np_new, batch_size, axis=0) # 4D
    
#     # convert to tensor
#     img_tensor = torch.tensor(img_np_new, dtype=torch.float32)
    
#     output = model(img_tensor)
#     print(output.size()) # torch.Size([4, 1, 2163, 2069]) -> 4 images, 1 channel, 2163x2069
    
    
class BasicCNN1(nn.Module):
    """
    A very basic CNN for detecting Bragg peaks in crystallography images.
    This model simplifies the architecture to its core components.
    """
    def __init__(self, input_channels=1, output_channels=1, heatmap_size=(2163, 2069)):
        super(BasicCNN1, self).__init__()
        
        # Define the heatmap size for upsampling
        self.heatmap_size = heatmap_size

        # Convolutional layer followed by pooling
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

        # Convolutional layer to generate heatmap with one channel
        self.heatmap_conv = nn.Conv2d(16, output_channels, kernel_size=3, stride=1, padding=1)

        # Upsampling layer to match the desired output size
        self.upsample = nn.Upsample(size=heatmap_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Applying a convolution, activation function, and pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Generating heatmap
        x = self.heatmap_conv(x)

        # Upsampling to match the input image size
        x = self.upsample(x)

        return x


class BasicCNN2(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, heatmap_size=(2163, 2069)):
        super(BasicCNN2, self).__init__()
        
        self.heatmap_size = heatmap_size
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.flattened_size = 256 * (heatmap_size[0] // 16) * (heatmap_size[1] // 16)  

        # Linear layers - you may adjust the sizes and number of linear layers
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, self.flattened_size)  

        # Convolutional layer for heatmap generation
        self.heatmap_conv = nn.Conv2d(256, output_channels, kernel_size=3, stride=1, padding=1)

        # Upsampling layer
        self.upsample = nn.Upsample(size=heatmap_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten the output for the linear layer
        x = x.view(-1, self.flattened_size)

        # Pass through linear layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Reshape output back to match the convolutional layer expected input
        x = x.view(-1, 256, self.heatmap_size[0] // 16, self.heatmap_size[1] // 16)

        # Generating heatmap
        x = self.heatmap_conv(x)

        # Upsampling to match the input image size
        x = self.upsample(x)

        return x

    
    
    
class DenseNetBraggPeakClassifier(nn.Module):
    """
    Model for detecting Bragg peaks in crystallography images using DenseNet.
    This model is adapted to focus on the peak detection task with input images being
    grayscale and outputs being heatmaps for Bragg peak locations.
    """
    def __init__(self, input_channels=1, output_channels=1, heatmap_size=(2163, 2069)):
        super(DenseNetBraggPeakClassifier, self).__init__()
        # Load a pre-trained DenseNet model, modifying it for 1-channel grayscale input
        # and to output a heatmap for Bragg peak locations.
        self.densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        
        # Adjust the first convolutional layer for 1-channel grayscale images
        self.densenet.features.conv0 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Use adaptive average pooling to produce a fixed size output while maintaining spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((heatmap_size[0] // 32, heatmap_size[1] // 32))
        
        # Remove DenseNet's classification layer (fully connected layer)
        self.densenet.classifier = nn.Identity()
        
        # Extra convolutional layer to transform the feature maps into a single-channel heatmap
        self.heatmap_conv = nn.Conv2d(1024, output_channels, kernel_size=1)  # 1024 for DenseNet121
        
        # Upsampling layer to resize the heatmap to the desired output size
        self.upsample = nn.Upsample(size=heatmap_size, mode='bilinear', align_corners=True)
            
    def forward(self, x):
        x = self.densenet.features(x)
        x = self.adaptive_pool(x)  # Reduce spatial size with adaptive pooling
        x = self.heatmap_conv(x)  # Convolution to generate heatmap
        x = self.upsample(x)  # Upsample to match desired heatmap size
        return x

class BasicCNN3(nn.Module):
    def __init__(self, input_channels=1, input_size=(2163, 2069)):
        super(BasicCNN3, self).__init__()
        
        # Convolutional layer followed by pooling
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        
        # Additional convolutional layers
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the flattened feature map.
        # This is an example calculation that needs to be adjusted
        # based on the actual size of your input images and the architecture's downsampling operations.
        reduced_size = input_size[0] // (4**3), input_size[1] // (4**3)  # Assuming three pooling layers each reducing size by a factor of 4
        self.fc_input_size = 64 * reduced_size[0] * reduced_size[1]  # Adjust this based on your network's architecture
        
        # Fully connected layer to produce a single output
        self.fc = nn.Linear(self.fc_input_size, 1)

    def forward(self, x):
        # Applying convolutions and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flattening the output for the fully connected layer
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layer to get to a single output value
        x = self.fc(x)
        
        return x


