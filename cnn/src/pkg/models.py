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

def get_models():
    """
    Returns a dictionary of available models.
    """
    models = {
        "ResNet50BraggPeakClassifier": ResNet50BraggPeakClassifier,
        "BasicCNN1": BasicCNN1,
        "BasicCNN2": BasicCNN2,
        "DenseNetBraggPeakClassifier": DenseNetBraggPeakClassifier,
        "BasicCNN3": BasicCNN3,
        "Multi_Class_CNN1": Multi_Class_CNN1
    }
    return models

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
    def __init__(self, input_channels=1, output_channels = 1, input_size=(2163, 2069)):
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
        self.fc = nn.Linear(self.fc_input_size, output_channels)

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


class Multi_Class_CNN1(nn.Module):
    def __init__(self, input_channels=1, input_size=(2163, 2069), output_channels=3):
        super(Multi_Class_CNN1, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=25, stride=5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Use a dummy input to pass through the conv layers to determine output size
        dummy_input = torch.autograd.Variable(torch.zeros(1, input_channels, *input_size))
        output_size = self._get_conv_output(dummy_input)
        
        self.fc1 = nn.Linear(output_size, 512)
        self.fc2 = nn.Linear(512, output_channels)
        
    def _get_conv_output(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        n_size = x.data.view(1, -1).size(1)
        return n_size
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # Excluding the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class ComparisonCNN(nn.Module):
    def __init__(self, input_channels=1, input_size=(2163, 2069), output_channels=3):
        super(ComparisonCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=25, stride=5, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)  
        
        dummy_input = torch.autograd.Variable(torch.zeros(1, input_channels, *input_size))
        output_size = self._get_conv_output(dummy_input)
        
        self.fc1 = nn.Linear(output_size, 512)
        self.fc2 = nn.Linear(512, output_channels)
        
    def _get_conv_output(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01))
        n_size = x.data.view(1, -1).size(1)
        return n_size
    
    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01))
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

class BaseCNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, input_size=(2163, 2069)):
        super(BaseCNN, self).__init__()
        
        # Parameters for the convolutional layer
        self.kernel_size = 25
        self.stride = 10
        self.padding = 1
        self.conv_output = 2

        # Define a single convolutional layer
        self.conv1 = nn.Conv2d(input_channels, self.conv_output, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        
        # Function to calculate output dimensions dynamically
        self.output_height = self.calculate_output_dimension(input_size[0], self.kernel_size, self.stride, self.padding)
        self.output_width = self.calculate_output_dimension(input_size[1], self.kernel_size, self.stride, self.padding)
        
        # Flatten the output dimensions for the fully connected layer
        self.fc_size = self.conv_output * self.output_height * self.output_width  # Dynamic number of features
        
        # Define a fully connected layer that maps to the output channels
        self.fc = nn.Linear(self.fc_size, output_channels)

    def calculate_output_dimension(self, input_dim, kernel_size, stride, padding):
        return ((input_dim + 2 * padding - kernel_size) // stride) + 1
    
    def forward(self, x):
        # Apply the convolutional layer followed by a ReLU activation function
        x = F.relu(self.conv1(x))
        
        # Flatten the output for the fully connected layer
        x = x.view(-1, self.fc_size)  # Dynamically flatten the tensor
        
        # Apply the fully connected layer
        x = self.fc(x)
        
        return x


    

class ResNetBinaryClassifier(nn.Module):
    def __init__(self, input_channels=1, input_size=(2163, 2069)):
        super(ResNetBinaryClassifier, self).__init__()

        # Initialize the ResNet model
        base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)  
        # Adjust ResNet to take input_channels other than 3 (RGB)
        self.first_conv_layer = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace the first conv layer of ResNet
        base_model.conv1 = self.first_conv_layer
        # Utilize the ResNet architecture but exclude the final fully connected layer
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        # Calculate the size of the flattened features from ResNet
        with torch.no_grad():
            self.feature_size = self._get_resnet_feature_size(input_channels, input_size)
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, 1)  # Output layer for binary classification
        self.dropout = nn.Dropout(0.5)

    def _get_resnet_feature_size(self, input_channels, input_size):
        dummy_input = torch.zeros(1, input_channels, *input_size)
        dummy_features = self.feature_extractor(dummy_input)
        return int(np.prod(dummy_features.size()))

    def forward(self, x):
        # Extract features
        x = self.feature_extractor(x)
        # Flatten the features
        x = x.view(-1, self.feature_size)
        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    
    
class Binary_Classification(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, input_size=(2163, 2069)):
        super(Binary_Classification, self).__init__()
        
        # Parameters for the convolutional layers
        self.kernel_size1 = 10
        self.stride1 = 1
        self.padding1 = 1
        self.kernel_size2 = 3  # Smaller kernel for capturing finer details
        self.stride2 = 1
        self.padding2 = 1

        # Number of groups for Group Normalization
        num_groups1 = 4  # For the first convolutional layer
        num_groups2 = 4  # For the second convolutional layer

        # Define the convolutional layers and group normalization
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1)
        self.gn1 = nn.GroupNorm(num_groups=num_groups1, num_channels=8)
        self.pool1 = nn.MaxPool2d(2, 2)  # Adding pooling to reduce dimensionality
        self.conv2 = nn.Conv2d(8, 16, kernel_size=self.kernel_size2, stride=self.stride2, padding=self.padding2)
        self.gn2 = nn.GroupNorm(num_groups=num_groups2, num_channels=16)

        # Dynamically calculate output dimensions after each layer
        out_height1 = self.calculate_output_dimension(input_size[0], self.kernel_size1, self.stride1, self.padding1)
        out_width1 = self.calculate_output_dimension(input_size[1], self.kernel_size1, self.stride1, self.padding1)
        out_height2 = self.calculate_output_dimension(out_height1 // 2, self.kernel_size2, self.stride2, self.padding2)  # Pooling reduces size
        out_width2 = self.calculate_output_dimension(out_width1 // 2, self.kernel_size2, self.stride2, self.padding2)
        
        # Flatten the output dimensions for the fully connected layer
        self.fc_size = 16 * out_height2 * out_width2
        self.fc = nn.Linear(self.fc_size, output_channels)

    def calculate_output_dimension(self, input_dim, kernel_size, stride, padding):
        return ((input_dim + 2 * padding - kernel_size) // stride) + 1

    def forward(self, x):
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        x = F.relu(self.gn2(self.conv2(x)))
        x = x.view(-1, self.fc_size)  # Dynamically flatten the tensor based on computed output sizes
        x = self.fc(x)
        return x
    
    

class DualInputCNN(nn.Module):
    def __init__(self, input_height=2163, input_width=2069):
        super(DualInputCNN, self).__init__()
        
        # Increased kernel size to 8 and adjusted padding
        kernel_size = 5
        stride = 2
        padding = 1  
        linear = 128
        
        # Define the first branch for noisy images
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Define the second branch for clean images
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Helper function to calculate output dimensions
        def calc_output_dim(input_dim, kernel_size, stride, padding):
            return (input_dim + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        
        # Calculate output dimensions after convolutions and pooling
        output_height = calc_output_dim(input_height, kernel_size, stride, padding)
        output_height = calc_output_dim(output_height, 2, 2, 0)  # Pooling layer
        output_width = calc_output_dim(input_width, kernel_size, stride, padding)
        output_width = calc_output_dim(output_width, 2, 2, 0)  # Pooling layer

        # Calculate the number of features for the fully connected layer
        num_features_per_branch = output_height * output_width * 32
        total_features = num_features_per_branch * 2  # Two branches

        # Combine features from both branches
        self.classifier = nn.Sequential(
            nn.Linear(total_features, linear),
            nn.ReLU(),
            nn.Linear(linear, 1),
        )
    
    def forward(self, noisy_img, clean_img):
        # Process each image through its respective branch
        features1 = self.branch1(noisy_img)
        features2 = self.branch2(clean_img)

        # Flatten the output from each branch and concatenate along dimension 1 (feature dimension)
        combined_features = torch.cat((features1.view(features1.size(0), -1), 
                                       features2.view(features2.size(0), -1)), dim=1)

        # Pass the combined features through the classifier to get the final output
        output = self.classifier(combined_features)
        return output
    
    
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


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
        # x = self.dropout(x)
        x = self.ca(x)
        x = self.sa(x)
        x = self.heatmap_conv(x)
        x = self.upsample(x)
        return x

class DnCNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, num_layers=17, features=64):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(input_channels, features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers-2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.GroupNorm(1, features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(features, output_channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self.threshold = 0.5  # Set a threshold for binarization

    def forward(self, x):
        x = self.dncnn(x)
        return x
