import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.densenet import DenseNet121_Weights
import os
import torch.nn.functional as F

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
    
class Multi_Class_CNN2(nn.Module):
    def __init__(self, input_channels=1, input_size=(2163, 2069), output_channels=3):
        super(Multi_Class_CNN2, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=10, stride=5, padding=1)
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
    
class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k, self.heads = k, heads

        self.tokeys    = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues  = nn.Linear(k, k * heads, bias=False)

        self.unifyheads = nn.Linear(heads * k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.toqueries(x).view(b, t, h, k)
        keys    = self.tokeys(x).view(b, t, h, k)
        values  = self.tovalues(x).view(b, t, h, k)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        queries = queries / (k ** (1/4))
        keys    = keys / (k ** (1/4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, k)

        out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return self.unifyheads(out)

class Multi_Class_CNN3(nn.Module):
    def __init__(self, input_channels=1, input_size=(2163, 2069), output_channels=3):
        super(Multi_Class_CNN3, self).__init__()

        # Initial Convolutional Layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=25, stride=5, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(0.25)

        # Adding more convolutional layers
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        # Use a dummy input to pass through the conv layers to determine output size
        dummy_input = torch.autograd.Variable(torch.zeros(1, input_channels, *input_size))
        output_size = self._get_conv_output(dummy_input)

        # Fully Connected Layers
        self.fc1 = nn.Linear(output_size, 512)
        self.fc2 = nn.Linear(512, output_channels)
        
    def _get_conv_output(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        n_size = x.data.view(1, -1).size(1)
        return n_size
    
    def forward(self, x):
        x = self.dropout1(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout1(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout2(self.pool2(F.relu(self.bn3(self.conv3(x)))))
        x = self.dropout2(self.pool2(F.relu(self.bn4(self.conv4(x)))))
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




class MultiClassCNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, input_size=(2163, 2069)):
        super(MultiClassCNN, self).__init__()
        
        # Parameters for the convolutional layer
        self.kernel_size = 10
        self.stride = 1
        self.padding = 1

        # Define a single convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        
        # Function to calculate output dimensions dynamically
        self.output_height = self.calculate_output_dimension(input_size[0], self.kernel_size, self.stride, self.padding)
        self.output_width = self.calculate_output_dimension(input_size[1], self.kernel_size, self.stride, self.padding)
        
        # Flatten the output dimensions for the fully connected layer
        self.fc_size = 8 * self.output_height * self.output_width  # Dynamic number of features
        
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
