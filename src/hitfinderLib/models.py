import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models



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
        self.fc_size_2 = (out_height2 * out_width2) // 23782
        
        self.fc1 = nn.Linear(self.fc_size_1, self.fc_size_2)
        self.fc2 = nn.Linear(self.fc_size_2 + 2, output_channels)

    def calculate_output_dimension(self, input_dim, kernel_size, stride, padding):
        return ((input_dim + 2 * padding - kernel_size) // stride) + 1

    def forward(self, x, camera_length, photon_energy):
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        x = F.relu(self.gn2(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        
        device = x.device
        camera_length = camera_length.to(device).float()
        photon_energy = photon_energy.to(device).float()
        
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
    
class Binary_Classification_SA_CA_Meta_Data(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, input_size=(2163, 2069)):
        super(Binary_Classification_SA_CA_Meta_Data, self).__init__()
        
        self.kernel_size1 = 10
        self.conv1_channels = 4
        self.stride1 = 1
        self.padding1 = 1
        self.kernel_size2 = 3
        self.conv2_channels = 8
        self.stride2 = 1
        self.padding2 = 1
        num_groups1 = 4  
        num_groups2 = 4  

        self.conv1 = nn.Conv2d(input_channels, self.conv1_channels, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1)
        self.gn1 = nn.GroupNorm(num_groups=num_groups1, num_channels=self.conv1_channels)
        self.pool1 = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(self.conv1_channels, self.conv2_channels, kernel_size=self.kernel_size2, stride=self.stride2, padding=self.padding2)
        self.gn2 = nn.GroupNorm(num_groups=num_groups2, num_channels=self.conv2_channels)

        out_height1 = self.calculate_output_dimension(input_size[0], self.kernel_size1, self.stride1, self.padding1)
        out_width1 = self.calculate_output_dimension(input_size[1], self.kernel_size1, self.stride1, self.padding1)
        out_height2 = self.calculate_output_dimension(out_height1 // 2, self.kernel_size2, self.stride2, self.padding2) 
        out_width2 = self.calculate_output_dimension(out_width1 // 2, self.kernel_size2, self.stride2, self.padding2)
        
        # self.fc_size_1 = 16 * out_height2 * out_width2
        self.fc_size_1 = 8790400
        # self.fc_size_2 = (out_height2 * out_width2) // 23782
        self.fc_size_2 = 256
        
        self.fc1 = nn.Linear(self.fc_size_1, self.fc_size_2)
        self.fc2 = nn.Linear(self.fc_size_2 + 2, output_channels)
        
        self.ca = ChannelAttention(self.conv1_channels)
        self.sa = SpatialAttention(self.conv2_channels)

    def calculate_output_dimension(self, input_dim, kernel_size, stride, padding):
        return ((input_dim + 2 * padding - kernel_size) // stride) + 1

    def forward(self, x, camera_length, photon_energy):
        print(f'Input shape: {x.shape}')
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        print(f'After conv1, gn1, pool1: {x.shape}')
        x = self.ca(x)
        print(f'After ChannelAttention: {x.shape}')
        x = F.relu(self.gn2(self.conv2(x)))
        print(f'After conv2, gn2: {x.shape}')
        x = self.sa(x)
        print(f'After SpatialAttention: {x.shape}')
        x = x.view(x.size(0), -1)
        print(f'After view (flatten): {x.shape}')
        x = F.relu(self.fc1(x))
        print(f'After fc1: {x.shape}')
        params = torch.stack((camera_length, photon_energy), dim=1)
        print(f'Params shape: {params.shape}')
        x = torch.cat((x, params), dim=1)
        print(f'After concatenation: {x.shape}')
        x = self.fc2(x)
        print(f'After fc2: {x.shape}')
        return x


class Binary_Classification_DenseNet(nn.Module):
    def __init__(self):
        super(Binary_Classification_DenseNet, self).__init__()
        self.densenet = torchvision.models.densenet121(pretrained=True)
        
        # Modify the first convolutional layer to accept 1-channel input instead of 3
        self.densenet.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()  # Remove the original classifier
        
        self.fc1 = nn.Linear(num_features + 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, camera_length, photon_energy):
        # Extract features from the DenseNet
        features = self.densenet.features(x)
        x = F.relu(features, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(features.size(0), -1)
        
        # Concatenate additional parameters
        params = torch.stack((camera_length, photon_energy), dim=1)
        x = torch.cat((x, params), dim=1)
        
        # Pass through the modified classifier
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x