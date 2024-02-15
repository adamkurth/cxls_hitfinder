import os
import glob
import h5py as h5
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter
from label_finder import(
    load_data, 
    load_file_h5,
    display_peak_regions, 
    validate,
    is_peak,
    view_neighborhood, 
    generate_labeled_image, 
    main,
    )                     

class PeakThresholdProcessor:
    def __init__(self, image_tensor, threshold_value=0):
        self.image_tensor = image_tensor
        self.threshold_value = threshold_value

    def set_threshold_value(self, new_threshold_value):
        self.threshold_value = new_threshold_value

    def get_coordinates_above_threshold(self):
        # convert to boolean mask
        mask = self.image_tensor > self.threshold_value
        # indices of True values in the mask
        coordinates = torch.nonzero(mask).cpu().numpy()
        return coordinates

    def get_local_maxima(self):
        # relies on 'find_peaks' which works on 1D arrays.
        image_1d = self.image_tensor.flatten().cpu().numpy()  # to numpy for compatibility with 'find_peaks'
        peaks, _ = find_peaks(image_1d, height=self.threshold_value)
        coordinates = [self.flat_to_2d(idx) for idx in peaks]
        return coordinates

    def flat_to_2d(self, index):
        rows, cols = self.image_tensor.shape
        return (index // cols, index % cols)

class ArrayRegion:
    def __init__(self, tensor):
        self.tensor = tensor
        self.x_center = 0
        self.y_center = 0
        self.region_size = 0

    def set_peak_coordinate(self, x, y):
        self.x_center = x
        self.y_center = y

    def set_region_size(self, size):
        max_printable_region = min(self.tensor.shape[0], self.tensor.shape[1]) // 2
        self.region_size = min(size, max_printable_region)

    def get_region(self):
        x_range = slice(self.x_center - self.region_size, self.x_center + self.region_size + 1)
        y_range = slice(self.y_center - self.region_size, self.y_center + self.region_size + 1)
        region = self.tensor[x_range, y_range] # tensor slicing
        return region

    def extract_region(self, x_center, y_center, region_size):
        self.set_peak_coordinate(x_center, y_center)
        self.set_region_size(region_size)
        return self.get_region()
    
class CCN(nn.Module):
    # CNN using pytorch
    def __init__(self, num_channels, img_height, img_width):
        super(CCN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1) # 32 neurons
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 64 neurons
        self.dropout = nn.Dropout(0.5)
        self.flattened_size = 64 * (img_height // 4) * (img_width // 4) # 64 neurons
        self.fc1 = nn.Linear(self.flattened_size, 128) # 128 neurons
        self.fc2 = nn.Linear(128, 2) # 2 for binary classification
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = nn.MaxPool2d(2)(x) # 2x2 pooling
        x = torch.relu(self.conv2(x)) # 64 neurons
        x = nn.MaxPool2d(2)(x) # 2x2 pooling
        x = x.view(x.size(0), -1) # flatten
        x = self.dropout(x) # regularization
        x = torch.relu(self.fc1(x)) # 128 neurons
        x = self.fc2(x) # 2 for binary classification
        return x
    
def is_peak(image_data, coordinate, neighborhood_size=3):
    x, y = coordinate
    region = ArrayRegion(image_data)
    
    neighborhood = region.extract_region(x, y, neighborhood_size)
    if torch.numel(neighborhood) == 0:  # neighborhood is empty
        return False
    
    center = neighborhood_size, neighborhood_size
    is_peak = neighborhood[center] == torch.max(neighborhood)
    return is_peak

def generate_labeled_image(image_data, peak_coordinates, neighborhood_size):
    labeled_image = torch.zeros_like(image_data)
    for (x, y) in peak_coordinates:
        if is_peak(image_data, (x, y), neighborhood_size):
            labeled_image[x, y] = 1  # label as peak
    print('Generated labeled image.')
    return labeled_image

def preprocess(image_data, labeled_image):
    # Convert numpy arrays to PyTorch tensors if needed
    if isinstance(image_data, np.ndarray):
        image_data = torch.from_numpy(image_data).float()
    if isinstance(labeled_image, np.ndarray):
        labeled_image = torch.from_numpy(labeled_image).float()

    # Add channel and batch dimensions if they're missing
    if len(image_data.shape) == 2:
        image_data = image_data.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

    # Move channel to the second dimension if it's in the last
    if len(image_data.shape) == 3:
        image_data = image_data.permute(2, 0, 1).unsqueeze(0)  # shape: [1, C, H, W]

    # If the data is already 4D but in the wrong order, fix it
    if len(image_data.shape) == 4 and image_data.shape[1] > image_data.shape[3]:
        image_data = image_data.permute(0, 3, 1, 2)  # shape: [B, C, H, W]

    return image_data, labeled_image


def data_preparation(image_data, labeled_data):
    # Preprocess the data to ensure it's in the right format
    image_tensor, labeled_tensor = preprocess(image_data, labeled_data)
    
    # Ensure the labeled_tensor is a 1D tensor of long type labels
    labeled_tensor = labeled_tensor.view(-1)[:image_tensor.shape[0]].long()
    
    # Flatten the spatial dimensions of the image_tensor
    num_samples = image_tensor.shape[0]
    flattened_image_tensor = image_tensor.view(image_tensor.shape[0], -1)

    print("Flattened image tensor shape:", flattened_image_tensor.shape)
    print("Reshaped labeled tensor shape:", labeled_tensor.shape)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(flattened_image_tensor, labeled_tensor, test_size=0.2)
    
    # Create DataLoader objects
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    print("Image tensor shape:", image_tensor.shape)
    print("Labeled tensor shape:", labeled_tensor.shape)

    return train_loader, test_loader

def train(train_loader, num_channels, img_height, img_width):
    model = CCN(num_channels, img_height, img_width)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader: 
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch: {epoch+1}, Loss: {running_loss/len(train_loader)}')
    return model

def evaluate_model(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on test images: {100 * correct / total}%')

def load_tensor():
    directory_path = '/Users/adamkurth/Documents/vscode/CXFEL_Image_Analysis/CXFEL/waterbackground_subtraction/images/'
    file_pattern = directory_path + '*.h5'
        
    tensor_list = []

    for file_path in glob.glob(file_pattern):
        with h5.File(file_path, 'r') as file:
            # Extract the dataset from the .h5 file
            data = file['data/data/entry'][:]
            tensor = torch.from_numpy(data)
            print(f'Loaded tensor from {file_path} with shape: {tensor.shape}')
            tensor_list.append(tensor)

    # If there are multiple tensors, stack them into a single tensor
    if tensor_list:
        tensor = torch.stack(tensor_list)
    else:
        raise ValueError("No .h5 files found or empty dataset in files.")

    return tensor, directory_path

def test_main():
    threshold = 1000
    
    tensor, file_path = load_tensor()
    
    # If image_data is a single image, unsqueeze to add a batch dimension
    if len(image_data.shape) == 3:
        image_data = image_data.unsqueeze(0)  # Add batch dimension if missing

    coordinates = main(file_path, threshold, display=False)
    coordinates = [tuple(coord) for coord in coordinates]
    
    labeled_image = generate_labeled_image(image_data, coordinates, neighborhood_size=5)

    # preprocessing
    image_tensor, labeled_tensor = preprocess(image_data, labeled_image)
    print("Preprocessed image tensor shape:", image_tensor.shape)
    print("Labeled tensor shape:", labeled_tensor.shape)

    # data prep
    X_train, X_test, y_train, y_test, train_loader, test_loader = data_preparation(image_tensor, labeled_tensor)
    
    num_channels = image_tensor.shape[1]
    img_height, img_width = image_tensor.shape[2:4]
    
    model = train(train_loader, num_channels, img_height, img_width)
    
if __name__ == '__main__':
    test_main()