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
import sys
from label_finder import(
    load_data, 
    load_file_h5,
    display_peak_regions, 
    # validate,
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

    def get_coordinates_above_threshold(self, image):
        # convert to boolean mask 'image' is a 2D tensor (H, W)
        mask = image > self.threshold_value
        coordinates = torch.nonzero(mask).cpu().numpy()
        return coordinates

    def get_local_maxima(self, image):
        # from 2D image to 1D
        image_1d = image.flatten().cpu().numpy()
        peaks, _ = find_peaks(image_1d, height=self.threshold_value)
        coordinates = [self.flat_to_2d(idx, image.shape[1]) for idx in peaks]
        return coordinates
    
    def flat_to_2d(self, index, width):
        return (index // width, index % width)
    
    def patch_method(self, image_tensor):
        # scale image_tensor to [0,1]
        normalized_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
        return normalized_tensor
    
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
    def __init__(self, img_height, img_width):
        super(CCN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # 32 neurons
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # 64 neurons
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 pooling
        self.dropout = nn.Dropout(0.5)  # Dropout layer with 50% probability
        self.flattened_size = 64 * (img_height // 4) * (img_width // 4) # 4x4 pooling
        self.fc1 = nn.Linear(self.flattened_size, 128) # 128 neurons
        self.fc2 = nn.Linear(128, 2) # 2 for binary classification
    
    def forward(self, x):
        print("Shape at the start of forward method:", x.shape)  # Debugging line
        
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x) 
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def preprocess():
    def load_tensor(directory_path):
        file_pattern = os.path.join(directory_path, '*.h5')
        tensor_list = []

        for file_path in glob.glob(file_pattern):
            with h5.File(file_path, 'r') as file:
                # Assuming 'entry/data/data' is the correct path within your .h5 files
                data = np.array(file['entry/data/data'][:])
                # Ensure data is 2D (H, W), if not, you might need to adjust or provide additional context
                if len(data.shape) == 3 and data.shape[2] == 1:
                    data = data[:, :, 0]  # If it's (H, W, 1), convert to (H, W)
                elif len(data.shape) != 2:
                    raise ValueError(f"Data in {file_path} has an unexpected shape: {data.shape}\n")

                tensor = torch.from_numpy(data).unsqueeze(0).float()  # Add a batch dimension (1, H, W)
                print(f'Loaded data from {file_path} with shape: {tensor.shape} \n')
                tensor_list.append(tensor)

        if not tensor_list:
            raise ValueError("No .h5 files found or empty dataset in files.")

        # single tensor with shape (N, H, W)
        combined_tensor = torch.cat(tensor_list, dim=0)

        print(f"Combined tensor shape: {combined_tensor.shape} \n")
        return combined_tensor, directory_path

    def is_peak(image_tensor, coordinate, neighborhood_size=3):
        x, y = coordinate
        region = ArrayRegion(image_tensor)
        
        neighborhood = region.extract_region(x, y, neighborhood_size)
        if torch.numel(neighborhood) == 0:  # empty
            return False
        
        center = neighborhood_size // 2, neighborhood_size // 2
        is_peak = neighborhood[center] == torch.max(neighborhood)
        return is_peak

    def find_coordinates(combined_tensor):
        coord_list_manual = []
        coord_list_script = []
        processor = PeakThresholdProcessor(combined_tensor, threshold_value=1000)
        confirmed_common_list = []

        for img_idx, img in enumerate(combined_tensor):
            print(f'Processing Image {img_idx}')
            # manual 
            coord_manual = processor.get_coordinates_above_threshold(img)
            coord_list_manual.append(coord_manual)
            # script
            coord_script = processor.get_local_maxima(img)
            coord_list_script.append(coord_script)
            # validate for img 
            confirmed_common, _, _ = validate(coord_manual, coord_script, img)
            confirmed_common_list.append(confirmed_common)
        return confirmed_common_list
        
    def validate(manual, script, image_array):
        manual_set = set([tuple(x) for x in manual])
        script_set = set([tuple(x) for x in script])
        
        common = manual_set.intersection(script_set)
        unique_manual = manual_set.difference(script_set)
        unique_script = script_set.difference(manual_set)
        print(f'common: {common}\n')
        print(f'unique_manual: {unique_manual}\n')
        print(f'unique_script: {unique_script}\n')
        
        confirmed_common = {coord for coord in common if is_peak(image_array, coord)}
        confirmed_unique_manual = {coord for coord in unique_manual if is_peak(image_array, coord)}
        confirmed_unique_script = {coord for coord in unique_script if is_peak(image_array, coord)}
        
        print(f'confirmed_common: {confirmed_common}\n')
        print(f'confirmed_unique_manual: {confirmed_unique_manual}\n')
        print(f'confirmed_unique_script: {confirmed_unique_script}\n')

        return confirmed_common, confirmed_unique_manual, confirmed_unique_script

    def generate_label_tensor(image_tensor, confirm_common_list, neighborhood_size=3):
        """Generate a tensor of the same shape as image_tensor, marking 1 at peaks and 0 elsewhere."""
        # N: number of images, 
        # H: height,
        # W: width
        label_tensor_list = []
        for img_idx, coordinates in enumerate(confirmed_common_list):
            label_tensor = np.array(image_tensor[img_idx, :, :])  # (H, W)
            
            for x, y in coordinates:
                if is_peak(image_tensor[img_idx, :, :], (x, y), neighborhood_size):
                    label_tensor[x, y] = 1 # peak

            label_tensor = torch.from_numpy(label_tensor).unsqueeze(0).float() 
            print(f'Label tensor shape: {label_tensor.shape}')
            label_tensor_list.append(label_tensor)
            
        combined_label_tensor = torch.stack(label_tensor_list)

        return combined_label_tensor
    
    # toggle between work, home, or agave directory
    def choose_path():
        chosen_path = sys.argv[1]
        if chosen_path is not None:
            return chosen_path
        else:
            raise ValueError("Invalid directory path provided: " + chosen_path)

    # work_dir = '/home/labuser/Development/adam/vscode/waterbackground_subtraction/images/'
    # home_dir = '/Users/adamkurth/Documents/vscode/CXFEL_Image_Analysis/CXFEL/waterbackground_subtraction/images/'
    # agave_dir = '/home/amkurth/Development/pattern_simulations/sim_3_3e5keV/'

    choice = choose_path()
    combined_tensor, directory_path = load_tensor(choice)
    print(f'Type of combined_label_tensor: {type(combined_tensor)}')
    print(f'Shape of combined_label_tensor: {combined_tensor.shape}')
    confirmed_common_list = find_coordinates(combined_tensor)
    label_tensor = generate_label_tensor(combined_tensor, confirmed_common_list)
    print(label_tensor)
    return combined_tensor, label_tensor, confirmed_common_list

def data_preparation(image_tensor, labeled_tensor):
    """Split the data into training and testing sets and create DataLoader objects."""
    num_images = image_tensor.shape[0]
    print(f'Number of images: {num_images}')
    
    image_tensor = image_tensor.unsqueeze(1) # Reshape from [N, H, W] to [N, 1, H, W]
    image_tensor = PeakThresholdProcessor(image_tensor).patch_method(image_tensor)
    print(f'Image tensor shape: {image_tensor.shape}') # [N, 1, H, W]
    
    # ensure labeled_tensor is a 1d tensor of long type
    labeled_tensor = labeled_tensor.view(-1)[:num_images].long()
    print(f'Corrected labeled tensor shape: {labeled_tensor.shape}')    
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(image_tensor, labeled_tensor, test_size=0.2)
    
    # Create DataLoader objects
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    print(f'Image tensor shape: {image_tensor.shape}')
    print(f'Labeled tensor shape: {labeled_tensor.shape}')
    print(f'Shape before entering the model: {image_tensor.shape}')
    return train_loader, test_loader

def train(train_loader, img_height, img_width):
    """Train the model and return the trained model."""
    # Create the model
    print(f'Running training method...\n')
    model = CCN(img_height, img_width)
    print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001) # lr: learning rate
    
    # Train the model
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            print(f'Images shape: {images.shape}')  # Should be [N, 1, H, W]
            print(f'Labels shape: {labels.shape}')  # Should be [N] or [N, 1]
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch: {epoch+1}, Loss: {running_loss/len(train_loader)}')
    return model

def evaluate_model(model, test_loader):
    """Evaluate the model and return the accuracy."""
    model.eval() 
    correct = 0
    total = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0) # total number of labels
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the {total} test images: {accuracy} %')
    return accuracy

if __name__ == '__main__':
    combined_tensor, label_tensor, confirmed_common_list = preprocess()
    train_loader, test_loader = data_preparation(combined_tensor, label_tensor)
    img_height, img_width = combined_tensor.shape[1], combined_tensor.shape[2]
    model = train(train_loader, img_height, img_width)
    accuracy = evaluate_model(model, test_loader)
    
    # labels shape: torch.Size([8]) - model is trained on batches of 8 images at a time
    # loss 