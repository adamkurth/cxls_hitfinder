import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
import matplotlib.pyplot as plt
from pkg import c, m, f 

# demo of ResNet50BraggPeakClassifier

# instances
paths = c.PathManager()
data = c.PeakImageDataset(paths)
prep = c.DataPreparation(paths, data, batch_size=5)
water_h5 = data.load_h5(paths.water_background_h5)
ip = c.ImageProcessor(water_h5)
p = c.PeakThresholdProcessor(threshold_value=10)

# clean sim/
paths.clean_sim() # moves all .err, .out, .sh files sim_specs
# if not already processed
# ip.process_directory(paths, p.threshold_value) 

# GPU/CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# model
model = m.ResNet50BraggPeakClassifier()
# loss function/ combines a Sigmoid layer and the BCELoss in one single class
criterion = nn.BCEWithLogitsLoss()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Criterion: ", criterion)
print("Optimizer: \n", optimizer)
print("Learning rate: ", optimizer.param_groups[0]['lr'])
# data loaders 
train_loader, test_loader = prep.prep_data()

# Train ResNet50BraggPeakClassifier
num_epochs = 5
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
train_losses = test_losses = []

logging.info('Staring training...')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for inputs, labels in train_loader: 
        peak_images, water_images = inputs[0].to(device), inputs[1].to(device)
        labels = labels.to(device)
        
        # zero parameter gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(peak_images)
        loss = criterion(outputs, labels)
        # backward pass/optimize
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # calculate accuracy
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct_predictions += (predicted == labels).float().sum()
        total_predictions += labels.size(0)
    
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}')

# Testing ResNet50BraggPeakClassifier
logging.info('Starting testing...')
model.eval()  
test_loss = correct_peaks = total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        peak_images, water_images = inputs
        # forward pass prediction
        peak_pred = model(peak_images) 
        # compute loss
        loss = criterion(peak_pred, labels)
        test_loss += loss.item()
        # convert predictions to binary to compare with labels 
        peak_prediction = torch.sigmoid(peak_pred) > 0.5 
        # calculate accuracy of peak detection
        correct_peaks += (peak_prediction == labels).sum().item()
        total += np.prod(labels.shape) 

    avg_test_loss = test_loss / len(test_loader)
    peak_detection_accuracy = correct_peaks / total
    test_losses.append(avg_test_loss)
    # log results 
    logging.info(f"Test Loss: {avg_test_loss:.4f}, Bragg Peak Detection Accuracy: {peak_detection_accuracy:.4f}")
    logging.info(f'Test loss: {test_loss}')
    
logging.info('Testing completed.')

# plotting
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Testing Loss')
plt.legend()
plt.show()

# end of ResNet50BraggPeakClassifier demo



