import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from pkg import u,m

clen_values, photon_energy_values = [1.5, 2.5, 3.5], [6000, 7000, 8000]
param_matrix = u.parameter_matrix(clen_values, photon_energy_values)
print(param_matrix, '\n')

dataset_dict = {
    '01': [clen_values[0], photon_energy_values[0]],
    '02': [clen_values[0], photon_energy_values[1]],
    '03': [clen_values[0], photon_energy_values[2]],
    '04': [clen_values[1], photon_energy_values[0]],
    '05': [clen_values[1], photon_energy_values[1]],
    '06': [clen_values[1], photon_energy_values[2]],
    '07': [clen_values[2], photon_energy_values[0]],
    '08': [clen_values[2], photon_energy_values[1]],
    '09': [clen_values[2], photon_energy_values[2]],
}


# parameters
dataset = '01'
print(dataset_dict[dataset])
clen, photon_energy = dataset_dict[dataset]
threshold = 1
# clen = 1.5 # meters 
# photon_energy = 6000 # eV/ 6 keV

# instances
pm = u.PathManager()
peak_paths, water_peak_paths, labels, water_background_path = pm.select_dataset(dataset=dataset)
p = u.Processor(paths=pm, dataset=dataset)
dm = u.DatasetManager(paths=pm, dataset=dataset, transform=None)


# p.process_directory(dataset=dataset, clen=clen, photon_energy=photon_energy)

# generate peak/overlay/label and upate attributes
# overlaying the correct water01.h5 path to the peak images
# p.process_directory(dataset=dataset, clen=clen, photon_energy=photon_energy)

# peak, label, overlay, background are valid types
u.check_attributes(paths=pm, dataset=dataset, type='label') 

train_loader, test_loader = u.prepare(data_manager=dm, batch_size=10)

model1 = m.BasicCNN1()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = {
    'num_epochs': 10,
    'num_classes': 2,
    'batch_size': train_loader.batch_size,
    'test_size': len(train_loader.dataset),
    'test_size': len(test_loader.dataset),
    'criterion': criterion,
    'optimizer': optimizer,
    'device': device,
    'model': model1,
}

# arguments: self, model, loader: list, criterion, optimizer, device, cfg: dict
t = u.TrainTestModels(model=model1, loader=[train_loader, test_loader], criterion=criterion, optimizer=optimizer, device=device, cfg=cfg)
t.train()


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



