import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt

from  torchvision import transforms
from pkg import c, m, f

"""Instances"""
paths = c.PathManager()
dataset = c.PeakImageDataset(paths=paths, transform=None, augment=True)
prep = c.DataPreparation(paths=paths, batch_size=5)

"""Clean sim/ directory"""
paths.clean_sim() # moves all .err, .out, .sh files sim_specs

"""checks"""
peak_paths = paths.__get_peak_images_paths__()
water_paths = paths.__get_water_images_paths__()
print('Number of Peak Images: ', len(peak_paths), 'Number of Water Images', len(water_paths))

print("Peak images path:", paths.peak_images_dir)
print("Water images path:", paths.water_images_dir)

"""Train/Test Data Loaders"""
train_loader, test_loader = prep.prep_data()

"""Protein Mapping"""
protein_to_idx = {
    '1IC6': 0,
    # To be developed
}

"""Models"""
model_res50 = m.CustomResNet50(num_proteins=3, num_camlengths=3, output_size=(50,50))

"""Loss/Optimizer"""
criterion_protein = criterion_camlength = criterion_peak = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_res50.parameters(), lr=0.001)

print("Criterion: ", criterion_protein, criterion_camlength, criterion_peak)
print("Optimizer: \n", optimizer)
print("Learning rate: ", optimizer.param_groups[0]['lr'])

"""Initial Setup"""
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
train_losses = []
test_losses = []

"""Training"""
logging.info('Staring training...')
num_epochs = 2

for epoch in range(num_epochs):
    model_res50.train()
    running_loss = 0.0
    batch_counter = 0
    
    for batch_index, ((peak_images, water_images), labels) in enumerate(train_loader, start=1):
        if batch_index == 1:
            logging.info(f"Epoch {epoch+1}/{num_epochs} - First batch label structure: {labels[0]} with type {type(labels[0])}")
        
        for label in labels:
            print(f"Label structure: {label} {type(label)}\n\n")
        # Extract the protein identifiers assuming they are always the first element in the label tuple
        protein_identifiers = labels[0] # gives tuple ('1IC6', '1IC6', '1IC6', '1IC6', '1IC6')

        try:
            labels_protein = torch.tensor([protein_to_idx[label] for label in labels[0]], dtype=torch.long).to(peak_images.device)
            labels_cam_len = labels[2].to(dtype=torch.long)
            labels_heatmap = prep.prep_labels_heatmap(labels)
            
        except KeyError as e:
            logging.error(f"KeyError with label: {e}")
            print(f"KeyError with label: {e}")
            print(labels[:5])
            continue

        optimizer.zero_grad()

        # multi-task learning: predicting protein and camlength
        protein_pred, camlen_pred, peak_heatmap_pred = model_res50((peak_images, water_images))

        protein_loss = criterion_protein(protein_pred, labels_protein.long())
        camlength_loss = criterion_camlength(camlen_pred, labels_cam_len.long())
        peak_heatmap_loss = criterion_peak(peak_heatmap_pred, labels_heatmap.long())
        
        total_loss = protein_loss + camlength_loss + peak_heatmap_loss
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        batch_counter += 1

        if (batch_index + 1) % 10 == 0:  # Log every 10 batches
            logging.info(f'Epoch {epoch+1}, Batch {batch_index + 1}: Loss = {running_loss/(batch_index+1)}')

    avg_loss_train = running_loss / len(train_loader)
    train_losses.append(avg_loss_train)
    logging.info(f'Epoch {epoch+1} Training Completed. Avg Loss: {avg_loss_train:.4f}')

"""Testing"""
logging.info('Starting testing...')
model_res50.eval()
test_loss = 0 
correct_protein = 0
correct_camlen = 0
total = 0

with torch.no_grad():
    for (peak_images, water_images), labels in test_loader:
        labels_protein = torch.tensor([protein_to_idx[protein] for protein in labels[0]], dtype=torch.long).to(peak_images.device)
        labels_camlen = labels[2].to(dtype=torch.long).to(peak_images.device)
        
        protein_pred, camlength_pred = model_res50((peak_images, water_images))
        
        loss_protein = criterion_protein(protein_pred, labels_protein)
        loss_camlen = criterion_camlength(camlength_pred, labels_camlen)
        
        _, predicted_protein = torch.max(protein_pred, 1)
        _, predicted_camlen = torch.max(camlength_pred, 1)
        correct_protein += (predicted_protein == labels_protein).sum().item()
        correct_camlen += (predicted_camlen == labels_camlen).sum().item()
        total += labels_protein.size(0)
    
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    protein_accuracy = correct_protein / total
    camlength_accuracy = correct_camlen / total

    logging.info(f"Test Loss: {avg_test_loss:.4f}, Protein Accuracy: {protein_accuracy:.4f}, Camera Length Accuracy: {camlength_accuracy:.4f}")
    
logging.info('Testing completed.')


"""Plotting"""
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Testing Loss')
plt.legend()
plt.show()





# """Training and Testing Functions"""
    
# def train(model, train_loader, criterion_protein, criterion_camlength, optimizer, num_epochs, protein_to_idx, device='cpu', log_interval=10):
#     train_losses = []
#     logging.info('Starting training...')
    
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         batch_counter = 0

#         for batch_index, ((peak_images, water_images), labels) in enumerate(train_loader, start=1):
#             if batch_index == 1:
#                 logging.info(f"Epoch {epoch+1}/{num_epochs} - First batch label structure: {labels[0]} with type {type(labels[0])}")

#             # Extract the protein identifiers assuming they are always the first element in the label tuple
#             protein_identifiers = labels[0] # gives tuple ('1IC6', '1IC6', '1IC6', '1IC6', '1IC6')

#             try:
#                 labels_protein = torch.tensor([protein_to_idx[protein] for protein in protein_identifiers], dtype=torch.long).to(peak_images.device)
#                 labels_cam_len = labels[2].to(dtype=torch.long).to(peak_images.device)
#             except KeyError as e:
#                 logging.error(f"KeyError with label: {e}")
#                 print(f"KeyError with label: {e}")
#                 print(labels[:5])
#                 continue

#             optimizer.zero_grad()

#             # multi-task learning: predicting protein and camlength
#             protein_pred, camlen_pred = model((peak_images, water_images))

#             protein_loss = criterion_protein(protein_pred, labels_protein.long())
#             camlength_loss = criterion_camlength(camlen_pred, labels_cam_len.long())

#             loss = protein_loss + camlength_loss
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             batch_counter += 1

#             if (batch_index + 1) % log_interval == 0:  # Log every log_interval batches
#                 logging.info(f'Epoch {epoch+1}, Batch {batch_index + 1}: Loss = {running_loss/(batch_index+1)}')

#         avg_loss_train = running_loss / len(train_loader)
#         train_losses.append(avg_loss_train)
#         logging.info(f'Epoch {epoch+1} Training Completed. Avg Loss: {avg_loss_train:.4f}')
    
#     return train_losses

# def test(model, test_loader, criterion_protein, criterion_camlength, protein_to_idx, device='cpu'):
#     test_losses = []
#     logging.info('Starting testing...')
#     model.eval()
#     test_loss = 0 
#     correct_protein = 0
#     correct_camlen = 0
#     total = 0

#     with torch.no_grad():
#         for (peak_images, water_images), labels in test_loader:
#             labels_protein = torch.tensor([protein_to_idx[protein] for protein in labels[0]], dtype=torch.long).to(peak_images.device)
#             labels_camlen = labels[2].to(dtype=torch.long).to(peak_images.device)
            
#             protein_pred, camlength_pred = model((peak_images, water_images))
            
#             loss_protein = criterion_protein(protein_pred, labels_protein)
#             loss_camlen = criterion_camlength(camlength_pred, labels_camlen)
#             loss = loss_protein + loss_camlen
#             test_loss += loss.item()

#             _, predicted_protein = torch.max(protein_pred, 1)
#             _, predicted_camlen = torch.max(camlength_pred, 1)
#             correct_protein += (predicted_protein == labels_protein).sum().item()
#             correct_camlen += (predicted_camlen == labels_camlen).sum().item()
#             total += labels_protein.size(0)
        
#         avg_test_loss = test_loss / len(test_loader)
#         test_losses.append(avg_test_loss)
#         protein_accuracy = correct_protein / total
#         camlength_accuracy = correct_camlen / total

#         logging.info(f"Test Loss: {avg_test_loss:.4f}, Protein Accuracy: {protein_accuracy:.4f}, Camera Length Accuracy: {camlength_accuracy:.4f}")
    
#     logging.info('Testing completed.')
#     return test_losses, protein_accuracy, camlength_accuracy

# def plot_losses(train_losses, test_losses, num_epochs):
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
#     plt.plot(range(1, num_epochs+1), test_losses, label='Testing Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training vs Testing Loss')
#     plt.legend()
#     plt.show()


# plot_losses(train_losses, test_losses, num_epochs=2)  
    
    
    
    
     