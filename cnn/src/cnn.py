import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import logging

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
model_res50 = m.CustomResNet50(num_proteins=3, num_camlengths=3)

"""Loss/Optimizer"""
criterion_protein, criterion_camlength = torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_res50.parameters(), lr=0.001)

print("Criterion: ", criterion_protein, criterion_camlength)
print("Optimizer: \n", optimizer)
print("Learning rate: ", optimizer.param_groups[0]['lr'])

"""Training"""
# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Staring training...')
num_epochs = 1

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
            labels_protein = torch.tensor([protein_to_idx[protein] for protein in protein_identifiers], dtype=torch.long).to(peak_images.device)
            labels_cam_len = labels[2].to(dtype=torch.long).to(peak_images.device)
        except KeyError as e:
            logging.error(f"KeyError with label: {e}")
            print(f"KeyError with label: {e}")
            print(labels[:5])
            continue

        # print(f'Beginning training:\n Current Epoch: {epoch} with {len(labels_protein), len(labels_cam_len)} labels')
        optimizer.zero_grad()

        # multi-task learning: predicting protein and camlength
        protein_pred, camlength_pred = model_res50((peak_images, water_images))

        protein_loss = criterion_protein(protein_pred, labels_protein.long())
        camlength_loss = criterion_camlength(camlength_pred, labels_cam_len.long())

        loss = protein_loss + camlength_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_counter += 1

        if batch_index == 1 or batch_index % 10 == 0:
            logging.info(f'Epoch: {epoch+1}, Batch: {batch_index}, Running Loss: {running_loss / batch_index}')
    
    # epoch summary
    avg_loss = running_loss / len(train_loader)
    logging.info(f'Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}')
