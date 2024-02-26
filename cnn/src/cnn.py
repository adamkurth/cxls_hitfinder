import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from pkg import models as m
from pkg import classes as c
from pkg import functions as f

"""instances"""
paths = c.PathManager()
paths.clean_sim() # moves all .err, .out, .sh files sim_specs


dataset = c.PeakImageDataset(paths=paths, transform=transforms.ToTensor(), augment=True)
prep = c.DataPreparation(paths=paths, batch_size=5)

"""checks"""
peak_paths = paths.__get_peak_images_paths__()
water_paths = paths.__get_water_images_paths__()
print('Number of Peak Images: ', len(peak_paths), 'Number of Water Images', len(water_paths))

"""train and test data loaders"""
train_loader, test_loader = prep.prep_data()

"""protein mapping"""
protein_to_idx = {
    '1IC6': 0,
    # To be developed
}

"""model"""
model_res50 = m.CustomResNet50(num_proteins=3, num_camlengths=3)

"""loss and optimizer"""
criteron_protein = nn.CrossEntropyLoss()
criteron_camlength = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_res50.parameters(), lr=0.001)

"""train"""
num_epochs = 10

for epoch in range(num_epochs):
    model_res50.train()
    running_loss = 0.0
    for (peak_images, water_images), labels in train_loader:
        for label in labels:
            print("Label structure:", label)

        labels_protein = torch.tensor([protein_to_idx[label['protein']] for label in labels], dtype=torch.long).to(peak_images.device)
        labels_cam_len = torch.tensor([label['cam_length_label'] for label in labels], dtype=torch.long).to(peak_images.device)
        optimizer.zero_grad()

        # multi-task learning
        protein_pred, camlength_pred = model_res50((peak_images, water_images))

        protein_loss = criteron_protein(protein_pred, labels_protein.long())
        camlength_loss = criteron_camlength(camlength_pred, labels_cam_len.long())

        loss = protein_loss + camlength_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
