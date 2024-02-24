
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

dataset = c.PeakImageDataset(paths=paths, transform=transforms.ToTensor(), augment=False)
prep = c.DataPreparation(paths=paths, batch_size=5)

"""checks"""
peak_paths = paths.__get_peak_images_paths__()
water_paths = paths.__get_water_images_paths__()
print('Number of Peak Images: ', len(peak_paths), 'Number of Water Images', len(water_paths))    


"""train and test data loaders"""
train_loader, test_loader = prep.prep_data()

"""model"""
model_res50 = m.CustomResNet50(num_proteins=3, num_camlengths=3)

"""loss and optimizer"""
criteron = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_res50.parameters(), lr=0.001)

"""train""" 
num_epochs = 10

for epoch in range(num_epochs):
    model_res50.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # one hot encoding to label encoding
        _, labels = torch.max(labels, dim=1)
        
        optimizer.zero_grad()
        outputs = model_res50(images)
        loss = criteron(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    


        
        



