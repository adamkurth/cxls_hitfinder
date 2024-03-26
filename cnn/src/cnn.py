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

# runs process_directory.py, 
#   checks uniformity of images/
#   generates all, with corresponding water/*/.h5
#       labels/ 
#       peaks_water_overlay/ 

%run process_directory.py ../../images 
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


