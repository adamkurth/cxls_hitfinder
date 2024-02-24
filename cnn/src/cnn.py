
import classes as cl 
import functions as fn
import models as md
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

"""instances"""
paths = cl.PathManager()
paths.clean_sim() # moves all .err, .out, .sh files sim_specs 

dataset = cl.PeakImageDataset(paths=paths, transform=transforms.ToTensor(), augment=False)
prep = cl.DataPreparation(paths, batch_size=32)

"""checks"""
peak_paths = paths.__get_peak_images_paths__()
water_paths = paths.__get_water_images_paths__()
print('Number of Peak Images: ', len(peak_paths), 'Number of Water Images', len(water_paths))    


"""train and test data loaders"""
train_loader, test_loader = prep.prep_data()

"""model"""
model_res50 = md.CustomResNet50(num_proteins=10, num_camlengths=3)

water_background = dataset.__load_h5__(paths.__get_path__('water_background_h5'))

dataset.__len__()
dataset.__get_item__(0)
dataset.__preview__(0)

# # loss and optimizer
# criteron = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # train 
# num_epochs = 10

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criteron(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    


        
        



