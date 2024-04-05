import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pkg import *
from torch.cuda.amp import GradScaler, autocast

class TrainTestModels:
    """ 
    This class trains, tests, and plots the loss, accuracy, and confusion matrix of a model.
    There are two methods for training: test_model_no_freeze and test_model_freeze.
    """
    
    def __init__(self, model: nn.Module, loader: list, criterion, optimizer, device, cfg: dict, feature: str, labels: list) -> None:
        """ 
        Takes the arguments for training and testing and makes them available to the class.

        Args:
            model: PyTorch model
            loader: list of torch.utils.data.DataLoader where loader[0] is the training set and loader[1] is the testing set
            criterion: PyTorch loss function
            optimizer: PyTorch optimizer
            device: torch.device which is either 'cuda' or 'cpu'
            cfg: dict which holds the configuration parameters num_epochs, batch_size, and num_classes
        """
        self.model = model
        self.loader = loader
        self.train_loader, self.test_loader = loader[0], loader[1]
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = cfg['num_epochs']
        self.device = device
        self.batch = cfg['batch_size']
        self.classes = cfg['num_classes']
        self.plot_train_accuracy = np.zeros(self.epochs)
        self.plot_train_loss = np.zeros(self.epochs)
        self.plot_test_accuracy = np.zeros(self.epochs)
        self.plot_test_loss = np.zeros(self.epochs)

        self.cm = np.zeros((self.classes,self.classes), dtype=int)
        self.threshold = cfg['threshold']
        self.logger = logging.getLogger(__name__)
        self.feature = feature # peak, photon_energy, clen
        self.labels = labels
        self.scaler = GradScaler()

           
    def train(self, epoch:int) -> None:
        running_loss_train = accuracy_train = predictions = total_predictions = 0.0

        self.model.train()
        for inputs, labels, attributes  in self.loader[0]:  # Assuming self.loader[0] is the training data loader
            # peak_images, overlay_images = inputs
            # peak_images, overlay_images, labels = peak_images.to(self.device), overlay_images.to(self.device), labels.to(self.device)
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            
            # Encapsulating the forward pass and loss calculation inside the autocast context
            with autocast():
                score = self.model(inputs)
                image_attribute = attributes[self.feature]
                
                if self.feature == 'peak':
                    image_attribute = image_attribute.reshape(-1, 1).to(self.device)
                    image_attribute = image_attribute.float()  # Convert target to float
                elif self.feature == 'photon_energy':
                    photon_energy_holder = torch.zeros_like(image_attribute, dtype=torch.long).to(self.device)
                    photon_energy_holder[image_attribute == 6e3] = 1
                    photon_energy_holder[image_attribute == 7e3] = 2
                    photon_energy_holder[image_attribute == 8e3] = 3
                    image_attribute = photon_energy_holder
                elif self.feature == 'clen':
                    photon_energy_holder = torch.zeros_like(image_attribute, dtype=torch.long).to(self.device)
                    photon_energy_holder[image_attribute == 1.5] = 1
                    photon_energy_holder[image_attribute == 2.5] = 2
                    photon_energy_holder[image_attribute == 3.5] = 3
                    image_attribute = photon_energy_holder
                    
                loss = self.criterion(score, image_attribute.to(self.device))
            
            # Scales loss. Calls backward to create scaled gradients
            self.scaler.scale(loss).backward()
            # Unscales gradients and calls or skips optimizer.step()
            self.scaler.step(self.optimizer)
            # Updates the scale for next iteration
            self.scaler.update()

            running_loss_train += loss.item()

            if self.feature == 'peak':
                predicted = (torch.sigmoid(score) > self.threshold).long()  # Assuming 'score' is the output of your model
            elif self.feature == 'photon_energy' or 'clen':
                _, predicted = torch.max(score, 1)
                    
            accuracy_train += (predicted == image_attribute.to(self.device)).float().sum()
            total_predictions += torch.numel(image_attribute)
            
        loss_train = running_loss_train / len(self.loader[0])  # Assuming you want to average over all batches
        self.plot_train_loss[epoch] = loss_train
        self.logger.info(f'Train loss: {loss_train}')
        print(f'Train loss: {loss_train}')

        accuracy_train /= total_predictions
        self.plot_train_accuracy[epoch] = accuracy_train
        self.logger.info(f'Train accuracy: {accuracy_train}')
        print(f'Train accuracy: {accuracy_train}')
        
        
    def test(self, epoch:int) -> None:
        """ 
        This function test the model and prints the loss and accuracy of the testing sets per epoch.
        """
        # print(f'Model testing: {self.model.__class__.__name__}')
            
        running_loss_test = accuracy_test = predicted = total = 0.0
        self.model.eval()
        with torch.no_grad(), autocast():
            for inputs, labels, attributes in self.loader[1]:
                # peak_images, _ = inputs
                # peak_images = peak_images.to(self.device)
                # labels = labels.to(self.device)
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                score = self.model(inputs)
                                                
                image_attribute = attributes[self.feature]
                
                if self.feature == 'peak':
                    image_attribute = image_attribute.reshape(-1, 1).to(self.device)
                    image_attribute = image_attribute.float()  # Convert target to float
                elif self.feature == 'photon_energy':
                    photon_energy_holder = torch.zeros_like(image_attribute, dtype=torch.long).to(self.device)
                    photon_energy_holder[image_attribute == 6e3] = 1
                    photon_energy_holder[image_attribute == 7e3] = 2
                    photon_energy_holder[image_attribute == 8e3] = 3
                    image_attribute = photon_energy_holder
                elif self.feature == 'clen':
                    photon_energy_holder = torch.zeros_like(image_attribute, dtype=torch.long).to(self.device)
                    photon_energy_holder[image_attribute == 1.5] = 1
                    photon_energy_holder[image_attribute == 2.5] = 2
                    photon_energy_holder[image_attribute == 3.5] = 3
                    image_attribute = photon_energy_holder
                    
                loss = self.criterion(score, image_attribute)
         
                running_loss_test += loss.item()  # Convert to Python number with .item()
                
                if self.feature == 'peak':
                    predicted = (torch.sigmoid(score) > self.threshold).long()  # Assuming 'score' is the output of your model
                elif self.feature == 'photon_energy' or 'clen':
                    _, predicted = torch.max(score, 1)
                    
                accuracy_test += (predicted == image_attribute.to(self.device)).float().sum()
                accuracy_test += (predicted == image_attribute.to(self.device)).float().sum()
                total += torch.numel(image_attribute)

        loss_test = running_loss_test/self.batch
        self.plot_test_loss[epoch] = loss_test

        accuracy_test /= total
        self.plot_test_accuracy[epoch] = accuracy_test

        self.logger.info(f'Test loss: {loss_test}')
        self.logger.info(f'Test accuracy: {accuracy_test}')
        print(f'Test loss: {loss_test}')
        print(f'Test accuracy: {accuracy_test}')          
                    
    def plot_loss_accuracy(self) -> None:
        """ 
        This function plots the loss and accuracy of the training and testing sets per epoch.
        """
        plt.plot(range(self.epochs), self.plot_train_accuracy, marker='o', color='red')
        plt.plot(range(self.epochs), self.plot_test_accuracy, marker='o', color='orange', linestyle='dashed')
        plt.plot(range(self.epochs), self.plot_train_loss ,marker='o',color='blue')
        plt.plot(range(self.epochs), self.plot_test_loss ,marker='o',color='teal',linestyle='dashed')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss/accuracy')
        plt.legend(['accuracy train','accuracy test','loss train','loss test'])
        plt.show()
    

    def plot_confusion_matrix(self) -> None:
        """ 
        This function plots the confusion matrix of the testing set.
        """
        all_labels = []
        all_predictions = []

        with torch.no_grad(), autocast():
            for inputs, labels, attributes in self.loader[1]:  # Assuming self.loader[1] is the testing data loader
                # peak_images, _ = inputs
                # peak_images = peak_images.to(self.device)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                score = self.model(inputs)

                # Flatten and append labels to all_labels
                image_attribute = attributes[self.feature].reshape(-1).cpu().numpy()
                all_labels.extend(image_attribute)
                
                
                if self.feature == 'peak':
                    predicted = (torch.sigmoid(score) > self.threshold).long().cpu()  # Assuming 'score' is the output of your model
                elif self.feature == 'photon_energy':
                    _, predicted = torch.max(score, 1)
                    predicted = predicted.cpu()
                    predicted = torch.where(predicted == 1, torch.tensor(6000), predicted)
                    predicted = torch.where(predicted == 2, torch.tensor(7000), predicted)
                    predicted = torch.where(predicted == 3, torch.tensor(8000), predicted)
                elif self.feature == 'clen':
                    _, predicted = torch.max(score, 1)
                    predicted - torch.where(predicted == 1, torch.tensor(1.5), predicted)
                    predicted - torch.where(predicted == 2, torch.tensor(2.5), predicted)
                    predicted - torch.where(predicted == 3, torch.tensor(3.5), predicted)
                                        
                all_predictions.extend(predicted)

        # No need to reshape - arrays should already be flat
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        # Compute confusion matrix
        self.cm = confusion_matrix(all_labels, all_predictions, labels=self.labels, normalize='true')

        # Plotting the confusion matrix
        plt.matshow(self.cm, cmap="Blues")
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def get_confusion_matrix(self) -> np.ndarray:
        """ 
        This function returns the confusion matrix of the testing set.
        """
        return self.cm
    
    def epoch_loop(self) -> None: 
        """
        This function loops through the number of epochs and trains and tests the model.
        """

        self.logger.info(f'Model training and testing: {self.model.__class__.__name__}')
        print(f'Model testing and validating: {self.model.__class__.__name__}')     
        print(f'Looking for the feature: {self.feature}')  
        
        for epoch in range(self.epochs):
            self.logger.info('-- epoch '+str(epoch)) 
            print('-- epoch '+str(epoch)) 

            self.train(epoch)
            self.test(epoch)
            
    def get_loss_accuracy(self) -> dict:
        """ 
        This function returns the loss and accuracy of the training and testing sets.
        """
        return {'train loss': self.plot_train_loss, 'train accuracy': self.plot_train_accuracy, 'test loss': self.plot_test_loss, 'test accuracy': self.plot_test_accuracy}
    
    def print_state_dict(self) -> None:
        """
        This function prints the model's state_dict and optimizer's state_dict.
        """
        
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            print(var_name, "\t", self.optimizer.state_dict()[var_name])
    
    def save_model(self, path:str) -> None:
        """
        This function saves the model's state_dict to a specified path. This can be used to load the trained model later.
        Save as .pt file.
        root: /cnn/models
        Args:
            path (str): Path to save the model's state_dict.
        """
        torch.save(self.model.state_dict(), path)
    