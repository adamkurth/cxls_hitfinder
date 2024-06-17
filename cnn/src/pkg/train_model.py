import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pkg import *
import torch.optim.lr_scheduler as lrs
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

class TrainModel:
    
    def __init__(self, cfg: dict, attributes: dict):
        self.logger = logging.getLogger(__name__)
        
        self.train_loader = cfg['train data']
        self.test_loader = cfg['test data']
        self.batch_size = cfg['batch size']
        self.device = cfg['device']
        self.epochs = cfg['epochs']
        self.optimizer = cfg['optimizer']
        self.scheduler = cfg['scheduler']
        self.criterion = cfg['criterion']
        self.learning_rate = cfg['learning rate']
        self.model = cfg['model']
        
        self.camera_length = attributes['camera length']
        self.photon_energy = attributes['photon energy']
        self.peak = attributes['peak']
        
        self.plot_train_accuracy = np.zeros(self.epochs)
        self.plot_train_loss = np.zeros(self.epochs)
        self.plot_test_accuracy = np.zeros(self.epochs)
        self.plot_test_loss = np.zeros(self.epochs)
        
        
    def make_training_instances(self) -> None:
        """
        This function takes the strings from the sbatch input and makes them objects.
        """
        # model_class = getattr(m, self.model_arch)
        # return model_class()
        
        self.model = getattr(m, self.model)()
        self.optimizer = getattr(optim, self.optimizer)(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = getattr(lrs, self.scheduler)(self.optimizer, mode='min', factor=0.1, patience=3, threshold=0.1)
        self.criterion = getattr(nn, self.criterion)()
    
    def epoch_loop(self) -> None: 
        """
        This function loops through the number of epochs and trains and tests the model.
        """
        
        self.logger.info(f'Model training and testing: {self.model.__class__.__name__}')
        self.logger.info(f'Looking for the feature: {self.feature}')  
        print(f'Model testing and validating: {self.model.__class__.__name__}')     
        print(f'Looking for the feature: {self.feature}')  
        
        for epoch in range(self.epochs):
            self.logger.info('-- epoch '+str(epoch)) 
            print('-- epoch '+str(epoch)) 

            self.train(epoch)
            self.test(epoch)
            
            print(f"-- learning rate : {self.scheduler.get_last_lr()}")
            
    def train(self, epoch:int) -> None:
        
        """
        This function trains the model and prints the loss and accuracy of the training sets per epoch.
        """
        
        running_loss_train, accuracy_train, predictions, total_predictions = 0.0, 0.0, 0.0, 0.0

        self.model.train()
        
        for inputs, attributes in self.train_loader:
        
            inputs = inputs.unsqueeze(0).unsqueeze(0).to(self.device)

            self.optimizer.zero_grad()
            
            with autocast():
                
                score = self.model(inputs, attributes[self.camera_length], attributes[self.photon_energy])

                truth = attributes[self.peak].reshape(-1, 1).float().to(self.device)
                
                loss = self.criterion(score, truth)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss_train += loss.item()

            predictions (torch.sigmoid(score) > 0.5).long()
                    
            accuracy_train += (predictions == truth.float().sum())
            total_predictions += torch.numel(truth)
            
        loss_train = running_loss_train / len(self.train_loader)  
        
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
        
        running_loss_test, accuracy_test, predictions, total = 0.0, 0.0, 0.0, 0.0
        
        self.model.eval()
        with torch.no_grad():
            for inputs, attributes in self.test_loader:
                
                inputs = inputs.unsqueeze(0).unsqueeze(0).to(self.device)

                with autocast():
                    score = self.model(inputs, attributes[self.camera_length], attributes[self.photon_energy])
                             

                    truth = attributes[self.peak].reshape(-1, 1).float().to(self.device)
                    
                    loss = self.criterion(score, truth)
            
                running_loss_test += loss.item()  # Convert to Python number with .item()
                
                predictions = (torch.sigmoid(score) > 0.5).long()
                    
                accuracy_test += (predictions == truth).float().sum()
                total += torch.numel(truth)

        loss_test = running_loss_test/self.batch
        self.scheduler.step(loss_test)
        self.plot_test_loss[epoch] = loss_test

        accuracy_test /= total
        self.plot_test_accuracy[epoch] = accuracy_test

        self.logger.info(f'Test loss: {loss_test}')
        self.logger.info(f'Test accuracy: {accuracy_test}')
        print(f'Test loss: {loss_test}')
        print(f'Test accuracy: {accuracy_test}')   
        
    def plot_loss_accuracy(self, path:str = None) -> None:
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
        plt.title(f'Loss and Accuracy for {self.feature} with {self.model.__class__.__name__}')
        plt.legend(['accuracy train','accuracy test','loss train','loss test'])
        
        if path != None:
            plt.savefig(path)
            
        plt.show()
        
    def save_model(self, path:str) -> None:
        """
        This function saves the model's state_dict to a specified path. This can be used to load the trained model later.
        Save as .pt file.
        root: /cnn/models
        Args:
            path (str): Path to save the model's state_dict.
        """

        torch.save(self.model.state_dict(), path)
