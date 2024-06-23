import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import datetime
from . import models as m

class TrainModel:
    
    # ! This class may work better with an inheritance relation with the evaluation class and or the data management class. 
    
    def __init__(self, cfg: dict, attributes: dict, transfer_learning_state_dict: str) -> None:
        """
        This constructor breaks up the training configuration infomation dictionary and h5 metadata key dictionary.
        In addition, a logging object is created and global list are created for storing infomation about the training loss and accuracy. 

        Args:
            cfg (dict): Dictionary containing important information for training including: data loaders, batch size, training device, number of epochs, the optimizer, the scheduler, the criterion, the learning rate, and the model class. Everything besides the data loaders and device are arguments in the sbatch script.
            attributes (dict): Dictionary containing the names of the metadata contained in the h5 image files. These names could change depending on whom created the metadata, so the specific names are arguments in the sbatch script. 
        """
        self.logger = logging.getLogger(__name__)
        self.scaler = GradScaler()
        
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
        
        self.transfer_learning_path = transfer_learning_state_dict
        
        
    def make_training_instances(self) -> None:
        """
        This function takes the strings from the sbatch script and makes them objects.
        These strings are objects that are needed for the training. Objects declared here are :
        - the model
        - the optimizer
        - the learning rate scheduler
        - the loss criterion
        """
        # model_class = getattr(m, self.model_arch)
        # return model_class()
        
        self.model = getattr(m, self.model)().to(self.device)
        self.optimizer = getattr(optim, self.optimizer)(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = getattr(lrs, self.scheduler)(self.optimizer, mode='min', factor=0.1, patience=3, threshold=0.1)
        self.criterion = getattr(nn, self.criterion)()
        
    def load_model_state_dict(self) -> None:
        """
        This function loads in the state dict of a model if provided.
        """
        if self.transfer_learning_path != 'None':
            state_dict = torch.load(self.transfer_learning_path)
            self.model.load_state_dict(state_dict)
            self.model = self.model.eval() 
            self.model.to(self.device)
    
    def epoch_loop(self) -> None: 
        """
        This function loops through the training and testing functions by the number of epochs iterations.
        The train and test function are used back to back per epoch to optimize then perfom a second evalution on the perfomance of the model. 
        """
        
        self.logger.info(f'Model training and testing: {self.model.__class__.__name__}')
        # self.logger.info(f'Looking for the feature: {self.feature}')  
        print(f'Model testing and validating: {self.model.__class__.__name__}')     
        # print(f'Looking for the feature: {self.feature}')  
        
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
        
            inputs = inputs.unsqueeze(1).to(self.device, dtype=torch.float32)
            attributes = {key: value.to(self.device).float() for key, value in attributes.items()}

            self.optimizer.zero_grad()
            
            with autocast(enabled=False):
                print(inputs.shape)
                self.logger.info(inputs.shape)
                score = self.model(inputs, attributes[self.camera_length], attributes[self.photon_energy])
                truth = attributes[self.peak].reshape(-1, 1).float().to(self.device)
                loss = self.criterion(score, truth)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss_train += loss.item()

            predictions = (torch.sigmoid(score) > 0.5).long()
            
            # print(f'prediction = {predictions}')
            # self.logger.info(f'prediction = {predictions}')
            
            # print(f'truth = {truth}')
            # self.logger.info(f'truth = {truth}')
            
            accuracy_train += (predictions == truth).float().sum()
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
        This function test the model in evaluation mode and prints the loss and accuracy of the testing sets per epoch.
        """
        
        running_loss_test, accuracy_test, predictions, total = 0.0, 0.0, 0.0, 0.0
        
        self.model.eval()
        with torch.no_grad():
            for inputs, attributes in self.test_loader:
                
                inputs = inputs.unsqueeze(1).to(self.device, dtype=torch.float32)
                attributes = {key: value.to(self.device, dtype=torch.float32) for key, value in attributes.items()}


                with autocast(enabled=False):
                    score = self.model(inputs, attributes[self.camera_length], attributes[self.photon_energy])
                    truth = attributes[self.peak].reshape(-1, 1).float().to(self.device)
                    loss = self.criterion(score, truth)
            
                running_loss_test += loss.item()  # Convert to Python number with .item()
                
                predictions = (torch.sigmoid(score) > 0.5).long()
                    
                accuracy_test += (predictions == truth).float().sum()
                total += torch.numel(truth)

        loss_test = running_loss_test/len(self.test_loader)
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
        plt.title(f'Loss and Accuracy for with {self.model.__class__.__name__}')
        plt.legend(['accuracy train','accuracy test','loss train','loss test'])
        
        if path != None:
            now = datetime.datetime.now()
            formatted_date_time = now.strftime("%m%d%y-%H:%M")
            path = path + '/' + formatted_date_time + '-' + 'training_loss_accuracy.png'
            plt.savefig(path)
            
        plt.show()
        
    def save_model(self, path:str) -> None:
        """
        This function saves the model's state_dict to a specified path. This can be used to load the trained model later.
        Save as .pt file.

        Args:
            path (str): Path to save the model's state_dict.
        """

        torch.save(self.model.state_dict(), path)
        
    def get_model(self) -> nn.Module:
        """
        This function returns the trained model obkect. This is to get the trained model to evaluation without having to load the state dict. 

        Returns:
            nn.Module: The trained model object. 
        """
        return self.model
