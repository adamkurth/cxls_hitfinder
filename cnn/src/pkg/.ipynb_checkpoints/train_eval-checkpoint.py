import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from pkg import *
from torch.cuda.amp import GradScaler, autocast



"""
This file is for creating the new train_eval file using the new inputs and class structure for each specific feature. git 
"""


class TrainTestModels:
    """ 
    This class trains, tests, and plots the loss, accuracy, and confusion matrix of a model.
    There are two methods for training: test_model_no_freeze and test_model_freeze.
    """
    
    def __init__(self, cfg: dict, feature_class: object) -> None:
        """ 
        Takes the arguments for training and testing and makes them available to the class.

        Args:
            cfg: dict which holds shared configuration details.
            feature_class: class which holds the feature specific configuration details.
        """
        self.train_loader, self.test_loader = cfg['loader']
        self.optimizer = cfg['optimizer']
        self.device = cfg['device']
        self.batch = cfg['batch_size']
        self.scheduler = cfg['scheduler']
        
        self.feature_class = feature_class
        self.model = feature_class.get_model().to(self.device)
        self.criterion = feature_class.get_criterion()
        self.classes = feature_class.get_classes()
        self.feature = feature_class.get_feature()
        self.labels = feature_class.get_labels()
        self.learning_rate = feature_class.get_learning_rate()
        self.save_path = feature_class.get_save_path()
        self.epochs = feature_class.get_epochs()
        
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = self.scheduler(self.optimizer, mode='min', factor=0.1, patience=3, threshold=0.1)
        
        self.plot_train_accuracy = np.zeros(self.epochs)
        self.plot_train_loss = np.zeros(self.epochs)
        self.plot_test_accuracy = np.zeros(self.epochs)
        self.plot_test_loss = np.zeros(self.epochs)
        self.cm = np.zeros((self.classes,self.classes), dtype=int)
        
        self.logger = logging.getLogger(__name__)
        self.scaler = GradScaler()
        
        self.fpr = 0
        self.tpr = 0
        self.roc_auc = 0

    def train(self, epoch:int) -> None:
        
        """
        This function trains the model and prints the loss and accuracy of the training sets per epoch.
        """
        
        running_loss_train, accuracy_train, predictions, total_predictions = 0.0, 0.0, 0.0, 0.0

        self.model.train()
        for inputs, labels, attributes in self.train_loader:
            inputs[0] = inputs[0].unsqueeze(1)
            for i in inputs:
                # inputs, labels = inputs.to(self.device), labels.to(self.device)
                i = i.to(self.device)

                self.optimizer.zero_grad()
                
                with autocast():
                    score = self.model(i)
                    image_attribute = attributes[self.feature]
                    
                    self.feature_class.format_image_attributes(image_attribute)
                    image_attribute = self.feature_class.get_formatted_image_attribute().to(self.device)
                    
                    # print("Input shape:", score.shape)
                    # print("----- score : ", score)
                    # print("Target shape:", image_attribute.shape)
                    # print("----- target : ", image_attribute)

                    
                    loss = self.criterion(score, image_attribute)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss_train += loss.item()

                self.feature_class.format_prediction(score)
                predictions = self.feature_class.get_formatted_prediction()
                        
                accuracy_train += (predictions == image_attribute.to(self.device)).float().sum()
                total_predictions += torch.numel(image_attribute)
            
        loss_train = running_loss_train / len(self.train_loader)  # Assuming you want to average over all batches
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
            for inputs, labels, attributes in self.test_loader:
                inputs, labels = inputs[1].to(self.device), labels.to(self.device)

                with autocast():
                    score = self.model(inputs)
                                                        
                    image_attribute = attributes[self.feature]
                    
                    self.feature_class.format_image_attributes(image_attribute)
                    image_attribute = self.feature_class.get_formatted_image_attribute().to(self.device)
                        
                    loss = self.criterion(score, image_attribute)
            
                running_loss_test += loss.item()  # Convert to Python number with .item()
                
                self.feature_class.format_prediction(score)
                predictions = self.feature_class.get_formatted_prediction()
                    
                accuracy_test += (predictions == image_attribute.to(self.device)).float().sum()
                total += torch.numel(image_attribute)

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

    def plot_confusion_matrix(self, path:str = None) -> None:
        """ 
        This function plots the confusion matrix of the testing set.
        """
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels, attributes in self.test_loader:  # Assuming self.loader[1] is the testing data loader
                inputs, labels = inputs[1].to(self.device), labels.to(self.device)

                with autocast():
                    score = self.model(inputs)

                # Flatten and append labels to all_labels
                image_attribute = attributes[self.feature].reshape(-1)
                self.feature_class.format_image_attributes(image_attribute)
                image_attribute = self.feature_class.get_formatted_image_attribute().cpu().numpy()
                all_labels.extend(image_attribute)
                                
                self.feature_class.format_prediction(score)
                predictions = self.feature_class.get_formatted_prediction().cpu()
                                        
                all_predictions.extend(predictions)

        # No need to reshape - arrays should already be flat
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        # Compute confusion matrix
        # print(f'-- Labels      : {all_labels}')
        # print(f'-- Predictions : {all_predictions}')
        self.cm = confusion_matrix(all_labels, all_predictions, labels=self.labels, normalize='true')

        # Plotting the confusion matrix
        plt.matshow(self.cm, cmap="Blues")
        plt.title(f'Confusion Matrix for {self.feature} with {self.model.__class__.__name__}')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if path != None:
            plt.savefig(path)
            
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
            
            print(f"-- learning rate : {self.scheduler.get_last_lr()}") 
            
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
    
    def save_model(self, path:str=None) -> None:
        """
        This function saves the model's state_dict to a specified path. This can be used to load the trained model later.
        Save as .pt file.
        root: /cnn/models
        Args:
            path (str): Path to save the model's state_dict.
        """
        if path != None:
            torch.save(self.model.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), self.save_path)
            
    def find_optimal_threshold(self) -> None:
        """
        This function finds the optimal threshold for the model.
        """
        
        self.model.eval()
        probabilities = []
        true_labels = []
        
        with torch.no_grad():
            for inputs, labels, attributes in self.train_loader:
                inputs, labels = inputs[1].to(self.device), labels.to(self.device)
                
                image_attribute = attributes[self.feature]
                self.feature_class.format_image_attributes(image_attribute)
                image_attribute = self.feature_class.get_formatted_image_attribute().to(self.device)
                
                score = self.model(inputs)
                
                probabilities.extend(torch.sigmoid(score).cpu().numpy())
                true_labels.extend(image_attribute.cpu().numpy())
                
        probabilities = np.array(probabilities)
        true_labels = np.array(true_labels)
        
        self.fpr, self.tpr, thresholds = roc_curve(true_labels, probabilities)
        self.roc_auc = auc(self.fpr, self.tpr)
        
        print(f'--- ROC AUC : {self.roc_auc}')
        
        optimal_idx = np.argmax(self.tpr - self.fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f'--- Optimal threshold : {optimal_threshold}')
        
        self.feature_class.set_threshold(optimal_threshold)
        
        print(f'--- confirm threshold : {self.feature_class.get_threshold()}')
                
    def plot_roc_curve(self, path:str = None) -> None:
        """
        This function plots the ROC curve of the model.
        """
        
        plt.figure()
        plt.plot(self.fpr, self.tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % self.roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        if path != None:
            plt.savefig(path)
            
        plt.show()
