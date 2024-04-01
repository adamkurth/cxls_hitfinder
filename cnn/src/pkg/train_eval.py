import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class TrainTestModels:
    """ 
    This class trains, tests, and plots the loss, accuracy, and confusion matrix of a model.
    There are two methods for training: test_model_no_freeze and test_model_freeze.
    """
    
    def __init__(self, model: object, loader: list, criterion, optimizer, device, cfg: dict) -> None:
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
        self.threshold = 0.5
        self.logger = logging.getLogger(__name__)


    def train(self, epoch:int) -> None:
        """
        This function trains the model without freezing the parameters in the case of transfer learning.
        This will print the loss and accuracy of the training sets per epoch.
        """

        # print(f'Model training: {self.model.__class__.__name__}')

        running_loss_train = accuracy_train = predictions = total_predictions = 0.0

        self.model.train()
        for inputs, labels in self.loader[0]:  # Assuming self.loader[0] is the training data loader
            peak_images, overlay_images = inputs
            peak_images, overlay_images, labels = peak_images.to(self.device), overlay_images.to(self.device), labels.to(self.device)

            temp = np.count_nonzero(np.array(labels.cpu()))
            print(temp)

            self.optimizer.zero_grad()
            score = self.model(peak_images)
            predictions = (torch.sigmoid(score) > self.threshold).long()  
            truth = (torch.sigmoid(labels) > self.threshold).long()
            predictions = predictions.any().item()
            truth = truth.any().item()
            predictions = torch.tensor([float(predictions)], requires_grad=True)
            truth = torch.tensor([float(truth)])
            # loss = self.criterion(score, labels)
            loss = self.criterion(predictions, truth)
            loss.backward()
            self.optimizer.step()
            running_loss_train += loss.item()  
            # predictions = (torch.sigmoid(score) > self.threshold).long()  
            # truth = (torch.sigmoid(labels) > self.threshold).long()
            # accuracy_train += (predictions == labels).float().sum()
            
            accuracy_train += (predictions == truth).float().sum()
            # total_predictions += np.prod(labels.shape)
            # total_predictions += torch.numel(labels)
            total_predictions += 1
            
        loss_train = running_loss_train / self.batch
        self.plot_train_loss[epoch] = loss_train
        self.logger.info(f'Train loss: {loss_train}')
        print(f'Train loss: {loss_train}')

        # If you want to uncomment these lines, make sure the calculation of accuracy_train is corrected as follows:
        accuracy_train /= total_predictions
        self.plot_train_accuracy[epoch] = accuracy_train
        self.logger.info(f'Train accuracy: {accuracy_train}')
        print(f'Train accuracy: {accuracy_train}')
            
    # def test_freeze(self) -> None:
    #     """ 
    #     This function trains the model with freezing the parameters of in the case of transfer learning.
    #     This will print the loss and accuracy of the testing sets per epoch.
    #     WIP
    #     """   
    #     pass
        
    def test(self, epoch:int) -> None:
        """ 
        This function test the model and prints the loss and accuracy of the testing sets per epoch.
        """
        # print(f'Model testing: {self.model.__class__.__name__}')
            
        running_loss_test = accuracy_test = predicted = total = 0.0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.loader[1]:
                peak_images, _ = inputs
                peak_images = peak_images.to(self.device)
                labels = labels.to(self.device)
                score = self.model(peak_images)
                predicted = (torch.sigmoid(score) > self.threshold).long()  # Assuming 'score' is the output of your model
                truth = (torch.sigmoid(labels) > self.threshold).long()
                predicted = predicted.any().item()
                truth = truth.any().item()
                predicted = torch.tensor([float(predicted)])
                truth = torch.tensor([float(truth)])
                
                loss = self.criterion(predicted, truth)
                # loss = self.criterion(score, labels)
                
                
                running_loss_test += loss.item()  # Convert to Python number with .item()
                # predicted = (torch.sigmoid(score) > self.threshold).long()  # Assuming 'score' is the output of your model
                # truth = (torch.sigmoid(labels) > self.threshold).long()
                
                # accuracy_test += (predicted == labels).float().sum()
                accuracy_test += (predicted == truth).float().sum()
                # total += np.prod(labels.shape)
                # total += torch.numel(labels)
                total += 1

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

        with torch.no_grad():
            for inputs, label in self.loader[1]:
                peak_images, _ = inputs
                peak_images = peak_images.to(self.device)
                label = label.to(self.device)

                score = self.model(peak_images).squeeze()
                predictions = (torch.sigmoid(score) > self.threshold).long()
                truth = (torch.sigmoid(label) > self.threshold).long()
                predictions = predictions.any().item()
                truth = truth.any().item()

                # all_labels.extend(label.cpu().numpy().flatten())
                # all_labels.extend(truth.cpu().numpy().flatten()) 
                # all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.append(truth)
                all_predictions.append(predictions)

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        self.cm = confusion_matrix(all_labels, all_predictions, normalize='true')

        plt.matshow(self.cm ,cmap="Blues")
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
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
    