import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from pkg import *
from torch.cuda.amp import GradScaler, autocast
import datetime


class ModelEvaluation:
    
    def __init__(self, cfg: dict, attributes: dict, trained_model: nn.Module):
        self.logger = logging.getLogger(__name__)
        
        self.test_loader = cfg['test data']
        self.batch_size = cfg['batch size']
        self.device = cfg['device']
        self.model = trained_model
        
        self.camera_length = attributes['camera length']
        self.photon_energy = attributes['photon energy']
        self.peak = attributes['peak']
        
        self.cm = np.zeros((self.classes,self.classes), dtype=int)
        self.all_labels = []
        self.all_predictions = []
        self.classification_report_dict = {}


    def run_testing_set(self) -> None:
        """ 
        Evaluates model post training. 
        """
        
        self.model.eval()
        with torch.no_grad():
            for inputs, attributes in self.test_loader:
                
                inputs = inputs.unsqueeze(0).unsqueeze(0).to(self.device)

                with autocast():
                    score = self.model(inputs, attributes[self.camera_length], attributes[self.photon_energy])
                             
                    truth = attributes[self.peak].reshape(-1, 1).float().to(self.device)
                                    
                predictions = (torch.sigmoid(score) > 0.5).long()
                
                self.all_labels.extend(torch.flatten(truth.cpu()))
                self.all_predictions.extend(torch.flatten(predictions))

        # No need to reshape - arrays should already be flat
        self.all_labels = np.array(self.all_labels)
        self.all_predictions = np.array(self.all_predictions)
        
    def make_classification_report(self) -> None:
        """
        This function creates a classification report for the model.
        """
        
        self.classification_report_dict = classification_report(self.all_labels, self.all_predictions, labels=self.labels, output_dict=True)
        [print(f"{key}: {value}") for key, value in self.classification_report_dict.items()]
        [self.logger.info(f"{key}: {value}") for key, value in self.classification_report_dict.items()]
        
    def get_classification_report(self) -> dict:
        """
        This function returns the classification report for the model.
        """
        return self.classification_report_dict

        
    def plot_confusion_matrix(self, path:str = None) -> None:
        """ 
        This function plots the confusion matrix of the testing set.
        """
        
        self.cm = confusion_matrix(self.all_labels, self.all_predictions, labels=self.labels, normalize='true')

        # Plotting the confusion matrix
        plt.matshow(self.cm, cmap="Blues")
        plt.title(f'CM for {self.feature} {self.model.__class__.__name__}')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if path != None:
            now = datetime.datetime.now()
            formatted_date_time = now.strftime("%m%d%y-%H:%M")
            path = path + '/' + formatted_date_time + '-' + 'confusion_matrix.png'
            plt.savefig(path)
            
        plt.show()


    def get_sconfusion_matrix(self) -> np.ndarray:
        """ 
        This function returns the confusion matrix of the testing set.
        """
        return self.cm