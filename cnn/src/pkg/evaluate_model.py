import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from pkg import *
from torch.cuda.amp import GradScaler, autocast


class Model_Evaluation:
    
    def __init__(self, cfg: dict, feature_class: object) -> None:

        self.train_loader, self.test_loader = cfg['loader']
        self.device = cfg['device']
        
        self.feature_class = feature_class
        self.model = feature_class.get_model().to(self.device)

        self.classes = feature_class.get_classes()
        self.feature = feature_class.get_feature()
        self.labels = feature_class.get_labels()

        self.save_path = feature_class.get_save_path()

        self.cm = np.zeros((self.classes,self.classes), dtype=int)
        self.all_labels = []
        self.all_predictions = []
        self.classification_report_dict = {}
        
        self.logger = logging.getLogger(__name__)
        self.scaler = GradScaler()
        
    def load_model(self, model_state_path) -> None:
        model_path = model_state_path
        state_dict = torch.load(model_path)
        model = self.feature_class.get_model()
        model.load_state_dict(state_dict)
        self.model = model.eval() 
        self.model.to(self.device)
   

    def run_model(self) -> None:
        """ 
        Evaluates model post training. 
        """

        with torch.no_grad():
            for inputs, labels, attributes in self.test_loader:  # Assuming self.loader[1] is the testing data loader
                # inputs, labels = inputs[1].to(self.device), labels.to(self.device)
                # inputs[0] = inputs[0].unsqueeze(1)
                # labels = labels.to(self.device)
                # inputs[0], inputs[1] = inputs[0].to(self.device), inputs[1].to(self.device)
                model_input = inputs[1].to(self.device)

                with autocast():
                    if self.feature == 'peak':
                        score = self.model(model_input, attributes['clen'], attributes['photon_energy'])
                    else:
                        score = self.model(model_input)

                # Flatten and append labels to all_labels
                if self.feature != 'peak_location':                           
                    image_attribute = attributes[self.feature]
                    self.feature_class.format_image_attributes(image_attribute)
                    true_value = self.feature_class.get_formatted_image_attribute().to(self.device)
                else:
                    true_value = labels.to(self.device)
                self.all_labels.extend(torch.flatten(true_value.cpu()))
                                
                self.feature_class.format_prediction(score)
                predictions = self.feature_class.get_formatted_prediction().cpu()
                                        
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
            plt.savefig(path)
            
        plt.show()


    def get_confusion_matrix(self) -> np.ndarray:
        """ 
        This function returns the confusion matrix of the testing set.
        """
        return self.cm