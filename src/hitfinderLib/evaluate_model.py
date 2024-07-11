import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
import datetime
from torch.utils.data import DataLoader

from . import conf

class ModelEvaluation:
    
    def __init__(self, cfg: dict, attributes: dict, trained_model: nn.Module, testing_data: DataLoader) -> None:
        """
        This constructor breaks out important dictonaries, takes in the trained model, creates a logger object, and creates parameters to store evaluation metrics. 

        Args:
            cfg (dict): Dictionary containing important information for training including: data loaders, batch size, training device, number of epochs, the optimizer, the scheduler, the criterion, the learning rate, and the model class. 
            Everything besides the data loaders and device are arguments in the sbatch script.
            Not all parameters are relevant for evaluation and therefore are not given variables. 
            attributes (dict): Dictionary containing the names of the metadata contained in the h5 image files. These names could change depending on whom created the metadata, so the specific names are arguments in the sbatch script. 
            trained_model (nn.Module): This is a trained model taken from the training class. 
        """
        
        self.test_loader = testing_data
        self.batch_size = cfg['batch size']
        self.device = cfg['device']
        self.model = trained_model
        
        self.camera_length = conf.camera_length_key
        self.photon_energy = conf.photon_energy_key
        self.peak = conf.present_peaks_key
        
        self.cm = None
        self.all_labels = []
        self.all_predictions = []
        self.classification_report_dict = {}


    def run_testing_set(self) -> None:
        """ 
        This function runs the trained model in evaluation mode.
        This function creates arrays of labels and predictions to compare against each other for metrics. 
        """
        print(f'Running evaluation on model: {self.model.__class__.__name__}')
        self.model.eval()
        try:
            with torch.no_grad():
                for inputs, attributes, _ in self.test_loader:
                    
                    # inputs = inputs.unsqueeze(1).to(self.device, dtype=torch.float32)
                    inputs = inputs.to(self.device, dtype=torch.float32)
                    attributes = {key: value.to(self.device, dtype=torch.float32) for key, value in attributes.items()}

                    with autocast(enabled=False):
                        score = self.model(inputs, attributes[self.camera_length], attributes[self.photon_energy])
                        truth = attributes[self.peak].reshape(-1, 1).float().to(self.device)
                                    
                    predictions = (torch.sigmoid(score) > 0.5).long()
                    self.all_labels.extend(torch.flatten(truth.cpu()))
                    self.all_predictions.extend(torch.flatten(predictions.cpu()))
                    
            # No need to reshape - arrays should already be flat
            self.all_labels = np.array(self.all_labels)
            self.all_predictions = np.array(self.all_predictions)
        except RuntimeError as e:
            print(f"RuntimeError during training: {e}")  
        except AttributeError as e:
            print(f"AttributeError during training: {e}")
        except TypeError as e:
            print(f"TypeError during training: {e}")    
        except Exception as e:
            print(f"An unexpected error occurred during evaluation: {e}")
        
    def make_classification_report(self) -> None:
        """
        This function creates a classification report for the model and prints it.
        """
        try:
            print('Creating classification report...')
            self.classification_report_dict = classification_report(self.all_labels, self.all_predictions, output_dict=True)
            print('Classification Matrix: ')
            [print(f"{key}: {value}") for key, value in self.classification_report_dict.items()]
        except Exception as e:
            print(f"An error occurred while creating the clasification report: {e}")       
        
    def get_classification_report(self) -> dict:
        """
        This function returns the classification report for the model.
        
        Returns:
            dict: The classification report in the form of a dictionary. 
        """
        return self.classification_report_dict

        
    def plot_confusion_matrix(self, path:str = None) -> None:
        """ 
        This function plots the confusion matrix of the testing set.
        The values in this matrix are done so that the rows total to 1. 
        """
        
        try:
            print('Creating confusion matrix...')
            self.cm = confusion_matrix(self.all_labels, self.all_predictions, normalize='true')
        except Exception as e:
            print(f"An error occurred while creating the confusion matrix: {e}")       
        # Plotting the confusion matrix
        try:
            plt.matshow(self.cm, cmap="Blues")
            plt.title(f'CM for {self.model.__class__.__name__}')
            plt.colorbar()
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if path != None:
                now = datetime.datetime.now()
                formatted_date_time = now.strftime("%m%d%y-%H%M")
                path = path + '/' + formatted_date_time + '-' + 'confusion_matrix.png'
                plt.savefig(path)
                
                print(f'Confustion matrix saved to: {path}')
        except Exception as e:
            print(f"An error occurred while plotting confusion matrix: {e}")



    def get_confusion_matrix(self) -> np.ndarray:
        """ 
        This function returns the confusion matrix of the testing set.
        
        Returns:
            np.darray: The numpy array of the values in the confusion matrix.
        """
        return self.cm