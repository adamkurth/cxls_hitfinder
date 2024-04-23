import torch
import torch.nn as nn
import torch.optim as optim
# from torchviz import make_dot
import pkg.models as m
from pkg.functions import get_counts_weights, save_h5
from scipy.signal import find_peaks
import numpy as np

class Get_Configuration_Details:
    """
    This class is used to get configuration details for the different properties we are looking for. 
    """
    def __init__(self):
        self.threshold = None  
        self._formatted_image_attribute = None
        self._formatted_prediction = None

    def get_model(self) -> nn.Module:
        return self._model
    
    def get_criterion(self) -> nn.Module:
        return self._criterion
    
    def get_feature(self) -> str:
        return self._feature
    
    def get_classes(self) -> int:
        return self._classes
    
    def get_labels(self) -> list:
        return self._labels
    
    def format_image_attributes(self, image_attribute: torch.Tensor) -> None:
        """
        Formats the image attributes based on the attribute mapping.

        Args:
            image_attribute (torch.Tensor): The image attribute tensor.
        """
        if not self._attribute_mapping:  # Directly use the image attribute if no mapping is needed
            self._formatted_image_attribute = image_attribute.reshape(-1, 1).float()
            return
        # Use the attribute mapping for transformation
        holder = torch.zeros_like(image_attribute, dtype=torch.long)
        for original_value, new_value in self._attribute_mapping.items():
            holder[image_attribute == original_value] = new_value
        self._formatted_image_attribute = holder
        
    def format_prediction(self, score: torch.Tensor) -> None:
        """
        Formats the prediction based on the score and threshold.

        Args:
            score (torch.Tensor): The score tensor.
            threshold (float, optional): Peak threshold. Defaults to None.
        """
        if self._threshold !=  None:  # For binary classification cases
            self._formatted_prediction = (torch.sigmoid(score) > self._threshold).long()
        else:  # For multi-class cases
            _, predicted = torch.max(score, 1)
            self._formatted_prediction = predicted
            
    def get_formatted_image_attribute(self) -> torch.Tensor:
        return self._formatted_image_attribute
    
    def get_formatted_prediction(self) -> torch.Tensor:
        return self._formatted_prediction
    
    def get_learning_rate(self) -> float:
        return self._learning_rate
    
    def get_loss_weights(self) -> torch.Tensor:
        return self._weights
    
    def get_save_path(self) -> str:
        return self._save_path
    
    def get_model_diagram(self, filename:str, file_path:str, device) -> None:
        x = torch.randn(1, 1, 2163, 2069).to(device)
        vis_graph = make_dot(self._model(x), params=dict(self._model.named_parameters()))
        vis_graph.render(filename=filename, directory=file_path, format='png', view=True)
        # vis_graph.view()  
    
    def get_epochs(self) -> int:
        return self._epochs
    
    def get_threshold(self) -> float:
        return self._threshold
    
    def set_threshold(self, threshold: float) -> None:
        self._threshold = threshold
        

    
class Peak_Detection_Configuration(Get_Configuration_Details):
    """
    This class is the specific configureation for the peak detection model.

    Args:
        Get_Configuration_Details (class): Class used for retreiving configuration details.
    """
    def __init__(self, paths, datasets, device, save_path=None): 
        super().__init__()
        # self._model = m.Multi_Class_CNN2(output_channels=1)
        # self._model = m.MultiClassCNN(output_channels=1)
        # self._model = m.ResNetBinaryClassifier()
        self._model = m.DualInputCNN()
        self._feature = "peak"
        self._classes = 2
        self._labels = [0,1]
        self._attribute_mapping = {}
        self._threshold = 0.5
        self._learning_rate = 0.0001
        self._weights = get_counts_weights(paths, datasets, self._classes)
        self._criterion = nn.BCEWithLogitsLoss(pos_weight=self._weights.to(device))
        self._save_path = save_path
        self._epochs = 10


class Photon_Energy_Configuration(Get_Configuration_Details):
    """
    This class is the specific configureation for the photon energy model.

    Args:
        Get_Configuration_Details (class): Class used for retreiving configuration details.
    """ 
    def __init__(self, paths, datasets, device, save_path=None): 
        super().__init__()
        # self._model = m.Multi_Class_CNN1()
        self._model = m.MultiClassCNN()
        self._feature = "photon_energy"
        self._classes = 3
        self._labels = [0,1,2]
        self._attribute_mapping = {
            6e3: 0,
            7e3: 1,
            8e3: 2
        }
        self._threshold = None
        self._learning_rate = 0.000001
        self._weights = get_counts_weights(paths, datasets, self._classes)
        self._criterion = nn.CrossEntropyLoss(weight=self._weights.to(device))
        self._save_path = save_path
        self._epochs = 5

        
        
class Camera_Length_Configureation(Get_Configuration_Details):
    """
    This class is the specific configureation for the camera length  model.

    Args:
        Get_Configuration_Details (class): Class used for retreiving configuration details.
    """
    def __init__(self, paths, datasets, device, save_path=None): 
        super().__init__()
        self._model = m.MultiClassCNN()
        self._feature = "clen"
        self._classes = 3
        self._labels = [0,1,2]
        self._attribute_mapping = {
            0.15: 0,
            0.25: 1,
            0.35: 2
        }
        self._threshold = None
        self._learning_rate = 0.00001
        self._weights = get_counts_weights(paths, datasets, self._classes)
        self._criterion = nn.CrossEntropyLoss(weight=self._weights.to(device))
        self._save_path = save_path
        self._epochs = 5