import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report


class RunModel:
    def __init__(self, model_config_class, device):
        self.model_conf = model_config_class
        model_path = model_conf.get_save_path()
        model_state_dict = torch.load(model_path)
        model = model_conf.get_model()
        model.load_state_dict(model_state_dict)
        self.model = model.eval() 
        self.model.to(device)
        
    