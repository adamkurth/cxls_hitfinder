import torch

from . import train_model
from . import run_model

class CommonFunctions:
    
    def __init__(self) -> None:
        pass 
    
    
    def load_model_state_dict(self) -> None:
        """
        This function loads in the state dict of a model if provided.
        """
        if self.model_path != 'None':
            try:
                state_dict = torch.load(self.transfer_learning_path)
                self.model.load_state_dict(state_dict)
                self.model = self.model.eval() 
                self.model.to(self.device)
                
                print(f'The model state dict has been loaded into: {self.model.__class__.__name__}')
                
            except FileNotFoundError:
                print(f"Error: The file '{self.transfer_learning_path}' was not found.")
            except torch.serialization.pickle.UnpicklingError:
                print(f"Error: The file '{self.transfer_learning_path}' is not a valid PyTorch model file.")
            except RuntimeError as e:
                print(f"Error: There was an issue loading the state dictionary into the model: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        else:
            print(f'There is no model state dict to load into: {self.model.__class__.__name__}')
            

class SpecialCaseFunctions:
    
    pass