import torch
import numpy as np
from scipy.constants import h, c, e

from . import train_model
from . import run_model
from . import conf

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
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def reshape_input_data(data_array: np.ndarray) -> np.ndarray:
        """
        This function reshapes the input data array to the correct dimensions for the model.
        
        Args:
            data_array (np.ndarray): The input data array to be reshaped.
        
        Returns:
            np.ndarray: The reshaped input data array.
        """
        crop_height, crop_width = conf.eiger_4m_image_size
        
        batch_size, height, width  = data_array.shape
        
        # Calculate the center of the images
        center_y, center_x = height // 2, width // 2
        
        # Calculate the start and end indices for the crop
        start_y = center_y - crop_height // 2
        end_y = start_y + crop_height
        start_x = center_x - crop_width // 2
        end_x = start_x + crop_width
        
        data_array = data_array[:, start_y:end_y, start_x:end_x]
        
        print(f'Reshaped input data array from {height}, {width} to {crop_width}, {crop_height}.')
        
        return data_array
    
    @staticmethod
    def incident_photon_wavelength_to_energy(wavelength: float) -> float:
        """
        This function takes in the wavelength of an incident photon and returns the energy of the photon on eV (electron volts).

        Args:
            wavelength (float): The wavelength of the incident photon in Angstroms.

        Returns:
            float: _description_
        """
        
        energy_J = h * c / wavelength
        energy_eV = energy_J / e
        
        return energy_eV