import torch
import datetime
import os 
from . import models
from . import utils as u
from . import conf
import inspect
import importlib    


class RunModel:
    
    def __init__(self, cfg: dict, attributes: dict) -> None:
        """
        Initialize the RunModel class with model architecture, model path, output list path, h5 file paths, and device.

        Args:
            cfg (dict): Dictionary containing important information for training including: data loaders, batch size, training device, number of epochs, the optimizer, the scheduler, the criterion, the learning rate, and the model class. Everything besides the data loaders and device are arguments in the sbatch script.
            attributes (dict): Dictionary containing the names of the metadata contained in the h5 image files. These names could change depending on whom created the metadata, so the specific names are arguments in the sbatch script. 
        """
        self.device = cfg['device']
        self.model_arch = cfg['model']
        self.model_path = cfg['model_path']
        self.save_output_list = cfg['save_output_list']
        
        self.camera_length = conf.camera_length_key
        self.photon_energy = conf.photon_energy_key
        
        self.list_containing_peaks = []
        self.list_not_containing_peaks = []
        
        self.model = None
    
    def make_model_instance(self) -> None:
        """
        Create an instance of the model class specified by the model architecture.

        Returns:
            class instance: An instance of the specified model class.
        """
        try:
            self.model = getattr(models, self.model_arch)()
            print(f'Model object has been created: {self.model.__class__.__name__}')
        except AttributeError:
            print(f"Error: Model '{self.model_arch}' not found in the module.")
            print(f'Available models: {inspect.getmembers(models, inspect.isclass)}')
        except TypeError:
            print(f"Error: '{self.model_arch}' found in module is not callable.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    def load_model(self) -> None:
        """
        Load the state dictionary into the model class and prepare it for evaluation.
        """
        try:
            state_dict = torch.load(self.model_path)
            self.model.load_state_dict(state_dict)
            self.model.eval() 
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
        
    def classify_data(self, data_loader) -> None:
        """
        Classify the input data using the model and segregate the data based on the classification results.
        """
        print('Starting classification...')
        try:
            with torch.no_grad():
                for inputs, attributes, paths in data_loader:
                
                    inputs = inputs.to(self.device, dtype=torch.float32)
                    attributes = {key: value.to(self.device, dtype=torch.float32) for key, value in attributes.items()}
                    score = self.model(inputs, attributes[self.camera_length], attributes[self.photon_energy])
                    prediction = (torch.sigmoid(score) > 0.5).long()
                    
                    assert len(prediction) == len(paths), "Prediction and paths length mismatch."

                    # Segregate data based on prediction
                    for pred, path in zip(prediction, paths):
                        if pred.item() == 1:
                            self.list_containing_peaks.append(path)
                            print(f'Classified as containing peaks: {path}')
                        elif pred.item() == 0:
                            self.list_not_containing_peaks.append(path)
                            print(f'Classified as not containing peaks: {path}')

        except Exception as e:
            print(f"An unexpected error occurred while classifying data: {e}")
                
    def get_classification_results(self) -> tuple:
        """
        Get the classification results from the model.

        Returns:
            tuple: A tuple containing two lists - one with file paths containing peaks and one without peaks.
        """
        return (self.list_containing_peaks, self.list_not_containing_peaks)
    
    def create_model_output_lst_files(self) -> None:
        """
        Create .lst files for the classified data based on the model's predictions.
        """
        try:
            now = datetime.datetime.now()
            formatted_date_time = now.strftime("%m%d%y-%H%M")
            print(f'Formatted date and time: {formatted_date_time}')
            filename_peaks = f"found_peaks-{formatted_date_time}.lst"
            print(f'Filename peaks: {filename_peaks}')
            file_path_peaks = os.path.join(self.save_output_list, filename_peaks)
            print(f'File path peaks: {file_path_peaks}')
            
            filename_no_peaks = f"no_peaks-{formatted_date_time}.lst"
            file_path_no_peaks = os.path.join(self.save_output_list, filename_no_peaks)

            try:
                with open(file_path_peaks, 'w') as file:
                    for item in self.list_containing_peaks:
                        file.write(f"{item}\n")
                print(f"Created .lst file for predicted peak files. There are {len(self.list_containing_peaks)} files containing peaks.")
            except Exception as e:
                print(f"An error occurred while writing to {file_path_peaks}: {e}")

            try:
                with open(file_path_no_peaks, 'w') as file:
                    for item in self.list_not_containing_peaks:
                        file.write(f"{item}\n")
                print(f"Created .lst file for predicted empty files. There are {len(self.list_not_containing_peaks)} files without peaks.")
            except Exception as e:
                print(f"An error occurred while writing to {file_path_no_peaks}: {e}")

        except Exception as e:
            print(f"An unexpected error occurred while creating .lst files: {e}")
        
    def output_verification(self, size: int, events: int) -> None:
        """
        Verify that the number of input file paths matches the sum of the output file paths.

        This function compares the size of the input file path list to the sum of the sizes of the two output file path lists and logs the result.
        
        Args:
            size (int): The size of the input file path queue.
        """
        if size == len(self.list_containing_peaks) // events + len(self.list_not_containing_peaks) // events:
            print("There is the same amount of input files as output files.")
        else:
            print("OUTPUT VERIFICATION FAILED: The input paths do not match the output paths.")           
            print(f'Input H5 files: {size}\nOutput peak files: {len(self.list_containing_peaks)}\nOutput empty files: {len(self.list_not_containing_peaks)}')
