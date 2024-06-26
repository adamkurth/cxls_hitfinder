import torch
import datetime
import os 
from . import models as m

class RunModel:
    
    def __init__(self, model_arch: str, model_path: str, save_output_list: str, h5_file_paths: list, device: torch.device) -> None:
        """
        Initialize the RunModel class with model architecture, model path, output list path, h5 file paths, and device.

        Args:
            model_arch (str): The architecture name of the model.
            model_path (str): The file path to the saved model state dictionary.
            save_output_list (str): The directory path where the output list files will be saved.
            h5_file_paths (list): List of h5 file paths to process.
            device (torch.device): The device (CPU or GPU) on which the model will run.
        """
        self.device = device
        
        self.model_arch = model_arch
        self.model_path = model_path
        self.save_output_list = save_output_list
        self.h5_file_paths = h5_file_paths
        
        self.model = self.make_model_instance()
        
        self.load_model()
        
        self.list_containing_peaks = []
        self.list_not_containing_peaks = []
    
    def make_model_instance(self) -> None:
        """
        Create an instance of the model class specified by the model architecture.

        Returns:
            class instance: An instance of the specified model class.
        """
        try:
            model_class = getattr(m, self.model_arch)
            print(f'Model object has been created: {model_class.__name__}')
            return model_class()
        except AttributeError:
            print(f"Error: Model '{self.model_arch}' not found in the module.")
            return None
        except TypeError:
            print(f"Error: '{self.model_arch}' found in module is not callable.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
            
    def load_model(self) -> None:
        """
        Load the state dictionary into the model class and prepare it for evaluation.
        """
        try:
            model_path = self.model_path
            state_dict = torch.load(model_path)
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
        
    def classify_data(self, input_data: list, meta_data: list, camera_length_key: str, photon_energy_key: str) -> None:
        """
        Classify the input data using the model and segregate the data based on the classification results.

        Args:
            input_data (list): List of input data tensors.
            meta_data (list): List of metadata dictionaries corresponding to the input data.
            camera_length_key (str): The key for accessing camera length from the metadata.
            photon_energy_key (str): The key for accessing photon energy from the metadata.
        """
        print('Starting classification ...')
        try:
            if len(input_data) != len(self.h5_file_paths):
                print('Input data size does not match the number of file paths.')
                # return
            photon_energy_key = photon_energy_key.split('/')[-1]
            camera_length_key = camera_length_key.split('/')[-1]
            print(len(input_data))
            for index in range(len(input_data)):
                try:
                    # Prepare input data
                    input_data[index] = input_data[index].unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)

                    # Prepare metadata
                    camera_length = torch.tensor([meta_data[index][camera_length_key]], dtype=torch.float32).to(self.device)
                    photon_energy = torch.tensor([meta_data[index][photon_energy_key]], dtype=torch.float32).to(self.device)

                    # Model prediction
                    score = self.model(input_data[index], camera_length, photon_energy)
                    prediction = (torch.sigmoid(score) > 0.5).long()

                    # Segregate data based on prediction
                    if prediction == 1:
                        self.list_containing_peaks.append(self.h5_file_paths[index])
                    elif prediction == 0:
                        self.list_not_containing_peaks.append(self.h5_file_paths[index])

                except KeyError as e:
                    print(f"KeyError: Missing key in metadata at index {index} - {e}")
                except RuntimeError as e:
                    print(f"RuntimeError: A runtime error occurred at index {index} - {e}")
                except Exception as e:
                    print(f"An unexpected error occurred at index {index} - {e}")

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
            formatted_date_time = now.strftime("%m%d%y-%H:%M")
            
            filename_peaks = f"found_peaks-{formatted_date_time}.lst"
            file_path_peaks = os.path.join(self.save_output_list, filename_peaks)
            
            filename_no_peaks = f"no_peaks-{formatted_date_time}.lst"
            file_path_no_peaks = os.path.join(self.save_output_list, filename_no_peaks)

            try:
                with open(file_path_peaks, 'w') as file:
                    for item in self.list_containing_peaks:
                        file.write(f"{item}\n")
                print("Created .lst file for predicted peak files.")
            except Exception as e:
                print(f"An error occurred while writing to {file_path_peaks}: {e}")

            try:
                with open(file_path_no_peaks, 'w') as file:
                    for item in self.list_not_containing_peaks:
                        file.write(f"{item}\n")
                print("Created .lst file for predicted empty files.")
            except Exception as e:
                print(f"An error occurred while writing to {file_path_no_peaks}: {e}")

        except Exception as e:
            print(f"An unexpected error occurred while creating .lst files: {e}")
        
    def output_verification(self) -> None:
        """
        Verify that the number of input file paths matches the sum of the output file paths.

        This function compares the size of the input file path list to the sum of the sizes of the two output file path lists and logs the result.
        """
        if len(self.h5_file_paths) == len(self.list_containing_peaks) + len(self.list_not_containing_peaks):
            print("There is the same amount of input files as output files.")
        else:
            print("OUTPUT VERIFICATION FAILED: The input paths do not match the output paths.")           
            print(f'Input H5 files: {len(self.h5_file_paths)}\nOutput peak files: {len(self.list_containing_peaks)}\nOutput empty files: {len(self.list_not_containing_peaks)}')
