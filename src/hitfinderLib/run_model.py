from pkg import *
import logging
import torch
import datetime
import os 

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
        self.logger = logging.getLogger(__name__)
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
            return model_class()
        except AttributeError:
            print("Model not found.")
            self.logger.info("Model not found.")
            print(self.model_arch)
            self.logger.info(self.model_arch)
            return None
            
    def load_model(self) -> None:
        """
        Load the state dictionary into the model class and prepare it for evaluation.
        """
        model_path = self.model_path
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval() 
        self.model.to(self.device)
        
    def classify_data(self, input_data: list, meta_data: list, camera_length_key: str, photon_energy_key: str) -> None:
        """
        Classify the input data using the model and segregate the data based on the classification results.

        Args:
            input_data (list): List of input data tensors.
            meta_data (list): List of metadata dictionaries corresponding to the input data.
            camera_length_key (str): The key for accessing camera length from the metadata.
            photon_energy_key (str): The key for accessing photon energy from the metadata.
        """
        if len(input_data) != len(self.h5_file_paths):
            print('Input data size does not match the number of file paths.')
            self.logger.info('Input data size does not match the number of file paths.')
        
        for index in range(len(input_data)):
            input_data[index] = input_data[index].unsqueeze(0).unsqueeze(0).to(self.device)
            
            camera_length = torch.tensor([meta_data[index][camera_length_key]], dtype=torch.float32).to(self.device)
            photon_energy = torch.tensor([meta_data[index][photon_energy_key]], dtype=torch.float32).to(self.device)
             
            score = self.model(input_data[index], camera_length, photon_energy)
            prediction = (torch.sigmoid(score) > 0.5).long()
            
            if prediction == 1:
                self.list_containing_peaks.append(self.h5_file_paths[index])
            elif prediction == 0:
                self.list_not_containing_peaks.append(self.h5_file_paths[index])
                
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
        now = datetime.datetime.now()
        formatted_date_time = now.strftime("%m%d%y-%H:%M")
        
        filename_peaks = f"found_peaks-{formatted_date_time}.lst"
        file_path_peaks = os.path.join(self.save_output_list, filename_peaks)
        
        filename_no_peaks = f"no_peaks-{formatted_date_time}.lst"
        file_path_no_peaks = os.path.join(self.save_output_list, filename_no_peaks)
        
        with open(file_path_peaks, 'w') as file:
            for item in self.list_containing_peaks:
                file.write(f"{item}\n")
            
        print("Created .lst file for predicted peak files.")
        self.logger.info("Created .lst file for predicted peak files.")
        
        with open(file_path_no_peaks, 'w') as file:
            for item in self.list_not_containing_peaks:
                file.write(f"{item}\n")
            
        print("Created .lst file for predicted empty files.")
        self.logger.info("Created .lst file for predicted empty files.")
        
    def output_verification(self) -> None:
        """
        Verify that the number of input file paths matches the sum of the output file paths.

        This function compares the size of the input file path list to the sum of the sizes of the two output file path lists and logs the result.
        """
        if len(self.h5_file_paths) == len(self.list_containing_peaks) + len(self.list_not_containing_peaks):
            print("There is the same amount of input files as output files.")
            self.logger.info('There is the same amount of input files as output files.')
        else:
            print("OUTPUT VERIFICATION FAILED: The input paths do not match the output paths.")
            self.logger.info('OUTPUT VERIFICATION FAILED: The input paths do not match the output paths.')
            
            print(f'Input H5 files: {len(self.h5_file_paths)}\nOutput peak files: {len(self.list_containing_peaks)}\nOutput empty files: {len(self.list_not_containing_peaks)}')
            self.logger.info(f'Input H5 files: {len(self.h5_file_paths)}\nOutput peak files: {len(self.list_containing_peaks)}\nOutput empty files: {len(self.list_not_containing_peaks)}')
