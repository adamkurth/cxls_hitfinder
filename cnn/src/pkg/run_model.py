from pkg import *
import logging
import torch
import importlib
import datetime
import os 

class RunModel:
    
    def __init__(self, model_arch, model_path, save_output_list):
        self.logger = logging.getLogger(__name__)
        
        self.model_arch = model_arch
        self.model_path = model_path
        self.save_output_list = save_output_list
        
        self.model = self.make_model_instance()
        
        self.load_model()
        
        self.list_containing_peaks = None
        self.list_not_containing_peaks = None
        
    
    def make_model_instance(self):
        """
        This function makes a class instance of the model provided from the models.py

        Returns:
            class instance
        """
        module = importlib.import_module('models')
        try:
            model_class = getattr(module, self.model_arch)
            return model_class()
        except:
            print("Model not found.")
            self.logger.info("Model not found.")
            
    def load_model(self) -> None:
        """
        This function loads in the state dictionary to the model class. 
        """
        model_path = self.model_path
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model = self.model.eval() 
        self.model.to(self.device)
        
    def classify_data(self, input_data, photon_energy, camera_length, file_paths) -> None: 
        """
        This function takes input data and classifies the data. 
        """
        
        if len(input_data) != len(file_paths):
            print('Input data size does not match number of file paths.')
            self.logger.info('Input data size does not match number of file paths.')
        
        for index in len(input_data):
            score = self.model(input_data[index], camera_length, photon_energy)
            prediction = (torch.sigmoid(score) > 0.5).long()
            
            if prediction == 1:
                self.list_containing_peaks.append(file_paths[index])
            elif prediction == 0:
                self.list_not_containing_peaks(file_paths[index])
            
    def get_classification_results(self) -> tuple:
        """
        This function returns the results from the model classicication. 

        Returns:
            tuple: This tuple has two list, one containing images with predicted peaks and one without predicted peaks. 
        """
        
        return (self.list_containing_peaks, self.list_not_containing_peaks)
    
    def create_model_output_lst_files(self) -> None:
        """
        This model creates the lst files of the predictions. 
        """
        now = datetime.datetime.now()
        formatted_date_time = now.strftime("%m%d%y-%H:%M")
        
        filename_peaks = f"found_peaks-{formatted_date_time}.lst"
        file_path_peaks = os.path.join(self.save_output_list, filename_peaks)
        
        filename_no_peaks = f"no_peaks-{formatted_date_time}.lst"
        file_path_no_peaks = os.path.join(self.save_output_list, filename_no_peaks)
        
        with open(file_path_peaks, 'w') as file:
            file.write(self.list_containing_peaks)
            
        print("Created lst file for predicted peak files.")
        self.logger.info("Created lst file for predicted peak files.")
        
        with open(file_path_no_peaks, 'w') as file:
            file.write(self.list_not_containing_peaks)
            
        print("Created lst file for predicted empty files.")
        self.logger.info("Created lst file for predicted empty files.")
        
        
        