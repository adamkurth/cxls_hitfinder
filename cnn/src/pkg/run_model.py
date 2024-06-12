from pkg import *
import argparse
import logging
import torch
import importlib
import datetime

class RunHitfinder:
    
    def __init__(self, model_class, model_architecture):
        self.logger = logging.getLogger(__name__)
        
        self.parser = argparse.ArgumentParser(description='file path')
        self.args = self.arguments()
        self.h5_file_ilst = self.args.list
        self.model_arch = self.args.model
        self.model_path =self.args.dict
        
        self.model = self.make_model_instance()
        
        self.load_model()
        
        self.list_containing_peaks = None
        self.list_not_containing_peaks = None
        
    def arguments(self) -> str: 
        """
        This function is for adding an argument when running the python file. 
        It needs to take an lst file of the h5 files for the model use. 
        """
        self.parser.add_argument('-l', '--list', type=str, help='file path to h5 list file')
        self.parser.add_argument('-m', '--model', type=str, help='name of the model architecture')
        self.parser.add_argument('-d', '--dict', type=str, help='file path to the model state dict')
        self.parser.add_argument('-o', '--output', type=str, help='output file path for the lst files without file names')
        args = self.parser.parse_args()
        if args:
            return args
        else:
            print('Input file needed.')
            self.logger.info('Input file needed.')
    
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
        
    def classify_data(self, input_data) -> None: 
        """
        This function takes input data and classifies the data. 
        
        ! This input_data should be broken up so that it contains the image tensor, meta data, and file path (either as tuple or seperate params).
        ! input_data is being used as a dummy variable for now. 
        """
        for data in input_data:
            score = self.model(data)
            prediction = (torch.sigmoid(score) > 0.5).long()
            
            if prediction == 1:
                self.list_containing_peaks.append(input_data)
            elif prediction == 0:
                self.list_not_containing_peaks(input_data)
            
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
        pass 