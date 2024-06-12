from pkg import *
import argparse
import logging
import torch
import importlib

class RunHitfinder:
    
    def __init__(self, model_class, model_architecture):
        self.logger = logging.getLogger(__name__)
        
        self.parser = argparse.ArgumentParser(description='file path')
        self.args = self.arguments()
        self.h5_file_ilst = self.args.list
        self.model_arch = self.args.model
        self.model_path =self.args.dict
        
        self.model = None
        
    def arguments(self) -> str: 
        """
        This function is for adding an argument when running the python file. 
        It needs to take an lst file of the h5 files for the model use. 
        """
        self.parser.add_argument('-l', '--list', type=str, help='file path to h5 list file')
        self.parser.add_argument('-m', '--model', type=str, help='name of the model architecture')
        self.parser.add_argument('-d', '--dict', type=str, help='file path to the model state dict')
        args = self.parser.parse_args()
        if args:
            return args
        else:
            print('Input file needed.')
            self.logger.info('Input file needed.')
    
    def make_model_instance(self) -> None:
        module = importlib.import_module('classes')
        try:
            class = getattr(module, self.model_arch)
        instance = class_()
    
    def load_model(self) -> None:
        model_path = self.model_path
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model = self.model.eval() 
        self.model.to(self.device)