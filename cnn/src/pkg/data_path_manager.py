import logging
import h5py as h5
import torch

logger = logging.getLogger(__name__)

class Paths:
    
    def __init__(self, list_path):
        self.list_path = list_path
        self.h5_files = self.read_file_paths()
        self.h5_tensor_list = self.load_h5_tensor_list()
        
    def read_file_paths(self) -> list:
        """
        Read the file names from the lst file and put it in a string. 
        """
        with open(self.list_path, 'r') as file:
            return [line.strip() for line in file]
        
    def get_file_paths(self) -> list:
        """
        Return the list of file path strings. 
        """
        return self.h5_files
    
    def load_h5_tensor_list(self) -> None:
        """
        This function takes the list of h5 files and loads them into a pytorch tensor. 
        """
        tensor_list = []
        for file_path in self.h5_files: 
            try:
                with h5.File(file_path, 'r') as file:
                    tensor_list.append(torch.Tensor(file['entry/data/data']))
            except:
                print("Incorrect file path : ", file_path)
                logger.info("Incorrect file path : ", file_path)
                
        return tensor_list
                
    def get_h5_tensor_list(self) -> list:
        
        return self.h5_tensor_list
            
    
        