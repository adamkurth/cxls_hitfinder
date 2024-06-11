import argparse
import logging
import h5py as h5
import torch

class PathManager:
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='file path')
        self.list_path = self.arguments()
        self.h5_files = self.read_file_paths()
        self.h5_tensor_list = self.load_h5_tensor_list()

        
    def arguments(self) -> str: 
        """
        This function is for adding an argument when running the python file. 
        It needs to take an lst file of the h5 files for the model use. 
        """
        self.parser.add_argument('-f', '--file', type=str, help='file path to h5 list file')
        args = self.parser.parse_args()
        if args.file:
            return args.file
        else:
            print('Input file needed.')
        
    def read_file_paths(self) -> list:
        """
        Read the file names from the lst file and put it in a string. 
        """
        with open(self.list_path, 'r') as file:
            return file.readlines()
        
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
            with h5.File(file_path, 'r') as file:
                tensor_list.append(torch.Tensor(file['entry/data/data']))
        return tensor_list
                
    def get_h5_tensor_list(self) -> list:
        
        return self.h5_tensor_list
        
def main() -> None:
    """
    This main is used for testing this file seperate of the complete code base. 
    """
    logger = logging.getLogger(__name__)
    
    paths = PathManager()
    h5_file_path_list = paths.get_file_paths()
    print(h5_file_path_list)
    logger.info(h5_file_path_list)

    h5_tensor_list = paths.get_h5_tensor_list()
    print(h5_tensor_list)
    logger.info(h5_tensor_list)
    
if __name__ == '__main__':
    main()
    