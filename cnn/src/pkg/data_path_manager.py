import logging
import h5py as h5
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

class Paths:
    
    def __init__(self, list_path: list):
        self.list_path = list_path
        self.h5_files = self.read_file_paths()
        self.h5_tensor_list, self.h5_attr_list = self.load_h5_data()
        
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
    
    def load_h5_data(self) -> tuple:
        """
        This function takes the list of h5 files and loads them into a PyTorch tensor and pulls the metadata.
        """
        tensor_list = []
        attribute_list = []

        for file_path in self.h5_files: 
            try:
                with h5.File(file_path, 'r') as file:
                    # Load data into tensor
                    tensor_list.append(torch.Tensor(file['entry/data/data']))

                    # Retrieve attributes
                    attributes = {}
                    for attr in file.attrs:
                        try:
                            attributes[attr] = file.attrs.get(attr)
                        except:
                            attributes[attr] = None
                            print(f"Attribute '{attr}' not found in file.")
                            logger.info(f"Attribute '{attr}' not found in file.")
                    
                    attribute_list.append(attributes)
                    
            except:
                print("Incorrect file path: ", file_path)
                logger.info("Incorrect file path: ", file_path)
                    
        return tensor_list, attribute_list
                
    def get_h5_tensor_list(self) -> list:
        return self.h5_tensor_list
            
    def get_h5_attribute_list(self) -> list:
        return self.h5_attr_list

class Data(Dataset):
    
    def __init__(self, classification_data: list, attribute_data: list):
        self.data = classification_data
        
        self.train_loader = None
        self.test_loader = None
        
        self.image_data = classification_data
        self.meta_data = attribute_data
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple:
        return self.image_data[idx], self.meta_data[idx]
        
    def split_data(self, batch_size: int) -> None: 
        
        num_items = len(self.data)
        
        num_train = int(0.8 * num_items)
        num_test = num_items - num_train
        
        train_dataset, test_dataset = torch.utils.data.random_split(self.data, [num_train, num_test])
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)  
        
        print(f"Train size: {len(train_dataset)}")
        logger.info(f"Train size: {len(train_dataset)}")
        print(f"Test size: {len(test_dataset)}")
        logger.info(f"Test size: {len(test_dataset)}")
        
    def get_data_loaders(self) -> tuple:
        return self.train_loader, self.test_loader