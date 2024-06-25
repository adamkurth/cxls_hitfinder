import h5py as h5
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Paths:
    
    def __init__(self, list_path: list) -> None:
        """
        This constructor takes in an lst file file path and runs it through the functions in this class.

        Args:
            list_path (list): This is the file path to an lst file which holds the file paths to h5 files to read. 
        """
        self.list_path = list_path
        self.h5_files = self.read_file_paths()
        self.h5_tensor_list, self.h5_attr_list = self.load_h5_data()
        
    def read_file_paths(self) -> list:
        """
        Read the file names from the lst file and put it in a list of strings. 
        
        Returns:
            list: This is the list of h5 file paths to read data from.
        """
        try:
            with open(self.list_path, 'r') as file:
                file_paths = [line.strip() for line in file]
            
            print(f'Read file paths from {self.list_path}.')
            return file_paths
        except Exception as e:
            print(f"An error occurred while reading from {self.list_path}: {e}")
        
    def get_file_paths(self) -> list:
        """
        Return the list of file path strings. 
        
        Returns:
            list: This is the list of h5 file paths to read data from.
        """
        return self.h5_files
    
    def load_h5_data(self) -> tuple:
        """
        This function takes the list of h5 files and loads them into a PyTorch tensor and pulls the metadata.
        
        Returns:
            tuple: This tuple contains two list, one list being the full of image tensors and one list being full of metadata dictionaries. 
        """
        tensor_list = []
        attribute_list = []

        for file_path in self.h5_files: 
            try:
                with h5.File(file_path, 'r') as file:
                    # Convert the HDF5 dataset to a NumPy array
                    numpy_array = np.array(file['entry/data/data'])
                    # Convert the NumPy array to a PyTorch tensor
                    tensor = torch.tensor(numpy_array)
                    # Append the tensor to the tensor_list
                    tensor_list.append(tensor)

                    # Retrieve attributes
                    attributes = {}
                    for attr in file.attrs:
                        try:
                            attributes[attr] = file.attrs.get(attr)
                        except KeyError:
                            attributes[attr] = None
                            print(f"Attribute '{attr}' not found in file {file_path}.")
                    
                    attribute_list.append(attributes)
                
                print('All .h5 files have been loaded into a list of torch.Tensors.')
                    
            except OSError:
                print(f"Error: Could not open file {file_path}. The file might not exist or be corrupted.")
            except IOError:
                print(f"Error: An I/O error occurred while opening file {file_path}.")
            except Exception as e:
                print(f"An unexpected error occurred while opening file {file_path}: {e}")
                        
        return tensor_list, attribute_list
                
    def get_h5_tensor_list(self) -> list:
        """
        Return the list of h5 tensors.

        Returns:
            list: This is the list of h5 tensors.
        """
        return self.h5_tensor_list
            
    def get_h5_attribute_list(self) -> list:
        """
        Return the list of h5 file attributes.

        Returns:
            list: This is the list of h5 file attributes.
        """
        return self.h5_attr_list

class Data(Dataset):
    
    def __init__(self, classification_data: list, attribute_data: list) -> None:
        """
        Initialize the Data object with classification and attribute data.

        Args:
            classification_data (list): List of classification data, that being list of pytorch tensors.
            attribute_data (list): List of attribute data, that being list of metadata dictionaries.
        """
        self.train_loader = None
        self.test_loader = None
        self.image_data = classification_data
        self.meta_data = attribute_data
        self.data = list(zip(self.image_data, self.meta_data))
        
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image data and the metadata at the given index.
        """
        return self.data[idx]
        
    def split_data(self, batch_size: int) -> None:
        """
        Split the data into training and testing datasets and create data loaders for them.

        Args:
            batch_size (int): The size of the batches to be used by the data loaders.
        """
        try:
            num_items = len(self.data)
            if num_items == 0:
                raise ValueError("The dataset is empty.")
            
            num_train = int(0.8 * num_items)
            num_test = num_items - num_train

            try:
                train_dataset, test_dataset = torch.utils.data.random_split(self.data, [num_train, num_test])
            except Exception as e:
                print(f"An error occurred while splitting the dataset: {e}")
                return

            try:
                self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
                self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            except Exception as e:
                print(f"An error occurred while creating data loaders: {e}")
                return
            
            print(f"Train size: {len(train_dataset)}")
            print(f"Test size: {len(test_dataset)}")

        except ValueError as e:
            print(f"ValueError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
    def get_data_loaders(self) -> tuple:
        """
        Get the training and testing data loaders.

        Returns:
            tuple: A tuple containing the training and testing data loaders.
        """
        return self.train_loader, self.test_loader