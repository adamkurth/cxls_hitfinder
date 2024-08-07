import h5py as h5
import hdf5plugin
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from queue import Queue
import concurrent.futures 
from typing import Optional
from .utils import SpecialCaseFunctions


class Paths:
    
    def __init__(self, list_path: list, attribute_manager: str, attributes: dict, multievent: str, master_file: Optional[str]=None) -> None:
        """
        This constructor takes in an lst file file path and runs it through the functions in this class.

        Args:
            list_path (list): This is the file path to an lst file which holds the file paths to h5 files to read. 
        """
        self.list_path = list_path
        self.attribute_manager = attribute_manager
        self.attributes = attributes
        self.multievent = multievent
        self.master_file = master_file
        
        self.h5_files = Queue()
        self.h5_tensor_list, self.h5_attr_list, self.h5_file_list = [], [], []
        
    def read_file_paths(self) -> None:
        """
        Read the file names from the lst file and put it in a list or queue of strings. 
        Due to the size of the multievent files, the queue is used to process the files concurrently and load each file as its own dataloader.
        """
        try:
            with open(self.list_path, 'r') as file:
                for line in file:
                    if line.strip() == '' or line.strip() == '\n': 
                        continue
                    print(f'Adding to queue: {line.strip()}')
                    self.h5_files.put(line.strip())
            
            print(f'Read file paths from {self.list_path}.')
        except Exception as e:
            print(f"An error occurred while reading from {self.list_path}: {e}")

        
    def get_file_path_queue(self) -> list:
        """
        Return the list of file path strings. 
        
        Returns:
            list: This is the list of h5 file paths to read data from.
        """
        return self.h5_files
    
    def process_files(self) -> None:
        """
        This function serves as a wrapper for the functions that load the h5 data into tensors and metadata dictionaries.
        The multievent files are large enough to be processed concurrently, so this function will call the concurrent function if the multievent flag is set to true. 

        Args:
            attribute_manager (str): This is a string boolean value that tells the function if the attribute manager class from h5py is being used.
            attributes (dict): This is a dictionary of the metadata attributes to be found in the h5 files.
            multievent (str): This is a string boolean value that tells the function if the input .h5 files are multievent or not.
        """
        
        print('Processing .h5 files...')
        self.h5_tensor_list, self.h5_attr_list, self.h5_file_list = [], [], []
        
        if self.multievent == 'True' or self.multievent == 'true':
            print('Processing multievent files...')
            self.h5_file_list.append(self.h5_files.get())
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.load_h5_data)
                try:
                    future.result()
                except Exception as exc:
                    print(f'File generated an exception: {exc}')
                    
            base_file_name = self.h5_file_list[0]
            self.h5_file_list =[]
            for i in range(len(self.h5_tensor_list)):
                self.h5_file_list.append(base_file_name + ' - event: ' + str(1+i) + '/' + str(len(self.h5_tensor_list)))
        else: 
            print('Processing single event files...')
            for _ in range(1000):
                while not self.h5_files.empty():
                    self.h5_file_list.append(self.h5_files.get())
                else:
                    break
            self.load_h5_data()
    
    def load_h5_data(self) -> None:
        """
        This function takes in a list of h5 file file paths and loads in the images as tensors and puts the metadata into dictionaries. 
        There are two ways for the metadata to be taken out, one method uses to attribute manager class from h5py and the other finds metadata in a given file location.

        Args:
            attribute_manager (str): Tells this function if the attribute manager is being used.
            attributes (dict): This is a dictionary of the metadata attributes to be found in the h5 files.
        """

        for file_path in self.h5_file_list: 
            try:
                file_path = file_path.strip().replace('*', '')
                with h5.File(file_path, 'r') as file:
                    print(f'Reading file {file_path}')
                    numpy_array = np.array(file['entry/data/data']).astype(np.float32)
                    if numpy_array.shape[-2:] != (2163, 2069):
                        numpy_array = SpecialCaseFunctions.reshape_input_data(numpy_array)                      
                    tensor = torch.tensor(numpy_array)
                    if tensor.dim() > 2:
                        tensor = [*torch.split(tensor, 1, 0)]
                        self.h5_tensor_list.extend(tensor)
                    else:
                        tensor = tensor.unsqueeze(0)
                        self.h5_tensor_list.append(tensor)

                    self.read_metadata_attributes(file, file_path)
                         
            except OSError:
                print(f"Error: An I/O error occurred while opening file {file_path}")
            except Exception as e:
                print(f"An unexpected error occurred while opening file {file_path}: {e}")
        print('.h5 files have been loaded into a list of torch.Tensors.') 
        print(f'Number of tensors: {len(self.h5_tensor_list)}\nNumber of tensors with attributes: {len(self.h5_attr_list)}')  
        
    
    def read_metadata_attributes(self, file: h5.File, file_path: str) -> None: 
        """
        This function takes in a h5 file and extracts the metadata attributes from the file.
        This is only supposed to be used with load_h5_data function, and will not work independently.

        Args:
            file (h5.File): This is the h5 file object to extract metadata from.
            file_path (str): This is the file path to the h5 file.
        """
        
        camera_length = self.attributes['camera length']
        photon_energy = self.attributes['photon energy']
        peaks = self.attributes['peak']
        
        attributes = {}
        
        if self.attribute_manager == 'True' or self.attribute_manager == 'true': 
            for attr in file.attrs:
                try:
                    attributes[attr] = file.attrs.get(attr)
                except KeyError:
                    attributes[attr] = None
                    print(f"Attribute '{attr}' not found in file {file_path}.")
                    
            self.h5_attr_list.append(attributes) 
        
        elif self.master_file is not None:
            try:
                with h5.File(self.master_file, 'r') as master:
                    try:
                        if peaks is not None:
                            attributes['hit'] = master[peaks][()]
                            #converting units here is a temporary fix until a way to handle units is developed
                        attributes['detector_distance'] = master[camera_length][()]
                        attributes['incident_wavelength'] = SpecialCaseFunctions.incident_photon_wavelength_to_energy(master[photon_energy][()])
                        
                        for _ in range(len(self.h5_tensor_list)):
                            self.h5_attr_list.append(attributes)
                        
                    except KeyError:
                        print(f"Attributes not found in file {file_path}.")

            except OSError:
                print(f"Error: An I/O error occurred while opening file {file_path}")
            except Exception as e:
                print(f"An unexpected error occurred while opening file {file_path}: {e}")        
                
        else:
            try:
                if peaks is not None:
                    attributes['hit'] = file[peaks][()]
                attributes['Detector-Distance_mm'] = file[camera_length][()]
                attributes['X-ray-Energy_eV'] = file[photon_energy][()]
                
            except KeyError:
                attributes[attr] = None
                print(f"Attributes not found in file {file_path}.")
                
            self.h5_attr_list.append(attributes) 
                
    
                
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
    
    def get_h5_file_paths(self) -> list:
        """
        Return the list of h5 file paths.

        Returns:
            list: This is the list of h5 file paths.
        """
        return self.h5_file_list

######################################################################################
######################################################################################
######################################################################################

class Data(Dataset):
    
    def __init__(self, classification_data: list, attribute_data: list, h5_file_path: list, multievent: str) -> None:
        """
        Initialize the Data object with classification and attribute data.

        Args:
            classification_data (list): List of classification data, that being list of pytorch tensors.
            attribute_data (list): List of attribute data, that being list of metadata dictionaries.
            h5_file_path (list): List of h5 file paths.
            multievent (str): String boolean value if the input .h5 files are multievent or not.
        """
        self.train_loader = None
        self.test_loader = None
        self.inference_loader = None
        self.image_data = classification_data
        self.meta_data = attribute_data
        self.file_paths = h5_file_path
        self.data = list(zip(self.image_data, self.meta_data, self.file_paths))
        self.multievent = multievent
        
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.image_data)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image data and the metadata at the given index.
        """
        try:
            return self.data[idx]
        except Exception as e:
            print(f"An unexpected error occurred while getting item at index {idx}: {e}")
        
    def split_training_data(self, batch_size: int) -> None:
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
        
    def get_training_data_loaders(self) -> tuple:
        """
        Get the training and testing data loaders.

        Returns:
            tuple: A tuple containing the training and testing data loaders.
        """
        return self.train_loader, self.test_loader
    
    def inference_data_loader(self, batch_size) -> None: 
        """
        Puts the inference data into a dataloader for batch processing.

        Args:
            batch_size (int): The size of the batches to be used by the data loaders.
        """
        print('Making data loader...')
        try:
            num_items = len(self.data)
            if num_items == 0:
                raise ValueError("The dataset is empty.")
            
            try:
                self.inference_loader = DataLoader(self.data, batch_size=batch_size, shuffle=False, pin_memory=True)
                print('Data loader created.')
            except Exception as e:
                print(f"An error occurred while creating data loaders: {e}")
                return

        except ValueError as e:
            print(f"ValueError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    def get_inference_data_loader(self) -> DataLoader:
        """
        This function returns the inference data loader.

        Returns:
            DataLoader: The data loader for putting through the trained model. 
        """
        return self.inference_loader


