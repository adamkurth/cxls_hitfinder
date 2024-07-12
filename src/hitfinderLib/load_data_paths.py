import h5py as h5
import hdf5plugin
import numpy as np
import torch
from queue import Queue
import concurrent.futures 
from typing import Optional
from abc import ABC, abstractmethod

from .utils import SpecialCaseFunctions
from . import conf

class Paths(ABC):
    
    def __init__(self, list_path: list, attributes: dict, master_file: Optional[str]=None) -> None:
        """
        This constructor takes in an lst file file path and runs it through the functions in this class.

        Args:
            list_path (list): This is the file path to an lst file which holds the file paths to h5 files to read. 
            attributes (dict): This is a dictionary that holds the names of the metadata attributes to be read from the hdf5 files.
            master_file (Optional[str], optional): This is the file path to a master file that holds the file paths to h5 files to read. Defaults to None.
        """
        self._list_path = list_path
        self._attributes = attributes
        
        self._master_file = master_file
        self._master_dict = {}
        
        self._h5_files = Queue()
        self._h5_tensor_list, self._h5_attr_list, self._h5_file_list = [], [], []
        
        self._loaded_h5_tensor = None
        self._h5_file_path = None
        self._open_h5_file = None
        self._attribute_holding = {}
        self._number_of_events = 1
        
    @abstractmethod
    def read_file_paths(self) -> None:
        """
        Read the file names from the lst file and put it in a list or queue of strings. 
        Due to the size of the multievent files, the queue is used to process the files concurrently and load each file as its own dataloader.
        """
        try:
            with open(self._list_path, 'r') as file:
                for line in file:
                    if line.strip() == '' or line.strip() == '\n': 
                        continue
                    print(f'Adding to queue: {line.strip()}')
                    self._h5_files.put(line.strip())
            
            print(f'Read file paths from {self._list_path}.')
        except Exception as e:
            print(f"An error occurred while reading from {self._list_path}: {e}")
    
    @abstractmethod
    def get_file_path_queue(self) -> Queue:
        """
        Return the list of file path strings. 
        
        Returns:
            list: This is the list of h5 file paths to read data from.
        """
        return self._h5_files
    
    @abstractmethod
    def process_files(self) -> None:
        """
        This function serves as a wrapper for the functions that load the h5 data into tensors and metadata dictionaries.
        The multievent files are large enough to be processed concurrently, so this function will call the concurrent function if the multievent flag is set to true. 
        """
        print('Processing .h5 files...')
        self._h5_tensor_list, self._h5_attr_list, self._h5_file_list = [], [], []

    @abstractmethod
    def load_h5_data(self) -> None:
        """
        This function takes in a list of h5 file file paths and loads in the images as tensors and puts the metadata into dictionaries. 
        There are two ways for the metadata to be taken out, one method uses to attribute manager class from h5py and the other finds metadata in a given file location.
        """

        try:            
            file_path = self._h5_file_path.strip().replace('*', '')

            print(f'Reading file {file_path}')
            self._open_h5_file = h5.File(file_path, 'r')
            
            numpy_array = np.array(self._open_h5_file['entry/data/data']).astype(np.float32)
            if numpy_array.shape[-2:] != conf.eiger_4m_image_size:
                numpy_array = SpecialCaseFunctions.reshape_input_data(numpy_array)                      
            self._loaded_h5_tensor = torch.tensor(numpy_array)
            
            self.read_metadata_attributes()
            self._open_h5_file.close()
                    
        except OSError:
            print(f"Error: An I/O error occurred while opening file {file_path}")
        except Exception as e:
            print(f"An unexpected error occurred while opening file {file_path}: {e}")
    
    @abstractmethod
    def read_metadata_attributes(self) -> None:
        """
        This function takes in a h5 file and extracts the metadata attributes from the file.
        This is only supposed to be used with load_h5_data function, and will not work independently.

        Args:
            file (h5.File): This is the h5 file object to extract metadata from.
            file_path (str): This is the file path to the h5 file.
        """
        print('Retrieving metadata attributes...')
        if len(self._master_dict) != 0:
            return

        self._attribute_holding = {}

        if self._master_file is not None:
            self.load_master_file()
                
        try:
            if len(self._open_h5_file.attrs.keys()) != 0:
                self.read_attribute_manager()
        except:
            pass
        
        try:
            if len(self._attribute_holding) == 0:
                self.read_attributes_from_file()
    
        except KeyError:
            print(f"ERROR: Attribute not found in file: {self._open_h5_file}.")
        
        
        if self._master_file is not None:
            self._master_dict = self._attribute_holding
        else:
            self._h5_attr_list.append(self._attribute_holding)

    @abstractmethod
    def load_master_file(self) -> None:
        """
        This function loads in the master file and sets the file path to the master file.
        """
        self._h5_file_path = self._master_file
        try:
            self._open_h5_file = h5.File(self._master_file, 'r')
        except OSError:
            print(f"Error: An I/O error occurred while opening file {self._master_file}")
        except Exception as e:
            print(f"An unexpected error occurred while opening file master file: {e}") 
            
    @abstractmethod
    def populate_attributes_from_master_dict(self) -> None:
        """
        This function populates the attribute list with the master dictionary.
        """
        for i in range(self._number_of_events):
            self._h5_attr_list.append(self._master_dict)
            
    @abstractmethod
    def read_attribute_manager(self) -> None:
        """
        This function reads the attributes from the attribute manager in the h5 file.
        """
        print('Reading attributes from attribute manager.')
        try:
            for attr in self._open_h5_file.attrs:
                self._attribute_holding[attr] = self._open_h5_file.attrs.get(attr)
        except KeyError:
            print(f"ERROR: Attribute not found in file: {self._h5_file_path}.")
            
    @abstractmethod
    def read_attributes_from_file(self) -> None:
        """
        This function reads the attributes from the h5 file.
        """
        camera_length = self._attributes[conf.camera_length_key]
        photon_energy = self._attributes[conf.photon_energy_key]
        peak = self._attributes[conf.present_peaks_key]
        
        if peak is not None:
            self._attribute_holding[conf.present_peaks_key] = self._open_h5_file[peak][()]
        self._attribute_holding[conf.camera_length_key] = self._open_h5_file[camera_length][()]
        self._attribute_holding[conf.photon_energy_key] = self._open_h5_file[photon_energy][()]
                
    @abstractmethod
    def get_h5_tensor_list(self) -> list:
        """
        Return the list of h5 tensors.

        Returns:
            list: This is the list of h5 tensors.
        """
        return self._h5_tensor_list
    
    @abstractmethod
    def get_h5_attribute_list(self) -> list:
        """
        Return the list of h5 file attributes.

        Returns:
            list: This is the list of h5 file attributes.
        """
        return self._h5_attr_list
    
    @abstractmethod
    def get_h5_file_paths(self) -> list:
        """
        Return the list of h5 file paths.

        Returns:
            list: This is the list of h5 file paths.
        """
        return self._h5_file_list
    
    @abstractmethod
    def get_event_count(self) -> int:
        """
        Return the number of events in the multievent files.

        Returns:
            int: This is the number of events in the multievent files.
        """
        return self._number_of_events
    
###########################################################
    
class PathsSingleEvent(Paths):
    
    def __init__(self, list_path: list, attributes: dict, master_file: Optional[str]=None) -> None:
        super().__init__(list_path, attributes, master_file)
        
    def read_file_paths(self) -> None:
        super().read_file_paths()
    
    def get_file_path_queue(self) -> Queue:
        return super().get_file_path_queue()
    
    def process_files(self) -> None:
        super().process_files()
        
        print('Processing single event files...')
        
        number_of_files_to_load = conf.single_event_data_loader_size
        for _ in range(number_of_files_to_load):
            while not self._h5_files.empty():
                self._h5_file_list.append(self._h5_files.get())
            else:
                break

        for file in self._h5_file_list:
            self._h5_file_path = file
            self.load_h5_data()
            
        print('.h5 files have been loaded into a list of torch.Tensors.') 
        print(f'Number of tensors: {len(self._h5_tensor_list)}\nNumber of tensors with attributes: {len(self._h5_attr_list)}')     
        
    def load_h5_data(self) -> None:
        super().load_h5_data()
        
        try:
            tensor = self._loaded_h5_tensor.unsqueeze(0)
            self._h5_tensor_list.append(tensor)
            
            if self._master_file is not None:
                self.populate_attributes_from_master_dict()
            
        except Exception as e:
            print(f"An unexpected error occurred while loading data: {e}")  
        
    def read_metadata_attributes(self) -> None:
        super().read_metadata_attributes()
        
    def load_master_file(self) -> None:
        super().load_master_file()
        
    def populate_attributes_from_master_dict(self) -> None:
        super().populate_attributes_from_master_dict()
        
    def read_attribute_manager(self) -> None:
        super().read_attribute_manager()
        
    def read_attributes_from_file(self) -> None:
        super().read_attributes_from_file()
        
    def get_h5_tensor_list(self) -> list:
        return super().get_h5_tensor_list()
    
    def get_h5_attribute_list(self) -> list:
        return super().get_h5_attribute_list()
    
    def get_h5_file_paths(self) -> list:
        return super().get_h5_file_paths()
    
    def get_event_count(self) -> int:
        return super().get_event_count()
    
###########################################################

class PathsMultiEvent(Paths):
    
    def __init__(self, list_path: list, attributes: dict, master_file: Optional[str]=None) -> None:
        super().__init__(list_path, attributes, master_file)
        
    def read_file_paths(self) -> None:
        super().read_file_paths()
    
    def get_file_path_queue(self) -> Queue:
        return super().get_file_path_queue()
    
    def process_files(self) -> None:
        super().process_files()
        
        print('Processing  multievent files...')
        
        base_file_name = self._h5_files.get()
        self._h5_file_path = base_file_name
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.load_h5_data)
            try:
                future.result()
            except Exception as exc:
                print(f'File generated an exception: {exc}')
        
        self._h5_file_list = []
        for i in range(len(self._h5_tensor_list)):
            self._h5_file_list.append(base_file_name + ' - event: ' + str(1+i) + '/' + str(len(self._h5_tensor_list)))
            
        print('.h5 files have been loaded into a list of torch.Tensors.') 
        print(f'Number of tensors: {len(self._h5_tensor_list)}\nNumber of tensors with attributes: {len(self._h5_attr_list)}')     
        
    def load_h5_data(self) -> None:
        super().load_h5_data()
        
        try:
            tensor = [*torch.split(self._loaded_h5_tensor, 1, dim=0)]
            self._h5_tensor_list.extend(tensor)
            self._number_of_events = len(tensor)
            
            if self._master_file is not None:
                self.populate_attributes_from_master_dict()
            
        except Exception as e:
            print(f"An unexpected error occurred while loading data: {e}")
        
    def read_metadata_attributes(self) -> None:
        super().read_metadata_attributes()
        
    def load_master_file(self) -> None:
        super().load_master_file()
        
    def populate_attributes_from_master_dict(self) -> None:
        super().populate_attributes_from_master_dict()
        
    def read_attribute_manager(self) -> None:
        super().read_attribute_manager()
        
    def read_attributes_from_file(self) -> None:
        super().read_attributes_from_file()
        
    def get_h5_tensor_list(self) -> list:
        return super().get_h5_tensor_list()
    
    def get_h5_attribute_list(self) -> list:
        return super().get_h5_attribute_list()
    
    def get_h5_file_paths(self) -> list:
        return super().get_h5_file_paths()
    
    def get_event_count(self) -> int:
        return super().get_event_count()