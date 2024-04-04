import os
import h5py as h5
import numpy as np
from typing import Any
from glob import glob
from torch.utils.data import DataLoader
import torch
from typing import Union, List, Dict


def convert2int(datasets: List[Union[str, int]]) -> List[int]:
    """
    Converts a list of dataset identifiers (either string or integer) to integers.
    
    Args:
        datasets (List[Union[str, int]]): The list of dataset strings or integers.
    
    Returns:
        List[int]: The datasets converted to integers.
    """
    converted = []
    for dataset in datasets:
        if isinstance(dataset, str):
            converted.append(int(dataset))
        elif isinstance(dataset, int):
            converted.append(dataset)
        else:
            raise ValueError("Invalid dataset type. Expected str or int within the list.")
    return converted

def convert2str(datasets: List[Union[str, int]]) -> List[str]:
    """
    Converts a list of dataset identifiers (either string or integer) to strings formatted as '01', '02', etc.
    
    Args:
        datasets (List[Union[str, int]]): The list of dataset strings or integers.
    
    Returns:
        List[str]: The datasets converted to strings, with leading zeros for single-digit numbers.
    """
    converted = []
    for dataset in datasets:
        if isinstance(dataset, str):
            converted.append(dataset.zfill(2))
        elif isinstance(dataset, int):
            converted.append(str(dataset).zfill(2))
        else:
            raise ValueError("Invalid dataset type. Expected str or int within the list.")
    return converted

def convert2str_single(dataset: Union[str, int]) -> str:
    """
    Converts a single dataset identifier (either string or integer) to a string formatted as '01', '02', etc.
    
    Args:
        dataset (Union[str, int]): The dataset string or integer.
    
    Returns:
        str: The dataset converted to a string, with leading zeros for single-digit numbers.
    """
    if isinstance(dataset, str):
        return dataset.zfill(2)
    elif isinstance(dataset, int):
        return str(dataset).zfill(2)
    else:
        raise ValueError("Invalid dataset type. Expected str or int.")

def load_h5(file_path:str) -> np.ndarray:
    with h5.File(file_path, 'r') as file:
        return np.array(file['entry/data/data'])

def save_h5(file_path:str, data:np.ndarray, save_parameters:bool, params:list) -> None:
    with h5.File(file_path, 'w') as file:
        file.create_dataset('entry/data/data', data=data)
    if save_parameters:
        assign_attributes(file_path=file_path, clen=params[0], photon_energy=params[1])
    print(f"File saved: {file_path}")

def parameter_matrix(clen_values: list, photon_energy_values: list) -> None:
    # limited to 2d for now 
    dtype = [('clen', float), ('photon_energy', float)]
    matrix = np.zeros((len(clen_values), len(photon_energy_values)), dtype=dtype)
    for i, clen in enumerate(clen_values):
        for j, photon_energy in enumerate(photon_energy_values):
            matrix[i, j] = (clen, photon_energy)
    return matrix

def assign_attributes(file_path: str, **kwargs: Any):
    """
    Assigns arbitrary attributes to an HDF5 file located at file_path.
    """
    with h5.File(file_path, 'a') as f:
        for key, value in kwargs.items():
            f.attrs[key] = value
    print(f"Attributes {list(kwargs.keys())} assigned to {file_path}")

def parse_attributes(paths: object, params:list) -> dict:
    """
    Assigns attributes including a type relevance flag to files based on directory type.

    assuming non-empty images
    peaks/ -> True
    labels/ -> True
    overlay/ -> True
    water/ -> False
    """
    paths.refresh_all()
    dataset = paths.dataset

    # Initialize the lists of file paths for each directory type
    peak_path_list, overlay_path_list, label_path_list, background_path_list = paths.init_lists(dataset)
    
    # Mapping of directory types to their respective path lists and relevance to the task
    dir_mappings = {
        'peak': (peak_path_list, True),
        'overlay': (overlay_path_list, True),
        'label': (label_path_list, True),
        'background': (background_path_list, False),
    }
    for d, (path_list, is_relavant) in dir_mappings.items():
        for f in path_list:
            assign_attributes(file_path=f, clen=params[0], photon_energy=params[1], peak=is_relavant)
            print(f"Attributes assigned to {f}")
    print("Attributes assigned to all files.")
    
def get_params(datasets:List[int]) -> dict:
    datasets = convert2str(datasets=datasets)
    clen_values, photon_energy_values = [1.5, 2.5, 3.5], [6000, 7000, 8000]
    dataset_dict = {
        '01': [clen_values[0], photon_energy_values[0]],
        '02': [clen_values[0], photon_energy_values[1]],
        '03': [clen_values[0], photon_energy_values[2]],
        '04': [clen_values[1], photon_energy_values[0]],
        '05': [clen_values[1], photon_energy_values[1]],
        '06': [clen_values[1], photon_energy_values[2]],
        '07': [clen_values[2], photon_energy_values[0]],
        '08': [clen_values[2], photon_energy_values[1]],
        '09': [clen_values[2], photon_energy_values[2]],
    }
    params_dict = {}
    for dataset in datasets:
        params_dict[dataset] = {
            'clen': dataset_dict[dataset][0],
            'photon_energy': dataset_dict[dataset][1]
        }
    return params_dict

def retrieve_attributes(file_path: str):
    """
    Retrieves specified attributes from an HDF5 file located at the given file path.
    
    Args:
        file_path (str): The path to the HDF5 file.
    
    Returns:
        dict: A dictionary containing the attributes of the HDF5 file.
    """
    attributes = {}

    # Open the HDF5 file located at file_path and retrieve its attributes
    with h5.File(file_path, 'r') as file:
        for attr in file.attrs:
            try:
                attributes[attr] = file.attrs.get(attr)
            except KeyError:
                attributes[attr] = None
                print(f"Attribute '{attr}' not found in file.")
                
    return attributes

def check_attributes(paths: object, datasets: List[str], dir_type: str) -> bool:
    conform = True
    params = get_params(datasets=datasets)
    for dataset in datasets:
        files = paths.fetch_paths_by_type(dataset=dataset, dir_type=dir_type)
        exp_clen, exp_photon_energy = params.get(dataset)['clen'], params.get(dataset)['photon_energy']
        for path in files:
            attributes = retrieve_attributes(path)
            act_clen, act_photon_energy = attributes['clen'], attributes['photon_energy']
            if act_clen == exp_clen or act_photon_energy == exp_photon_energy:
                pass
            else:
                conform = False
                print(f'Error: {path} does not match expected attributes')
                print(f'Expected: clen={exp_clen}, photon_energy={exp_photon_energy}')
                print(f'Actual: clen={act_clen}, photon_energy={act_photon_energy}')
        return conform
    
def get_counts(paths: object, datasets:List[int]) -> None:
    """
    Counts and reports the number of 'normal' and 'empty' images in the specified directories
    for the selected dataset, using the paths to access directory paths.
    """
    # Refresh the lists in PathManager to ensure they are up-to-date
    paths.refresh_all()

    # Directories to check
    directory_types = ['peaks', 'labels', 'peaks_water_overlay']
    
    for dataset in datasets:
        # Loop through each directory type and count files
        dataset = convert2str_single(dataset)
        for directory_type in directory_types:
            directory_path = os.path.join(paths.images_dir, directory_type, dataset)
            all_files = glob(os.path.join(directory_path, '*.h5'))  # Corrected usage here
            normal_files = [file for file in all_files if 'empty' not in os.path.basename(file)]
            empty_files = [file for file in all_files if 'empty' in os.path.basename(file)]

            # Reporting the counts
            print(f"Directory: {directory_type}/{dataset}")
            print(f"\tTotal files: {len(all_files)}")
            print(f"\tNormal images: {len(normal_files)}")
            print(f"\tEmpty images: {len(empty_files)}")

def prepare(data_manager: object, batch_size:int=32) -> tuple:
    """
    Prepares and splits the data into training and testing datasets.
    Applies transformations and loads them into DataLoader objects.

    :param data_manager: An instance of DatasetManager, which is a subclass of torch.utils.data.Dataset.
    :param batch_size: The size of each data batch.
    :return: A tuple containing train_loader and test_loader.
    """
    # Split the dataset into training and testing sets.
    num_items = len(data_manager)
    num_train = int(0.8 * num_items)
    num_test = num_items - num_train
    train_dataset, test_dataset = torch.utils.data.random_split(data_manager, [num_train, num_test])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)    
        
    print("\nData prepared.")
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches in train_loader: {len(train_loader)} \n")
    
    return train_loader, test_loader # returns train/test tensor data loaders
    