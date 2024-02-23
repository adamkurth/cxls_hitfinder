import os
import numpy as np
import h5py
import torch
import shutil
import re
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import namedtuple

class PeakImageDataset(Dataset):

    def __init__(self, peak_image_paths, water_image_paths, transform=None, augment=False):
        assert len(peak_image_paths) == len(water_image_paths), "The number of peak images must match the number of water images."
        self.peak_image_paths = peak_image_paths
        self.water_image_paths = water_image_paths
        self.transform = transform or self.default_transform()
        self.augment = augment
        
    def __len__(self):
        # returns number of images in the dataset
        num_peaks = len(self.peak_image_paths)
        num_waters = len(self.water_image_paths)
        return (num_peaks, num_waters) # returns a tuple of the number of peak and water images

    def __get_item__(self, idx):
        # gets peak/water image and returns numpy array and applies transform
        peak_image = self.__load_h5__(self.peak_image_paths[idx])
        water_image = self.__load_h5__(self.water_image_paths[idx])
        
        if self.augment:
            peak_image = self.augment_image(peak_image)
            water_image = self.augment_image(water_image)
        
        if self.transform:
            peak_image = self.transform(peak_image)
            water_image = self.transform(water_image)
    
        return peak_image, water_image 
        
    def __load_h5__(self, image_path):
        with h5py.File(image_path, 'r') as f:
            image = np.array(f['entry/data/data'])
        return image

    def augment_image(self, image):
        pil_image = transforms.ToPILImage()(image)
        rotated_image = pil_image.rotate(90)
        return transforms.ToTensor()(rotated_image)
    
    def default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float()), #ensure float type
            transforms.Normalize(mean=[0.5], std=[0.5]) #normalize to [-1, 1]
        ])
    
    def __preview__(self, idx, image_type='peak'):
        image_path = self.peak_image_paths[idx] if image_type == 'peak' else self.water_image_paths[idx]
        image = self.__load_h5__(image_path)
        # visualize outliers 
        plt.imshow(image, cmap='viridis')
        plt.colorbar()
        plt.title(f'{image_type.capitalize()} Image at Index {idx}')
        plt.axis('off')
        plt.show()
        
class PathManager:
    def __init__(self):
        # grabs peaks and processed images from images directory /
        self.root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.images_dir = os.path.join(self.root, 'images') #/images
        self.sim_dir = os.path.join(self.root, 'sim')  #/sim
        self.sim_specs_dir = os.path.join(self.sim_dir, 'sim_specs') # sim/sim_specs
        # KEY
        self.peak_images_dir = os.path.join(self.images_dir, 'peaks') # images/peaks
        self.water_images_dir = os.path.join(self.images_dir, 'data') # images/data
        # 
        # built just in case further development is needed
        self.processed_images_dir = os.path.join(self.images_dir, 'processed_images') # images/processed_images
        self.preprocessed_images_dir = os.path.join(self.images_dir, 'preprocessed_images') # images/preprocessed_images
        self.label_images_dir = os.path.join(self.images_dir, 'labels') # images/labels
        self.pdb_dir = os.path.join(self.sim_dir, 'pdb') # /sim/pdb
        self.sh_dir = os.path.join(self.sim_dir, 'sh') # /sim/sh
        self.water_background_h5 = os.path.join(self.sim_dir, 'water_background.h5') # /sim/water_background.h5
    
    def __get_path__(self, path_name):
        # returns the path of the path_name
        paths_dict = {
            'root': self.root,
            'images_dir': self.images_dir,
            'sim_dir': self.sim_dir,
            'sim_specs_dir': self.sim_specs_dir,
            'peak_images_dir': self.peak_images_dir,
            'water_images_dir': self.water_images_dir,
            'processed_images_dir': self.processed_images_dir,
            'preprocessed_images_dir': self.preprocessed_images_dir,
            'label_images_dir': self.label_images_dir,
            'pdb_dir': self.pdb_dir,
            'sh_dir': self.sh_dir,
            'water_background_h5': self.water_background_h5,
        }
        return paths_dict.get(path_name, None)

    def __get_all_paths__(self):
        # returns a namedtuple of all paths for easier access
        PathsOutput = namedtuple('PathsOutput', ['root', 'images_dir', 'sim_dir', 'peak_images_dir', 'water_images_dir', 'processed_images_dir', 'label_images_dir', 'pdb_dir', 'sh_dir', 'water_background_h5'])
        return PathsOutput(self.root, self.images_dir, self.sim_dir, self.peak_images_dir, self.processed_images_dir, self.label_images_dir, self.pdb_dir, self.sh_dir, self.water_background_h5)
    
    def __get_peak_images_paths__(self):
        # searches the peak images directory for .h5 files and returns a list of paths
        return [os.path.join(self.peak_images_dir, f) for f in os.listdir(self.peak_images_dir) if f.endswith('.h5')]    
    
    def __get_water_images_paths__(self):
        # searches the water images directory (images/data)for .h5 files and returns a list of paths
        return [os.path.join(self.water_images_dir, f) for f in os.listdir(self.water_images_dir) if f.endswith('.h5')]
    
    def __get_processed_images_paths__(self):
        # searches the processed images directory for .h5 files and returns a list of paths
        return [os.path.join(self.processed_images_dir, f) for f in os.listdir(self.processed_images_dir) if f.startswith('processed')]
    
    def __get_label_images_paths__(self):
        # searches the label images directory for .h5 files and returns a list of paths
        return [os.path.join(self.label_images_dir, f) for f in os.listdir(self.label_images_dir) if f.startswith('label')]
    
    def __get_pdb_path__(self, pdb_file):
        # returns the .pdb file path of the file name in the pdb directory
        return os.path.join(self.pdb_dir, pdb_file)
    
    def __get_sh_path__(self, sh_file):
        # returns the .sh file path of the file name in the sh directory
        return os.path.join(self.sh_dir, sh_file)

    def clean_sim(self):
        file_types = ['.err', '.out', '.sh']
        files_moved = False  # flag to check if any files were moved

        # manage the .err, .out, .sh files into sim_specs
        for file_type in file_types:
            for f in os.listdir(self.sim_dir):
                if f.endswith(file_type):
                    source_path = os.path.join(self.sim_dir, f)
                    dest_path = os.path.join(self.sim_specs_dir, f)
                    
                    # checks to ensure that the file is not already in the sim_specs directory
                    if not os.path.exists(dest_path):
                        print(f"Moving {source_path} to {dest_path}\n")
                        shutil.move(source_path, dest_path)
                        
                        # copy the .sh file to the sim directory
                        if file_type == '.sh':
                            sh_copy_path = os.path.join(self.sim_dir, f)
                            print(f"Copying {dest_path} to {sh_copy_path}\n")
                            shutil.copy(dest_path, sh_copy_path)
                        
                        files_moved = True
                        
        if not files_moved:
            print("clean_sim did not move any files\n\n")
            

class DataPreparation:
    def __init__(self, paths, batch_size=32):
        self.paths = paths
        self.batch_size = batch_size
        
    def prep_data(self):
        peak_image_paths = self.paths.__get_peak_images_paths__()
        water_images_paths = self.paths.__get_water_images_paths__()
        
        # ensure matching number of paths of peak and water images         
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float()), #ensure float type
            transforms.Normalize(mean=[0.5], std=[0.5]) #normalize to [-1, 1]
        ])
        
        dataset = PeakImageDataset(peak_image_paths, water_images_paths, transform=transform)
        num_items = len(dataset)
        num_train = int(0.8 * num_items)
        num_test = num_items - num_train
        
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])
    
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        print("Data prepared.")
        return train_loader, test_loader


def sim_parameters(paths):
    """
    Reads the .pdb and .sh files and returns a dictionary of simulation parameters.

    Parameters:
    - Paths: An instance of the Paths class that contains the paths to the .pdb and .sh files.

    Returns:
    - combined_params: A dictionary containing the simulation parameters extracted from the .pdb and .sh files.
        The dictionary includes the following keys:
        - geom: The geometry parameter from the .sh file.
        - cell: The cell parameter from the .sh file.
        - number: The number parameter from the .sh file.
        - output_name: The output_name parameter from the .sh file.
        - photon_energy: The photon_energy parameter from the .sh file.
        - nphotons: The nphotons parameter from the .sh file.
        - a: The 'a' parameter from the .pdb file.
        - b: The 'b' parameter from the .pdb file.
        - c: The 'c' parameter from the .pdb file.
        - alpha: The 'alpha' parameter from the .pdb file.
        - beta: The 'beta' parameter from the .pdb file.
        - gamma: The 'gamma' parameter from the .pdb file.
        - spacegroup: The spacegroup parameter from the .pdb file.
    """
    def read_pdb(path):
        UnitcellParams = namedtuple('UnitcellParams', ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'spacegroup'])
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('CRYST1'):
                    tokens = line.split()
                    a, b, c = float(tokens[1]), float(tokens[2]), float(tokens[3])
                    alpha, beta, gamma = float(tokens[4]), float(tokens[5]), float(tokens[6])
                    spacegroup = ' '.join(tokens[7:-1])  # Exclude the last element
        return UnitcellParams(a, b, c, alpha, beta, gamma, spacegroup)._asdict()

    def read_sh(path):
        ShParams = namedtuple('ShParams', [
            'geom', 'cell', 'number', 'output_name', 'sf', 'pointgroup',
            'min_size', 'max_size', 'spectrum', 'cores', 'background',
            'beam_bandwidth', 'photon_energy', 'nphotons', 'beam_radius', 'really_random'
        ])
        
        params = {key: None for key in ShParams._fields}
        
        with open(path, 'r') as file:
            content = file.read()
        param_patterns = {
            'geom': r'-g\s+(\S+)',
            'cell': r'-p\s+(\S+)',
            'number': r'--number=(\d+)',
            'output_name': r'-o\s+(\S+)',
            'sf': r'-i\s+(\S+)',
            'pointgroup': r'-y\s+(\S+)',
            'min_size': r'--min-size=(\d+)',
            'max_size': r'--max-size=(\d+)',
            'spectrum': r'--spectrum=(\S+)',
            'cores': r'-s\s+(\d+)',
            'background': r'--background=(\d+)',
            'beam_bandwidth': r'--beam-bandwidth=([\d.]+)',
            'photon_energy': r'--photon-energy=(\d+)',
            'nphotons': r'--nphotons=([\d.e+-]+)',
            'beam_radius': r'--beam-radius=([\d.]+)',
            'really_random': r'--really-random=(True|False)'
        }
        for key, pattern in param_patterns.items():
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                value = match.group(1)
                if value.isdigit():
                    params[key] = int(value)
                elif value.replace('.', '', 1).isdigit():
                    params[key] = float(value)
                elif value == 'True':
                    params[key] = True
                elif value == 'False':
                    params[key] = False
                else:
                    params[key] = value

        return ShParams(**params)._asdict()        
    
    pdb_path = os.path.join(paths.root, 'sim', 'pdb', '1ic6.pdb') # hardcoded for now
    sh_path = os.path.join(paths.root, 'sim', 'submit_7keV_clen01.sh') # hardcode for now
    
    unitcell_params_dict = read_pdb(pdb_path)
    sh_params_dict = read_sh(sh_path)
    
    essential_keys_sh = ['geom', 'cell', 'number', 'output_name', 'photon_energy', 'nphotons']
    essential_sh_params = {key: sh_params_dict[key] for key in essential_keys_sh}
    
    combined_params = {**essential_sh_params, **unitcell_params_dict}
    return combined_params

def main():
    # instances
    paths = PathManager()
    peak_paths = paths.__get_peak_images_paths__()
    water_paths = paths.__get_peak_images_paths__()
    dataset = PeakImageDataset(peak_image_paths=peak_paths, water_image_paths=water_paths, transform=transforms.ToTensor(), augment=False)
    item = dataset.__get_item__(0) 
    print(item.__len__())
    # dataset.__preview__(2, image_type='peak')
    # dataset.__preview__(2, image_type='water')


    data_preparation = DataPreparation(paths, batch_size=32)
    paths.clean_sim() # moves all .err, .out, .sh files sim_specs 
    train_loader, test_loader = data_preparation.prep_data()

    sim_dict = sim_parameters(paths)
    print(sim_dict)

if __name__ == "__main__":
    main()