import os
import numpy as np
import h5py
import torch
import shutil
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import namedtuple

# import sys
# sys.path.append('/Users/adamkurth/Documents/vscode/CXFEL_Image_Analysis/CXFEL/pdb_rcsb_api_scripts')
# import pdb_rcsb_api_scripts as pdb_api

class PeakImageDataset(Dataset):
    def __init__(self, peak_image_paths, water_image_paths, transform=None):
        self.peak_image_paths = peak_image_paths
        self.water_image_paths = water_image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.peak_image_paths)
    
    def __getitem__(self, idx):
        h5_path = 'entry/data/data'
        with h5py.File(self.image_paths[idx], 'r') as f:
            peak_image = np.array(f[h5_path])
        with h5py.File(self.label_paths[idx], 'r') as f:
            water_image = np.array(f[h5_path])
                
        if self.transform:
            peak_image = self.transform(peak_image)
            water_image = self.transform(water_image)
                    
        return peak_image, water_image
    
    def augment(self, peak_images, water_images, label_images):
        pass
    
    
class Paths:
    def __init__(self):
        # grabs peaks and processed images from images directory /
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.images_dir = os.path.join(self.root_dir, 'images') #/images
        self.sim_dir = os.path.join(self.root_dir, 'sim')  #/sim
        self.sim_specs_dir = os.path.join(self.sim_dir, 'sim_specs') # sim/sim_specs
        self.peak_images_dir = os.path.join(self.images_dir, 'peaks') # images/peaks
        self.water_images_dir = os.path.join(self.images_dir, 'data') # images/data
        self.processed_images_dir = os.path.join(self.images_dir, 'processed_images') # images/processed_images
        self.preprocessed_images_dir = os.path.join(self.images_dir, 'preprocessed_images') # images/preprocessed_images
        self.label_images_dir = os.path.join(self.images_dir, 'labels') # images/labels
        self.pdb_dir = os.path.join(self.sim_dir, 'pdb') # /sim/pdb
        self.sh_dir = os.path.join(self.sim_dir, 'sh') # /sim/sh
        self.water_background_h5 = os.path.join(self.sim_dir, 'water_background.h5') # /sim/water_background.h5
    
    def __get_path__(self, path_name):
        # returns the path of the path_name
        paths_dict = {
            'root_dir': self.root_dir,
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
        return paths_dict.get(path_name)
        
    def __get_all_paths__(self):
        # returns a namedtuple of all paths for easier access
        PathsOutput = namedtuple('PathsOutput', ['root_dir', 'images_dir', 'sim_dir', 'peak_images_dir', 'processed_images_dir', 'label_images_dir', 'pdb_dir', 'sh_dir', 'water_background_h5'])
        return PathsOutput(self.root_dir, self.images_dir, self.sim_dir, self.peak_images_dir, self.processed_images_dir, self.label_images_dir, self.pdb_dir, self.sh_dir, self.water_background_h5)
    
    def __get_peak_images_paths__(self):
        # searches the peak images directory for .h5 files and returns a list of paths
        return [os.path.join(self.peak_images_dir, f) for f in os.listdir(self.peak_images_dir) if f.endswith('.h5')]    
    
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
                        print(f"Moving {source_path} to {dest_path}")
                        shutil.move(source_path, dest_path)
                        
                        # copy the .sh file to the sim directory
                        if file_type == '.sh':
                            sh_copy_path = os.path.join(self.sim_dir, f)
                            print(f"Copying {dest_path} to {sh_copy_path}")
                            shutil.copy(dest_path, sh_copy_path)
                        
                        files_moved = True
                        
        if not files_moved:
            print("clean_sim did not move any files")
            
        
        
        
def prep_data(peak_image_paths, water_image_path):
    found_peak_image_paths = [os.path.join(peak_image_paths, f) for f in os.listdir(peak_image_paths) if f.endswith('.h5')] 
    found_water_image_paths = [os.path.join(water_image_path, f) for f in os.listdir(water_image_path) if f.startswith('processed')]
    
    # print(found_peak_image_paths)
    # print(found_water_image_path)
    
    # ensure matching number of paths of peak and water images 
    assert len(found_peak_image_paths) == len(found_water_image_paths), "Mismatch in the number of peak and water images."
    
    # join the data with water/peak images, then split 80/20
    train_peak_images, test_peak_images, train_water_images, test_water_images = train_test_split(found_peak_image_paths, found_water_image_paths, test_size=0.2, random_state=42)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()), #ensure float type
        transforms.Normalize((0.5,), (0.5,)) #normalize to [-1, 1]
    ])
    
    train_dataset = PeakImageDataset(train_peak_images, train_water_images, transform=transform)
    test_dataset = PeakImageDataset(test_peak_images, test_water_images, transform=transform)
    
    # print(len(train_dataset), len(test_dataset))
    
    return train_dataset, test_dataset
    
def get_relative_paths():
    # want to be universal, repo specific not local specific
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    background_h5_path = os.path.join(root_path, 'sim', 'water_background.h5')
    peak_images_path = os.path.join(root_path, 'images', 'peaks')
    processed_images_path = os.path.join(root_path, 'images', 'data')
    label_output_path = os.path.join(root_path, 'images', 'labels')
    # namedtuple 
    Paths = namedtuple('Paths', ['peak_images', 'processed_images', 'label_output', 'background_h5', 'root'])
    Paths = Paths(peak_images_path, processed_images_path, label_output_path, background_h5_path, root_path)
    
    return Paths

def sim_parameters(Paths):
    import re
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
    
    pdb_path = os.path.join(Paths.root, 'sim', 'pdb', '1ic6.pdb') # hardcoded for now
    sh_path = os.path.join(Paths.root, 'sim', 'submit_7keV_clen01.sh') # hardcode for now
    
    unitcell_params_dict = read_pdb(pdb_path)
    sh_params_dict = read_sh(sh_path)
    
    essential_keys_sh = ['geom', 'cell', 'number', 'output_name', 'photon_energy', 'nphotons']
    essential_sh_params = {key: sh_params_dict[key] for key in essential_keys_sh}
    
    combined_params = {**essential_sh_params, **unitcell_params_dict}
    return combined_params
    


# # def main_dataprep():
# Paths = get_relative_paths()
# peak_images_path = Paths.peak_images
# water_images_path = Paths.processed_images
# label_images_path = Paths.label_output
# background_h5_path = Paths.background_h5
# root_path = Paths.root

# sim_parameters(Paths)
# train_dataset, test_dataset = prep_data(peak_images_path, water_images_path)

# grabs .pdb and .sh files and returns a dictionary of parameters
# combined_params = sim_parameters(Paths) 
# outputs dict of geom, cell, number, output_name, photon_energy, nphotons, a, b, c, alpha, beta, gamma, spacegroup
# print(combined_params) 

paths = Paths()
paths.clean_sim()
all_paths = paths.__get_all_paths__()
print(all_paths, "\n\n")

path = paths.__get_path__('root_dir')
print('root path', path, "\n\n")

sim_path = paths.__get_path__('sim_dir')
print('sim path', sim_path, "\n\n")

peak_images_paths = paths.__get_peak_images_paths__()
print("Peak Images Paths:", peak_images_paths, "\n\n")

processed_images_paths = paths.__get_processed_images_paths__()
print("Processed Images Paths:", processed_images_paths, "\n\n")

label_images_paths = paths.__get_label_images_paths__()
print("Label Images Paths:", label_images_paths, "\n\n")

pdb_path = paths.__get_pdb_path__('1ic6.pdb')
print("PDB Path:", pdb_path, "\n\n")

sh_path = paths.__get_sh_path__('submit_7keV_clen01.sh')
print("SH Path:", sh_path, "\n\n")



# if __name__ == '__main__':
#     main_dataprep()
    
