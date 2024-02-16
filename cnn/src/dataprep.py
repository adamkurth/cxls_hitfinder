import os
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import namedtuple

class PeakImageDataset(Dataset):
    def __init__(self, image_paths, water_images, transform=None):
        self.image_paths = image_paths
        # self.label_paths = label_paths # add label_paths arg
        self.water_images = water_images
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        h5_path = 'entry/data/data'
        with h5py.File(self.image_paths[idx], 'r') as f:
            peak_image = np.array(f[h5_path])
        with h5py.File(self.label_paths[idx], 'r') as f:
            water_image = np.array(f[h5_path])
        # with h5py.File(self.label_paths[idx], 'r') as f:
        #     label_image = np.array(f[h5_path])
                
        if self.transform:
            peak_image = self.transform(peak_image)
            water_image = self.transform(water_image)
            # label_image = self.transform(label_image)
        
        return peak_image, water_image
  
  
def prep_data(peak_image_paths, water_image_path):
    found_peak_image_paths = [os.path.join(peak_image_paths, f) for f in os.listdir(peak_image_paths) if f.endswith('.h5')] 
    found_water_image_path = [os.path.join(water_image_path, f) for f in os.listdir(water_image_path) if f.startswith('processed')]
    
    # print(found_peak_image_paths)
    # print(found_water_image_path)
    
    # join the data with water/peak images, then split 80/20
    train_image_paths, test_image_paths, train_water_paths, test_water_paths = train_test_split(found_peak_image_paths, found_water_image_path, test_size=0.2, random_state=42)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()) #ensure float type
    ])
    
    train_dataset = PeakImageDataset(train_image_paths, train_water_paths, transform=transform)
    test_dataset = PeakImageDataset(test_image_paths, test_water_paths, transform=transform)
    
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

def params(Paths):
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
    
    
    
    
    
    

# def main():
Paths = get_relative_paths()
peak_images_path = Paths.peak_images
water_images_path = Paths.processed_images
label_images_path = Paths.label_output
background_h5_path = Paths.background_h5
root_path = Paths.root

params(Paths)
train_dataset, test_dataset = prep_data(peak_images_path, water_images_path)

# grabs .pdb and .sh files and returns a dictionary of parameters
combined_params = params(Paths) 
print(combined_params)


# if __name__ == '__main__':
#     main()
    
