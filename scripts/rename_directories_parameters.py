import os
import re

def extract_parameters(dir_name):
    """Extracts parameters from directory name."""
    match = re.match(r'master_(\d{2})_(\d+)keV_clen(\d{2})', dir_name)
    if match:
        return {
            'dataset': match.group(1),
            'photon_energy': match.group(2),
            'clen': match.group(3)
        }
    return None

def rename_files_in_directory(root_dir):
    for root, dirs, files in os.walk(root_dir):
        # Skip the 'water' directory
        if 'water' in dirs:
            dirs.remove('water')

        for dir_name in dirs:
            parameters = extract_parameters(dir_name)
            if parameters:
                dataset_dir = os.path.join(root, dir_name)
                rename_files(dataset_dir, parameters)

def rename_files(dataset_dir, params):
    sub_dirs = ['peaks', 'labels', 'peak_water_overlay']
    for sub_dir in sub_dirs:
        path = os.path.join(dataset_dir, sub_dir)
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.endswith('.h5'):
                    new_name = generate_new_name(filename, params, sub_dir)
                    os.rename(os.path.join(path, filename), os.path.join(path, new_name))
                    print(f'Renamed {filename} to {new_name}')

def generate_new_name(filename, params, sub_dir):
    base_name = filename.split('.')[0]
    index = base_name.split('_')[-1]
    if sub_dir == 'peaks':
        new_name = f'img_{params["photon_energy"]}keV_clen{params["dataset"]}_{index}.h5'
    elif sub_dir == 'labels':
        new_name = f'label_img_{params["photon_energy"]}keV_clen{params["dataset"]}_{index}.h5'
    else:  # peak_water_overlay
        new_name = f'overlay_img_{params["photon_energy"]}keV_clen{params["dataset"]}_{index}.h5'
    return new_name

# Example usage
root_directory = '../../images/'  # Adjust to your root directory
rename_files_in_directory(root_directory)
