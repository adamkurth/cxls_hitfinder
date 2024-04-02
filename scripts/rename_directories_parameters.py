import os
import re

def extract_parameters(dir_name):
    """Extracts keV and clen values from directory name."""
    match = re.match(r'(\d{2})_(0\.\d{2})m_(\d+)keV', dir_name)
    if match:
        dataset, clen_meter, keV = match.groups()
        # Map dataset numbers to clen values as described
        clen_map = {"01": "01", "02": "01", "03": "01", 
                    "04": "02", "05": "02", "06": "02",
                    "07": "03", "08": "03", "09": "03"}
        clen = clen_map.get(dataset)
        if clen:
            return {
                'dataset': dataset,
                'clen': clen,
                'keV': keV
            }
    return None

def rename_files_in_directory(root_dir):
    for root, dirs, files in os.walk(root_dir):
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
    index = base_name.split('_')[-1]  # Assuming index is the last part after '_'
    new_name_format = "{prefix}_img_{keV}keV_clen{clen}_{index}.h5"
    prefix = ''  # Default for peaks
    if sub_dir == 'labels':
        prefix = 'label'
    elif sub_dir == 'peak_water_overlay':
        prefix = 'overlay'
    return new_name_format.format(prefix=prefix, keV=params['keV'], clen=params['clen'], index=index)

# Example usage
root_directory = '../../MASTER_COPY/'  # Adjust as per your structure
rename_files_in_directory(root_directory)
