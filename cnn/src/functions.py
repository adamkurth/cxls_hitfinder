import os 
import re
from collections import namedtuple


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