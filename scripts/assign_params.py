import os
import h5py as h5
from typing import Any, Dict

def assign_attributes(file_path: str, **kwargs: Any):
    """Assigns arbitrary attributes to an HDF5 file without individual confirmation."""
    with h5.File(file_path, 'a') as f:
        for key, value in kwargs.items():
            f.attrs[key] = value
    print(f"Attributes {list(kwargs.keys())} assigned to {file_path}.")

def get_params() -> Dict[str, Dict[str, Any]]:
    """Returns parameter settings for given dataset IDs."""
    # Define your clen values and photon energy values
    clen_values, photon_energy_values = [0.15, 0.25, 0.35], [6000, 7000, 8000]
    dataset_params = {
        '01': {'clen': clen_values[0], 'photon_energy': photon_energy_values[0]},
        '02': {'clen': clen_values[0], 'photon_energy': photon_energy_values[1]},
        '03': {'clen': clen_values[0], 'photon_energy': photon_energy_values[2]},
        '04': {'clen': clen_values[1], 'photon_energy': photon_energy_values[0]},
        '05': {'clen': clen_values[1], 'photon_energy': photon_energy_values[1]},
        '06': {'clen': clen_values[1], 'photon_energy': photon_energy_values[2]},
        '07': {'clen': clen_values[2], 'photon_energy': photon_energy_values[0]},
        '08': {'clen': clen_values[2], 'photon_energy': photon_energy_values[1]},
        '09': {'clen': clen_values[2], 'photon_energy': photon_energy_values[2]},
    }
    return dataset_params

def process_directory(directory: str):
    """Goes through the specified directory, confirms attribute assignment once per dataset directory."""
    dataset_params = get_params()
    
    for dataset, params in dataset_params.items():
        dataset_dir = os.path.join(directory, dataset)
        if os.path.exists(dataset_dir):
            print(f"Attributes to be assigned for dataset {dataset}: {params}")
            confirmation = input(f"Proceed with assigning attributes for dataset {dataset}? (y/n): ")
            if confirmation.lower() == 'y':
                for file in os.listdir(dataset_dir):
                    if file.endswith('.h5'):
                        file_path = os.path.join(dataset_dir, file)
                        assign_attributes(file_path, **params)
                print(f"\nCompleted processing for dataset {dataset} in {directory}.")
            else:
                print(f"Skipped attribute assignment for dataset {dataset}.")
        else:
            print(f"Dataset directory {dataset} not found in {directory}.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Assigns attributes to HDF5 files within specific dataset directories, with directory-level confirmation.")
    parser.add_argument("directory", help="Path to the specific parent directory to process, e.g., images/water")
    args = parser.parse_args()

    process_directory(args.directory)