"""
This script validates and optionally processes image directories.

It checks the expected directory structure and the presence of specific file types. If the validation passes,
or if the --force argument is used, it proceeds with processing (generating labels, overlays, and images). 
Otherwise, it halts with an error message.

Usage:
    python script_name.py <images_directory> [--force]
"""
import os
from glob import glob
import argparse
import shutil
from typing import Union
from pkg import *

def clear_directory(directory_path : str) -> None:
    """
    Removes all .h5 files from the specified directory.
    """
    for file in glob(os.path.join(directory_path, "*.h5")):
        os.remove(file)

def create_and_populate_dirs(target_path:str):
    """
    Creates and populates directories if they don't exist.

    Parameters:
    - target_path: The base directory to create and populate.

    Prints messages indicating whether new directories were created or if they already existed.
    """
    made_new = False
    sub_dirs = ['01', '02', '03', '04', '05', '06', '07', '08', '09'] # selected datasets 
    for sub_dir in sub_dirs:
        dir_path = os.path.join(target_path, sub_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Creating directory '{dir_path}'...")
            made_new = True
    
    if made_new:
        print(f"New directories created in '{target_path}'.")
    else: 
        print(f"Directories '01' through '09' already created in '{target_path}'.")

def validate_and_prepare_directories(base_path:str, force:bool, regenerate:bool) -> Union[bool, bool]:
    """
    Validates the directory structure and file counts within the specified base path.

    Parameters:
    - base_path: The base directory to validate.
    - force: Whether to force the processing.
    - regenerate: Whether to clear existing processed data and regenerate.

    Returns:
    - A tuple (bool, bool): First bool indicates if the structure is valid, and second bool
      indicates if processing should proceed.
    """
    is_valid = True
    proceed_with_processing = False
    
    dirs = ["peaks", "labels", "peaks_water_overlay", "water"]
    for i in range(1, 10):  # 01 to 09
        dataset_dir = f"{i:02d}"
        counts = {}
        for dir_name in dirs:
            dir_path = os.path.join(base_path, dir_name, dataset_dir)
            if regenerate and dir_name in ['labels', 'peaks_water_overlay']:
                clear_directory(dir_path)
                print(f'Cleared existing images in {dir_path} due to regeneration request.\n')

            file_count = len(glob(os.path.join(dir_path, "*.h5")))
            counts[dir_name] = file_count

        if counts['peaks'] and (counts['labels'] == 0 and counts['peaks_water_overlay'] == 0):
            print(f"File count mismatch in dataset {dataset_dir}: {counts}")
            if input(f"Detected no processed images for dataset {dataset_dir}. Would you like to generate them? (y/n): ").lower() == 'y':
                proceed_with_processing = True
            else:
                is_valid = False
        elif force or all(counts[dir] == counts['peaks'] for dir in ["labels", "peaks_water_overlay"]):
            proceed_with_processing = True
        else:
            print(f"File count mismatch in dataset {dataset_dir}: {counts}")
            is_valid = False
        
    return is_valid, proceed_with_processing

def process_data(paths:path.PathManager, image_directory:path.PathManager, percent_empty=0.3) -> None:
    """
    Processes data in the specified directory.

    Parameters:
    - paths: Path management object.
    - image_directory: The directory containing data to process.
    - percent_empty: Percentage of empty images to generate.
    """
    processor = process.Processor(paths=paths, datasets=paths.datasets)
    parameters = processor.get_parameters()
    clen, photon_energy = parameters.get('clen'), parameters.get('photon_energy')
    
    # Step 0: Remove all existing empty images.
    processor.cleanup()
    
    # Step 1: Process existing data to generate labels, overlays, and possibly more.
    processor.process_directory()
    processor.cleanup_authenticator()
    
    # Step 2: Calculate the number of empty images to add based on the percentage.
    processor.process_empty(percent_empty=percent_empty)
    
def main(images_dir:str, force: bool=False, regenerate: bool=False, percent_empty: float = 0.3)-> None:
    """
    Main function to control the script's logic.

    Validates directories and decides whether to proceed with processing based on command-line flags.

    Parameters:
    - images_dir: The base directory containing image data.
    - force: Flag to force processing regardless of validation outcome.
    - regenerate: Flag to clear existing data and regenerate.
    - percent_empty: Percentage of empty images to add.
    """
    for dir_name in  ["labels", "peaks", "peaks_water_overlay", "water"]:
        create_and_populate_dirs(os.path.join(images_dir, dir_name))
    
    is_valid, proceed_with_processing = validate_and_prepare_directories(base_path=images_dir, force=force, regenerate=False)

    if proceed_with_processing:
        print("Proceeding with data processing...")
        datasets = input("Enter dataset numbers (separated by commas): ").split(",")
        datasets = [str(int(d)).zfill(2) for d in datasets]
        paths = path.PathManager(datasets=f.convert2str(datasets))
        process_data(paths=paths, image_directory=images_dir, percent_empty=percent_empty)
    else:
        print("Processing halted. Please resolve the issues and try again.")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and process image directories.")
    parser.add_argument("images_dir", type=str, help="The directory containing the images to process.")
    parser.add_argument("--force", action="store_true", help="Force processing even if validation fails.")
    parser.add_argument("--regenerate", action="store_true", help="Clear existing data and regenerate.")
    parser.add_argument("--percent_empty", type=float, default=0.3, help="Percentage of empty images to add.")
    args = parser.parse_args()
    
    main(images_dir=args.images_dir, force=args.force, regenerate=args.regenerate, percent_empty=args.percent_empty)
