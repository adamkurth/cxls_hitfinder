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
from pkg import u

def create_and_populate_dirs(target_path):
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

def validate_directories(base_path):
    """
    Validates the directory structure and file counts within the specified base path.

    Parameters:
    - base_path: The base directory to validate.

    Returns True if the directory structure and file counts are as expected, False otherwise.
    """
    expected_dirs = ["peaks", "labels", "peaks_water_overlay", "water"]
    is_valid = True
    
    for i in range(1, 10): # 01 to 09
        dataset_dir = f"{i:02d}"
        counts = {}
        for dir_name in expected_dirs: 
            dir_path = os.path.join(base_path, dir_name, dataset_dir) 
            file_count = len(glob(os.path.join(dir_path, "*.h5")))
            counts[dir_name] = file_count
            
            # check that images/water/dataset only contains 1 .h5 file
            if dir_name == "water" and file_count != 1:
                print(f"Error: Directory '{dir_path}' does not contain exactly 1 .h5 file.")
                is_valid = False
            
        if not (counts["peaks"] == counts["labels"] == counts["peaks_water_overlay"]):
            print(f"Mismatch found: '{base_path}/peaks/{dataset_dir}' has {counts['peaks']},\n '{base_path}/labels/{dataset_dir}' has {counts['labels']},\n '{base_path}/peaks_water_overlay/{dataset_dir}' has {counts['peaks_water_overlay']} images to compare.\n")
            is_valid = False
            
    return is_valid

def process_data(directory):
    """
    Processes data in the specified directory.

    Parameters:
    - directory: The directory containing data to process.
    """
    dataset_num = input("Enter dataset number: ")
    dataset = dataset_num.zfill(2) # string (ex '01')
    
    pm = u.PathManager()
    p = u.Processor(paths=pm, dataset=dataset)
    clen, photon_energy = p.get_parameters()
    p.process_directory(dataset=dataset, clen=clen, photon_energy=photon_energy)
    
def main(images_dir, force=False):
    """
    Main function to control the script's logic.

    Validates directories and decides whether to proceed with processing based on command-line flags.

    Parameters:
    - images_dir: The base directory containing image data.
    - force: Flag to force processing regardless of validation outcome.
    """
    for dir_name in  ["labels", "peaks", "peaks_water_overlay", "water"]:
        create_and_populate_dirs(os.path.join(images_dir, dir_name))
    
    if validate_directories(images_dir) or force:
        if force:
            print("Force-processing flag is used; proceeding with data processing despite potential issues.")
        else:
            print("Directory structure and image count verification completed successfully. Proceeding with data processing.")
        process_data(images_dir)  # step 1: generate labels, overlays, and images
    else:
        print("Errors detected during directory structure and image count verification. Please resolve these issues before proceeding.")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and process image directories.")
    parser.add_argument("images_dir", type=str, help="The directory containing the images to process.")
    parser.add_argument("--force", action="store_true", help="Force processing even if validation fails.")
    args = parser.parse_args()

    main(args.images_dir, args.force)
