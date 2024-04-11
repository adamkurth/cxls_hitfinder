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
from pkg import *

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

def validate_directories(base_path:str):
    """
    Validates the directory structure and file counts within the specified base path.

    - If `peaks_water_overlay/` and `labels/` directories have zero files, it suggests
      that the script should proceed with generating labels and overlays.
    - If these directories have a nonzero count that matches `peaks/`, it implies
      the processing has already been done.

    Parameters:
    - base_path: The base directory to validate.

    Returns:
    - A tuple (bool, bool): First bool indicates if the structure is valid, and second bool
      indicates if processing should proceed.
    """
    is_valid = True
    proceed_with_processing = False
    
    for i in range(1, 10): # 01 to 09
        dataset_dir = f"{i:02d}"
        counts = {}
        dirs = ["peaks", "labels", "peaks_water_overlay", "water"]
        for dir_name in dirs: 
            dir_path = os.path.join(base_path, dir_name, dataset_dir) 
            file_count = len(glob(os.path.join(dir_path, "*.h5")))
            counts[dir_name] = file_count
            
            # check that images/water/dataset only contains 1 .h5 file
            if dir_name == "water" and file_count != 1:
                print(f"Error: Directory '{dir_path}' does not contain exactly 1 .h5 file for water background.\n")
                is_valid = False
        # after collecting counts, perform checks
        if counts.get('peaks_water_overlay') == counts.get('labels') == 0:
            # Indicates that processing should occur as no overlays or labels have been generated yet.
            proceed_with_processing = True
            print(f"Directories for dataset {dataset_dir} are ready for processing (Step 1).")
        elif counts.get('peaks_water_overlay') == counts.get('labels') == counts.get('peaks'):
            # Indicates that processing has likely already occurred as counts match.
            print(f"Directories for dataset {dataset_dir} have been processed. Matching file counts found.")
        else:
            print(f"Mismatch found in dataset {dataset_dir}: Peaks: {counts['peaks']}, Labels: {counts['labels']}, Overlays: {counts['peaks_water_overlay']}.\n")
            is_valid = False
        
    return is_valid, proceed_with_processing

def process_data(paths:path.PathManager, image_directory:path.PathManager, percent_empty=0.3):
    """
    Processes data in the specified directory.

    Parameters:
    - directory: The directory containing data to process.
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
    
def main(images_dir, force=False, percent_empty: float = 0.3):
    """
    Main function to control the script's logic.

    Validates directories and decides whether to proceed with processing based on command-line flags.

    Parameters:
    - images_dir: The base directory containing image data.
    - force: Flag to force processing regardless of validation outcome.
    """
    for dir_name in  ["labels", "peaks", "peaks_water_overlay", "water"]:
        create_and_populate_dirs(os.path.join(images_dir, dir_name))
    
    is_valid, proceed_with_processing = validate_directories(images_dir)
    if force or (is_valid and proceed_with_processing):
        if force:
            print("Force-processing flag is used; proceeding with data processing despite potential issues.")
        else:
            print("Directory structure and image count verification completed successfully. Proceeding with data processing.")
        print("\nProceeding with data processing...")
        # generate labels, overlays, and images, then add empty images
        
        datasets = input("Enter dataset numbers (separated by commas): ").split(",")
        datasets = [str(int(d)).zfill(2) for d in datasets]
        paths = path.PathManager(datasets=f.convert2str(datasets))
        process_data(paths=paths,image_directory=images_dir, percent_empty=percent_empty)
    elif is_valid and not proceed_with_processing:
        print("Processing not required. Directories already contain processed data.")
    else:
        print("Errors detected during directory structure and image count verification. Please resolve these issues before proceeding.")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and process image directories.")
    parser.add_argument("images_dir", type=str, help="The directory containing the images to process.")
    parser.add_argument("--force", action="store_true", help="Force processing even if validation fails.")
    parser.add_argument("--percent_empty", type=float, default=0.3, help="Percentage of empty images to add.")
    args = parser.parse_args()
    
    main(images_dir=args.images_dir, force=args.force, percent_empty=args.percent_empty)