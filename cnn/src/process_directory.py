import os
import sys
from glob import glob
from pkg import u

# Directory creation and validation functions
def create_and_populate_dirs(target_path):
    made_new = False
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        made_new = True
    
    for i in range(1, 10):
        dir_path = os.path.join(target_path, f"{i:02d}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            made_new = True
    
    if made_new:
        print(f"Made new directories in '{target_path}'.")
    else:
        print(f"Directories '01' through '09' already created in '{target_path}'.")

def validate_directories(base_path):
    for i in range(1, 10):
        dirs = {
            "peaks": os.path.join(base_path, "peaks", f"{i:02d}"),
            "labels": os.path.join(base_path, "labels", f"{i:02d}"),
            "peaks_water_overlay": os.path.join(base_path, "peaks_water_overlay", f"{i:02d}"),
            "water": os.path.join(base_path, "water", f"{i:02d}")
        }

        counts = {key: len(glob(os.path.join(path, "*.h5"))) for key, path in dirs.items()}

        if counts["water"] != 1:
            print(f"Error: Directory '{dirs['water']}' does not contain exactly 1 .h5 file.")
            return False

        if counts["peaks"] == counts["labels"] == counts["peaks_water_overlay"]:
            print(f"All directories in '{dirs['peaks']}', '{dirs['labels']}', and '{dirs['peaks_water_overlay']}' have {counts['peaks']} images, as expected.")
        else:
            print(f"Mismatch found: {dirs['peaks']} has {counts['peaks']}, {dirs['labels']} has {counts['labels']}, {dirs['peaks_water_overlay']} has {counts['peaks_water_overlay']} images to compare.")
            return False
    return True

# Main processing function
def process_data(directory):
    clen_values, photon_energy_values = [1.5, 2.5, 3.5], [6000, 7000, 8000]
    param_matrix = u.parameter_matrix(clen_values, photon_energy_values)
    print(param_matrix, '\n')

    dataset_dict = {
        '01': [clen_values[0], photon_energy_values[0]],
        '02': [clen_values[0], photon_energy_values[1]],
        '03': [clen_values[0], photon_energy_values[2]],
        '04': [clen_values[1], photon_energy_values[0]],
        '05': [clen_values[1], photon_energy_values[1]],
        '06': [clen_values[1], photon_energy_values[2]],
        '07': [clen_values[2], photon_energy_values[0]],
        '08': [clen_values[2], photon_energy_values[1]],
        '09': [clen_values[2], photon_energy_values[2]],
    }
    
    dataset_number = input("Enter dataset number: ")
    dataset = dataset_number.zfill(2)
    print(f'Parameter values of dataset {dataset}: {dataset_dict[dataset]}')

    clen, photon_energy = dataset_dict[dataset]

    pm = u.PathManager()
    p = u.Processor(paths=pm, dataset=dataset)
    p.process_directory(dataset=dataset, clen=clen, photon_energy=photon_energy)

def main(images_dir):
    target_dirs = ["labels", "peaks", "peaks_water_overlay", "water"]
    for dir_name in target_dirs:
        create_and_populate_dirs(os.path.join(images_dir, dir_name))
    
    # Perform directory validation
    if validate_directories(images_dir):
        print("Directory structure and image count verification completed successfully.")
        user_decision = input("Do you want to proceed with data processing? (yes/no): ").strip().lower()
        if user_decision == 'yes':
            print("Proceeding with data processing.")
            process_data(images_dir)
        else:
            print("Skipping data processing based on user input.")
    else:
        print("Errors detected during directory structure and image count verification. Please resolve these issues before proceeding.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <relative_path_to_images_directory>")
        sys.exit(1)
    main(sys.argv[1])