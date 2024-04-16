import os
import re
import shutil
import sys

def extract_kev_clen(filename):
    patterns = [
        r'master_[0-9]+_([0-9]+)keV_clen([0-9]+)_',
        r'img_([0-9]+)keV_clen([0-9]+)_',
        r'processed_img_([0-9]+)keV_clen([0-9]+)_'
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            kev, clen = match.groups()
            return kev, clen
    print(f"Error: Filename '{filename}' does not contain keV and clen values.", file=sys.stderr)
    return None, None

def get_prefix(subdir, file):
    if subdir.endswith('/labels'):
        return "label_" if file.startswith("label") else ""
    elif subdir.endswith('/peak_water_overlay'):
        return "processed_"  # For all files in this directory, regardless of starting with 'overlay_' or 'processed_'
    elif subdir.endswith('/peaks'):
        return ""
    return "img_"  # Default prefix for other directories

def process_directory(subdir, files, img_count):
    for file in files:
        # Pass 'empty' files as is
        if file.startswith("empty"):
            continue
        
        filepath = os.path.join(subdir, file)
        prefix = get_prefix(subdir, file)
        
        if prefix != "":  # If specific prefix is assigned, it means we need to process the file
            kev, clen = extract_kev_clen(file)
            if kev is None or clen is None:
                print(f"Skipping file: {file}", file=sys.stderr)
                continue

            new_filename_format = f"{prefix}{kev}keV_clen{clen}_{img_count:05d}.h5"
            new_filepath = os.path.join(subdir, new_filename_format)
            shutil.move(filepath, new_filepath)
            print(f"Renamed '{file}' to '{new_filename_format}'")
            img_count += 1
    return img_count

def reformat_filenames(directory):
    img_count = 1
    water_directory = os.path.join(directory, 'water')

    for subdir, dirs, files in os.walk(directory):
        if subdir == water_directory:
            print(f"Skipping water directory: {subdir}", file=sys.stderr)
            continue

        filtered_files = [file for file in files if file.endswith('.h5')]
        img_count = process_directory(subdir, filtered_files, img_count)

def main(directory):
    if not directory:
        print("Error: No directory provided.\nUsage: script.py <directory>", file=sys.stderr)
        return 1

    reformat_filenames(directory)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: No directory provided.\nUsage: script.py <directory>", file=sys.stderr)
        sys.exit(1)
    
    target_directory = sys.argv[1]
    main(target_directory)
