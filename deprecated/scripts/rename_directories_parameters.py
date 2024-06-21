import os
import re

def parse_directory_name(dir_name):
    """
    Parses the directory name to extract keV and clen values.
    Assumes directory name format: '01_0.15m_6keV'
    """
    match = re.match(r"(\d{2})_(0\.\d{2})m_(\d+)keV", dir_name)
    if not match:
        return None

    dataset, clen_meter, keV = match.groups()
    # Map clen_meter to clen value
    clen_map = {"0.15": "01", "0.25": "02", "0.35": "03"}
    clen = clen_map[clen_meter]
    return keV, clen

def rename_files_in_dir(path, keV, clen):
    """
    Renames all .h5 files in the given path according to the keV and clen values.
    """
    for filename in os.listdir(path):
        if filename.endswith('.h5'):
            # Extract the index from the original filename
            match = re.search(r"(\d+)\.h5$", filename)
            if match:
                index = match.group(1).zfill(5)  # Ensure the index is 5 digits
                new_filename = f"img_{keV}keV_clen{clen}_{index}.h5"
                os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
                print(f"Renamed {filename} to {new_filename}")


def main():
    base_path = "../../MASTER/"
    for dir_name in os.listdir(base_path):
        full_path = os.path.join(base_path, dir_name)
        if os.path.isdir(full_path):  # Ensure it's a directory
            parsed = parse_directory_name(dir_name)
            if parsed:
                keV, clen = parsed
                print(f"Processing {dir_name} -> keV: {keV}, clen: {clen}")
                rename_files_in_dir(full_path, keV, clen)
            else:
                print(f"Skipping unrecognized directory format: {dir_name}")

if __name__ == "__main__":
    main()
