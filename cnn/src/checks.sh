#!/bin/bash

# Check if the argument is provided
if [[ $# -ne 1 ]]; then
    printf "Usage: %s <relative_path_to_images_directory>\n" "$0" >&2
    exit 1
fi

# Assign the provided argument to a variable
images_dir=$1

# Set the pipefail option to ensure that the script exits if a pipe command fails
set -o pipefail

# Define the directories where directories "01" through "09" should be created
declare -a target_dirs=("labels" "peaks" "peaks_water_overlay" "water")

# Create and populate directories
create_and_populate_dirs() {
    local target_path="$1"
    
    # Create the target directory if it does not exist
    mkdir -p "$target_path"

    # Create directories "01" through "09" within the target directory
    for i in {1..9}; do
        mkdir -p "${target_path}/$(printf "%02d" "$i")"
    done

    if [[ $? -eq 0 ]]; then
        printf "Directories '01' through '09' have been created in '%s'.\n" "$target_path"
    else
        printf "Directories '01' through '09' already exist in '%s'.\n" "$target_path"
    fi
}

# Count .h5 files in each dataset
count_h5_files() {
    local base_path="$1"
    for dir in "${target_dirs[@]}"; do
        for i in {1..9}; do
            local folder_path="${base_path}/${dir}/$(printf "%02d" "$i")"
            local count=$(find "$folder_path" -maxdepth 1 -name "*.h5" | wc -l)
            printf "Directory '%s' contains %d .h5 files.\n" "$folder_path" "$count"
        done
    done
}

# Main function to control script flow
main() {
    # Loop through the target directories to create and populate them
    for dir in "${target_dirs[@]}"; do
        create_and_populate_dirs "${images_dir}/${dir}"
    done

    # Count .h5 files in each directory
    count_h5_files "$images_dir"

    printf "Directory structure verification and .h5 files counting completed.\n"
}

# Execute the main function
main "$@"
