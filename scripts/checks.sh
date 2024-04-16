#!/bin/bash

# Check if the argument is provided
if [[ $# -ne 1 ]]; then
    printf "Usage: %s <relative_path_to_images_directory>\n" "$0" >&2
    return 1
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
    local made_new=false
    
    # Create the target directory if it does not exist
    if mkdir -p "$target_path"; then
        made_new=true
    fi

    # Create directories "01" through "09" within the target directory
    for i in {1..9}; do
        if mkdir -p "${target_path}/$(printf "%02d" "$i")"; then
            made_new=true
        fi
    done

    if $made_new; then
        printf "Directories '01' through '09' already created in '%s'.\n" "$target_path" # logic here is reversed  
    else
        printf "Made new directories in '%s'.\n" "$target_path"
    fi
}

# Validate directory constraints
validate_directories() {
    local base_path="$1"
    for i in {1..9}; do
        local dir_peaks="${base_path}/peaks/$(printf "%02d" "$i")"
        local dir_labels="${base_path}/labels/$(printf "%02d" "$i")"
        local dir_peaks_water_overlay="${base_path}/peaks_water_overlay/$(printf "%02d" "$i")"
        local dir_water="${base_path}/water/$(printf "%02d" "$i")"

        local count_peaks=$(find "$dir_peaks" -maxdepth 1 -name "*.h5" | wc -l)
        local count_labels=$(find "$dir_labels" -maxdepth 1 -name "*.h5" | wc -l)
        local count_peaks_water_overlay=$(find "$dir_peaks_water_overlay" -maxdepth 1 -name "*.h5" | wc -l)
        local count_water=$(find "$dir_water" -maxdepth 1 -name "*.h5" | wc -l)

        # Check for water directory constraint
        if [ "$count_water" -ne 1 ]; then
            printf "Error: Directory '%s' does not contain exactly 1 .h5 file.\n" "$dir_water"
            return 1
        fi

        # Check for consistency across peaks, labels, and peaks_water_overlay
        if [ "$count_peaks" -eq "$count_labels" ] && [ "$count_peaks" -eq "$count_peaks_water_overlay" ]; then
            printf "All directories in '%s', '%s', and '%s' have %d images, as expected.\n" "$dir_peaks" "$dir_labels" "$dir_peaks_water_overlay" "$count_peaks"
        else
            printf "Mismatch found: %s has %d, %s has %d, %s has %d images to compare.\n" "$dir_peaks" "$count_peaks" "$dir_labels" "$count_labels" "$dir_peaks_water_overlay" "$count_peaks_water_overlay"
            return 1
        fi
    done
}

# Main function to control script flow
main() {
    # Loop through the target directories to create and populate them
    for dir in "${target_dirs[@]}"; do
        create_and_populate_dirs "${images_dir}/${dir}"
    done

    # Validate directories against the constraints
    if validate_directories "$images_dir"; then
        printf "Directory structure and image count verification completed successfully.\n"
    else
        printf "Errors detected during directory structure and image count verification.\n"
    fi
}

# Execute the main function
main "$@"
