#!/bin/bash

# Check if the argument is provided
if [[ $# -ne 1 ]]; then
    printf "Usage: %s <relative_path_to_images_directory>\n" "$0" >&2
    exit 1
fi

# Assign the provided argument to a variable
images_dir=$1

# Define the directories where directories "01" through "09" should be created
declare -a target_dirs=("labels" "peaks" "peaks_water_overlay" "water")

# Loop through the target directories
for dir in "${target_dirs[@]}"; do
  target_path="${images_dir}/${dir}"

  # Create the target directory if it does not exist
  mkdir -p "$target_path"

  # Create directories "01" through "09" within the target directory
  for i in {1..9}; do
    mkdir -p "${target_path}/$(printf "%02d" "$i")"
  done

  printf "Directories '01' through '09' created in '%s'.\n" "$dir"
done

printf "Directories creation process completed for all target directories in '%s'.\n" "$images_dir"
