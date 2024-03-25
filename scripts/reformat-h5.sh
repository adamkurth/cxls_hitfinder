#!/bin/bash

# Extract keV and clen values from a filename, handling both original and already renamed files
extract_kev_clen() {
  local filename=$1
  local kev clen

  # First pattern matches original naming convention, second pattern matches already renamed files
  if [[ $filename =~ master_[0-9]+_([0-9]+)keV_clen([0-9]+)_ ]] || [[ $filename =~ img_([0-9]+)keV_clen([0-9]+)_ ]]; then
    kev=${BASH_REMATCH[1]}
    clen=${BASH_REMATCH[2]}
    echo "$kev $clen"
  else
    printf "Error: Filename does not contain keV and clen values.\n" >&2
    return 1
  fi
}

# Function to reformat filenames in all subdirectories
reformat_filenames() {
  local directory=$1
  local img_count=1

  # Reset IFS to default to avoid issues in file paths with spaces
  IFS=$' \t\n'

  find "$directory" -type d -print0 | while IFS= read -r -d '' subdir; do
    for file in "$subdir"/*.h5; do
      if [[ -f $file ]]; then
        local filename=$(basename -- "$file")
        read -r kev clen <<< $(extract_kev_clen "$filename")
        if [[ $? -ne 0 ]]; then
          printf "Skipping file: %s\n" "$filename" >&2
          continue
        fi

        local new_filename_format="img_${kev}keV_clen${clen}_$(printf "%05d" "$img_count").h5"
        local new_filepath="${subdir}/${new_filename_format}"
        mv -- "$file" "$new_filepath"
        printf "Renamed '%s' to '%s'\n" "$file" "$new_filepath"
        ((img_count++))
      fi
    done
  done
}

# Main function to control script flow
main() {
  if [[ $# -eq 0 ]]; then
    printf "Error: No directory provided.\nUsage: %s <directory>\n" "$0" >&2
    return 1
  fi

  local target_directory=$1

  reformat_filenames "$target_directory"
}

# Script execution starts here
main "$@"
