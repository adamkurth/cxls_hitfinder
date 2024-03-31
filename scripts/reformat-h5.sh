#!/bin/bash

set -euo pipefail

# Extract keV and clen values from a filename
extract_kev_clen() {
  local filename="$1"
  local kev clen

  if [[ $filename =~ master_[0-9]+_([0-9]+)keV_clen([0-9]+)_ ]] || [[ $filename =~ img_([0-9]+)keV_clen([0-9]+)_ ]]; then
    kev="${BASH_REMATCH[1]}"
    clen="${BASH_REMATCH[2]}"
    printf "%s %s\n" "$kev" "$clen"
  else
    printf "Error: Filename '%s' does not contain keV and clen values.\n" "$filename" >&2
    return 1
  fi
}

# Reformat filenames, excluding files in water/ directory and starting with "empty"
reformat_filenames() {
  local directory="$1"
  local img_count=1

  while IFS= read -r -d '' subdir; do
    # Skip water/ directory
    if [[ $subdir =~ /water$ ]]; then
      printf "Skipping water directory: %s\n" "$subdir" >&2
      continue
    fi

    local files=("$subdir"/*[!empty]*.h5)  # Exclude files starting with "empty"
    for file in "${files[@]}"; do
      # Skip if no file found due to the glob
      [[ -e $file ]] || continue

      local filename
      filename=$(basename -- "$file")
      
      local kev clen
      if ! read -r kev clen <<< "$(extract_kev_clen "$filename")"; then
        printf "Skipping file: %s\n" "$filename" >&2
        continue
      fi

      local new_filename_format
      new_filename_format="img_${kev}keV_clen${clen}_$(printf "%05d" "$img_count").h5"
      local new_filepath
      new_filepath="${subdir}/${new_filename_format}"
      
      mv -- "$file" "$new_filepath"
      printf "Renamed '%s' to '%s'\n" "$filename" "$new_filename_format"
      ((img_count++))
    done
  done < <(find "$directory" -type d -print0 | grep -vzZ "$directory/water")
}

main() {
  if [[ $# -eq 0 ]]; then
    printf "Error: No directory provided.\nUsage: %s <directory>\n" "$0" >&2
    return 1
  fi

  local target_directory="$1"
  reformat_filenames "$target_directory"
}

main "$@"
