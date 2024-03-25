#!/bin/bash

# Extract keV and clen values from a filename
extract_kev_clen() {
  local filename=$1
  local kev clen

  if [[ $filename =~ master_[0-9]+_([0-9]+)keV_clen([0-9]+)_ ]]; then
    kev=${BASH_REMATCH[1]}
    clen=${BASH_REMATCH[2]}
    echo "$kev $clen"
  else
    printf "Error: Filename does not contain keV and clen values.\n" >&2
    return 1
  fi
}

# Reformat filenames based on extracted keV and clen values
reformat_filenames() {
  local directory=$1
  local subdirs=("01" "02" "03" "04" "05" "06" "07" "08" "09")

  for subdir in "${subdirs[@]}"; do
    local path="$directory/$subdir"
    if [[ ! -d $path ]]; then
      printf "Directory '%s' not found.\n" "$path" >&2
      continue
    fi

    local files_found=0
    for file in "$path"/*.h5; do
      if [[ -f $file ]]; then
        files_found=1
        local filename=$(basename -- "$file")
        read -r kev clen <<< $(extract_kev_clen "$filename")
        if [[ $? -ne 0 ]]; then
          printf "Skipping file: %s\n" "$filename" >&2
          continue
        fi

        local new_filename="img_${kev}keV_clen${clen}.h5"
        mv -- "$file" "$path/$new_filename"
        printf "Renamed '%s' to '%s'\n" "$filename" "$new_filename"
      fi
    done

    if [[ $files_found -eq 0 ]]; then
      printf "No .h5 files found in '%s'.\n" "$path" >&2
    fi
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
