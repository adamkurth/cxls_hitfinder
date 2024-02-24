#!/bin/bash

# Function to reformat filenames within a given directory based on their extension
reformat_filenames() {
  local directory=$1
  local prefix="img_7keV_clen01"
  local h5_count=1
  local sh_count=1
  local err_count=1
  local out_count=1

  # Check if directory is provided
  if [[ -z "$directory" ]]; then
    printf "Usage: %s <directory>\n" "$0" >&2
    return 1
  fi

  # Ensure the directory exists
  if [[ ! -d "$directory" ]]; then
    printf "Directory '%s' not found.\n" "$directory" >&2
    return 1
  fi

  # Change to the specified directory
  cd "$directory" || return 1

  # Handle .h5 files with counting
  for file in *.h5; do
    # Skip if no matching files are found
    [[ -e $file ]] || continue

    local new_name="${prefix}_$(printf "%05d" "$h5_count").h5"
    printf "Renaming '%s' to '%s'\n" "$file" "$new_name"
    mv -- "$file" "$new_name"
    ((h5_count++))
  done

  # Handle .sh, .err, and .out files by renaming with individual counts for each extension
  for file in *.{sh,err,out}; do
    # Skip if no matching files are found
    [[ -e $file ]] || continue

    local extension="${file##*.}"
    local new_name

    # Determine the correct new name based on the extension
    case "$extension" in
      sh)
        new_name="${prefix}.sh"
        ((sh_count++))
        ;;
      err)
        new_name="${prefix}.err"
        ((err_count++))
        ;;
      out)
        new_name="${prefix}.out"
        ((out_count++))
        ;;
    esac

    printf "Renaming '%s' to '%s'\n" "$file" "$new_name"

    # Handle potential conflicts by appending a count to the filename
    if [[ -e $new_name ]]; then
      local base="${new_name%.*}"
      local counter=2 # Start from 2 since 1 is implied in the initial name
      local conflict_name="${base}_${counter}.${extension}"
      while [[ -e $conflict_name ]]; do
        ((counter++))
        conflict_name="${base}_${counter}.${extension}"
      done
      new_name=$conflict_name
    fi

    mv -- "$file" "$new_name"
  done

  # Return to the original directory
  cd - > /dev/null || return 1
}

# Main function to handle script logic
main() {
  local target_directory=$1

  # Validate input
  if [[ -z "$target_directory" ]]; then
    printf "Error: No directory specified.\nUsage: %s <directory>\n" "$0" >&2
    return 1
  fi

  # Call the reformat function with the provided directory
  reformat_filenames "$target_directory"
}

# Script execution starts here
main "$@"
