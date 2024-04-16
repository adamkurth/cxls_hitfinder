#!/bin/bash

# from Agave
# use this script to create the directory structure for the experiment
# edit this for further development

# Directory naming components
camera_lengths=("0.15m" "0.25m" "0.35m")
photon_energies=("6keV" "7keV" "8keV")

# Function to pad the experiment ID
pad_id() {
    printf "%02d" $1
}

# Find the highest current ID to ensure we increment correctly
max_id=0
for dir in */ ; do
    id=$(echo $dir | cut -d'_' -f1)
    [[ $id =~ ^[0-9]+$ ]] && ((id > max_id)) && max_id=$id
done

# Increment max_id for the next directory creation
max_id=$((max_id+1))

# Create directories based on the incremented ID
for camera_length in "${camera_lengths[@]}"; do
    for photon_energy in "${photon_energies[@]}"; do
        if [[ $max_id -le 9 ]]; then  # Ensure we only create up to 09 directories for now
            dirname=$(pad_id $max_id)_${camera_length}_${photon_energy}
            if [ ! -d "$dirname" ]; then  # Check if the directory does not already exist
                mkdir -p "$dirname"
                echo "Created directory: $dirname"
            else
                echo "Directory already exists: $dirname, skipping..."
            fi
            max_id=$((max_id+1))  # Increment for the next directory
        fi
    done
    if [[ $max_id -gt 9 ]]; then  # Exit the loop once we reach the target count
        break
    fi
done

echo "Directory creation complete."

# edit this for further development