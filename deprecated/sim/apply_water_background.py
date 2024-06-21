import os
import h5py
import numpy as np
import reborn

# export PYTHONPATH="/Users/adamkurth/Documents/vscode/CXFEL_Image_Analysis/CXFEL/reborn_dev:$PYTHONPATH"

def load_hdf5_image(file_path):
    """Load an HDF5 image."""
    with h5py.File(file_path, 'r') as f:
        return np.array(f['entry/data/data'])

def apply_water_background(image_array, background_array):
    """Apply the water background to the image. This function is a placeholder
    and should be replaced with actual logic to apply the background."""
    # This is a simplified example. Replace it with the actual logic for combining the images.
    return image_array + background_array

def process_directory(directory_path, background_path):
    """Process all HDF5 images in a directory by applying a water background."""
    background = load_hdf5_image(background_path)
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.h5') and filename != 'water_background.h5':
            full_path = os.path.join(directory_path, filename)
            
            print(f"Processing {filename}...")
            image = load_hdf5_image(full_path)
            processed_image = apply_water_background(image, background)
            
            # Save the processed image
            save_path = os.path.join(directory_path, f"processed_{filename}")
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('entry/data/data', data=processed_image)
            
            print(f"Saved processed image to {save_path}")

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_images_path = os.path.join(base_path, 'sim', 'water_images')
    print(processed_images_path)
    
    temp_path = os.path.join(base_path, 'images', 'preprocessed_images')
    print(temp_path)

    water_background_path = os.path.join(base_path, 'sim/water_background.h5')
    print(water_background_path)    
    
    
    # image_path = input("Enter the directory path containing the HDF5 files: ")
    # image_path = os.path.expanduser(directory) # Expand ~ to the user's home directory
    
    # images path 
    
    # for water_background.h5
    # /Users/adamkurth/Documents/vscode/CXFEL_Image_Analysis/CXFEL/cxls_hitfinder/sim/water_background.h5
    
    # background_path = input("Enter the full path to the water background HDF5 file: ")
    # background_path = os.path.expanduser(background_path) # Expand ~ to the user's home directory
    
    confirmation = input(f"Process all HDF5 files in {temp_path} \n\n ... and apply the water background from {water_background_path} \n\n ... while outputting processes images in {processed_images_path}? \n\n (yes/no): ")
    
    if confirmation.lower() == 'yes':
        process_directory(processed_images_path, water_background_path)
    else:
        print("Operation canceled.")

if __name__ == "__main__":
    main()
