import os
import concurrent.futures
from pkg import c, m, f

def main():
    """
    Reads the water_background.h5 file, processes the peak images in the peak_images_dir,
    applies the water background, generates label heatmaps for detecting Bragg peaks,
    and updates the HDF5 files with a 'clen' attribute for peak images, water images, and label images.
    """
    paths = c.PathManager()
    threshold_value = 1  # threshold value for peak detection 
    # from peaks images this is pure signal
    clen_values = {0.1, 0.2, 0.3} # in mm

<<<<<<< HEAD
    # load water noise images
    water_background = f.load_h5(paths.get_path('water_background_h5'))
    # water_background = c.ImageProcessor().load_h5_image(paths.get_path('water_background_h5'))

    # clen from user 
    clen = 0.1  # DEBUG: remove this line

    # clen_input = input("Enter the 'clen' value (e.g., 0.1, 0.2, 0.3) in mm: ")
    # try:
    #     clen = float(clen_input)
    # except ValueError:
    #     print("Invalid 'clen' value. Please enter a numeric value.")
    #     return
    
    ## DEBUG: remove this block
    # confirm clen value
    # for i in range(1, 4):  # 3 confirmations
    #     confirm = input(f"Confirm 'clen' value {clen} mm (Attempt {i}/3, type 'y' to confirm): ")
    #     if confirm.lower() == 'y':
    #         print(f'clen confirmed {i}/3 times.')
    #         continue
    #     elif i == 3:
    #         print(f"'clen' value {clen} mm confirmed successfully.")
    #     else:
    #         print(f"'clen' confirmation failed at attempt {i}.")
    #         return
    
    ip = c.ImageProcessor(water_background)
    
    conf = input(f"Are you sure you want to process the peak images in '{paths.get_path('peak_images_dir')}'\n\nand apply 'water_background.h5' from '{paths.get_path('water_background_h5')}'\n\noutput processed images to '{paths.get_path('processed_images_dir')}'\n\nand output labels to '{paths.get_path('label_images_dir')}'\n\nwith 'clen' value {clen} mm? (y/n): ")
=======
    water_background = c.ImageProcessor.load_h5_image(paths.get_path('water_background_h5'))
    processor = c.ImageProcessor(water_background)

    conf = input(f"Are you sure you want to process ... \n\n the peak images: {paths.get_path('peak_images_dir')} \n\n ...  and apply (water_background.h5) {paths.get_path('water_background_h5')} \n\n ... and output processed: {paths.get_path('processed_images_dir')}, \n\n ... and output labels: {paths.get_path('label_images_dir')} \n\n (y/n): ")
>>>>>>> progress-Everett
    if conf.lower() == 'y':
        ip.process_directory(paths=paths, threshold_value=threshold_value, clen=clen)
    else:
        print("Operation cancelled.")

if __name__ == '__main__':
    main()
