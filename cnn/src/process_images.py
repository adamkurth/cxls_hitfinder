import os

from pkg import c, m, f

def main():
    """
    reads the water_background.h5 file and processes the peak images in the peak_images_dir
    """
    paths = c.PathManager()
    threshold_value = 1000  # Adjust as needed

    water_background = c.ImageProcessor.load_h5_image(paths.get_path('water_background_h5'))
    processor = c.ImageProcessor(water_background)

    conf = input(f"Are you sure you want to process ... \n\n the peak images: {paths.get_path('peak_images_dir')} \n\n ...  and apply (water_background.h5) {paths.get_path('water_background_h5')} \n\n ... and output processed: {paths.get_path('processed_images_dir')}, \n\n ... and output labels: {paths.get_path('label_images_dir')} \n\n (y/n): ")
    if conf.lower() == 'y':
        processor.process_directory(paths=paths, threshold_value=threshold_value)
    else:
        print("Operation cancelled.")
        
if __name__ == '__main__':
    main()
