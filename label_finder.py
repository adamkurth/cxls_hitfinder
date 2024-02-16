import os
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import glob
import h5py 
from scipy.signal import find_peaks

class PeakThresholdProcessor: 
    def __init__(self, image_array, threshold_value=0):
        self.image_array = image_array
        self.threshold_value = threshold_value
    
    def set_threshold_value(self, new_threshold_value):
        self.threshold_value = new_threshold_value
    
    def get_coordinates_above_threshold(self):  
        coordinates = np.argwhere(self.image_array > self.threshold_value)
        return coordinates
    
    def get_local_maxima(self):
        image_1d = self.image_array.flatten()
        peaks, _ = find_peaks(image_1d, height=self.threshold_value)
        coordinates = [self.flat_to_2d(idx) for idx in peaks]
        return coordinates
        
    def flat_to_2d(self, index):
        shape = self.image_array.shape
        rows, cols = shape
        return (index // cols, index % cols) 
    
class ArrayRegion:
    def __init__(self, array):
        self.array = array
        self.x_center = 0
        self.y_center = 0
        self.region_size = 0
    
    def set_peak_coordinate(self, x, y):
        self.x_center = x
        self.y_center = y
    
    def set_region_size(self, size):
        #limit that is printable in terminal
        self.region_size = size
        max_printable_region = min(self.array.shape[0], self.array.shape[1]) //2
        self.region_size = min(size, max_printable_region)
    
    def get_region(self):
        x_range = slice(self.x_center - self.region_size, self.x_center + self.region_size+1)
        y_range = slice(self.y_center - self.region_size, self.y_center + self.region_size+1)
        region = self.array[x_range, y_range]
        return region

    def extract_region(self, x_center, y_center, region_size):
            self.set_peak_coordinate(x_center, y_center)
            self.set_region_size(region_size)
            region = self.get_region()

            # Set print options for better readability
            np.set_printoptions(precision=8, suppress=True, linewidth=120, edgeitems=7)
            return region
    
def load_data():
    base_directory = os.path.dirname(os.path.abspath(__file__))
    # images_directory = os.path.join(base_directory, 'images')
    # test_image_path = os.path.join(images_directory, '9_18_23_high_intensity_3e8keV-2.h5')
    # file_path = test_image_path

    # for water background image
    file_path = os.path.join(base_directory, 'sim', 'water_images', 'processed_submit_7keV_clen01-2.h5')
    try:
        with h5.File(file_path, 'r') as f:
            data = f['entry/data/data'][:]
        return data, file_path
    except Exception as e:
        raise OSError(f"Failed to read {file_path}: {e}")
    
def load_file_h5(file_path):
    """Load an HDF5 file and return the data as a numpy array.

    Args:
        file_path (string): path to the HDF5 file.

    Returns:
        numpy array: loads data from entry/data/data
    """
    try:
        with h5.File(file_path, "r") as f:
            data = np.array(f["entry/data/data"][()])
            print(f"File loaded successfully: \n {file_path}")
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while loading the file: {str(e)}")
                  
def display_peak_regions(image_array, coordinates):
    """
    Display the peak regions on an image array.

    Parameters:
    image_array (numpy.ndarray): The image array to display.
    coordinates (list): List of (x, y) coordinates of the peak regions.

    Returns:
    None
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image_array, cmap='viridis')
    # plt.title("Intensity")
    for i, (x, y) in enumerate(coordinates, 1):
        plt.scatter(y, x, marker='x', color='red')
        plt.text(y, x, f'{i+1}', color='white', ha='right')    

    plt.title(f"Peak Regions (size={image_array.shape})")
    plt.show()

def is_peak(image_data, coordinate, neighborhood_size=3):
    """
    Check if the specified coordinate is a peak within its neighborhood.
    """
    x, y = coordinate
    region = ArrayRegion(image_data)
    
    neighborhood = region.extract_region(x, y, neighborhood_size)    
    center = neighborhood_size, neighborhood_size
    if coordinate is type(int):
        is_peak = neighborhood[center] == np.max(neighborhood) and neighborhood[center] > 0
    else: 
        is_peak = neighborhood[center] == np.max(neighborhood)
    # print(f'Peak found at {coordinate}' if is_peak else f'No peak found at {coordinate}')
    return is_peak

def validate(manual, script, image_array):
    """
    Validates the results of manual and script-based peak detection by comparing the coordinates.
    
    Args:
        manual (list): List of coordinates from manual peak detection.
        script (list): List of coordinates from script-based peak detection.
        image_array (ndarray): Array representing the image.
        
    Returns:
        tuple: A tuple containing three sets of confirmed coordinates:
            - confirmed_common: Coordinates that are common to both manual and script-based detection and are confirmed as peaks.
            - confirmed_unique_manual: Coordinates that are unique to manual detection and are confirmed as peaks.
            - confirmed_unique_script: Coordinates that are unique to script-based detection and are confirmed as peaks.
    """
    
    manual_set = set(manual)
    script_set = set(script)
    common = manual_set.intersection(script_set)
    unique_manual = manual_set.difference(script_set)
    unique_script = script_set.difference(manual_set)
    
    print(f'common: {common}\n')
    print(f'unique_manual: {unique_manual}\n')
    print(f'unique_script: {unique_script}\n')
    
    confirmed_common = {coord for coord in common if is_peak(image_array, coord)}
    confirmed_unique_manual = {coord for coord in unique_manual if is_peak(image_array, coord)}
    confirmed_unique_script = {coord for coord in unique_script if is_peak(image_array, coord)}
    
    print(f'confirmed_common: {confirmed_common}\n')
    print(f'confirmed_unique_manual: {confirmed_unique_manual}\n')
    print(f'confirmed_unique_script: {confirmed_unique_script}\n')

    return confirmed_common, confirmed_unique_manual, confirmed_unique_script

def view_neighborhood(coordinates, image_data):
    """
    Displays the neighborhood of a given coordinate in an image.

    Args:
        coordinates (list): List of coordinate tuples (x, y).
        image_data (ndarray): 2D array representing the image data.

    Returns:
        None
    """
    
    coordinates = list(coordinates)
    
    print('\n', "List of coordinates:")
    for i, (x, y) in enumerate(coordinates, 1):
        print(f'{i}. ({x}, {y})')

    while True:
        ans = input(f'Which coordinate do you want to view? (1-{len(coordinates)} or "q" to quit) \n')

        if ans.lower() == "q":
            print("Exiting")
            break

        try:
            ans = int(ans) - 1  # Convert to 0-based index
            if 0 <= ans < len(coordinates):
                coordinate = coordinates[ans]
                x, y = coordinate
                
                region = ArrayRegion(image_data)
                neighborhood = region.extract_region(x_center=x, y_center=y, region_size=3)
                                
                # Determine if the coordinate is a peak
                center = neighborhood.shape[0] // 2, neighborhood.shape[1] // 2
                is_peak = neighborhood[center] == np.max(neighborhood)
                
                print(f'Neighborhood for ({x}, {y}):')
                print(neighborhood)
                
                if is_peak:
                    print("This is a peak.")
                else:
                    print("This is not a peak.")
        
                # continue?
                cont = input('Do you want to view another neighborhood? (Y/n) ').strip().lower()
                if cont in ['n', 'no']:
                    print("Exiting")
                    break
                else:
                    view_neighborhood(coordinates, image_data)  # Recursive call
            else:
                print(f"Please enter a number between 1 and {len(coordinates)}.")
                
        except ValueError:
            print("Invalid choice. Please enter a number or 'q' to quit.")
        except IndexError:
            print("Invalid choice. Try again.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            
def generate_labeled_image(image_data, peak_coordinates, neighborhood_size):
    """
    Generate a labeled image based on detected peaks.
    """
    labeled_image = np.zeros_like(image_data)
    for (x, y) in peak_coordinates:
        if is_peak(image_data, (x, y), neighborhood_size):
            labeled_image[x, y] = 1 # label as peak
    print('Generated labeled image.')
    return labeled_image

def save_labeled_h5(labeled_array):
    """
    Save the labeled image array to an HDF5 file.
    """
    generate_image = input("Do you want to generate a new .h5 image? (yes/no): ")
    if generate_image.lower() == "yes":
        file_name = input("Enter the file name for the new .h5 image: ")
        with h5py.File(file_name, "w") as f:
            entry = f.create_group("entry/data/data")
            entry.create_dataset("labeled_array", data=labeled_array)
            # path for image is: entry/data/data/labeled_array
        print(f"New .h5 image '{file_name}' generated.")
    else:
        print("No new .h5 image generated.")
      
def visualize(conf_common, conf_unique_manual, conf_unique_script, image_array):
    """
    Visualizes the peaks found in an image.

    Parameters:
    conf_common (list): List of common peak coordinates.
    conf_unique_manual (list): List of unique manual peak coordinates.
    conf_unique_script (list): List of unique script peak coordinates.
    image_array (numpy.ndarray): Array representing the image.

    Returns:
    None
    """
    
    def add_jitter(coordinates, jitter=10):
        """
        Adds random jitter to a list of coordinates.
        """
        return [(x + np.random.uniform(-jitter, jitter), 
                 y + np.random.uniform(-jitter, jitter)) for x, y in coordinates]
        
    common = add_jitter(list(conf_common))
    manual = add_jitter(list(conf_unique_manual))
    script = add_jitter(list(conf_unique_script))
    
    plt.imshow(image_array, cmap='viridis')
    
    if common:
        common_x, common_y = zip(*common)
        plt.scatter(common_x, common_y, color='red', s=50, marker='o', alpha=0.7, label='Common Peaks')
            
    if manual:
        manual_x, manual_y = zip(*manual)
        plt.scatter(manual_x, manual_y, color='blue', s=50, marker='s', alpha=0.7, label='Unique Manual')    
    
    if script:
        script_x, script_y = zip(*script)
        plt.scatter(script_x, script_y, color='green', s=50, marker='^', alpha=0.7, label='Unique Script')
    
    plt.title('Peaks Found')
    plt.legend()
    plt.show()

def main(file_path, threshold_value, display=True):
    """
    Main function to process an image file and display detected peaks.
    """
    image_array = load_file_h5(file_path) # load_file_h5
    threshold_processor = PeakThresholdProcessor(image_array, threshold_value)
    coordinates = threshold_processor.get_coordinates_above_threshold()
    # display    
    print(f'Found {len(coordinates)} peaks above threshold {threshold_value}')
    if display:
            display_peak_regions(image_array, coordinates)
    return coordinates

if __name__ == "__main__":
    image_data, file_path = load_data() # searches in images/
    threshold = 1000
    coordinates = main(file_path, threshold, display=True)

    # converts to list
    coordinates = [tuple(coord) for coord in coordinates]   
    # print results
    print('\n', f'threshold: {threshold} \n')
    print('\n', f'manually found coordinates {coordinates}\n')

    threshold_processor = PeakThresholdProcessor(image_data, threshold)
    peaks = threshold_processor.get_local_maxima()
    
    # validates that peaks in script are the same as manually found peaks
    confirmed_common_peaks, confirmed_unique_manual, confirmed_unique_script = validate(coordinates, peaks, image_data) # manually found and script found
    confirmed_common_peaks = list(confirmed_common_peaks)
    
    # prints menu to view neighborhood
    view_neighborhood(confirmed_common_peaks, image_data)
    
    # return labeled array for training
    labeled_array = generate_labeled_image(image_data, confirmed_common_peaks, neighborhood_size=5)
    save_labeled_h5(labeled_array)
    
    # visualize(confirmed_common_peaks, confirmed_unique_manual, confirmed_unique_script, image_data)