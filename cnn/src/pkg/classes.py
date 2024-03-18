import os
import re
import h5py as h5
import torch
import shutil
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image, to_tensor
from skimage.feature import peak_local_max
from functools import lru_cache
from scipy.signal import find_peaks


class PathManager:
    def __init__(self):
        self.current_path = os.path.abspath(__file__)
        self.root = self.re_root(self.current_path)
        self.setup_directories()
        
    def setup_directories(self) -> None:
        self.images_dir = os.path.join(self.root, 'images')
        self.peak_images_dir = os.path.join(self.images_dir, 'peaks')
        self.water_images_dir = os.path.join(self.images_dir, 'data')
        self.processed_images_dir = os.path.join(self.images_dir, 'processed_images')
        self.preprocessed_images_dir = os.path.join(self.images_dir, 'preprocessed_images')
        self.label_images_dir = os.path.join(self.images_dir, 'labels')
        self.sim_dir = os.path.join(self.root, 'sim')
        self.sim_specs_dir = os.path.join(self.sim_dir, 'sim_specs')
        self.pdb_dir = os.path.join(self.sim_dir, 'pdb')
        self.sh_dir = os.path.join(self.sim_dir, 'sh')
        self.water_background_h5 = os.path.join(self.sim_dir, 'water_background.h5')
        
    def re_root(self, current_path: str) -> str:
        match = re.search("cxls_hitfinder", current_path)
        if match:
            root = current_path[:match.end()]
            return root
        else:
            raise Exception("Could not find the root directory. (cxls_hitfinder)\n", "Current working dir:", self.current_path)

<<<<<<< HEAD
    @lru_cache(maxsize=32)
    def get_path(self, path_name:str) -> str:
=======
    def get_path(self, path_name:str):
        # returns the path of the path_name
>>>>>>> progress-Everett
        paths_dict = {
            'root': self.root,
            'images_dir': self.images_dir,
            'sim_dir': self.sim_dir,
            'sim_specs_dir': self.sim_specs_dir,
            'peak_images_dir': self.peak_images_dir,
            'water_images_dir': self.water_images_dir,
            'processed_images_dir': self.processed_images_dir,
            'preprocessed_images_dir': self.preprocessed_images_dir,
            'label_images_dir': self.label_images_dir,
            'pdb_dir': self.pdb_dir,
            'sh_dir': self.sh_dir,
            'water_background_h5': self.water_background_h5,
        }
        return paths_dict.get(path_name, None)

    @lru_cache(maxsize=32)
    def get_peak_image_paths(self) -> list:
        return [os.path.join(self.peak_images_dir, f) for f in os.listdir(self.peak_images_dir) if f.endswith('.h5')]

    @lru_cache(maxsize=32)
    def get_water_image_paths(self) -> list:
        return [os.path.join(self.water_images_dir, f) for f in os.listdir(self.water_images_dir) if f.endswith('.h5')]

    @lru_cache(maxsize=32)
    def get_processed_images_paths(self) -> list:
        return [os.path.join(self.processed_images_dir, f) for f in os.listdir(self.processed_images_dir) if f.startswith('processed')]

    @lru_cache(maxsize=32)
    def get_label_images_paths(self) -> list:
        return [os.path.join(self.label_images_dir, f) for f in os.listdir(self.label_images_dir) if f.startswith('label')]
    
    # @lru_cache(maxsize=32)
    # def get_pdb_path(self, pdb_file):
    #     # returns the .pdb file path of the file name in the pdb directory
    #     return os.path.join(self.pdb_dir, pdb_file)
    # @lru_cache(maxsize=32)
    # def get_sh_path(self, sh_file):
    #     # returns the .sh file path of the file name in the sh directory
    #     return os.path.join(self.sh_dir, sh_file)
    
    @lru_cache(maxsize=32)
    def clean_sim(self) -> None:
        file_types = ['.err', '.out', '.sh']
        files_moved = False  # flag to check if any files were moved

        # manage the .err, .out, .sh files into sim_specs
        for file_type in file_types:
            for f in os.listdir(self.sim_dir):
                if f.endswith(file_type):
                    source_path = os.path.join(self.sim_dir, f)
                    dest_path = os.path.join(self.sim_specs_dir, f)

                    # checks to ensure that the file is not already in the sim_specs directory
                    if not os.path.exists(dest_path):
                        print(f"Moving {source_path} to {dest_path}\n")
                        shutil.move(source_path, dest_path)

                        # copy the .sh file to the sim directory
                        if file_type == '.sh':
                            sh_copy_path = os.path.join(self.sim_dir, f)
                            print(f"Copying {dest_path} to {sh_copy_path}\n")
                            shutil.copy(dest_path, sh_copy_path)
                        files_moved = True
        if not files_moved:
            print("clean_sim did not move any files\n\n")

class PeakImageDataset(Dataset):
    # for PyTorch DataLoader
    def __init__(self, paths, transform=None):
        self.peak_image_paths = paths.get_peak_image_paths()
        self.water_image_paths = paths.get_water_image_paths()
        self.label_image_paths = paths.get_label_images_paths()
        self.transform = transform if transform is not None else TransformToTensor()
        # checks 
        # assert len(self.peak_image_paths) == len(self.water_image_paths) == len(self.label_image_paths), "Mismatch in dataset sizes"
        print(f"Number of peak images: {len(self.peak_image_paths)}")
        print(f"Number of water images: {len(self.water_image_paths)}")
        print(f"Number of label images: {len(self.label_image_paths)}")

    def __getitem__(self, idx:int) -> tuple:
        peak_image = self.load_h5(self.peak_image_paths[idx])
        water_image = self.load_h5(self.water_image_paths[idx])
        label_image = self.load_h5(self.label_image_paths[idx])

        if self.transform:
            peak_image = self.transform(peak_image) 
            water_image = self.transform(water_image) # dimensions: C x H x W
            label_image = self.transform(label_image)
        return (peak_image, water_image), label_image

<<<<<<< HEAD
    def __len__(self) -> int:
=======
    #     # retrieve 
    #     match = re.search(r'(processed_)?img_7keV_clen(\d{2})_\d+\.h5', filepath)
    #     if match:
    #         camera_length_label = int(match.group(2)) - 1  # Convert '01', '02', '03', to 0, 1, 2, etc.
    #         camera_length = float(f"0.{match.group(2)}")
    #         protein = default_protein
    #         return (protein, camera_length, camera_length_label)
    #     else:
    #         print(f'Warning: Filename does not match expected pattern: {filepath}')
    #     return ('default', 0.0, -1)

    def __getitem__(self, idx):
        try: 
            peak_image = self.load_h5(self.peak_image_paths[idx])
            water_image = self.load_h5(self.water_image_paths[idx])
            label_image = self.load_h5(self.label_image_paths[idx])

            if self.transform:
                peak_image = self.transform(peak_image) 
                water_image = self.transform(water_image) # dimensions: B x C x H x W
                ### Changed long to float 
                label_image = self.transform(label_image).float() # long tensor for cross-entropy loss
            return (peak_image, water_image), label_image
        except Exception as e:
            raise IndexError(f"Error accessing index {idx}: {str(e)}")

    def __len__(self):
>>>>>>> progress-Everett
        return len(self.peak_image_paths)

    def load_h5(self, filepath:str) -> np.ndarray:
        with h5.File(filepath, 'r') as file:
            return np.array(file['entry/data/data'])
        
class TransformToTensor:
    def __call__(self, image:np.ndarray) -> torch.Tensor:
        """
        Converts a numpy array to a PyTorch tensor while ensuring:
        - The output is in B x C x H x W format for a single image.
        - The tensor is of dtype float32.
        - For grayscale images, a dummy channel is added to ensure compatibility.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            torch.Tensor: The transformed image tensor.
        """
        if image.ndim == 2:  # For grayscale images
            image = np.expand_dims(image, axis=0)  # Add channel dimension 
        else : 
            raise ValueError(f"Image has invalid dimensions: {image.shape}")
        image_tensor = torch.from_numpy(image).float().to(dtype=torch.float32)  # Convert to tensor with dtype=torch.float32
        return image_tensor # dimensions: C x H x W 
    
class DataPreparation:
    def __init__(self, paths, dataset, batch_size=32):
        self.paths = paths
        self.batch_size = batch_size
        self.dataset = dataset
    
    def prep_data(self) -> tuple:
        """
        Prepares and splits the data into training and testing datasets.
        Applies transformations and loads them into DataLoader objects.
        """
        # Split the dataset into training and testing sets.
        num_items = len(self.dataset)
        num_train = int(0.8 * num_items)
        num_test = num_items - num_train
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [num_train, num_test])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)    
            
        print("Data prepared.")
        print(f"Train size: {len(train_dataset)}")
        print(f"Test size: {len(test_dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches in train_loader: {len(train_loader)} \n\n")
        
        return train_loader, test_loader # returns train/test tensor data loaders
    
class PeakThresholdProcessor: 
    def __init__(self, threshold_value=0) -> None:
        self.threshold_value = threshold_value

    def set_threshold_value(self, new:int) -> None:
        self.threshold_value = new
    
    def get_local_maxima(self) -> list:
        image_1d = self.image.flatten()
        peaks, _ = find_peaks(image_1d, height=self.threshold_value)
        coordinates = [self.flat_to_2d(idx) for idx in peaks]
        return coordinates
        
    def flat_to_2d(self, index:int, shape:tuple) -> tuple:
        rows, cols = shape
        return (index // cols, index % cols) 
    
    def process_image(self, image_tensor:torch.Tensor) -> np.ndarray:
        image_np = image_tensor.squeeze().cpu().numpy()
        return np.argwhere(image_np > self.threshold_value)
    
class ImageProcessor:
    # with concurrent processing
    def __init__(self, water_background_array:np.ndarray) -> None:
        self.water_background_array = water_background_array

    @staticmethod
    def load_h5_image(file_path:str) -> np.ndarray:        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            with h5.File(file_path, 'r') as f:
                data = f['entry/data/data']
                return np.array(data) 
        except KeyError:
            raise ValueError(f"Dataset 'entry/data/data' not found in file: {file_path}")
        except IOError as e:
            if 'unable to open file' in str(e).lower():
                raise IOError(f"File cannot be opened, might be corrupt or not an HDF5 file: {file_path}")
            else:
                raise IOError(f"Failed to read the file {file_path}: {e}")
    
    @staticmethod
    def update_clen(file_path:str, clen:float) -> None:
        """
        Updates an HDF5 file with the clen values.
        """
        try:
            with h5.File(file_path, 'a') as f:  # Open in append mode
                f.attrs['clen'] = clen
                print(f"\t'clen' value updated for {file_path}")     
        except Exception as e:
            print(f"Error updating 'clen' value for {file_path}: {e}")
            raise e
               
    def apply_water_background(self, peak_image_array: np.ndarray) -> np.ndarray:
        return peak_image_array + self.water_background_array

    def heatmap(self, peak_image_array: np.ndarray, threshold_value=0, min_distance=3) -> tuple:
        """
        Generates a heatmap based on detected Bragg peaks in an image.

        Args:
            peak_image_array: The peak image array.
            threshold_value: The intensity threshold for detecting peaks.
            min_distance: The minimum number of pixels separating peaks.

        Returns:
            A labeled tensor of the heatmap where peaks are detected, 
            with the same dimensions as the input array.
        """
        coordinates = peak_local_max(peak_image_array, min_distance=min_distance, threshold_abs=threshold_value)
        labeled_array = np.zeros(peak_image_array.shape)
        labeled_array[tuple(coordinates.T)] = 1 # efficiently assigns 1 to the coordinates
        return to_tensor(labeled_array).unsqueeze(0), coordinates  # Ensure it matches expected dimensions [C, H, W]        
    
    def process_directory(self, paths, threshold_value:int, clen:float) -> None:
        """
        Processes all peak images in the directory, applies the water background,
        generates label heatmaps for detecting Bragg peaks, and updates the HDF5 files
        with a 'clen' attribute for peak images, water images, and label images.
        """
        try:
            # directories 
            peaks_path = paths.get_path('peak_images_dir')
            processed_out_path = paths.get_path('processed_images_dir')
            label_out_path = paths.get_path('label_images_dir')
                    
            # process peak/water/heatmap images 
            for f in os.listdir(peaks_path):
                if f.endswith('.h5') and f != 'water_background.h5':
                    full_path = os.path.join(peaks_path, f)
                    image = self.load_h5_image(full_path)
                    processed_image = self.apply_water_background(image)
                    heatmap_tensor, _ = self.heatmap(processed_image, threshold_value)
                    
                    processed_save_path = os.path.join(processed_out_path, f"processed_{f}")
                    labeled_save_path = os.path.join(label_out_path, f"labeled_{f}")
                    
                    # save processed images and update with clen values
                    # processed
                    with h5.File(processed_save_path, 'w') as pf:
                        pf.create_dataset('entry/data/data', data=processed_image)
                        print(f"\tProcessed images saved: {processed_save_path}")
                    
                    # heatmap
                    with h5.File(labeled_save_path, 'w') as lf:
                        lf.create_dataset('entry/data/data', data=heatmap_tensor.squeeze().cpu().numpy())
                        print(f"\tLabeled images saved: {processed_save_path}")

                    # update clen for peak/water/heatmap images
                    self.update_clen(processed_save_path, clen) # processed
                    self.update_clen(labeled_save_path, clen) # heatmap
                    self.update_clen(full_path, clen) # peak
                    
                print(f"Processed and labeled images saved:\n {processed_save_path} --> {labeled_save_path}\n\n")            
            print("Processing complete and 'clen' values updated.")

        except Exception as e:
            print(f"Error processing directory: {e}")
            raise e
        
