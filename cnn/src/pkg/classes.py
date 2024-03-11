import os
import re
import h5py as h5
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from torchvision import transforms
from collections import namedtuple
from skimage.feature import peak_local_max
from scipy.signal import find_peaks
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image

class PathManager:
    def __init__(self):
        # grabs peaks and processed images from images directory
        self.current_path = os.path.abspath(__file__)
        self.root = self.__re_root__(self.current_path)
        self.images_dir = os.path.join(self.root, 'images') #/images
        self.sim_dir = os.path.join(self.root, 'sim')  #/sim
        self.sim_specs_dir = os.path.join(self.sim_dir, 'sim_specs') # sim/sim_specs
        # KEY
        self.peak_images_dir = os.path.join(self.images_dir, 'peaks') # images/peaks
        self.water_images_dir = os.path.join(self.images_dir, 'data') # images/data
        #
        # built just in case further development is needed
        self.processed_images_dir = os.path.join(self.images_dir, 'processed_images') # images/processed_images
        self.preprocessed_images_dir = os.path.join(self.images_dir, 'preprocessed_images') # images/preprocessed_images
        self.label_images_dir = os.path.join(self.images_dir, 'labels') # images/labels
        self.pdb_dir = os.path.join(self.sim_dir, 'pdb') # /sim/pdb
        self.sh_dir = os.path.join(self.sim_dir, 'sh') # /sim/sh
        self.water_background_h5 = os.path.join(self.sim_dir, 'water_background.h5') # /sim/water_background.h5

    def __re_root__(self, current_path):
        match = re.search("cxls_hitfinder", current_path)
        if match:
            root = current_path[:match.end()]
            return root
        else:
            raise Exception("Could not find the root directory. (cxls_hitfinder)\n", "Current working dir:", self.current_path)

    def __get_path__(self, path_name):
        # returns the path of the path_name
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

    def __get_all_paths__(self):
        # returns a namedtuple of all paths for easier access
        PathsOutput = namedtuple('PathsOutput', ['root', 'images_dir', 'sim_dir', 'peak_images_dir', 'water_images_dir', 'processed_images_dir', 'label_images_dir', 'pdb_dir', 'sh_dir', 'water_background_h5'])
        return PathsOutput(self.root, self.images_dir, self.sim_dir, self.peak_images_dir, self.processed_images_dir, self.label_images_dir, self.pdb_dir, self.sh_dir, self.water_background_h5)

    def __get_peak_images_paths__(self):
        # searches the peak images directory for .h5 files and returns a list of paths
        return [os.path.join(self.peak_images_dir, f) for f in os.listdir(self.peak_images_dir) if f.endswith('.h5')]

    def __get_water_images_paths__(self):
        # searches the water images directory (images/data)for .h5 files and returns a list of paths
        return [os.path.join(self.water_images_dir, f) for f in os.listdir(self.water_images_dir) if f.endswith('.h5')]

    def __get_processed_images_paths__(self):
        # searches the processed images directory for .h5 files and returns a list of paths
        return [os.path.join(self.processed_images_dir, f) for f in os.listdir(self.processed_images_dir) if f.startswith('processed')]

    def __get_label_images_paths__(self):
        # searches the label images directory for .h5 files and returns a list of paths
        return [os.path.join(self.label_images_dir, f) for f in os.listdir(self.label_images_dir) if f.startswith('label')]

    def __get_pdb_path__(self, pdb_file):
        # returns the .pdb file path of the file name in the pdb directory
        return os.path.join(self.pdb_dir, pdb_file)

    def __get_sh_path__(self, sh_file):
        # returns the .sh file path of the file name in the sh directory
        return os.path.join(self.sh_dir, sh_file)

    def clean_sim(self):
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

# class PeakImageDataset(Dataset):
#     def __init__(self, paths, transform=None, augment=False):
#         self.peak_image_paths = paths.__get_peak_images_paths__()
#         self.water_image_paths = paths.__get_water_images_paths__()
#         assert len(self.peak_image_paths) == len(self.water_image_paths), "The number of peak images must match the number of water images."
#         self.transform = TransformToTensor() if transform is None else transform
#         self.augment = augment
#         self.labels = [self._extract_labels(path) for path in self.peak_image_paths]

#     def __len__(self):
#         # returns number of images in the dataset
#         return len(self.peak_image_paths)

#     def _extract_labels(self, filepath):
#         default_protein = '1IC6'
#         default_camera_len = 0.1 # in mm
#         default_camera_len_label = 0

#         match = re.search(r'(processed_)?img_7keV_clen(\d{2})_\d+\.h5', filepath)
#         # needs the specific name pattern to extract the protein and camera length
#         if match:
#             camera_length_label = int(match.group(2)) - 1  # Convert '01', '02', '03', to 0, 1, 2, etc.
#             camera_length = float(f"0.{match.group(2)}")
#             protein = default_protein
#             return (protein, camera_length, camera_length_label)
#         else:
#             print(f'Warning: Filename does not match expected pattern: {filepath}')
#         return ('default', 0.0, -1)

#     def __getitem__(self, idx):
#         # gets peak/water image and returns numpy array and applies transform
#         peak_image_path = self.peak_image_paths[idx]
#         water_image_path = self.water_image_paths[idx]

#         peak_image = self.__load_h5__(peak_image_path)
#         water_image = self.__load_h5__(water_image_path)

#         # Ensure image is in the correct format for the transformations

class PeakImageDataset(Dataset):
    def __init__(self, paths, transform=None, augment=False):
        self.peak_image_paths = paths.__get_peak_images_paths__()
        self.water_image_paths = paths.__get_water_images_paths__()
        assert len(self.peak_image_paths) == len(self.water_image_paths), "The number of peak images must match the number of water images."
        self.transform = TransformToTensor() if transform is None else transform
        self.augment = augment
        self.labels = [self._extract_labels(path) for path in self.peak_image_paths]

    def __len__(self):
        # returns number of images in the dataset
        return len(self.peak_image_paths)

    def _extract_labels(self, filepath):
        default_protein = '1IC6'
        default_camera_len = 0.1 # in mm
        default_camera_len_label = 0

        # retrieve 

        match = re.search(r'(processed_)?img_7keV_clen(\d{2})_\d+\.h5', filepath)
        if match:
            camera_length_label = int(match.group(2)) - 1  # Convert '01', '02', '03', to 0, 1, 2, etc.
            camera_length = float(f"0.{match.group(2)}")
            protein = default_protein
            return (protein, camera_length, camera_length_label)
        else:
            print(f'Warning: Filename does not match expected pattern: {filepath}')
        return ('default', 0.0, -1)

    def __getitem__(self, idx):
        # gets peak/water image and returns numpy array and applies transform
        peak_image_path = self.peak_image_paths[idx]
        water_image_path = self.water_image_paths[idx]

        peak_image = self.__load_h5__(peak_image_path)
        water_image = self.__load_h5__(water_image_path)

        # Ensure image is in the correct format for the transformations
        peak_image = self.transform(peak_image)
        water_image = self.transform(water_image)

        labels = self._extract_labels(peak_image_path)
        
        # check for shape
        # print(f"Peak Image Shape: {peak_image.shape}, Water Image Shape: {water_image.shape}")
        # print(f"Protein: {labels[0]}, Camera Length: {labels[1]}, Label Camera Length: {labels[2]}")

        return (peak_image, water_image), labels

    def __load_h5__(self, image_path):
        with h5.File(image_path, 'r') as f:
            image = np.array(f['entry/data/data'])
            # If image is not already in [H, W, C] format, adjust accordingly before return
            if image.ndim == 2:  # For grayscale images, add a channel dimension
                image = np.expand_dims(image, axis=-1)  # Add channel dimension at the last axis
            return image

    def augment_image(self, image, to_tensor=False):
        # FIXME: This is a temporary implementation. Need to replace with albumentations
        # assuming image is np.array and needs to be converted to PIL for augmentation
        if image.ndim == 3 and image.shape[0] in [1,3,4]: #check if CxHxW
            image = image.squeeze(0)
        elif image.ndim == 2: #HxW
            pass
        else:
            raise ValueError(f'Unexpected image shape: {image.shape}')

        pil_image = Image.fromarray(image)
        rotated_image = pil_image.rotate(90)

        if to_tensor:
            return self.default_transform()(rotated_image)
        else:
            return rotated_image

    def default_transform(self):
        return transforms.Compose([
            TransformToTensor(),
        ])

    def __preview__(self, idx, image_type='peak'):
        image_path = self.peak_image_paths[idx] if image_type == 'peak' else self.water_image_paths[idx]
        image = self.__load_h5__(image_path)
        # visualize outliers
        plt.imshow(image, cmap='viridis')
        plt.colorbar()
        plt.title(f'{image_type.capitalize()} Image at Index {idx}')
        plt.axis('off')
        plt.show()

class TransformToTensor:
    def __call__(self, image):
        """
        Converts a numpy array to a PyTorch tensor while ensuring:
        - The output is in C x H x W format.
        - The tensor is of dtype float32.
        - For grayscale images, a dummy channel is added to ensure compatibility.

        Args:
            pic (numpy.ndarray): The input image.

        Returns:
            torch.Tensor: The transformed image tensor.
        """
        # convert the np.array to float32 without changing its range
        pic = image.astype(np.float32)

        # ensure input is HWC format for compatibility without F.to_tensor
        if pic.ndim == 2: # If grayscale, add a dummy channel
            pic = np.expand_dims(pic, axis=-1)

        # Convert the numpy array (HWC) to a PyTorch tensor (CHW)
        # Note: F.to_tensor automatically scales the input based on its dtype
        tensor = F.to_tensor(pic)
        return tensor

class DataPreparation:
    def __init__(self, paths, batch_size=32):
        self.paths = paths
        self.batch_size = batch_size

    def prep_data(self):
        # ensure matching number of paths of peak and water images
        transform = transforms.Compose([
            TransformToTensor(),
        ])

        dataset = PeakImageDataset(self.paths, transform=transform, augment=True)
        num_items = len(dataset)
        num_train = int(0.8 * num_items)
        num_test = num_items - num_train

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        
        print("Data prepared.")
        print(f"Train size: {len(train_dataset)}")
        print(f"Test size: {len(test_dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches: {len(train_loader)} \n\n")

        return train_loader, test_loader # returns train/test tensor data loaders

    def generate_heatmaps(self, batch_images, processor):
        batch_heatmaps = []
        for image_tensor in batch_images:
            peak_coords = processor._process_image(image_tensor)  # Process each image
            heatmap = np.zeros(image_tensor.squeeze().shape)
            for y, x in peak_coords:
                heatmap[y, x] = 1  # Set peak positions to 1
            heatmap_tensor = torch.tensor(heatmap).unsqueeze(0)  # Convert to tensor and adjust shape
            batch_heatmaps.append(heatmap_tensor)
        return torch.stack(batch_heatmaps) # return batch of heatmaps
    
class PeakThresholdProcessor: 
    def __init__(self, threshold_value=0):
        self.threshold_value = threshold_value

    def _set_threshold_value(self, new):
        self.threshold_value = new
    
    def _get_local_maxima(self):
        image_1d = self.image.flatten()
        peaks, _ = find_peaks(image_1d, height=self.threshold_value)
        coordinates = [self.flat_to_2d(idx) for idx in peaks]
        return coordinates
        
    def _flat_to_2d(self, index, shape):
        rows, cols = shape
        return (index // cols, index % cols) 
    
    def _process_image(self, image_tensor):
        image_np = image_tensor.squeeze().cpu().numpy()
        return np.argwhere(image_np > self.threshold_value  )
    
class ImageProcessor:
    def __init__(self, water_background_array):
        self.water_background_array = water_background_array

    @staticmethod
    def load_h5_image(file_path):        
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

    def apply_water_background(self, peak_image_array):
        return peak_image_array + self.water_background_array

    def find_peaks_and_label(self, peak_image_array, threshold_value=0, min_distance=5):
        Output = namedtuple('Out', ['coordinates', 'labeled_array'])
        coordinates = peak_local_max(peak_image_array, min_distance=min_distance, threshold_abs=threshold_value)
        labeled_array = np.zeros(peak_image_array.shape)
        for y, x in coordinates:
            labeled_array[y, x] = 1

        return Output(coordinates, labeled_array)

    def process_directory(self, paths, threshold_value):
        try:
            peaks_path = paths.__get_path__('peak_images_dir')
            processed_out_path = paths.__get_path__('processed_images_dir')
            label_out_path = paths.__get_path__('label_images_dir')

            for file in os.listdir(peaks_path):
                if file.endswith('.h5') and file != 'water_background.h5':
                    full_path = os.path.join(peaks_path, file)
                    image = self.load_h5_image(full_path)
                    processed_image = self.apply_water_background(image)

                    Out = self.find_peaks_and_label(processed_image, threshold_value)
                    labeled_image = Out.labeled_array
                    coordinates = Out.coordinates

                    processed_save_path = os.path.join(processed_out_path, f"processed_{file}")
                    labeled_save_path = os.path.join(label_out_path, f"labeled_{file}")

                with h5.File(processed_save_path, 'w') as pf:
                    pf.create_dataset('entry/data/data', data=processed_image)
                with h5.File(labeled_save_path, 'w') as lf:
                    lf.create_dataset('entry/data/data', data=labeled_image)

                print(f"Processed and labeled images saved:\n {processed_save_path} --> {labeled_save_path}")

        except Exception as e:
            print(f"Error processing directory: {e}")