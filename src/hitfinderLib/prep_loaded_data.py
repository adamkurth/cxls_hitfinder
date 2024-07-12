import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from . import conf


class Data(Dataset):
    
    def __init__(self, classification_data: list, attribute_data: list, h5_file_path: list, use_transform: bool) -> None:
        """
        Initialize the Data object with classification and attribute data.

        Args:
            classification_data (list): List of classification data, that being list of pytorch tensors.
            attribute_data (list): List of attribute data, that being list of metadata dictionaries.
            h5_file_path (list): List of h5 file paths.
            multievent (str): String boolean value if the input .h5 files are multievent or not.
        """
        self.train_loader = None
        self.test_loader = None
        self.inference_loader = None
        self.image_data = classification_data
        self.meta_data = attribute_data
        self.file_paths = h5_file_path
        self.data = list(zip(self.image_data, self.meta_data, self.file_paths))
        
        self.use_transform = use_transform
        self.transforms = None
        if self.use_transform:
            self.make_transform()
        
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.image_data)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image data and the metadata at the given index.
        """
        try:
            if self.use_transform:
                image = self.transforms(self.image_data[idx])
                return image, self.meta_data[idx], self.file_paths[idx]
            else:
                return self.data[idx]
        except Exception as e:
            print(f"An unexpected error occurred while getting item at index {idx}: {e}")
            
    def make_transform(self) -> None:
        """
        If the transfom flag is true, this function creates the global variable for the transform for image data. 
        """
        self.transforms = v2.Compose([
            v2.toPILImage(),
            v2.Resize(conf.eiger_4m_image_size),
            v2.ToTensor(),
        ])
            
        
        
    def split_training_data(self, batch_size: int) -> None:
        """
        Split the data into training and testing datasets and create data loaders for them.

        Args:
            batch_size (int): The size of the batches to be used by the data loaders.
        """
        try:
            num_items = len(self.data)
            if num_items == 0:
                raise ValueError("The dataset is empty.")
            
            num_train = int(0.8 * num_items)
            num_test = num_items - num_train

            try:
                train_dataset, test_dataset = torch.utils.data.random_split(self.data, [num_train, num_test])
            except Exception as e:
                print(f"An error occurred while splitting the dataset: {e}")
                return

            try:
                self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
                self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            except Exception as e:
                print(f"An error occurred while creating data loaders: {e}")
                return
            
            print(f"Train size: {len(train_dataset)}")
            print(f"Test size: {len(test_dataset)}")

        except ValueError as e:
            print(f"ValueError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
    def get_training_data_loaders(self) -> tuple:
        """
        Get the training and testing data loaders.

        Returns:
            tuple: A tuple containing the training and testing data loaders.
        """
        return self.train_loader, self.test_loader
    
    def inference_data_loader(self, batch_size) -> None: 
        """
        Puts the inference data into a dataloader for batch processing.

        Args:
            batch_size (int): The size of the batches to be used by the data loaders.
        """
        print('Making data loader...')
        try:
            num_items = len(self.data)
            if num_items == 0:
                raise ValueError("The dataset is empty.")
            
            try:
                self.inference_loader = DataLoader(self.data, batch_size=batch_size, shuffle=False, pin_memory=True)
                print('Data loader created.')
            except Exception as e:
                print(f"An error occurred while creating data loaders: {e}")
                return

        except ValueError as e:
            print(f"ValueError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    def get_inference_data_loader(self) -> DataLoader:
        """
        This function returns the inference data loader.

        Returns:
            DataLoader: The data loader for putting through the trained model. 
        """
        return self.inference_loader