import argparse
import logging
from pkg import *
import torch

def arguments(parser) -> argparse.ArgumentParser:
    """
    This function is for adding arguments to configure the parameters used for training different models.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which the arguments will be added.
        
    Returns:
        argparse.ArgumentParser: The parser with the added arugments.
    """
    parser.add_argument('-l', '--list', type=str, help='file path to the lst file for the model to use')
    parser.add_argument('-m', '--model', type=str, help='name of the model architecture')
    parser.add_argument('-o', '--output', type=str, help='output file path for training results')
    parser.add_argument('-d', '--dict', type=str, help='output state dict for the trained model to me loaded and used later')
    
    parser.add_argument('-e', '--epoch', type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch', type=int, help='batch size for training')
    
    args = parser.parse_args()
    
    if args:
        return args
    else:
        print('Input needed.')
        logger.info('Input needed.')

def main():
    parser = argparse.ArgumentParser(description='Model training arguments.')
    device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
    print(device)
    logger.info(device)
    
    args = arguments(parser)
    h5_file_list = args.list
    model_arch = args.model
    training_results = args.output
    model_dict_save_path = args.dict
    
    num_epoch = args.epoch
    batch_size = args.batch
    
    data_manager = data_path_manager.Paths(h5_file_list)

    h5_tensor_list = data_manager.get_h5_tensor_list()
    h5_attribute_list = data_manager.get_h5_attribute_list()
    
    training_data_manager = data_path_manager.Data(h5_tensor_list, h5_attribute_list)
    training_data_manager.split_data(batch_size)
    train_loader, test_loader = training_data_manager.get_data_loaders()

    
if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    main()