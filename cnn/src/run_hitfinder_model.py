import argparse
import logging
from pkg import *
import torch


def arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser: 
    """
    This function is for adding an argument when running the python file. 
    It needs to take an lst file of the h5 files for the model use. 
    
    Args:
        parser (argparse.ArgumentParser): The argument parser to which the arguments will be added.
        
    Returns:
        argparse.ArgumentParser: The parser with the added arugments.
    """
    parser.add_argument('-l', '--list', type=str, help='file path to the lst file for the model to use')
    parser.add_argument('-m', '--model', type=str, help='name of the model architecture')
    parser.add_argument('-d', '--dict', type=str, help='file path to the model state dict')
    parser.add_argument('-o', '--output', type=str, help='output file path for the lst files without file names')
    
    args = parser.parse_args()
    if args:
        return args
    else:
        print('Input needed.')
        logger.info('Input needed.')


def main():
    parser = argparse.ArgumentParser(description='parameters for running the model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    logger.info(device)

    args = arguments(parser)
    h5_file_list = args.list
    model_arch = args.model
    model_path = args.dict
    save_output_list = args.output 
    
    data_manager = data_path_manager.Paths(h5_file_list)
    
    h5_file_paths = data_manager.get_file_paths()
    h5_tensor_list = data_manager.get_h5_tensor_list()
    h5_attribute_list = data_manager.get_h5_attribute_list()
    
    process_data = run_model.RunModel(model_arch, model_path, save_output_list, h5_file_paths, device)
    process_data.make_model_instance()
    process_data.load_model()
    process_data.classify_data(h5_tensor_list, h5_attribute_list) 
    process_data.create_model_output_lst_files()
    process_data.output_verification()

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    main()