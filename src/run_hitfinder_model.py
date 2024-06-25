import argparse
from hitfinderLib import *
import torch
import datetime


def arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser: 
    """
    This function is for adding an argument when running the python file. 
    It needs to take an lst file of the h5 files for the model use. 
    
    Args:
        parser (argparse.ArgumentParser): The argument parser to which the arguments will be added.
        
    Returns:
        argparse.ArgumentParser: The parser with the added arugments.
    """
    parser.add_argument('-l', '--list', type=str, help='File path to the .lst file containing file paths to the .h5 file to run through the model.')
    parser.add_argument('-m', '--model', type=str, help='Name of the model architecture class found in models.py that corresponds to the model state dict.')
    parser.add_argument('-d', '--dict', type=str, help='File path to the model state dict .pt file.')
    parser.add_argument('-o', '--output', type=str, help='Output file path only for the .lst files after classification.')
    
    parser.add_argument('-cl', '--camera_length', type=str, help='Attribute name for the camera length parameter.')
    parser.add_argument('-pe', '--photon_energy', type=str, help='Attribute name for the photon energy parameter.')
    
    try:
        args = parser.parse_args()
        print("Parsed arguments:")
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
            
        return args
    
    except argparse.ArgumentError as e:
        print(f"Argument error: {e}")
    
    except argparse.ArgumentTypeError as e:
        print(f"Argument type error: {e}")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """
    This main function is the flow of logic for running a trained model. Here parameter arugments are assigned to variables.
    Classes for data management and using the model are declared and the relavent functions for the process are called following declaration in blocks. 
    """
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%m%d%y-%H:%M")
    print(f'Starting hitfinder model: {formatted_date_time}')
    
    parser = argparse.ArgumentParser(description='Parameters for running a model.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'This model will be running on: {device}')

    args = arguments(parser)
    h5_file_list = args.list
    model_arch = args.model
    model_path = args.dict
    save_output_list = args.output 
    
    camera_length = args.camera_length
    photon_energy = args.photon_energy
    
    data_manager = data_path_manager.Paths(h5_file_list)
    
    h5_file_paths = data_manager.get_file_paths()
    h5_tensor_list = data_manager.get_h5_tensor_list()
    h5_attribute_list = data_manager.get_h5_attribute_list()
    
    process_data = run_model.RunModel(model_arch, model_path, save_output_list, h5_file_paths, device)
    process_data.make_model_instance()
    process_data.load_model()
    process_data.classify_data(h5_tensor_list, h5_attribute_list, camera_length, photon_energy) 
    process_data.create_model_output_lst_files()
    process_data.output_verification()

if __name__ == '__main__':
    main()