import argparse
from hitfinderLib import *
import torch
import datetime
from queue import Queue


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
    parser.add_argument('-b', '--batch', type=int, help='Batch size for data running through the model.')
    parser.add_argument('-me', '--multievent', type=str, help='True or false value for if the input .h5 files are multievent or not.')
    parser.add_argument('-mf', '--master_file', type=str, default=None, help='File path to the master file containing the .lst files.')
    
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
    peaks = None
    batch_size = args.batch
    multievent = args.multievent
    
    master_file = args.master_file
    if master_file == 'None' or master_file == 'none':
        master_file = None
    
    attributes = {
        'clen': camera_length,
        'photon_energy': photon_energy,
        'peak': peaks
    } 
    
    cfg = {
        'model': model_arch,
        'model_path': model_path,
        'save_output_list': save_output_list,
        'device': device,
    }
    
    # path_manager = data_path_manager.Paths(h5_file_list, attributes, multievent, master_file)
    if multievent == 'True' or multievent == 'true':
        path_manager = load_data_paths.PathsMultiEvent(h5_file_list,  attributes,  master_file)
    else:
        path_manager = load_data_paths.PathsSingleEvent(h5_file_list, attributes, master_file)
    
    path_manager.read_file_paths()
    h5_file_path_queue = path_manager.get_file_path_queue()
    
    queue_size = h5_file_path_queue.qsize()
    
    process_data = run_model.RunModel(cfg, attributes)
    process_data.make_model_instance()
    process_data.load_model()
    
    while not h5_file_path_queue.empty():
        path_manager.process_files()
        
        h5_file_paths = path_manager.get_h5_file_paths()
        h5_tensor_list = path_manager.get_h5_tensor_list()
        h5_attribute_list = path_manager.get_h5_attribute_list()
        events = path_manager.get_event_count()
        
        data_manager = prep_loaded_data.Data(h5_tensor_list, h5_attribute_list, h5_file_paths, multievent)
        data_manager.inference_data_loader(batch_size)
        data_loader = data_manager.get_inference_data_loader()
        
        process_data.classify_data(data_loader) 
        
        
    process_data.create_model_output_lst_files()
    process_data.output_verification(queue_size, events)

if __name__ == '__main__':
    main()