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
    parser.add_argument('-op', '--optimizer', type=str, help='training optimizer function')
    parser.add_argument('-s', '--scheduler', type=str, help='training learning rate scheduler')
    parser.add_argument('-c', '--criterion', type=str, help='training loss function')
    parser.add_argument('-lr', '--learning_rate', type=float, help='training learning rate')
    
    parser.add_argument('-cl', '--camera_length', type=str, help='attribute name for the camera length parameter')
    parser.add_argument('-pe', '--photon_energy', type=str, help='attribute name for the camera length parameter')
    parser.add_argument('-pk', '--peaks', type=str, help='attribute name for is there are peaks present')
    
    args = parser.parse_args()
    
    if args:
        return args
    else:
        print('Input needed.')
        logger.info('Input needed.')

def main():
    parser = argparse.ArgumentParser(description='Model training arguments.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    logger.info(device)
    
    args = arguments(parser)
    h5_file_list = args.list
    model_arch = args.model
    training_results = args.output
    model_dict_save_path = args.dict
    
    num_epoch = args.epoch
    batch_size = args.batch
    optimizer = args.optimizer
    scheduler = args.scheduler
    criterion = args.criterion
    learning_rate = args.learning_rate
    
    camera_length = args.camera_length
    photon_energy = args.photon_energy
    peak = args.peaks
    
    attributes = {
        'camera length': camera_length,
        'photon energy': photon_energy,
        'peak': peak
    }
    
    path_manager = data_path_manager.Paths(h5_file_list)

    h5_tensor_list = path_manager.get_h5_tensor_list()
    h5_attribute_list = path_manager.get_h5_attribute_list()
    
    data_manager = data_path_manager.Data(h5_tensor_list, h5_attribute_list)
    data_manager.split_data(batch_size)
    train_loader, test_loader = data_manager.get_data_loaders()
    
    cfg = {
        'train data': train_loader,
        'test data': test_loader,
        'batch size': batch_size,
        'device': device,
        'epochs': num_epoch,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'learning rate': learning_rate,
        'model': model_arch
    }

    training_manager = train_model.TrainModel(cfg, attributes)
    training_manager.make_training_instances()
    training_manager.epoch_loop()
    training_manager.plot_loss_accuracy(training_results)
    training_manager.save_model(model_dict_save_path)
    trained_model = training_manager.get_model()
    
    evaluation_manager = evaluate_model.ModelEvaluation(cfg, attributes, trained_model)
    evaluation_manager.run_testing_set()
    evaluation_manager.make_classification_report()
    evaluation_manager.plot_confusion_matrix(training_results)
    
    
    
if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    main()