
import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from pkg import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training for Advanced Applications')
    parser.add_argument('--datasets', nargs='+', type=int, default=[1, 4], help='List of dataset indices')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu)')
    return parser.parse_args()


# Main function encapsulating the training logic
def main(args):
    print('PyTorch version:', torch.__version__)
    print('Torchvision version:', torchvision.__version__)
    print('NMS loaded successfully')
    print('CUDA available:', torch.cuda.is_available())
    print('CUDA version:', torch.version.cuda)
    print('CUDNN version:', torch.backends.cudnn.version())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    # Path and data management setup
    myPaths = path.PathManager(datasets=args.datasets)
    myProcessor = process.Processor(paths=myPaths, datasets=args.datasets)
    logging.info(f'Parameters: {myProcessor.get_parameters()}')

    myDataManager = data.DatasetManager(paths=myPaths, datasets=args.datasets, transform=None)
    train_loader, test_loader = f.prepare(data_manager=myDataManager, batch_size=args.batch_size)

    # Optimizer and scheduler setup
    optimizer = optim.Adam
    scheduler = ReduceLROnPlateau

    cfg = {
        "loader": [train_loader, test_loader],
        'batch_size': train_loader.batch_size,
        'optimizer': optimizer,
        'device': device,
        'scheduler': scheduler
    }

    # Model configurations
    model_configs = [
        eval.Peak_Detection_Configuration(myPaths, args.datasets, device, save_path='../models/peak_model.pt'),
        eval.Photon_Energy_Configuration(myPaths, args.datasets, device, save_path='../models/photon_model.pt'),
        eval.Camera_Length_Configureation(myPaths, args.datasets, device, save_path='../models/clen_model.pt')
    ]

    for config in model_configs:
        model = train_eval.TrainTestModels(cfg, config)
        model.epoch_loop()
        model.plot_loss_accuracy()
        model.plot_confusion_matrix()
        model.get_confusion_matrix()
        model.save_model()
        logging.info(f'Model: {model} Training Complete')

    # Cleanup resources
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    logging.info('Cleaned up CUDA resources')
    
if __name__ == '__main__':
    args = parse_arguments()
    main(args)
