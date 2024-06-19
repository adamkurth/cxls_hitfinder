<<<<<<< HEAD

import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pkg import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training for Peak Detection')
    parser.add_argument('--datasets', nargs='+', type=int, default=[1, 4], help='List of dataset indices')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs to train')
    return parser.parse_args()

def main(args):
    datasets = args.datasets
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    myPaths = path.PathManager(datasets=datasets)
    myProcessor = process.Processor(paths=myPaths, datasets=datasets)
    print('Parameters:', myProcessor.get_parameters())

    myDataManager = data.DatasetManager(paths=myPaths, datasets=datasets, transform=None)
    train_loader, test_loader = f.prepare(data_manager=myDataManager, batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    optimizer = optim.Adam
    scheduler = scheduler = ReduceLROnPlateau

    cfg = {
        "loader": [train_loader, test_loader],
        'num_epochs': 15,
        'batch_size': train_loader.batch_size,
        'optimizer': optimizer,
        'device': device,
        'scheduler': scheduler
        }

    peak_config = eval.Peak_Detection_Configuration(myPaths, datasets, device, save_path='../models/peak_model.pt')
    print(peak_config.get_loss_weights())
    photon_config = eval.Photon_Energy_Configuration(myPaths, datasets, device, save_path='../models/photon_model.pt')
    print(photon_config.get_loss_weights())
    clen_config = eval.Camera_Length_Configureation(myPaths, datasets, device, save_path='../models/clen_model.pt')
    print(clen_config.get_loss_weights())

    a = train_eval.TrainTestModels(cfg, peak_config)
    a.epoch_loop()
    a.plot_loss_accuracy()
    a.plot_confusion_matrix()
    a.get_confusion_matrix()
    a.save_model()

    print(f'Model: {a} Training Complete')

    b = train_eval.TrainTestModels(cfg, photon_config)
    b.epoch_loop()
    b.plot_loss_accuracy()
    b.plot_confusion_matrix()
    b.get_confusion_matrix()
    b.save_model()

    print(f'Model: {b} Training Complete')

    c = train_eval.TrainTestModels(cfg, clen_config)
    c.epoch_loop()
    c.plot_loss_accuracy()
    c.plot_confusion_matrix()
    c.get_confusion_matrix()
    c.save_model()

    print(f'Model: {c} Training Complete')

    # reload(pipe)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    pipeline = pipe.ModelPipeline(peak_config, photon_config, clen_config, device)

    train_loader, test_loader = f.prepare(data_manager=myDataManager, batch_size=1)

    for inputs, labels, attributes in train_loader:
        print(f'-- attributes: {attributes}')
        results = pipeline.run(inputs.to(device))
        print(f'-- results: {results}')
        break
    
if __name__ == '__main__':
    args = parse_arguments()
    main(args)
=======
"""
This script is used to process and train a dataset with an optional preprocessing step.

Usage:
    python main.py [--dataset DATASET] [--process_dir] [--images_path IMAGES_PATH]

Arguments:
    --dataset DATASET: The dataset number to process and train. Default is "01".
    --process_dir: Flag to force the execution of process_directory.py. By default, it is skipped.
    --images_path IMAGES_PATH: The path to the images directory. Default is "../../images".
"""
import os
import sys
import subprocess
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import argparse 
from pkg import *

# Configure logging
def configure_logging(): 
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_process_directory(process:bool, images_path:str):
    try:
        if process:
            script_path = os.path.join(os.path.dirname(__file__), "process_directory.py")
            subprocess.run(["python", script_path, images_path], check=True)
            logging.info("Successfully executed process_directory.py.")
        else:
            logging.info("Skipping execution of process_directory.py due to default or flag usage.")
    except subprocess.CalledProcessError as e:
        logging.error("Failed to execute process_directory.py. Error: %s", e)
        sys.exit(1)

def get_data_dict(dataset):
    clen_values, photon_energy_values = [1.5, 2.5, 3.5], [6000, 7000, 8000]
    param_matrix = f.parameter_matrix(clen_values, photon_energy_values)
    logging.info(f"\nParameter matrix:\n{param_matrix}\n")
    dataset_dict = {
        '01': [clen_values[0], photon_energy_values[0]],
        '02': [clen_values[0], photon_energy_values[1]],
        '03': [clen_values[0], photon_energy_values[2]],
        '04': [clen_values[1], photon_energy_values[0]],
        '05': [clen_values[1], photon_energy_values[1]],
        '06': [clen_values[1], photon_energy_values[2]],
        '07': [clen_values[2], photon_energy_values[0]],
        '08': [clen_values[2], photon_energy_values[1]],
        '09': [clen_values[2], photon_energy_values[2]],
    }
    return dataset_dict

def main(dataset:str, process_dir:bool, images_path:str):
    configure_logging()
    run_process_directory(process_dir, images_path)

    # instances
    myPaths = path.PathManager(dataset=dataset)
    myProcessor = process.Processor(paths=myPaths, dataset=dataset)
    clen, photon_energy = myProcessor.get_parameters()
    print(f"clen: {clen}, photon_energy: {photon_energy}")
    
    # peak, label, overlay, background are valid types
    myDataManager = data.DatasetManager(paths=myPaths, dataset=dataset, parameters=(clen, photon_energy), transform=None)

    # peak, label, overlay are valid types
    f.check_attributes(paths=myPaths, dataset=dataset, type='peak') 
    f.check_attributes(paths=myPaths, dataset=dataset, type='overlay')
    f.check_attributes(paths=myPaths, dataset=dataset, type='label')
    
    # train/test loaders
    train_loader, test_loader = f.prepare(data_manager=myDataManager, batch_size=10)
    
    # model, criterion, optimizer
    model1 = m.BasicCNN3()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model1.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = {
        'num_epochs': 10,
        'num_classes': 2,
        'batch_size': train_loader.batch_size,
        'test_size': len(train_loader.dataset),
        'test_size': len(test_loader.dataset),
        'criterion': criterion,
        'optimizer': optimizer,
        'device': device,
        'model': model1,
    }
    # arguments: self, model, loader: list, criterion, optimizer, device, cfg: dict
    t = train_eval.TrainTestModels(model=model1, loader=[train_loader, test_loader], criterion=criterion, optimizer=optimizer, device=device, cfg=cfg)
    logging.info("Starting model training...")
    t.train(epoch=2)    
    logging.info("Script execution completed.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and train dataset with optional preprocessing step.")
    parser.add_argument("--dataset", type=str, default="01", help="Dataset number to process and train.")
    parser.add_argument("--process_dir", action="store_true", help="Force execution of process_directory.py; defaults to skipping.")
    parser.add_argument("--images_path", type=str, default="../../images", help="Path to the images directory.")
    args = parser.parse_args()
    
    main(args.dataset, args.process_dir, args.images_path)
>>>>>>> progress-Everett
