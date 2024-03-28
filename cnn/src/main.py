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
from pkg import u, m

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

def data_prep(dataset):
    clen_values, photon_energy_values = [1.5, 2.5, 3.5], [6000, 7000, 8000]
    param_matrix = u.parameter_matrix(clen_values, photon_energy_values)
    
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

    logging.info(f"Dataset {dataset} selected with parameters: {dataset_dict[dataset]}")
    clen, photon_energy = dataset_dict[dataset]
    return clen, photon_energy

def main(dataset:str, process_dir:bool, images_path:str):
    configure_logging()
    run_process_directory(process_dir, images_path)
    clen, photon_energy = data_prep(dataset)
    
    # instances
    pm = u.PathManager()
    # peak_paths, water_peak_paths, labels, water_background_path = pm.select_dataset(dataset=dataset) 
    p = u.Processor(paths=pm, dataset=dataset)

    # peak, label, overlay, background are valid types
    dm = u.DatasetManager(paths=pm, dataset=dataset, parameters=(clen, photon_energy), transform=None, include_water_background=True, percent_water_repeat=0.35)

    # peak, label, overlay, background are valid types
    u.check_attributes(paths=pm, dataset=dataset, type='peak') 
    u.check_attributes(paths=pm, dataset=dataset, type='overlay')
    u.check_attributes(paths=pm, dataset=dataset, type='label')
    u.check_attributes(paths=pm, dataset=dataset, type='background')
    
    # test/train loaders
    train_loader, test_loader = u.prepare(data_manager=dm, batch_size=10)
    
    # model, criterion, optimizer
    model1 = m.BasicCNN1()
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
    t = u.TrainTestModels(model=model1, loader=[train_loader, test_loader], criterion=criterion, optimizer=optimizer, device=device, cfg=cfg)
    logging.info("Starting model training...")
    t.train()    
    logging.info("Script execution completed.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and train dataset with optional preprocessing step.")
    parser.add_argument("--dataset", type=str, default="01", help="Dataset number to process and train.")
    parser.add_argument("--process_dir", action="store_true", help="Force execution of process_directory.py; defaults to skipping.")
    parser.add_argument("--images_path", type=str, default="../../images", help="Path to the images directory.")
    
    args = parser.parse_args()
    
    main(args.dataset, args.process_dir, args.images_path)