
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
