import os 
import re
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
    

def sim_parameters(paths):
    """
    Reads the .pdb and .sh files and returns a dictionary of simulation parameters.

    Parameters:
    - Paths: An instance of the Paths class that contains the paths to the .pdb and .sh files.

    Returns:
    - combined_params: A dictionary containing the simulation parameters extracted from the .pdb and .sh files.
        The dictionary includes the following keys:
        - geom: The geometry parameter from the .sh file.
        - cell: The cell parameter from the .sh file.
        - number: The number parameter from the .sh file.
        - output_name: The output_name parameter from the .sh file.
        - photon_energy: The photon_energy parameter from the .sh file.
        - nphotons: The nphotons parameter from the .sh file.
        - a: The 'a' parameter from the .pdb file.
        - b: The 'b' parameter from the .pdb file.
        - c: The 'c' parameter from the .pdb file.
        - alpha: The 'alpha' parameter from the .pdb file.
        - beta: The 'beta' parameter from the .pdb file.
        - gamma: The 'gamma' parameter from the .pdb file.
        - spacegroup: The spacegroup parameter from the .pdb file.
    """
    def read_pdb(path):
        UnitcellParams = namedtuple('UnitcellParams', ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'spacegroup'])
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('CRYST1'):
                    tokens = line.split()
                    a, b, c = float(tokens[1]), float(tokens[2]), float(tokens[3])
                    alpha, beta, gamma = float(tokens[4]), float(tokens[5]), float(tokens[6])
                    spacegroup = ' '.join(tokens[7:-1])  # Exclude the last element
        return UnitcellParams(a, b, c, alpha, beta, gamma, spacegroup)._asdict()

    def read_sh(path):
        ShParams = namedtuple('ShParams', [
            'geom', 'cell', 'number', 'output_name', 'sf', 'pointgroup',
            'min_size', 'max_size', 'spectrum', 'cores', 'background',
            'beam_bandwidth', 'photon_energy', 'nphotons', 'beam_radius', 'really_random'
        ])
        
        params = {key: None for key in ShParams._fields}
        
        with open(path, 'r') as file:
            content = file.read()
        param_patterns = {
            'geom': r'-g\s+(\S+)',
            'cell': r'-p\s+(\S+)',
            'number': r'--number=(\d+)',
            'output_name': r'-o\s+(\S+)',
            'sf': r'-i\s+(\S+)',
            'pointgroup': r'-y\s+(\S+)',
            'min_size': r'--min-size=(\d+)',
            'max_size': r'--max-size=(\d+)',
            'spectrum': r'--spectrum=(\S+)',
            'cores': r'-s\s+(\d+)',
            'background': r'--background=(\d+)',
            'beam_bandwidth': r'--beam-bandwidth=([\d.]+)',
            'photon_energy': r'--photon-energy=(\d+)',
            'nphotons': r'--nphotons=([\d.e+-]+)',
            'beam_radius': r'--beam-radius=([\d.]+)',
            'really_random': r'--really-random=(True|False)'
        }
        for key, pattern in param_patterns.items():
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                value = match.group(1)
                if value.isdigit():
                    params[key] = int(value)
                elif value.replace('.', '', 1).isdigit():
                    params[key] = float(value)
                elif value == 'True':
                    params[key] = True
                elif value == 'False':
                    params[key] = False
                else:
                    params[key] = value

        return ShParams(**params)._asdict()        
    
    pdb_path = os.path.join(paths.root, 'sim', 'pdb', '1ic6.pdb') # hardcoded for now
    sh_path = os.path.join(paths.root, 'sim', 'submit_7keV_clen01.sh') # hardcode for now
    
    unitcell_params_dict = read_pdb(pdb_path)
    sh_params_dict = read_sh(sh_path)
    
    essential_keys_sh = ['geom', 'cell', 'number', 'output_name', 'photon_energy', 'nphotons']
    essential_sh_params = {key: sh_params_dict[key] for key in essential_keys_sh}
    
    combined_params = {**essential_sh_params, **unitcell_params_dict}
    return combined_params

# def __preview__(self, image_path):
#     try:
#         image = self.__load_h5__(image_path)
#         # visualize outliers 
#         plt.imshow(image, cmap='viridis')
#         plt.colorbar()
#         plt.title(f'{image_type.capitalize()} Image at Index {idx}')
#         plt.axis('off')
#         plt.show()
#     except Exception as e:
#         print(e)
#         print(f'Error: Could not load the image at {image_path}')


def train_test_model(model, loader, criterion, optimizer, epochs, device, N, batch, classes):
  
    """
    This function trains, test, and plots the loss, accuracy, and confusion matrix of a model.

    Args:
        model: PyTorch model
        loader: list of torch.utils.data.DataLoader
        criterion: PyTorch loss function
        optimizer: PyTorch optimizer
        epochs: int
        device: torch.device
        N: list of int
        batch: list of int
        classes: int

    Returns:
        None
    """
  
    print(f'Model: {model.__class__.__name__}')

    plot_train_accuracy = np.zeros(epochs)
    plot_train_loss = np.zeros(epochs)
    plot_test_accuracy = np.zeros(epochs)
    plot_test_loss = np.zeros(epochs)

    print('Training and testing the model...')

    for epoch in range(epochs):
        print('-- epoch '+str(epoch)) 
        # train
        running_loss_train = 0.0
        accuracy_train = 0.0
        predictions = 0.0
        total_predictions = 0.0
        model.train()
        for inputs, labels in loader[0]:
            peak_images, _ = inputs
            peak_images = peak_images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            score = model(peak_images)
            loss = criterion(score, labels)
            
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()  # Convert to Python number with .item()
            predictions = (torch.sigmoid(score) > 0.5).long()  # Assuming 'score' is the output of your model
            accuracy_train += (predictions == labels).float().sum().item()
            total_predictions += np.prod(labels.shape)
    # test
        running_loss_test = 0.0
        accuracy_test = 0.0
        predicted = 0.0
        total = 0.0
        
        cm_test = np.zeros((classes,classes), dtype=int)
        all_labels = []
        all_predictions = []
        
        model.eval()
        with torch.no_grad():
            for inputs, labels in loader[1]:
                peak_images, _ = inputs
                peak_images = peak_images.to(device)
                labels = labels.to(device)
                
                score = model(peak_images)
                loss = criterion(score, labels)
                running_loss_test += loss.item()  # Convert to Python number with .item()
                predicted = (torch.sigmoid(score) > 0.5).long()  # Assuming 'score' is the output of your model
                accuracy_test += (predicted == labels).float().sum().item()
                total += np.prod(labels.shape)
                
                # Extend the all_labels and all_predictions lists
                # Convert tensors to CPU numpy arrays for sklearn compatibility
                all_labels.extend(labels.cpu().numpy().flatten())  # Flatten in case it's not already 1D
                all_predictions.extend(predictions.cpu().numpy().flatten())

    # statistics

        loss_train = running_loss_train/batch[0]
        plot_train_loss[epoch] = loss_train

        loss_test = running_loss_test/batch[1]
        plot_test_loss[epoch] = loss_test

        accuracy_train /= total_predictions
        plot_train_accuracy[epoch] = accuracy_train

        accuracy_test /= total
        plot_test_accuracy[epoch] = accuracy_test

        print('loss (train, test): {:.4f}, {:.4f}'.format(loss_train,loss_test))
        print('accuracy (train, test): {:.4f}, {:.4f}'.format(accuracy_train,accuracy_test))


  # plotting loss, accuracy, and confusion matrix
    plt.plot(range(epochs), plot_train_accuracy,marker='o',color='red')
    plt.plot(range(epochs), plot_test_accuracy ,marker='o',color='orange',linestyle='dashed')
    plt.plot(range(epochs), plot_train_loss ,marker='o',color='blue')
    plt.plot(range(epochs), plot_test_loss ,marker='o',color='teal',linestyle='dashed')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss/accuracy')
    plt.legend(['accuracy train','accuracy test','loss train','loss test'])
    plt.show()


    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Calculate the confusion matrix
    cm_test = confusion_matrix(all_labels, all_predictions)



    plt.matshow(cm_test,cmap="Blues")
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    torch.cuda.empty_cache()