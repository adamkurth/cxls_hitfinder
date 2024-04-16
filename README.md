# `cxls_hitfinder` Project

### Description
Will be ongoing project to implement machine learning techniques for peakfinding in crystallography experimental imaging data.

For use at CXLS beamline, this software aims to identify the Bragg peaks, camera length (m), and photon energy (keV), of an image based on the training data. This software will be continuing development, so please check the most recent update to `main` branch for the most recent updates.

### 1. Installation

1. Clone the repository:

```bash
# local
git clone --recurse-submodules https://github.com/adamkurth/cxls_hitfinder.git
```
or using GitLab:
    
```bash
# local
git clone --recurse-submodules https://gitlab.com/adamkurth/cxls_hitfinder.git
```

Or if you forget to add `--recurse-submodules`, then run the following command:

```bash
git submodule update --init --recursive
```

2. Change into the directory:

```bash
# local
cd cxls_hitfinder
```

#### 1.1 Local Linux Installation
 To install Apptainer, run the following command:

```bash

``` 

#### 1.2 Docker Installation

1. Run the Docker file to generate the Docker image:

```bash
# local
docker buildx build -t cxls_hitfinder .
```

2. Save the Docker image:

```bash
# local
docker save cxls_hitfinder > cxls_hitfinder.tar
```

To transfer the Docker image to a remote machine, use `scp`:
Note: Please ensure that the remote machine has Docker installed, and has cloned the repository on the remote machine.

On remote machine:
```bash
# On SOL
git clone https://github.com/adamkurth/cxls_hitfinder.git
cd cxls_hitfinder
```
Then, on the local machine, transfer the Docker image to the remote machine using `scp`:
```bash
# local
scp cxls_hitfinder.tar <ASURITE>@sol.asu.edu:/home/<ASURITE>/cxls_hitfinder/
```

3. Load the Docker image on the respective machine:
```bash
# on SOL
docker load < cxls_hitfinder.tar
```

4. Run the Docker image:
```bash
# On SOL
docker run -d --name my_cxls_hitfinder_instance cxls_hitfinder
```

5. Access the Docker container:
```bash
docker exec -it my_cxls_hitfinder_instance bash
```

#### 1.3 Apptainer (formerly Singularity) Installation

##### 1.3.1 Apptainer Installation For MacOS

For Mac with (Apple Silicon or Intel), you can't install Apptainer (formerly Singularity) directly because it's designed to run on Linux. However, you can use a Linux virtual machine on your Mac and then install Apptainer within that VM. Lima is a tool that sets up a Linux VM for you on a Mac, and once it's running, you can install Linux-compatible tools like Apptainer inside it.

1. Install Lima:

```bash
brew install lima
```

2. Start the VM:

```bash
limactl start default
```

1. Access the Lima VM:

```bash
lima start default
```

4. Install Apptainer:

```bash
sudo apt-get update
sudo apt-get install singularity-container
```

5. Run Apptainer:

```bash
singularity shell default
```

---

### 1. Directory Structure

The `checks.sh` script will ensure the proper directories are made in the `images/` directory. The `images/` directory is where the images will be stored. The `images/` directory will have the following structure:

```bash
cnn_hitfinder/
    ├── cnn/
    │   ├── src/
    │   │   ├── pkg/
    │   │   │   ├── __init__.py
    │   │   │   ├── util.py
    │   │   │   ├── models.py
    │   │   │   ├── functions.py
    │   │   ├── cnn.py
    │   │   ├── test.ipynb
    │   │   ├── etc.
    ├── images/
    │   ├── peaks/
    │   ├── labels/
    │   ├── peaks_water_overlay/
    │   └── water/
    ├── scripts/
    │   └── 
    ├── requirements.txt
    └── README.md
```

#### Photon Energy and Camera Length Combinations

*Note that every folder in `images` contains `01` through `09` corresponding to the camlen/keV combination*

Parameter matrix for `camlen` and `keV`:

| Dataset (01-09) | camlen (m) | photon energy (keV) |
|---------------|------------|---------------------|
| `01`          | 0.15        | 6                   |
| `02`          | 0.15        | 7                   |
| `03`          | 0.15        | 8                   |
| `04`          | 0.25        | 6                   |
| `05`          | 0.25        | 7                   |
| `06`          | 0.25        | 8                   |
| `07`          | 0.35        | 6                   |
| `08`          | 0.35        | 7                   |
| `09`          | 0.35        | 8                   |

Contents of `images/`:
- `images/peaks` contains the Bragg peaks. 
- `images/labels` contains the labels for the Bragg peaks, (i.e. above a threshold of 5). 
- `images/peaks_water_overlay` contains the Bragg peaks overlayed with the respective keV water image. 
- `images/water` contains the different keV water images. 

The following bash script will make the needed directories in `images/`:

```bash
cd cnn/src/
bash checks.sh
```

### 2. Dataset Selection

1. Inside of the `cnn/src/pkg/util.py` file, the following classes are used:

   - `PathManager`: This class is used to manage the paths of the datasets.
   - `DatasetManager`: This class is used to manage the datasets themselves.
   - `Processor`: This class is used to process the selected dataset.

   The `select_dataset` function of the `PathManager` class is used to select the dataset with the identifier '01'. The `DatasetManager` class is then used to manage the dataset, with the `transform` parameter set to `None` in this case. Please note that for dataset '01', the `clen` and `photon_energy` parameters are set to 1.5 meters and 6000 eV respectively.

2. The `Processor` class is responsible for processing the dataset. It includes functions for loading the data, detecting peaks, and generating labels. The `process_directory` function is used to process all the images in the `images/peaks/` directory of the selected dataset (e.g., '01'). It uses the corresponding `images/water/water{0*}.h5` files to generate overlay images. For example, it generates `images/water/01/water01.h5` for all images in `images/peaks/01/*`.

1. The `Processor` class is responsible for processing the dataset, using pure signal images in `images/peaks/` (of respective dataset), then generating labeled images (heatmaps), and overlay images with the corresponding keV water image. 
   - Namely, the `process_directory` function uses all the images in `images/peaks/` of the selected dataset (e.g. '01'), and uses the respective `images/water/water{0*}.h5` to generate the overlay images (e.g. `images/water/01/water01.h5` to all images in `images/peaks/01/*.h5`).


   ---

## Rational Behind the Model

### Problem:

- *Long term goal*: We will simulate diffraction intensity images of a protein, and then use a CNN to predict the protein structure given the diffraction intensity images.

- *Short term goal*: Will simulate diffraction intensity images of protein 1IC6.pdb (at first), then apply water background noise to all the "*peak only*" images, then use both peak-only and peak + water noise images to train a CNN to predict the protein structure. Each 10,000 (20,000 including water-noise images) images will be generated at a given `clen` or camera length away from the interaction point. 
  
- Given the 20,000 images, we want predict (a) camera length away from the interaction point, and (b) the protein structure from the peak patterns on the image.

    --- 

#### Base Architecture:

1. **ResNet**: a Residual Network (ResNet), could serve as the backbone for feature extraction, thanks to its deep architecture and ability to combat vanishing gradients, allowing it to learn rich features even from complex images. 

    **Overview**: renowned for its ability to train very deep networks by usng skip connections or shortcuts to jump over some layers. These connections help mitigate the *vanishing gradient problem*, which is the problem of the gradients becoming increasingly small as the network becomes deeper, making it hard to train. Thus, this allows for deeper network architectures without degredation in performance.

    **Why**: The ResNet variants (ResNet-50, ResNet-101, ResNet-152) enables it to learn a wide variety of features from diffraction images which is crucial for identifying subtle patterns indicitive of the protein structure. Its ability to effectively propogate gradients through many layers makes it ideal for learning from complex, high-dimensional data images typically in crystallagraphy.

2. **U-Net**: Originally for biomedical imaging segmentation, this architecture could be adapted to peak detection. Its ability to capture context at different resulutions (image sense of the word) and precisely localize the features makes it suitable for identifying the peaks in in noisy backgrounds.

   **Overview**: Originally designed for biomedical image segmentation, U-Net's architecture is unique for its use of a contracting path to capture context and a symmetric expanding path that enables precise localization.

   **Why**: The U-Net can be particularly effective if the task involves not just classifying the entire image but also identifying the specific regions within the image that correspond to peaks or other features relevant to protein structure prediction. Its structure is advantageous for tasks requiring precise spacial localization of features, which could be useful for peak detection with noisy backgrounds.
   
3. **Inception Networks (GoogLeNet)**: The Inception Network is known for its efficient use of computational resources, thanks to its inception module. This module is a combination of 1x1, 3x3, and 5x5 convolutions, as well as max pooling, which allows the network to learn from features at different scales.

   **Overview**: Inception networks utilize inception modules, which allow the network to choose from different kernel sizes at each block, enabling it to adaptively capture information at various scales.

   **Why**: The varying sizes of the convolutional kernels in the inception modules allow the network to be efficient in terms of computation and parameters while being powerful in capturing features at different scales, which is critical for analyzing diffraction patterns that may vary significantly in size and shape across different protein structures.

4. **Attention Mechanisms (e.g., SENet)**: 
    **Overview**: Squeeze-and-Excitation Networks (SENet) introduce a mechanism to recalibrate channel-wise feature responses by explicitly modelling interdependencies between channels. This attention mechanism enhances the representational power of the network.
    
    **Why**: Given the variability in the significance of different features in the diffraction images, especially with the introduction of noise, the ability of SENet to focus on the most relevant features could improve the accuracy of predicting protein structures from the images.

5. **DenseNet (Densely Connected Convolution Networks)**:
    
    **Overview**: DenseNet improves upon the idea of skip connections in ResNet by connecting each layer to every other layer in a feed-forward fashion. This architecture ensures maximum information flow between layers in the network.
    
    **Why**: The feature reuse in DenseNet makes it particularly efficient for learning from limited data, which might be beneficial given the specific nature of the protein structures and the complexity introduced by noise in the images. DenseNet's efficiency and compactness could lead to better performance on tasks with high computational complexity like protein structure prediction from crystallography images.

    ---

#### Proposed Architecture:

- **ResNet + Attention Mechanisms**: 
   **Quickly**: (See ResNet above) ResNet architectures, especially deeper versions like ResNet-101 or ResNet-152, are highly capable of capturing complex patterns in images due to their deep structure supported by residual connections. These connections help in training very deep networks by addressing the vanishing gradient problem, making them highly effective for tasks requiring detailed feature extraction from images, such as identifying specific patterns in diffraction intensity images indicative of protein structures.

    **Enhancements**: To further boost the performance of a ResNet-based model, you could integrate attention mechanisms, such as the Squeeze-and-Excitation blocks or non-local blocks, which would allow the model to focus more on relevant features within the diffraction images. This is *particularly useful when dealing with images with variable noise levels, as it helps in distinguishing between noise and meaningful signal.*

- **Transformer-based Models for Vision (ViT)**: 
    **Quickly**: Given the recent success of Transformer models in various domains, including natural language processing and image recognition tasks, a Vision Transformer (ViT) model could be another excellent choice. Transformers *treat the image as a sequence of patches and learn global dependencies between them*, which could be particularly beneficial for understanding the global structure of protein based on local diffraction patterns.

    **Advantages**: The Transformer architecture's ability to *capture long-range interactions between different parts of the image could provide a significant advantage in predicting the protein structure from diffraction intensity images.* This aspect is crucial when the relationship between different peaks (or features) in the image directly influences the inferred protein structure.


- #### Considerations: 
  - Given the computational resouces at your disposal, and the goal of pushing the boundaries of what's achievable with current deep learning techniques in crystallography, starting with a ResNet-based architecture enhanced with attention mechanisms offers a solid foundation with proven capabilities in image processing tasks. This approach combines the strength of deep convolutional networks with the ability to focus on the most informative parts of an image, making it well-suited for the complexities of your task.
  - Simultaneously, *exploring Transformer-based models tailored for vision tasks could offer groundbreaking insights*, especially in terms of learning global dependencies and handling the complex patterns found in diffraction intensity images.
  - **Prototyping**: Given the novelty of the task and the resources available, an iterative approach that involves prototyping with both architectures, evaluating performance, and then refining or combining models based on empirical results would be the most effective strategy. This process allows for the discovery of the most suitable model architecture or combination thereof for predicting protein structures from diffraction intensity images accurately. 


# OUT OF DATE BELOW


### 2. Convolution Neural Network (CNN) for Peak Classification

1. Please ensure that you're logged into AGAVE **first**.
2. Use AGAVE to run `ccn_test.py` using the `run_ccn_SLURM.sh` script.

- Use *gpu* if available for significantly faster computation.

```bash {"id":"01HK8P6S3V98JZJ0QETSV5B8R9"}
./run_ccn_SLURM.sh <RUN> <TASKS> <PARTITION> <QOS> <HOURS> <PATH> <TAG>
# example
./run_ccn_SLURM.sh test1 8 publicgpu wildfire 4 
```

3. Then use this command to watch the job:

- Copy the given `JOB_ID` and paste this

```bash {"id":"01HK8P6S45CBZZ0915DE6Y303F"}
watch -n 2 squeue -j <JOB_ID>
# exmaple
watch "squeue -u amkurth"
```

- The output files are:
   - errors: (`.err`)
   - slurm: (`.slurm`)
   - output: (`.out`)

***
# Convolution Neural Network for Peak Classification (`ccn_test.py`)

The script `ccn_test.py` is under development, but designed for peak detection and classification in 2D image data, utilizing deep learning techniques with PyTorch. Below is a detailed breakdown of the script's components, functionalities, and the mathematical concepts they incorporate:

## Purpose:

The script aims to:

1. **Load and Preprocess Data**: It reads 2D image data from `.h5` files.
2. **Detect Peaks**: It identifies significant points or peaks in the images which could represent important features or areas of interest.
3. **Classify Peaks**: It uses a Convolutional Neural Network (CNN) to classify these peaks into categories (binary in this case).

## Components:

### 1. Import Statements:

```bash {"id":"01HKDKNZ9HFNCK32VZJEF1PA6D"}
import os
import glob
import h5py as h5
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter
import sys
from label_finder import(
    load_data, 
    load_file_h5,
    display_peak_regions, 
    # validate,
    is_peak,
    view_neighborhood, 
    generate_labeled_image, 
    main,
    )                     
```

### 2. PeakThresholdProcessor (Class):

- **Purpose**: To process each image and identify potential peaks.
- **Methods**:
   - `set_threshold_value`: Sets a threshold. Any value above this is considered a potential peak.
   - `get_coordinates_above_threshold`: Returns coordinates of all points above the threshold, indicating potential peaks. Mathematically, for an image \(I\) and a threshold \(T\), the mask \(M\) is defined as:

      $$M(x, y) = \begin{cases}
      1 & \text{if } I(x, y) > T \\
      0 & \text{otherwise}
      \end{cases}$$

   - `get_local_maxima`: Uses the `find_peaks` method to identify actual peaks from the potential ones in a flattened version of the image.
   - `flat_to_2d`: Converts flat indices to 2D coordinates.
   - `patch_method`: Normalizes the image data to a range of [0,1].

### 3. ArrayRegion (Class):

- **Purpose**: Manages a specific region around a peak in an image.
- **Methods**:
   - `set_peak_coordinate` & `set_region_size`: Define the region around a peak.
   - `get_region` & `extract_region`: Extract the specified region from the image, mathematically represented as:

      $$\text{Region} = I[x_c-s:x_c+s, y_c-s:y_c+s]$$

      where $(x_c, y_c)$ are the center coordinates and \(s\) is the size of the region.

### 4. CCN (Class):

- **Purpose**: Defines a Convolutional Neural Network for binary classification of peaks.
- **Details**: Includes layers like Convolutional, MaxPooling, Dropout, and Fully Connected layers. The forward pass can be represented as a series of mathematical operations: convolutions, ReLU activations, pooling, and linear transformations.

### 5. preprocess (Function):

- **Objective**: To load `.h5` image data, detect peaks, and prepare labels.
- **Process**:
   - `load_tensor`: Loads image data from files and converts them into tensors suitable for PyTorch.
   - `is_peak`: Determines whether a given point is a peak based on its neighborhood.
   - `find_coordinates`: Uses `PeakThresholdProcessor` to find peaks.
   - `validate`: Confirms peaks by comparing manual and script-detected ones.
   - `generate_label_tensor`: Creates a label tensor marking 1 at peaks and 0 elsewhere.

### 6. data_preparation (Function):

- **Objective**: To split the dataset into training and testing sets and create DataLoader objects for model training.
- __Process__: Reshapes tensors, splits the data using `train_test_split`, and prepares DataLoader objects for training and testing.

### 7. train (Function):

- **Objective**: To train the CNN model with the training data.
- **Process**: Initializes the model, sets up loss function and optimizer, and iteratively updates the model weights over multiple epochs based on the loss gradient. The loss function used is the Cross-Entropy Loss, mathematically expressed as:

   $$\text{Loss} = -\sum_{c=1}^{M} y_{o,c} \log(p_{o,c})$$

   where \(M\) is the number of classes (binary in this case), \(y\) is the binary indicator (0 or 1) if class label \(c\) is the correct classification for observation \(o\), and \(p\) is the predicted probability observation \(o\) is of class \(c\).

### 8. evaluate_model (Function):

- **Objective**: To evaluate the trained model's performance on test data.
- **Process**: Tests the model on unseen data and calculates accuracy based on the number of correctly classified instances.

## Execution Flow:

1. **Preprocessing**: The script starts with preprocessing the data. It loads the `.h5` files, detects peaks, and generates labels.
2. **Data Preparation**: Next, it prepares the data for training by splitting it into training and test sets and creating DataLoader objects.
3. **Model Training**: The CNN model is then trained on the training set.
4. **Evaluation**: Finally, the trained model is evaluated on the test set to determine its accuracy.

## Conclusion:

This script provides an end-to-end solution for detecting and classifying peaks in 2D image data using deep learning. It encapsulates the entire process from data loading and preprocessing to model training and evaluation, making it a comprehensive tool for image analysis tasks involving peak detection and classification. The use of PyTorch and other scientific libraries in Python allows for efficient processing and flexibility in handling various data formats and model architectures.

***

# Support Vector Machine (SVM) Classification (`sim.py`)

The Support Vector Machine (SVM) is a supervised machine learning algorithm capable of performing classification, regression, and outlier detection. This implementation focuses on using SVM for classification purposes, particularly for classifying peaks in crystallography diffraction images.

## Methodology:

### **Data Preparation**:
The data is first flattened and standardized. Flattening is necessary as SVMs take in 1D feature vectors. Standardization (z-score normalization) ensures that each feature contributes equally to the distance calculations.

$$X_{flattened} = X.reshape(-1, 1)$$
$$z = \frac{(x - \mu)}{\sigma}$$

### **Model Training**: 
We use `LinearSVC` from scikit-learn's `svm` module for large datasets and support explicit setting of `dual=False` when the number of samples is greater than the number of features for better performance.

- The `LinearSVC` model is defined with a maximum number of iterations `max_iter=1000`. This value might need adjustment based on the specifics of the dataset.
- The `dual` parameter is set to `False` based on the warning suggesting its future change and to improve performance.

### Hyperparameter Tuning: 
Hyperparameter tyning is performed using `GridSearchCV`. This exhaustively searches over the specified parameter values for an estimator, in this case, the SVM classifier, to find the combination of paramters that results in the best performance. 

### Cross-Validation: 
Cross-Validation is used to ensure that the model's performance is robust and not dependent on the specific way the data is split. The model is trained and evaluated several times, each time with a different split of the data into training and testing sets. 

### Feature Engineering: 
Feature engineering involves creating or transforming features to improve the model's performance. This might include applying transformations like PCA to reduce dimensionality to capture more complex relationships in the data.

### Model Evaluation: 
Post-training, the model is evaluated using a test set. Key metrics provided include a classification report (precision, recall, f1-score) and a confusion matrix.

- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives. High precision relates to a low false positive rate.

    $$Precision = \frac{TP}{TP + FP}$$

- **Recall (Sensitivity)**: The ratio of correctly predicted positive observations to all observations in the actual class.

    $$Recall = \frac{TP}{TP + FN}$$

- **F1 Score**: The weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account.

    $$F1 Score = 2*\frac{Precision * Recall}{Precision + Recall}$$

- **Confusion Matrix**: A table used to describe the performance of the classification model on a set of test data for which the true values are known. It allows visualization of the performance of the algorithm.

Confusion Matrix: 
- True Negative (top left) negative samples correctly identified of not peak
- False Positive (top right) negative samples incorrectly identified as peak
- False Negative (bottom left) positive samples incorrectly identified as not peak
- True Positive (bottom right) positive samples correctly identified as peak

## Usage:
To use this SVM implementation:

1. **Load your data**: Ensure your data is in the correct format. For image data, each pixel or region can be a feature.
2. **Call `svm_classification`**: Pass your features and labels to the function. Optionally, you can downsample your data for quicker testing and development.
3. **Interpret the results**: Use the provided metrics and confusion matrix to understand the model's performance and make any necessary adjustments.

## Functions and Class Descriptions:

- `PeakThresholdProcessor`: Processes image arrays based on a threshold to identify potential peak regions.
- `ArrayRegion`: Extracts and handles specific regions from the array, typically centered around peaks.
- `load_data`: Loads image data from specified paths.
- `svm_hyperparameter_tuning`: Performs grid search to find the best parameters for the SVM model.
- `svm_cross_validation`: Evaluates the SVM model using cross-validation.
- `apply_pca`: Applies Principal Component Analysis (PCA) for dimensionality reduction.
- `svm_classification`: The main function for SVM classification, incorporating data preparation, model training, hyperparameter tuning, and evaluation.
- `downsample_data`: Optionally downsamples the data to make the model training faster and more manageable.

## Future Enhancements:

- **Expand Hyperparameter Space**: Explore a wider range of parameters and kernels in hyperparameter tuning.
- **Automated Feature Selection**: Implement automated feature selection techniques to identify the most informative features.
- **Model Interpretability**: Enhance model interpretability with techniques like feature importance scores or SHAP values.
- **Scalability**: Optimize the code for scalability and performance, especially for very large datasets.


***
