# Peakfinding using Machine Learning

Will be ongoing project to implement machine learning techniques for peakfinding in crystallography experimental imaging data.

### Install and Run

- Ensure that all of the requirements are downloaded using

```bash {"id":"01HKAN88JS7KRAF93Y0GSMWXGX"}
pip install -r requirements.txt
```

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
